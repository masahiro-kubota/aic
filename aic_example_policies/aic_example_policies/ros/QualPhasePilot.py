#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp
from transforms3d.quaternions import mat2quat, quat2mat

from aic_example_policies.ros.PublicTrialPosePilot import (
    PublicTrialPosePilot,
    SC_REPLAY_TRAJECTORY,
)
from aic_example_policies.ros.learned_port_pipeline import (
    GroundTruthPortDatasetWriter,
    LearnedPortInference,
)
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo


def _stamp_to_float(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def _pose_to_dict(pose: Pose | None) -> dict | None:
    if pose is None:
        return None
    return {
        "position": {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
        },
        "orientation": {
            "x": float(pose.orientation.x),
            "y": float(pose.orientation.y),
            "z": float(pose.orientation.z),
            "w": float(pose.orientation.w),
        },
    }


def _build_learned_runtime_aux_vector(
    current_pose: Pose,
    feature_summary: dict | None,
    step_index: int = 1,
    max_steps: int = 8,
) -> np.ndarray:
    image_shape = (
        list(feature_summary.get("image_shape", [1152, 1024]))
        if isinstance(feature_summary, dict)
        else [1152, 1024]
    )
    image_width = max(float(image_shape[0]), 1.0)
    image_height = max(float(image_shape[1]), 1.0)
    center_camera = (
        feature_summary.get("center_camera", {})
        if isinstance(feature_summary, dict)
        else {}
    )
    union_bbox = center_camera.get("union_bbox_xywh")
    if union_bbox is not None and len(union_bbox) == 4:
        x, y, width, height = [float(value) for value in union_bbox]
        center_u = x + width * 0.5
        center_v = y + height * 0.5
        feature_values = np.array(
            [
                1.0,
                (center_u / image_width) * 2.0 - 1.0,
                (center_v / image_height) * 2.0 - 1.0,
                width / image_width,
                height / image_height,
                min(float(center_camera.get("component_count", 0)) / 20.0, 1.0),
            ],
            dtype=np.float32,
        )
    else:
        feature_values = np.zeros(6, dtype=np.float32)
    step_fraction = float(max(step_index - 1, 0)) / float(max(max_steps - 1, 1))
    return np.array(
        [
            float(current_pose.position.x),
            float(current_pose.position.y),
            float(current_pose.position.z),
            float(current_pose.orientation.x),
            float(current_pose.orientation.y),
            float(current_pose.orientation.z),
            float(current_pose.orientation.w),
            *feature_values.tolist(),
            step_fraction,
        ],
        dtype=np.float32,
    )


def _wrench_to_dict(wrench_msg) -> dict:
    return {
        "force": {
            "x": float(wrench_msg.wrench.force.x),
            "y": float(wrench_msg.wrench.force.y),
            "z": float(wrench_msg.wrench.force.z),
        },
        "torque": {
            "x": float(wrench_msg.wrench.torque.x),
            "y": float(wrench_msg.wrench.torque.y),
            "z": float(wrench_msg.wrench.torque.z),
        },
    }


def _controller_state_to_dict(controller_state) -> dict:
    return {
        "tcp_pose": _pose_to_dict(controller_state.tcp_pose),
        "reference_tcp_pose": _pose_to_dict(controller_state.reference_tcp_pose),
        "tcp_error": [float(v) for v in controller_state.tcp_error],
        "tcp_velocity": {
            "linear": {
                "x": float(controller_state.tcp_velocity.linear.x),
                "y": float(controller_state.tcp_velocity.linear.y),
                "z": float(controller_state.tcp_velocity.linear.z),
            },
            "angular": {
                "x": float(controller_state.tcp_velocity.angular.x),
                "y": float(controller_state.tcp_velocity.angular.y),
                "z": float(controller_state.tcp_velocity.angular.z),
            },
        },
        "target_mode": int(controller_state.target_mode.mode),
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _image_to_bgr(image_msg) -> np.ndarray:
    image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
        image_msg.height, image_msg.width, 3
    )
    if image_msg.encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image.copy()


def _mask_components(mask: np.ndarray, min_area_px: int) -> list[dict]:
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    components = []
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if int(area) < min_area_px:
            continue
        components.append(
            {
                "area_px": int(area),
                "bbox_xywh": [int(x), int(y), int(w), int(h)],
                "centroid_uv": [float(centroids[idx][0]), float(centroids[idx][1])],
            }
        )
    components.sort(key=lambda item: item["area_px"], reverse=True)
    return components


def _union_bbox(components: list[dict]) -> list[int] | None:
    if not components:
        return None
    x0 = min(component["bbox_xywh"][0] for component in components)
    y0 = min(component["bbox_xywh"][1] for component in components)
    x1 = max(
        component["bbox_xywh"][0] + component["bbox_xywh"][2] for component in components
    )
    y1 = max(
        component["bbox_xywh"][1] + component["bbox_xywh"][3] for component in components
    )
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


@dataclass(frozen=True)
class PhaseTarget:
    entrance_position: tuple[float, float, float]
    entrance_quat_wxyz: tuple[float, float, float, float]
    port_position: tuple[float, float, float]
    port_quat_wxyz: tuple[float, float, float, float]
    push_distance_m: float


@dataclass
class PhaseEvent:
    phase: str
    sim_time_sec: float
    note: str


class DebugRun:
    def __init__(self, stage: str, task: Task):
        base_dir = Path(
            os.environ.get(
                "AIC_QUAL_DEBUG_ROOT",
                "/home/masa/ws_aic_runtime/qualification_debug",
            )
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_task_id = task.id.replace("/", "_") or "task"
        self.root = base_dir / f"{timestamp}_{stage}_{safe_task_id}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.task = task
        self.phase_events: list[PhaseEvent] = []
        self.snapshot_count = 0
        self.command_samples: list[dict] = []

    def _write_json(self, path: Path, payload: dict | list) -> None:
        path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))

    def _snapshot_dir(self, label: str) -> Path:
        self.snapshot_count += 1
        snapshot_dir = self.root / f"{self.snapshot_count:02d}_{label}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        return snapshot_dir

    def log_phase(self, phase: str, sim_time_sec: float, note: str = "") -> None:
        self.phase_events.append(
            PhaseEvent(phase=phase, sim_time_sec=sim_time_sec, note=note)
        )
        self._write_json(
            self.root / "phase_timeline.json",
            {"events": [asdict(event) for event in self.phase_events]},
        )

    def save_target_metadata(self, payload: dict) -> None:
        self._write_json(self.root / "target_metadata.json", payload)

    def log_command_sample(
        self,
        phase: str,
        pose: Pose,
        sim_time_sec: float,
        note: str = "",
    ) -> None:
        self.command_samples.append(
            {
                "phase": phase,
                "sim_time_sec": float(sim_time_sec),
                "note": note,
                "pose": _pose_to_dict(pose),
            }
        )
        self._write_json(self.root / "command_samples.json", self.command_samples)

    def save_observation_snapshot(
        self, label: str, observation: Observation | None, extra: dict | None = None
    ) -> None:
        snapshot_dir = self._snapshot_dir(label)
        payload = {"label": label, "stage": self.stage, "extra": extra or {}}
        if observation is None:
            payload["observation"] = None
            self._write_json(snapshot_dir / "metadata.json", payload)
            return

        left = _image_to_bgr(observation.left_image)
        center = _image_to_bgr(observation.center_image)
        right = _image_to_bgr(observation.right_image)

        cv2.imwrite(str(snapshot_dir / "left.png"), left)
        cv2.imwrite(str(snapshot_dir / "center.png"), center)
        cv2.imwrite(str(snapshot_dir / "right.png"), right)

        rendered = []
        for name, image in [("left", left), ("center", center), ("right", right)]:
            canvas = image.copy()
            cv2.putText(
                canvas,
                name,
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
            rendered.append(canvas)
        montage = cv2.hconcat(rendered)
        cv2.imwrite(str(snapshot_dir / "montage.png"), montage)

        payload["observation"] = {
            "left_stamp_sec": _stamp_to_float(observation.left_image.header.stamp),
            "center_stamp_sec": _stamp_to_float(observation.center_image.header.stamp),
            "right_stamp_sec": _stamp_to_float(observation.right_image.header.stamp),
            "controller_state": _controller_state_to_dict(
                observation.controller_state
            ),
            "wrist_wrench": _wrench_to_dict(observation.wrist_wrench),
            "camera_info": {
                "center": {
                    "width": int(observation.center_camera_info.width),
                    "height": int(observation.center_camera_info.height),
                    "k": [float(v) for v in observation.center_camera_info.k],
                }
            },
        }
        self._write_json(snapshot_dir / "metadata.json", payload)

    def finalize(
        self,
        success: bool,
        note: str,
        final_observation: Observation | None,
        extra: dict | None = None,
    ) -> None:
        if final_observation is not None:
            self.save_observation_snapshot("final", final_observation, extra or {"note": note})
        self._write_json(
            self.root / "summary.json",
            {
                "stage": self.stage,
                "task": {
                    "id": self.task.id,
                    "plug_type": self.task.plug_type,
                    "plug_name": self.task.plug_name,
                    "port_type": self.task.port_type,
                    "port_name": self.task.port_name,
                    "target_module_name": self.task.target_module_name,
                    "time_limit": int(self.task.time_limit),
                },
                "success": bool(success),
                "note": note,
                "phase_event_count": len(self.phase_events),
                "command_sample_count": len(self.command_samples),
            },
        )


class QualPhasePilot(Policy):
    _CENTER_CAMERA_FRAME = "center_camera/optical"
    _CAMERA_FRAMES = {
        "left": "left_camera/optical",
        "center": "center_camera/optical",
        "right": "right_camera/optical",
    }
    _SFP_VISUAL_TEMPLATES = {
        "nic_card_mount_0": {
            "union_center_uv": (538.0, 565.0),
            "union_size_px": (306.0, 332.0),
            "camera_depth_m": 0.18,
            "nominal_quat_wxyz": (0.184036, 0.9825145, -0.0277635, -0.0049785),
        },
        "nic_card_mount_1": {
            "union_center_uv": (539.0, 454.0),
            "union_size_px": (270.0, 252.0),
            "camera_depth_m": 0.22,
            "nominal_quat_wxyz": (0.183998, 0.982521, -0.027775, -0.005039),
        },
    }
    _SFP_MULTI_CAMERA_TEMPLATES = {
        "nic_card_mount_0": {
            "left": {"union_center_uv": (631.5, 573.0), "union_size_px": (245.0, 388.0)},
            "center": {
                "union_center_uv": (537.5, 569.5),
                "union_size_px": (307.0, 333.0),
            },
            "right": {
                "union_center_uv": (475.0, 572.5),
                "union_size_px": (216.0, 237.0),
            },
        },
        "nic_card_mount_1": {
            "left": {"union_center_uv": (633.0, 573.5), "union_size_px": (248.0, 387.0)},
            "center": {
                "union_center_uv": (537.0, 573.5),
                "union_size_px": (312.0, 339.0),
            },
            "right": {
                "union_center_uv": (473.0, 577.0),
                "union_size_px": (218.0, 242.0),
            },
        },
    }
    _SC_CENTER_RESIDUAL_TEMPLATE = {
        "cyan_centroid_uv": (626.4, 759.0),
        "nominal_depth_m": 0.055,
    }
    _SC_SUBMISSION_COARSE_TEMPLATE = {
        "cyan_centroid_uv": (520.0, 620.0),
        "desired_area_px": 5500.0,
        "nominal_depth_m": 0.085,
    }
    _SC_SUBMISSION_FINE_TEMPLATE = {
        "cyan_centroid_uv": (626.4, 759.0),
        "desired_area_px": 8500.0,
        "nominal_depth_m": 0.06,
    }
    _SC_LEARNED_HOVER_TEMPLATE = {
        "cyan_centroid_uv": (570.7, 344.1),
        "nominal_depth_m": 0.402,
    }
    _SC_LEARNED_INSERT_MID_TEMPLATE = {
        "cyan_centroid_uv": (573.8, 518.9),
        "nominal_depth_m": 0.291,
    }
    _SC_LEARNED_INSERT_FINAL_TEMPLATE = {
        "cyan_centroid_uv": (572.8, 727.6),
        "nominal_depth_m": 0.241,
    }
    _SC_LEARNED_PRIMITIVE_SEGMENTS_BASE = (
        (-0.0269, 0.0057, -0.0732),
        (-0.0272, 0.0082, -0.0605),
    )
    _SC_MAGENTA_REFERENCE_ANGLE_DEG = 8.4
    _SC_NOMINAL_QUAT_WXYZ = (0.337997, 0.662025, 0.668874, 0.009412)
    _SFP_TOOL_INSERTION_AXIS = (
        0.000159431701,
        -0.350186974,
        0.936679805,
    )
    _SFP_POST_SERVO_INSERT_TRANSLATION_DELTAS_BASE = {
        "nic_card_mount_0": (0.000728, 0.031514, -0.117908),
        "nic_card_mount_1": (0.000797, 0.067410, -0.113470),
    }
    _SFP_POST_SERVO_INSERT_ROTATION_DELTA_ROTVEC_BASE = (
        -0.37067,
        0.01032,
        -0.05592,
    )
    _SFP_POST_SERVO_INSERT_SEGMENT_FRACTIONS = (0.34, 0.33, 0.33)
    _SC_TOOL_INSERTION_AXIS = (
        0.43965921,
        -0.45945677,
        0.77175077,
    )
    _SFP_INSERTION_VECTOR_BASE = (0.0, -0.000579, -0.045797)
    _DEV_TARGETS = {
        ("sfp", "nic_card_mount_0", "sfp_port_0"): PhaseTarget(
            entrance_position=(-0.384371, 0.213751, 0.237210),
            entrance_quat_wxyz=(0.184074, 0.982508, -0.027752, -0.004918),
            port_position=(-0.384371, 0.213172, 0.191413),
            port_quat_wxyz=(0.184074, 0.982508, -0.027752, -0.004918),
            push_distance_m=0.015,
        ),
        ("sfp", "nic_card_mount_1", "sfp_port_0"): PhaseTarget(
            entrance_position=(-0.384399, 0.252639, 0.234351),
            entrance_quat_wxyz=(0.183998, 0.982521, -0.027775, -0.005039),
            port_position=(-0.384399, 0.252060, 0.188555),
            port_quat_wxyz=(0.183998, 0.982521, -0.027775, -0.005039),
            push_distance_m=0.015,
        ),
        ("sc", "sc_port_1", "sc_port_base"): PhaseTarget(
            entrance_position=(-0.482564, 0.292170, 0.067884),
            entrance_quat_wxyz=(0.337997, 0.662025, 0.668874, 0.009412),
            port_position=(-0.482552, 0.292168, 0.052244),
            port_quat_wxyz=(0.337997, 0.662025, 0.668874, 0.009412),
            push_distance_m=0.008,
        ),
    }

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._stage = os.environ.get("AIC_QUAL_STAGE", "m0")
        self._hold_duration_sec = float(os.environ.get("AIC_QUAL_M0_HOLD_SEC", "3.0"))
        self._approach_duration_sec = float(
            os.environ.get("AIC_QUAL_APPROACH_SEC", "2.5")
        )
        self._entrance_hold_sec = float(
            os.environ.get("AIC_QUAL_ENTRANCE_HOLD_SEC", "0.5")
        )
        self._align_duration_sec = float(
            os.environ.get("AIC_QUAL_ALIGN_SEC", "2.5")
        )
        self._align_hold_sec = float(
            os.environ.get("AIC_QUAL_ALIGN_HOLD_SEC", "0.5")
        )
        self._insert_duration_sec = float(
            os.environ.get("AIC_QUAL_INSERT_SEC", "1.0")
        )
        self._final_hold_sec = float(
            os.environ.get("AIC_QUAL_FINAL_HOLD_SEC", "2.5")
        )
        self._command_period_sec = float(
            os.environ.get("AIC_QUAL_COMMAND_PERIOD_SEC", "0.02")
        )
        self._learned_dataset_root = os.environ.get("AIC_LEARNED_PORT_DATASET_ROOT")
        self._learned_collection_split = os.environ.get(
            "AIC_LEARNED_PORT_DATASET_SPLIT", "train"
        )
        self._learned_sc_model_dir = os.environ.get("AIC_QUAL_LEARNED_SC_MODEL_DIR")
        self._learned_sfp_model_dir = os.environ.get("AIC_QUAL_LEARNED_SFP_MODEL_DIR")
        self._learned_model_dir = self._learned_sc_model_dir
        self._enable_sc_refinement = (
            os.environ.get("AIC_QUAL_ENABLE_SC_REFINEMENT", "false").lower() == "true"
        )
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._learned_dataset_writer: GroundTruthPortDatasetWriter | None = None
        self._learned_port_inference_by_plug: dict[str, LearnedPortInference] = {}
        self._public_trial_pilot = PublicTrialPosePilot(parent_node)
        self.get_logger().info(f"QualPhasePilot.__init__() stage={self._stage}")

    def _task_key(self, task: Task) -> tuple[str, str, str]:
        return (task.plug_type, task.target_module_name, task.port_name)

    def _sfp_visual_template_for_task(self, task: Task) -> dict:
        return self._SFP_VISUAL_TEMPLATES.get(
            task.target_module_name,
            self._SFP_VISUAL_TEMPLATES["nic_card_mount_0"],
        )

    def _quat_xyzw_to_wxyz(self, quat: Quaternion) -> tuple[float, float, float, float]:
        return (quat.w, quat.x, quat.y, quat.z)

    def _quat_wxyz_to_xyzw(
        self, quat: tuple[float, float, float, float]
    ) -> Quaternion:
        return Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0])

    def _interpolate_pose(self, start: Pose, end: Pose, fraction: float) -> Pose:
        start_pos = np.array(
            [start.position.x, start.position.y, start.position.z], dtype=float
        )
        end_pos = np.array(
            [end.position.x, end.position.y, end.position.z], dtype=float
        )
        xyz = (1.0 - fraction) * start_pos + fraction * end_pos
        q = quaternion_slerp(
            self._quat_xyzw_to_wxyz(start.orientation),
            self._quat_xyzw_to_wxyz(end.orientation),
            fraction,
        )
        quat_wxyz = (
            self._quat_xyzw_to_wxyz(end.orientation)
            if q is None
            else (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        )
        return Pose(
            position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
            orientation=self._quat_wxyz_to_xyzw(quat_wxyz),
        )

    def _pose_from_target(
        self,
        position: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
    ) -> Pose:
        return Pose(
            position=Point(x=position[0], y=position[1], z=position[2]),
            orientation=self._quat_wxyz_to_xyzw(quat_wxyz),
        )

    def _push_pose(self, target: PhaseTarget) -> Pose:
        entrance = np.array(target.entrance_position, dtype=float)
        port = np.array(target.port_position, dtype=float)
        insertion_axis = port - entrance
        insertion_axis /= max(np.linalg.norm(insertion_axis), 1e-6)
        pushed = port + target.push_distance_m * insertion_axis
        return Pose(
            position=Point(x=float(pushed[0]), y=float(pushed[1]), z=float(pushed[2])),
            orientation=self._quat_wxyz_to_xyzw(target.port_quat_wxyz),
        )

    def _wait_for_observation(
        self,
        get_observation: GetObservationCallback,
        timeout_sec: float = 5.0,
        newer_than_sec: float | None = None,
    ) -> Observation | None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            observation = get_observation()
            if observation is None:
                time.sleep(0.05)
                continue
            if newer_than_sec is None:
                return observation
            observation_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)
            if observation_stamp_sec > newer_than_sec + 1e-6:
                return observation
            # Use wall-time polling here. If sim time stalls or the camera timestamp
            # stops updating, a sim-time sleep can deadlock the whole task callback.
            self._wait_for_sim_progress(0.05)
        return None

    def _wait_for_sim_progress(
        self,
        duration_sec: float,
        wall_timeout_scale: float = 6.0,
        min_wall_timeout_sec: float = 0.25,
        poll_period_sec: float = 0.01,
    ) -> bool:
        if duration_sec <= 0.0:
            return True
        start_sim_sec = self.time_now().nanoseconds / 1e9
        deadline = time.monotonic() + max(
            min_wall_timeout_sec, duration_sec * wall_timeout_scale
        )
        while time.monotonic() < deadline:
            sim_elapsed_sec = self.time_now().nanoseconds / 1e9 - start_sim_sec
            if sim_elapsed_sec + 1e-4 >= duration_sec:
                return True
            time.sleep(poll_period_sec)
        return False

    def _hold_pose(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        duration_sec: float,
        debug_run: DebugRun | None = None,
        phase_name: str = "hold",
    ) -> None:
        steps = max(1, int(duration_sec / self._command_period_sec))
        for step in range(steps):
            self.set_pose_target(move_robot=move_robot, pose=pose)
            if debug_run is not None and (step == 0 or step == steps - 1):
                debug_run.log_command_sample(
                    phase_name,
                    pose,
                    self.time_now().nanoseconds / 1e9,
                    note=f"hold step {step + 1}/{steps}",
                )
            self._wait_for_sim_progress(self._command_period_sec)

    def _move_for_duration(
        self,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
        end_pose: Pose,
        duration_sec: float,
        debug_run: DebugRun | None = None,
        phase_name: str = "move",
    ) -> None:
        steps = max(1, int(duration_sec / self._command_period_sec))
        log_stride = max(1, steps // 4)
        for step in range(1, steps + 1):
            fraction = step / steps
            target_pose = self._interpolate_pose(start_pose, end_pose, fraction)
            self.set_pose_target(move_robot=move_robot, pose=target_pose)
            if debug_run is not None and (
                step == 1 or step == steps or step % log_stride == 0
            ):
                debug_run.log_command_sample(
                    phase_name,
                    target_pose,
                    self.time_now().nanoseconds / 1e9,
                    note=f"interp {step}/{steps}",
                )
            self._wait_for_sim_progress(self._command_period_sec)

    def _extract_feature_summary(
        self, task: Task, observation: Observation | None
    ) -> dict:
        if observation is None:
            return {"available": False}

        center = _image_to_bgr(observation.center_image)
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        summary = {
            "available": True,
            "task_key": list(self._task_key(task)),
            "image_shape": [int(center.shape[1]), int(center.shape[0])],
        }

        if task.plug_type == "sfp":
            green_mask = cv2.inRange(
                hsv,
                np.array([40, 40, 20], dtype=np.uint8),
                np.array([90, 255, 255], dtype=np.uint8),
            )
            green_components = _mask_components(green_mask, min_area_px=80)
            green_components.sort(key=lambda item: item["centroid_uv"][0])
            summary["center_camera"] = {
                "detector": "green_mask",
                "component_count": len(green_components),
                "components": green_components,
                "union_bbox_xywh": _union_bbox(green_components),
            }
        elif task.plug_type == "sc":
            cyan_mask = cv2.inRange(
                hsv,
                np.array([80, 80, 80], dtype=np.uint8),
                np.array([110, 255, 255], dtype=np.uint8),
            )
            magenta_mask = cv2.inRange(
                hsv,
                np.array([130, 80, 80], dtype=np.uint8),
                np.array([170, 255, 255], dtype=np.uint8),
            )
            cyan_components = _mask_components(cyan_mask, min_area_px=80)
            magenta_components = _mask_components(magenta_mask, min_area_px=80)
            summary["center_camera"] = {
                "detector": "cyan_magenta_masks",
                "cyan_components": cyan_components,
                "magenta_components": magenta_components,
                "cyan_union_bbox_xywh": _union_bbox(cyan_components),
                "magenta_union_bbox_xywh": _union_bbox(magenta_components),
            }
        else:
            summary["center_camera"] = {"detector": "none"}
        return summary

    def _lookup_camera_rotation_base_from_optical(
        self, camera_frame: str
    ) -> np.ndarray | None:
        pose = self._lookup_camera_pose_base_from_optical(camera_frame)
        if pose is None:
            return None
        rotation_base_from_optical, _ = pose
        return rotation_base_from_optical

    def _lookup_camera_pose_base_from_optical(
        self, camera_frame: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            transform = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                camera_frame,
                Time(),
            )
        except Exception as ex:
            self.get_logger().warn(f"{camera_frame} TF lookup failed: {ex}")
            return None
        quat_wxyz = (
            transform.transform.rotation.w,
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
        )
        translation_base = np.array(
            [
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
                float(transform.transform.translation.z),
            ],
            dtype=float,
        )
        return quat2mat(quat_wxyz), translation_base

    def _lookup_frame_pose_base(
        self, frame_name: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            transform = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                frame_name,
                Time(),
            )
        except Exception as ex:
            self.get_logger().warn(f"{frame_name} TF lookup failed: {ex}")
            return None
        quat_wxyz = (
            transform.transform.rotation.w,
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
        )
        translation_base = np.array(
            [
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
                float(transform.transform.translation.z),
            ],
            dtype=float,
        )
        return quat2mat(quat_wxyz), translation_base

    def _lookup_frame_transform_base(
        self, frame_name: str
    ) -> tuple[tuple[float, float, float, float], np.ndarray] | None:
        try:
            transform = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                frame_name,
                Time(),
            )
        except Exception as ex:
            self.get_logger().warn(f"{frame_name} TF lookup failed: {ex}")
            return None
        quat_wxyz = (
            float(transform.transform.rotation.w),
            float(transform.transform.rotation.x),
            float(transform.transform.rotation.y),
            float(transform.transform.rotation.z),
        )
        translation_base = np.array(
            [
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
                float(transform.transform.translation.z),
            ],
            dtype=float,
        )
        return quat_wxyz, translation_base

    def _port_frame_name(self, task: Task) -> str:
        return f"task_board/{task.target_module_name}/{task.port_name}_link"

    def _project_point_base_to_camera(
        self,
        point_base: np.ndarray,
        camera_name: str,
        camera_info: CameraInfo,
    ) -> dict | None:
        camera_pose = self._lookup_camera_pose_base_from_optical(
            self._CAMERA_FRAMES[camera_name]
        )
        if camera_pose is None:
            return None
        rotation_base_from_optical, translation_base = camera_pose
        point_optical = rotation_base_from_optical.T @ (point_base - translation_base)
        if float(point_optical[2]) <= 1e-6:
            return None
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        u = fx * float(point_optical[0]) / float(point_optical[2]) + cx
        v = fy * float(point_optical[1]) / float(point_optical[2]) + cy
        return {
            "uv": [float(u), float(v)],
            "point_optical": [float(value) for value in point_optical],
            "visible": bool(
                0.0 <= u < float(camera_info.width) and 0.0 <= v < float(camera_info.height)
            ),
        }

    def _learned_dataset_writer_or_none(self) -> GroundTruthPortDatasetWriter | None:
        if self._learned_dataset_root is None:
            return None
        if self._learned_dataset_writer is None:
            self._learned_dataset_writer = GroundTruthPortDatasetWriter(
                self._learned_dataset_root,
                split_name=self._learned_collection_split,
            )
        return self._learned_dataset_writer

    def _learned_port_inference_or_none(
        self, plug_type: str = "sc"
    ) -> LearnedPortInference | None:
        model_dir = (
            self._learned_sc_model_dir
            if plug_type == "sc"
            else self._learned_sfp_model_dir
        )
        if model_dir is None:
            return None
        if plug_type not in self._learned_port_inference_by_plug:
            self._learned_port_inference_by_plug[plug_type] = LearnedPortInference.load(
                model_dir
            )
        return self._learned_port_inference_by_plug[plug_type]

    def _learned_phase_template_uvz(
        self,
        plug_type: str,
        task: Task,
        template_name: str,
        fallback_uvz: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        estimator = self._learned_port_inference_or_none(plug_type)
        if estimator is None:
            return fallback_uvz
        task_key = "|".join(self._task_key(task))
        manifest = getattr(estimator, "manifest", {})
        task_templates = manifest.get("phase_templates", {}).get(task_key, {})
        template = task_templates.get(template_name)
        if template is None:
            return fallback_uvz
        center_uvz = template.get("center_uvz_median")
        if not isinstance(center_uvz, list) or len(center_uvz) != 3:
            return fallback_uvz
        return (
            float(center_uvz[0]),
            float(center_uvz[1]),
            float(center_uvz[2]),
        )

    def _capture_gt_port_sample(
        self,
        task: Task,
        observation: Observation | None,
        phase: str,
        extra: dict | None = None,
        label_overrides: dict | None = None,
    ) -> dict | None:
        writer = self._learned_dataset_writer_or_none()
        if writer is None or observation is None:
            return None
        port_pose = self._lookup_frame_pose_base(self._port_frame_name(task))
        if port_pose is None:
            return None
        teacher_hover_pose = self._gt_teacher_gripper_pose(
            task,
            self._port_frame_name(task),
            slerp_fraction=1.0,
            position_fraction=1.0,
            z_offset=0.16 if task.plug_type == "sfp" else 0.18,
            update_integrator=False,
        )
        teacher_insert_pose = self._gt_teacher_gripper_pose(
            task,
            self._port_frame_name(task),
            slerp_fraction=1.0,
            position_fraction=1.0,
            z_offset=0.08 if task.plug_type == "sfp" else 0.11,
            update_integrator=False,
        )
        _, target_point_base = port_pose
        per_camera: dict[str, dict] = {}
        images_bgr: dict[str, np.ndarray] = {}
        camera_infos: dict[str, dict] = {}
        for camera_name in ("left", "center", "right"):
            image_bgr, camera_info = self._camera_view(observation, camera_name)
            if image_bgr is None or camera_info is None:
                return None
            projected = self._project_point_base_to_camera(
                target_point_base, camera_name, camera_info
            )
            if projected is None:
                return None
            per_camera[camera_name] = projected
            images_bgr[camera_name] = image_bgr
            camera_infos[camera_name] = {
                "width": int(camera_info.width),
                "height": int(camera_info.height),
                "k": [float(value) for value in camera_info.k],
            }
        observation_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)
        labels = {
            "port_frame": self._port_frame_name(task),
            "target_point_base": [float(value) for value in target_point_base],
            "current_tcp_pose": _pose_to_dict(observation.controller_state.tcp_pose),
            "teacher_hover_pose": _pose_to_dict(teacher_hover_pose),
            "teacher_insert_pose": _pose_to_dict(teacher_insert_pose),
            "per_camera": per_camera,
            "camera_info": camera_infos,
        }
        if label_overrides:
            labels.update(label_overrides)
        return writer.append_sample(
            task=task,
            stage=self._stage,
            phase=phase,
            observation_stamp_sec=observation_stamp_sec,
            images_bgr=images_bgr,
            labels=labels,
            extra=extra,
        )

    def _gt_teacher_gripper_pose(
        self,
        task: Task,
        port_frame: str,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
        update_integrator: bool = True,
    ) -> Pose | None:
        port_tf = self._lookup_frame_transform_base(port_frame)
        plug_tf = self._lookup_frame_transform_base(
            f"{task.cable_name}/{task.plug_name}_link"
        )
        gripper_tf = self._lookup_frame_transform_base("gripper/tcp")
        if port_tf is None or plug_tf is None or gripper_tf is None:
            return None

        q_port, port_xyz = port_tf
        q_plug, plug_xyz = plug_tf
        q_gripper, gripper_xyz = gripper_tf

        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)
        if q_gripper_slerp is None:
            return None

        tip_x_error = float(port_xyz[0] - plug_xyz[0])
        tip_y_error = float(port_xyz[1] - plug_xyz[1])
        integral_x = 0.0
        integral_y = 0.0
        if update_integrator:
            if reset_xy_integrator:
                self._tip_x_error_integrator = 0.0
                self._tip_y_error_integrator = 0.0
            else:
                self._tip_x_error_integrator = float(
                    np.clip(
                        self._tip_x_error_integrator + tip_x_error,
                        -self._max_integrator_windup,
                        self._max_integrator_windup,
                    )
                )
                self._tip_y_error_integrator = float(
                    np.clip(
                        self._tip_y_error_integrator + tip_y_error,
                        -self._max_integrator_windup,
                        self._max_integrator_windup,
                    )
                )
            integral_x = self._tip_x_error_integrator
            integral_y = self._tip_y_error_integrator
        elif reset_xy_integrator:
            integral_x = 0.0
            integral_y = 0.0
        else:
            integral_x = 0.0
            integral_y = 0.0

        i_gain = 0.15
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )
        target_x = port_xyz[0] + i_gain * integral_x
        target_y = port_xyz[1] + i_gain * integral_y
        target_z = port_xyz[2] + z_offset - plug_tip_gripper_offset[2]
        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )
        return Pose(
            position=Point(
                x=float(blend_xyz[0]),
                y=float(blend_xyz[1]),
                z=float(blend_xyz[2]),
            ),
            orientation=Quaternion(
                w=float(q_gripper_slerp[0]),
                x=float(q_gripper_slerp[1]),
                y=float(q_gripper_slerp[2]),
                z=float(q_gripper_slerp[3]),
            ),
        )

    def _camera_view(
        self, observation: Observation | None, camera_name: str
    ) -> tuple[np.ndarray, CameraInfo] | tuple[None, None]:
        if observation is None:
            return None, None
        if camera_name == "left":
            return _image_to_bgr(observation.left_image), observation.left_camera_info
        if camera_name == "right":
            return _image_to_bgr(observation.right_image), observation.right_camera_info
        return _image_to_bgr(observation.center_image), observation.center_camera_info

    def _sc_feature_for_camera(
        self, observation: Observation | None, camera_name: str
    ) -> dict | None:
        if observation is None:
            return None
        image, _ = self._camera_view(observation, camera_name)
        if image is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(
            hsv,
            np.array([80, 80, 80], dtype=np.uint8),
            np.array([110, 255, 255], dtype=np.uint8),
        )
        cyan_components = _mask_components(cyan_mask, min_area_px=80)
        magenta_mask = cv2.inRange(
            hsv,
            np.array([130, 80, 80], dtype=np.uint8),
            np.array([170, 255, 255], dtype=np.uint8),
        )
        magenta_components = _mask_components(magenta_mask, min_area_px=80)
        if not cyan_components:
            return None
        largest = max(cyan_components, key=lambda item: item["area_px"])
        magenta_angle_deg = None
        if magenta_components:
            ys, xs = np.nonzero(magenta_mask)
            if len(xs) >= 5:
                rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
                (_, _), (w, h), angle_deg = rect
                if w < h:
                    angle_deg += 90.0
                magenta_angle_deg = float(angle_deg)
        return {
            "camera_name": camera_name,
            "cyan_components": cyan_components,
            "cyan_centroid_uv": tuple(largest["centroid_uv"]),
            "cyan_bbox_xywh": tuple(largest["bbox_xywh"]),
            "cyan_area_px": int(largest["area_px"]),
            "magenta_components": magenta_components,
            "magenta_union_bbox_xywh": _union_bbox(magenta_components),
            "magenta_angle_deg": magenta_angle_deg,
        }

    def _sfp_union_feature_for_camera(
        self, observation: Observation | None, camera_name: str
    ) -> dict | None:
        if observation is None:
            return None
        image, _ = self._camera_view(observation, camera_name)
        if image is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(
            hsv,
            np.array([40, 40, 20], dtype=np.uint8),
            np.array([90, 255, 255], dtype=np.uint8),
        )
        components = _mask_components(green_mask, min_area_px=80)
        components.sort(key=lambda item: item["centroid_uv"][0])
        union_bbox = _union_bbox(components)
        feature_summary = {
            "available": True,
            "center_camera": {
                "detector": "green_mask",
                "component_count": len(components),
                "components": components,
                "union_bbox_xywh": union_bbox,
            },
        }
        if union_bbox is None or not components:
            return None
        x, y, w, h = union_bbox
        largest = max(components, key=lambda item: item["area_px"])
        return {
            "feature_summary": feature_summary,
            "union_bbox_xywh": union_bbox,
            "union_center_uv": (float(x + w / 2.0), float(y + h / 2.0)),
            "union_size_px": (float(w), float(h)),
            "largest_component": largest,
        }

    def _sfp_union_feature(self, observation: Observation | None) -> dict | None:
        return self._sfp_union_feature_for_camera(observation, "center")

    def _sfp_multi_camera_template_for_task(self, task: Task) -> dict:
        return self._SFP_MULTI_CAMERA_TEMPLATES.get(
            task.target_module_name,
            self._SFP_MULTI_CAMERA_TEMPLATES["nic_card_mount_0"],
        )

    def _sc_center_feature(self, observation: Observation | None) -> dict | None:
        return self._sc_feature_for_camera(observation, "center")

    def _pixel_ray_optical(
        self,
        camera_info: CameraInfo,
        uv: tuple[float, float],
    ) -> np.ndarray | None:
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        if fx <= 1e-6 or fy <= 1e-6:
            return None
        u, v = uv
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
        norm = np.linalg.norm(ray)
        if norm <= 1e-6:
            return None
        return ray / norm

    def _camera_ray_base(
        self,
        observation: Observation | None,
        camera_name: str,
        uv: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if observation is None:
            return None
        _, camera_info = self._camera_view(observation, camera_name)
        if camera_info is None:
            return None
        ray_optical = self._pixel_ray_optical(camera_info, uv)
        if ray_optical is None:
            return None
        camera_pose = self._lookup_camera_pose_base_from_optical(
            self._CAMERA_FRAMES[camera_name]
        )
        if camera_pose is None:
            return None
        rotation_base_from_optical, translation_base = camera_pose
        ray_base = rotation_base_from_optical @ ray_optical
        ray_norm = np.linalg.norm(ray_base)
        if ray_norm <= 1e-6:
            return None
        return translation_base, ray_base / ray_norm

    def _triangulate_point_from_rays(
        self,
        rays: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray | None:
        if len(rays) < 2:
            return None
        lhs = np.zeros((3, 3), dtype=float)
        rhs = np.zeros(3, dtype=float)
        for origin, direction in rays:
            direction = direction / max(np.linalg.norm(direction), 1e-6)
            projector = np.eye(3, dtype=float) - np.outer(direction, direction)
            lhs += projector
            rhs += projector @ origin
        try:
            solution, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(solution)):
            return None
        return solution

    def _sc_triangulated_feature(self, observation: Observation | None) -> dict | None:
        if observation is None:
            return None
        per_camera: dict[str, dict] = {}
        rays: list[tuple[np.ndarray, np.ndarray]] = []
        for camera_name in ("left", "center", "right"):
            feature = self._sc_feature_for_camera(observation, camera_name)
            if feature is None:
                continue
            per_camera[camera_name] = feature
            ray = self._camera_ray_base(
                observation,
                camera_name,
                feature["cyan_centroid_uv"],
            )
            if ray is not None:
                rays.append(ray)
        triangulated_point_base = self._triangulate_point_from_rays(rays)
        if triangulated_point_base is None or "center" not in per_camera:
            return None

        center_pose = self._lookup_camera_pose_base_from_optical(self._CENTER_CAMERA_FRAME)
        if center_pose is None:
            return None
        rotation_base_from_optical, translation_base = center_pose
        point_center_optical = rotation_base_from_optical.T @ (
            triangulated_point_base - translation_base
        )
        center_feature = per_camera["center"]
        return {
            "triangulated_point_base": triangulated_point_base,
            "point_center_optical": point_center_optical,
            "camera_features": per_camera,
            "center_feature": center_feature,
        }

    def _sc_learned_feature(
        self,
        task: Task,
        observation: Observation | None,
        *,
        step_index: int = 1,
        max_steps: int = 1,
    ) -> dict | None:
        if observation is None:
            return None
        estimator = self._learned_port_inference_or_none("sc")
        if estimator is None:
            return None

        feature_summary = self._extract_feature_summary(task, observation)
        aux_vector = _build_learned_runtime_aux_vector(
            current_pose=observation.controller_state.tcp_pose,
            feature_summary=feature_summary,
            step_index=step_index,
            max_steps=max_steps,
        )

        left_image = _image_to_bgr(observation.left_image)
        center_image = _image_to_bgr(observation.center_image)
        right_image = _image_to_bgr(observation.right_image)
        predicted = cast(Any, estimator).predict_center_uvz(
            task=task,
            images_bgr={
                "left": left_image,
                "center": center_image,
                "right": right_image,
            },
            center_camera_k=[float(value) for value in observation.center_camera_info.k],
            aux_vector=aux_vector,
        )

        raw_center_feature = self._sc_feature_for_camera(observation, "center")
        predicted_center_uvz = np.array(predicted["center_uvz"], dtype=float)
        center_uv = tuple(predicted_center_uvz[:2])
        magenta_angle_deg = None
        if raw_center_feature is not None:
            center_uv = raw_center_feature["cyan_centroid_uv"]
            magenta_angle_deg = raw_center_feature.get("magenta_angle_deg")

        center_pose = self._lookup_camera_pose_base_from_optical(self._CENTER_CAMERA_FRAME)
        if center_pose is None:
            return None
        rotation_base_from_optical, translation_base = center_pose
        point_center_optical = self._desired_point_in_camera(
            observation.center_camera_info,
            center_uv,
            float(predicted_center_uvz[2]),
        )
        if point_center_optical is None:
            return None
        point_base = translation_base + (rotation_base_from_optical @ point_center_optical)
        return {
            "source": (
                "raw_center_uv_plus_learned_depth"
                if raw_center_feature is not None
                else "learned_center_uvz"
            ),
            "triangulated_point_base": point_base,
            "point_center_optical": point_center_optical,
            "center_feature": {
                "camera_name": "center",
                "cyan_centroid_uv": center_uv,
                "cyan_area_px": (
                    None
                    if raw_center_feature is None
                    else raw_center_feature.get("cyan_area_px")
                ),
                "magenta_angle_deg": magenta_angle_deg,
            },
            "predicted_center_uvz": predicted["center_uvz"],
            "raw_center_feature": raw_center_feature,
        }

    def _sfp_learned_feature(
        self,
        task: Task,
        observation: Observation | None,
        *,
        step_index: int = 1,
        max_steps: int = 1,
    ) -> dict | None:
        if observation is None:
            return None
        estimator = self._learned_port_inference_or_none("sfp")
        if estimator is None:
            return None

        feature_summary = self._extract_feature_summary(task, observation)
        aux_vector = _build_learned_runtime_aux_vector(
            current_pose=observation.controller_state.tcp_pose,
            feature_summary=feature_summary,
            step_index=step_index,
            max_steps=max_steps,
        )

        predicted = cast(Any, estimator).predict_center_uvz(
            task=task,
            images_bgr={
                "left": _image_to_bgr(observation.left_image),
                "center": _image_to_bgr(observation.center_image),
                "right": _image_to_bgr(observation.right_image),
            },
            center_camera_k=[float(value) for value in observation.center_camera_info.k],
            aux_vector=aux_vector,
        )
        predicted_center_uvz = np.array(predicted["center_uvz"], dtype=float)
        desired_uv = (float(predicted_center_uvz[0]), float(predicted_center_uvz[1]))
        point_center_optical = self._desired_point_in_camera(
            observation.center_camera_info,
            desired_uv,
            float(predicted_center_uvz[2]),
        )
        if point_center_optical is None:
            return None
        center_pose = self._lookup_camera_pose_base_from_optical(self._CENTER_CAMERA_FRAME)
        if center_pose is None:
            return None
        rotation_base_from_optical, translation_base = center_pose
        point_base = translation_base + (rotation_base_from_optical @ point_center_optical)
        raw_center_feature = self._sfp_union_feature(observation)
        return {
            "source": "learned_center_uvz",
            "triangulated_point_base": point_base,
            "point_center_optical": point_center_optical,
            "center_feature": {
                "union_center_uv": desired_uv,
                "union_size_px": (
                    None
                    if raw_center_feature is None
                    else raw_center_feature.get("union_size_px")
                ),
            },
            "predicted_center_uvz": [float(value) for value in predicted_center_uvz],
            "raw_center_feature": raw_center_feature,
        }

    def _desired_point_in_camera(
        self,
        camera_info: CameraInfo,
        desired_uv: tuple[float, float],
        desired_depth_m: float,
    ) -> np.ndarray | None:
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        if fx <= 1e-6 or fy <= 1e-6 or desired_depth_m <= 1e-6:
            return None
        u, v = desired_uv
        z = float(desired_depth_m)
        return np.array(
            [
                (float(u) - cx) * z / fx,
                (float(v) - cy) * z / fy,
                z,
            ],
            dtype=float,
        )

    def _rotvec_to_rotation_matrix(self, rotvec: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-9:
            return np.eye(3, dtype=float)
        axis = np.array(rotvec, dtype=float) / angle
        x, y, z = axis
        skew = np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=float,
        )
        return (
            np.eye(3, dtype=float)
            + math.sin(angle) * skew
            + (1.0 - math.cos(angle)) * (skew @ skew)
        )

    def _rotation_matrix_to_rotvec(self, rotation: np.ndarray) -> np.ndarray:
        trace = float(np.trace(rotation))
        angle = float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
        if angle < 1e-8:
            return np.zeros(3, dtype=float)
        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=float,
        )
        axis /= max(2.0 * math.sin(angle), 1e-6)
        return axis * angle

    def _pose_delta_base(
        self,
        current_pose: Pose,
        target_pose: Pose,
    ) -> tuple[np.ndarray, np.ndarray]:
        translation_delta = self._pose_position(target_pose) - self._pose_position(
            current_pose
        )
        current_rotation = quat2mat(self._quat_xyzw_to_wxyz(current_pose.orientation))
        target_rotation = quat2mat(self._quat_xyzw_to_wxyz(target_pose.orientation))
        rotation_delta = target_rotation @ current_rotation.T
        return translation_delta, self._rotation_matrix_to_rotvec(rotation_delta)

    def _clamp_sfp_teacher_step_delta(
        self,
        translation_delta_base: np.ndarray,
        rotation_delta_rotvec_base: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        translation_delta = np.array(translation_delta_base, dtype=float).copy()
        translation_delta[0] = float(np.clip(translation_delta[0], -0.010, 0.010))
        translation_delta[1] = float(np.clip(translation_delta[1], -0.020, 0.020))
        translation_delta[2] = float(np.clip(translation_delta[2], -0.020, 0.010))
        rotation_delta = np.array(rotation_delta_rotvec_base, dtype=float).copy()
        rotation_norm = float(np.linalg.norm(rotation_delta))
        if rotation_norm > 0.14:
            rotation_delta *= 0.14 / rotation_norm
        return translation_delta, rotation_delta

    def _apply_pose_residual_base(
        self,
        current_pose: Pose,
        translation_delta_base: np.ndarray,
        rotation_delta_rotvec_base: np.ndarray,
    ) -> Pose:
        current_quat_wxyz = self._quat_xyzw_to_wxyz(current_pose.orientation)
        current_rotation = quat2mat(current_quat_wxyz)
        residual_rotation = self._rotvec_to_rotation_matrix(rotation_delta_rotvec_base)
        target_rotation = residual_rotation @ current_rotation
        target_quat_wxyz = mat2quat(target_rotation)
        return self._make_pose(
            self._pose_position(current_pose) + np.array(translation_delta_base, dtype=float),
            (
                float(target_quat_wxyz[0]),
                float(target_quat_wxyz[1]),
                float(target_quat_wxyz[2]),
                float(target_quat_wxyz[3]),
            ),
        )

    def _sfp_post_servo_insert_delta(
        self, task: Task
    ) -> tuple[np.ndarray, np.ndarray]:
        translation = np.array(
            self._SFP_POST_SERVO_INSERT_TRANSLATION_DELTAS_BASE.get(
                task.target_module_name,
                self._SFP_POST_SERVO_INSERT_TRANSLATION_DELTAS_BASE[
                    "nic_card_mount_0"
                ],
            ),
            dtype=float,
        )
        rotation = np.array(
            self._SFP_POST_SERVO_INSERT_ROTATION_DELTA_ROTVEC_BASE, dtype=float
        )
        return translation, rotation

    def _run_sfp_submission_structured_teacher_insert(
        self,
        task: Task,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
    ) -> tuple[Pose, Observation | None, dict]:
        translation_delta_base, rotation_delta_rotvec_base = (
            self._sfp_post_servo_insert_delta(task)
        )
        current_observation: Observation | None = initial_observation
        current_pose = self._observed_tcp_pose(
            initial_observation,
            initial_observation.controller_state.reference_tcp_pose,
        )
        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        segment_records: list[dict[str, object]] = []

        for segment_index, fraction in enumerate(
            self._SFP_POST_SERVO_INSERT_SEGMENT_FRACTIONS, start=1
        ):
            segment_translation = fraction * translation_delta_base
            segment_rotation = fraction * rotation_delta_rotvec_base
            target_pose = self._apply_pose_residual_base(
                current_pose,
                translation_delta_base=segment_translation,
                rotation_delta_rotvec_base=segment_rotation,
            )
            debug_run.log_command_sample(
                "submission_sfp_structured_teacher_insert",
                target_pose,
                self.time_now().nanoseconds / 1e9,
                note=(
                    f"segment={segment_index}/{len(self._SFP_POST_SERVO_INSERT_SEGMENT_FRACTIONS)} "
                    f"fraction={fraction:.2f} "
                    f"dx={segment_translation[0]:.4f} dy={segment_translation[1]:.4f} dz={segment_translation[2]:.4f} "
                    f"rot=({segment_rotation[0]:.4f},{segment_rotation[1]:.4f},{segment_rotation[2]:.4f})"
                ),
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=0.30,
                debug_run=debug_run,
                phase_name=f"submission_sfp_structured_insert_segment_{segment_index}",
            )
            self._hold_pose(
                move_robot,
                target_pose,
                duration_sec=0.14,
                debug_run=debug_run,
                phase_name=f"submission_sfp_structured_insert_settle_{segment_index}",
            )
            next_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if next_observation is not None:
                current_observation = next_observation
                last_stamp_sec = _stamp_to_float(
                    next_observation.center_image.header.stamp
                )
                current_pose = self._observed_tcp_pose(next_observation, target_pose)
            else:
                current_pose = target_pose

            segment_records.append(
                {
                    "segment_index": segment_index,
                    "fraction": float(fraction),
                    "translation_delta_base": [
                        float(value) for value in segment_translation
                    ],
                    "rotation_delta_rotvec_base": [
                        float(value) for value in segment_rotation
                    ],
                    "target_pose": _pose_to_dict(target_pose),
                }
            )

        return current_pose, current_observation, {
            "translation_delta_base": [
                float(value) for value in translation_delta_base
            ],
            "rotation_delta_rotvec_base": [
                float(value) for value in rotation_delta_rotvec_base
            ],
            "segments": segment_records,
        }

    def _rotate_quat_about_camera_optical(
        self,
        quat_wxyz: tuple[float, float, float, float],
        delta_angle_deg: float,
        camera_frame: str,
    ) -> tuple[float, float, float, float] | None:
        rotation_base_from_optical = self._lookup_camera_rotation_base_from_optical(
            camera_frame
        )
        if rotation_base_from_optical is None:
            return None
        rotation_optical_from_base = rotation_base_from_optical.T
        delta_rad = math.radians(delta_angle_deg)
        c = math.cos(delta_rad)
        s = math.sin(delta_rad)
        rot_optical = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        rotated = (
            rotation_base_from_optical
            @ rot_optical
            @ rotation_optical_from_base
            @ quat2mat(quat_wxyz)
        )
        quat_rotated = mat2quat(rotated)
        return (
            float(quat_rotated[0]),
            float(quat_rotated[1]),
            float(quat_rotated[2]),
            float(quat_rotated[3]),
        )

    def _estimate_sc_insertion_quat_wxyz(
        self, observation: Observation | None
    ) -> tuple[float, float, float, float]:
        feature = self._sc_center_feature(observation)
        if feature is None or feature.get("magenta_angle_deg") is None:
            return self._SC_NOMINAL_QUAT_WXYZ
        delta_angle_deg = (
            float(feature["magenta_angle_deg"]) - self._SC_MAGENTA_REFERENCE_ANGLE_DEG
        )
        rotated = self._rotate_quat_about_camera_optical(
            self._SC_NOMINAL_QUAT_WXYZ,
            delta_angle_deg,
            self._CENTER_CAMERA_FRAME,
        )
        if rotated is None:
            return self._SC_NOMINAL_QUAT_WXYZ
        return rotated

    def _make_pose(
        self, position_xyz: np.ndarray, quat_wxyz: tuple[float, float, float, float]
    ) -> Pose:
        return Pose(
            position=Point(
                x=float(position_xyz[0]),
                y=float(position_xyz[1]),
                z=float(position_xyz[2]),
            ),
            orientation=self._quat_wxyz_to_xyzw(quat_wxyz),
        )

    def _pose_position(self, pose: Pose) -> np.ndarray:
        return np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=float,
        )

    def _pose_from_replay_row(
        self, row: tuple[float, float, float, float, float, float, float, float]
    ) -> Pose:
        _, x, y, z, qx, qy, qz, qw = row
        return Pose(
            position=Point(x=x, y=y, z=z),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )

    def _force_norm(self, observation: Observation | None) -> float:
        if observation is None:
            return 0.0
        force = observation.wrist_wrench.wrench.force
        return float(np.linalg.norm([force.x, force.y, force.z]))

    def _tool_axis_in_base(
        self,
        pose: Pose,
        axis_tool: tuple[float, float, float],
    ) -> np.ndarray:
        axis_tool_np = np.array(axis_tool, dtype=float)
        axis_tool_np /= max(np.linalg.norm(axis_tool_np), 1e-6)
        return quat2mat(self._quat_xyzw_to_wxyz(pose.orientation)) @ axis_tool_np

    def _observed_tcp_pose(
        self,
        observation: Observation | None,
        fallback_pose: Pose,
    ) -> Pose:
        if observation is None:
            return fallback_pose
        return observation.controller_state.tcp_pose

    def _run_tool_axis_push(
        self,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
        axis_tool: tuple[float, float, float],
        push_distance_m: float,
        duration_sec: float,
        hold_sec: float,
        debug_run: DebugRun,
        phase_prefix: str,
    ) -> Pose:
        axis_base = self._tool_axis_in_base(start_pose, axis_tool)
        axis_base /= max(np.linalg.norm(axis_base), 1e-6)
        pushed_pose = self._make_pose(
            self._pose_position(start_pose) + push_distance_m * axis_base,
            self._quat_xyzw_to_wxyz(start_pose.orientation),
        )
        self._move_for_duration(
            move_robot,
            start_pose,
            pushed_pose,
            duration_sec=duration_sec,
            debug_run=debug_run,
            phase_name=f"{phase_prefix}_push",
        )
        self._hold_pose(
            move_robot,
            pushed_pose,
            duration_sec=hold_sec,
            debug_run=debug_run,
            phase_name=f"{phase_prefix}_hold",
        )
        return pushed_pose

    def _run_sfp_closed_loop_insertion(
        self,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        push_schedule_m: tuple[float, ...] = (0.003, 0.003, 0.004, 0.004, 0.004, 0.003),
    ) -> tuple[Pose, Observation | None, dict]:
        current_observation: Observation | None = initial_observation
        current_pose = self._observed_tcp_pose(
            initial_observation,
            initial_observation.controller_state.reference_tcp_pose,
        )
        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        cycle_records: list[dict[str, object]] = []

        for cycle_index, push_distance_m in enumerate(push_schedule_m, start=1):
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                current_observation if current_observation is not None else initial_observation,
                get_observation,
                move_robot,
                debug_run,
                max_steps=8,
            )
            if aligned_observation is not None:
                current_observation = aligned_observation
                last_stamp_sec = _stamp_to_float(
                    aligned_observation.center_image.header.stamp
                )
            if aligned_pose is not None:
                current_pose = self._observed_tcp_pose(current_observation, aligned_pose)
            elif current_observation is not None:
                current_pose = current_observation.controller_state.tcp_pose

            feature_before = self._sfp_union_feature(current_observation)
            axis_base = self._tool_axis_in_base(current_pose, self._SFP_TOOL_INSERTION_AXIS)
            axis_base /= max(np.linalg.norm(axis_base), 1e-6)
            pre_push_position = self._pose_position(current_pose)
            target_pose = self._run_tool_axis_push(
                move_robot,
                current_pose,
                self._SFP_TOOL_INSERTION_AXIS,
                push_distance_m=push_distance_m,
                duration_sec=0.24,
                hold_sec=0.28,
                debug_run=debug_run,
                phase_prefix=f"submission_sfp_cycle_{cycle_index}",
            )
            pushed_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.6,
            )
            if pushed_observation is not None:
                current_observation = pushed_observation
                last_stamp_sec = _stamp_to_float(
                    pushed_observation.center_image.header.stamp
                )
            current_pose = self._observed_tcp_pose(current_observation, target_pose)

            feature_after = self._sfp_union_feature(current_observation)
            post_push_position = self._pose_position(current_pose)
            actual_advance_m = float(np.dot(post_push_position - pre_push_position, axis_base))
            tracking_error_m = None
            if current_observation is not None:
                tracking_error_m = float(
                    np.linalg.norm(current_observation.controller_state.tcp_error[:3])
                )

            cycle_record: dict[str, object] = {
                "cycle_index": cycle_index,
                "push_distance_m": float(push_distance_m),
                "actual_advance_m": actual_advance_m,
                "tracking_error_m": tracking_error_m,
                "pose": _pose_to_dict(current_pose),
                "feature_before": feature_before,
                "feature_after": feature_after,
            }
            cycle_records.append(cycle_record)
            debug_run.log_phase(
                f"submission_sfp_cycle_{cycle_index}_summary",
                self.time_now().nanoseconds / 1e9,
                (
                    f"push={push_distance_m:.4f} actual_advance={actual_advance_m:.4f} "
                    f"tracking_error={tracking_error_m if tracking_error_m is not None else float('nan'):.4f}"
                ),
            )

            if tracking_error_m is not None and tracking_error_m > 0.028:
                break

        return current_pose, current_observation, {"cycles": cycle_records}

    def _run_sfp_incremental_insertion(
        self,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        aligned_feature: dict | None,
        push_schedule_m: tuple[float, ...] = (
            0.004,
            0.004,
            0.004,
            0.004,
            0.004,
            0.004,
            0.003,
            0.003,
        ),
    ) -> tuple[Pose, Observation | None, dict]:
        current_observation: Observation | None = initial_observation
        current_pose = self._observed_tcp_pose(
            initial_observation,
            initial_observation.controller_state.reference_tcp_pose,
        )
        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        _ = aligned_feature
        insertion_quat_wxyz = self._quat_xyzw_to_wxyz(current_pose.orientation)
        insertion_axis_base = quat2mat(insertion_quat_wxyz) @ np.array(
            self._SFP_TOOL_INSERTION_AXIS, dtype=float
        )
        insertion_axis_base /= max(np.linalg.norm(insertion_axis_base), 1e-6)
        stalled_cycles = 0
        cycle_records: list[dict[str, object]] = []

        for cycle_index, push_distance_m in enumerate(push_schedule_m, start=1):
            pre_push_position = self._pose_position(current_pose)
            target_pose = self._make_pose(
                pre_push_position + push_distance_m * insertion_axis_base,
                insertion_quat_wxyz,
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=0.16,
                debug_run=debug_run,
                phase_name=f"submission_sfp_cycle_{cycle_index}_push",
            )
            self._hold_pose(
                move_robot,
                target_pose,
                duration_sec=0.20,
                debug_run=debug_run,
                phase_name=f"submission_sfp_cycle_{cycle_index}_hold",
            )
            pushed_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.35,
                newer_than_sec=last_stamp_sec,
            )
            if pushed_observation is None:
                pushed_observation = get_observation()
            if pushed_observation is not None:
                current_observation = pushed_observation
                last_stamp_sec = _stamp_to_float(
                    pushed_observation.center_image.header.stamp
                )
            observed_pose = self._observed_tcp_pose(current_observation, target_pose)
            current_pose = self._make_pose(
                self._pose_position(observed_pose),
                insertion_quat_wxyz,
            )

            post_push_position = self._pose_position(current_pose)
            actual_advance_m = float(
                np.dot(post_push_position - pre_push_position, insertion_axis_base)
            )
            tracking_error_m = None
            if current_observation is not None:
                tracking_error_m = float(
                    np.linalg.norm(current_observation.controller_state.tcp_error[:3])
                )

            if actual_advance_m < 0.0008:
                stalled_cycles += 1
            else:
                stalled_cycles = 0

            cycle_record: dict[str, object] = {
                "cycle_index": cycle_index,
                "push_distance_m": float(push_distance_m),
                "actual_advance_m": actual_advance_m,
                "tracking_error_m": tracking_error_m,
                "pose": _pose_to_dict(current_pose),
            }
            cycle_records.append(cycle_record)
            debug_run.log_phase(
                f"submission_sfp_cycle_{cycle_index}_summary",
                self.time_now().nanoseconds / 1e9,
                (
                    f"push={push_distance_m:.4f} actual_advance={actual_advance_m:.4f} "
                    f"tracking_error={tracking_error_m if tracking_error_m is not None else float('nan'):.4f}"
                ),
            )

            if stalled_cycles >= 2:
                break

        return current_pose, current_observation, {"cycles": cycle_records}

    def _run_sfp_submission_servo(
        self,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        max_steps: int = 24,
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        rotation_base_from_optical = self._lookup_camera_rotation_base_from_optical(
            self._CENTER_CAMERA_FRAME
        )
        if rotation_base_from_optical is None:
            return None, None, None

        initial_feature = self._sfp_union_feature(initial_observation)
        if initial_feature is None:
            return None, initial_observation, None

        desired_u = 0.5 * float(initial_observation.center_image.width)
        desired_v = 0.5 * float(initial_observation.center_image.height)
        desired_height = max(
            float(initial_feature["union_size_px"][1]) * 1.35,
            float(initial_feature["union_size_px"][1]) + 60.0,
        )
        desired_height = float(np.clip(desired_height, 180.0, 360.0))
        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)

        observation = initial_observation
        best_pose = initial_observation.controller_state.tcp_pose
        best_feature = initial_feature

        for step in range(max_steps):
            feature = self._sfp_union_feature(observation)
            if feature is None:
                break
            best_feature = feature
            current_pose = observation.controller_state.tcp_pose
            fx = float(observation.center_camera_info.k[0])
            fy = float(observation.center_camera_info.k[4])
            union_u, union_v = feature["union_center_uv"]
            _, union_h = feature["union_size_px"]
            du = union_u - desired_u
            dv = union_v - desired_v
            depth_scale = desired_height / max(union_h, 1.0)
            depth_error = depth_scale - 1.0
            aligned = (
                abs(du) < 18.0
                and abs(dv) < 18.0
                and union_h >= 0.92 * desired_height
            )
            debug_run.log_command_sample(
                "submission_sfp_servo",
                current_pose,
                self.time_now().nanoseconds / 1e9,
                note=(
                    f"step {step + 1}/{max_steps} du={du:.1f} dv={dv:.1f} "
                    f"height={union_h:.1f} desired_height={desired_height:.1f}"
                ),
            )
            if aligned:
                best_pose = current_pose
                break

            delta_cam = np.array(
                [
                    0.55 * 0.06 * du / max(fx, 1.0),
                    0.70 * 0.06 * dv / max(fy, 1.0),
                    float(np.clip(0.035 * depth_error, -0.015, 0.02)),
                ],
                dtype=float,
            )
            delta_cam[0] = float(np.clip(delta_cam[0], -0.012, 0.012))
            delta_cam[1] = float(np.clip(delta_cam[1], -0.012, 0.012))
            target_pose = self._make_pose(
                self._pose_position(current_pose) + (rotation_base_from_optical @ delta_cam),
                self._quat_xyzw_to_wxyz(current_pose.orientation),
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=0.18,
                debug_run=debug_run,
                phase_name="submission_sfp_servo",
            )
            best_pose = target_pose
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if observation is None:
                break
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        self._hold_pose(
            move_robot,
            best_pose,
            duration_sec=0.25,
            debug_run=debug_run,
            phase_name="submission_sfp_settle",
        )
        final_observation = self._wait_for_observation(
            get_observation,
            timeout_sec=0.8,
            newer_than_sec=last_stamp_sec,
        )
        if final_observation is None:
            final_observation = observation
        return best_pose, final_observation, best_feature

    def _run_sfp_submission_learned_servo(
        self,
        task: Task,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        desired_uv: tuple[float, float],
        desired_depth_m: float,
        max_steps: int = 16,
        position_gain: float = 0.72,
        max_step_m: float = 0.018,
        align_tol_px: float = 24.0,
        depth_tol_m: float = 0.015,
        phase_name: str = "submission_sfp_learned_servo",
        settle_phase_name: str = "submission_sfp_learned_settle",
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        learned_feature = self._sfp_learned_feature(
            task,
            initial_observation,
            step_index=1,
            max_steps=max_steps,
        )
        if learned_feature is None:
            return None, initial_observation, None

        desired_point_optical = self._desired_point_in_camera(
            initial_observation.center_camera_info,
            desired_uv,
            desired_depth_m,
        )
        if desired_point_optical is None:
            return None, initial_observation, learned_feature

        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        observation = initial_observation
        best_pose = initial_observation.controller_state.tcp_pose
        best_feature = learned_feature

        for step in range(max_steps):
            learned_feature = self._sfp_learned_feature(
                task,
                observation,
                step_index=step + 1,
                max_steps=max_steps,
            )
            if learned_feature is None:
                break
            best_feature = learned_feature
            current_pose = observation.controller_state.tcp_pose
            current_point_optical = learned_feature["point_center_optical"]
            center_uv = learned_feature["center_feature"]["union_center_uv"]
            du = float(center_uv[0] - desired_uv[0])
            dv = float(center_uv[1] - desired_uv[1])
            depth_error_m = float(current_point_optical[2] - desired_depth_m)
            aligned = (
                abs(du) < align_tol_px
                and abs(dv) < align_tol_px
                and abs(depth_error_m) < depth_tol_m
            )
            center_pose = self._lookup_camera_pose_base_from_optical(
                self._CENTER_CAMERA_FRAME
            )
            if center_pose is None:
                break
            rotation_base_from_optical, _camera_origin_base = center_pose
            optical_error = np.array(current_point_optical, dtype=float) - np.array(
                desired_point_optical, dtype=float
            )
            delta_base = rotation_base_from_optical @ optical_error
            delta_norm = float(np.linalg.norm(delta_base))
            if delta_norm > max_step_m:
                delta_base *= max_step_m / delta_norm
            else:
                delta_base *= position_gain
            debug_run.log_command_sample(
                phase_name,
                current_pose,
                self.time_now().nanoseconds / 1e9,
                note=(
                    f"step {step + 1}/{max_steps} learned_du={du:.1f} "
                    f"learned_dv={dv:.1f} depth={current_point_optical[2]:.3f} "
                    f"desired_depth={desired_depth_m:.3f} "
                    f"optical_err=({optical_error[0]:.4f},{optical_error[1]:.4f},{optical_error[2]:.4f}) "
                    f"delta_base=({delta_base[0]:.4f},{delta_base[1]:.4f},{delta_base[2]:.4f})"
                ),
            )
            if aligned:
                best_pose = current_pose
                break

            target_pose = self._make_pose(
                self._pose_position(current_pose) + delta_base,
                self._quat_xyzw_to_wxyz(current_pose.orientation),
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=0.18,
                debug_run=debug_run,
                phase_name=phase_name,
            )
            best_pose = target_pose
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if observation is None:
                break
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        self._hold_pose(
            move_robot,
            best_pose,
            duration_sec=0.25,
            debug_run=debug_run,
            phase_name=settle_phase_name,
        )
        final_observation = self._wait_for_observation(
            get_observation,
            timeout_sec=0.8,
            newer_than_sec=last_stamp_sec,
        )
        if final_observation is None:
            final_observation = observation
        return best_pose, final_observation, best_feature

    def _run_sfp_submission_learned_teacher_insert_residual(
        self,
        task: Task,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        max_iterations: int = 2,
    ) -> tuple[Pose | None, Observation | None, dict]:
        estimator = self._learned_port_inference_or_none("sfp")
        if estimator is None:
            return None, initial_observation, {"error": "missing_estimator"}
        if estimator.target_kind != "teacher_insert_delta6":
            return None, initial_observation, {
                "error": f"unexpected_target_kind:{estimator.target_kind}"
            }

        current_observation: Observation | None = initial_observation
        current_pose = initial_observation.controller_state.tcp_pose
        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        prediction_records: list[dict[str, object]] = []

        for iteration_index in range(1, max_iterations + 1):
            if current_observation is None:
                break
            current_feature_summary = self._extract_feature_summary(
                task, current_observation
            )
            prediction = cast(Any, estimator).predict_target_vector(
                task=task,
                images_bgr={
                    "left": _image_to_bgr(current_observation.left_image),
                    "center": _image_to_bgr(current_observation.center_image),
                    "right": _image_to_bgr(current_observation.right_image),
                },
                aux_vector=_build_learned_runtime_aux_vector(
                    current_pose=current_pose,
                    feature_summary=current_feature_summary,
                    step_index=iteration_index,
                    max_steps=max_iterations,
                ),
            )
            raw_vector = np.array(prediction["vector"], dtype=float)
            translation_delta = raw_vector[:3].copy()
            translation_delta[0] = float(np.clip(translation_delta[0], -0.035, 0.035))
            translation_delta[1] = float(np.clip(translation_delta[1], -0.035, 0.035))
            translation_delta[2] = float(np.clip(translation_delta[2], -0.025, 0.025))
            rotation_delta = raw_vector[3:].copy()
            rotation_norm = float(np.linalg.norm(rotation_delta))
            if rotation_norm > 0.45:
                rotation_delta *= 0.45 / rotation_norm

            target_pose = self._apply_pose_residual_base(
                current_pose,
                translation_delta_base=translation_delta,
                rotation_delta_rotvec_base=rotation_delta,
            )
            debug_run.log_command_sample(
                "submission_sfp_learned_insert_residual",
                target_pose,
                self.time_now().nanoseconds / 1e9,
                note=(
                    f"iter={iteration_index} "
                    f"dx={translation_delta[0]:.4f} dy={translation_delta[1]:.4f} dz={translation_delta[2]:.4f} "
                    f"rot_norm={float(np.linalg.norm(rotation_delta)):.4f}"
                ),
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=0.28,
                debug_run=debug_run,
                phase_name="submission_sfp_learned_insert_residual",
            )
            self._hold_pose(
                move_robot,
                target_pose,
                duration_sec=0.16,
                debug_run=debug_run,
                phase_name="submission_sfp_learned_insert_residual_settle",
            )
            next_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if next_observation is not None:
                current_observation = next_observation
                last_stamp_sec = _stamp_to_float(
                    next_observation.center_image.header.stamp
                )
                current_pose = self._observed_tcp_pose(next_observation, target_pose)
            else:
                current_pose = target_pose
            prediction_records.append(
                {
                    "iteration_index": iteration_index,
                    "raw_vector": [float(value) for value in raw_vector],
                    "applied_translation_delta_base": [
                        float(value) for value in translation_delta
                    ],
                    "applied_rotation_delta_rotvec_base": [
                        float(value) for value in rotation_delta
                    ],
                    "feature_summary": current_feature_summary,
                    "target_pose": _pose_to_dict(target_pose),
                }
            )
            if float(np.linalg.norm(translation_delta[:2])) < 0.003 and float(
                np.linalg.norm(rotation_delta)
            ) < 0.05:
                break

        return current_pose, current_observation, {"predictions": prediction_records}

    def _run_sfp_submission_learned_teacher_step_policy(
        self,
        task: Task,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        max_iterations: int = 8,
    ) -> tuple[Pose | None, Observation | None, dict]:
        estimator = self._learned_port_inference_or_none("sfp")
        if estimator is None:
            return None, initial_observation, {"error": "missing_estimator"}
        if estimator.target_kind != "teacher_step_delta6":
            return None, initial_observation, {
                "error": f"unexpected_target_kind:{estimator.target_kind}"
            }

        current_observation: Observation | None = initial_observation
        current_pose = self._observed_tcp_pose(
            initial_observation,
            initial_observation.controller_state.reference_tcp_pose,
        )
        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        prediction_records: list[dict[str, object]] = []

        for iteration_index in range(1, max_iterations + 1):
            if current_observation is None:
                break
            current_feature_summary = self._extract_feature_summary(
                task, current_observation
            )
            prediction = cast(Any, estimator).predict_target_vector(
                task=task,
                images_bgr={
                    "left": _image_to_bgr(current_observation.left_image),
                    "center": _image_to_bgr(current_observation.center_image),
                    "right": _image_to_bgr(current_observation.right_image),
                },
                aux_vector=_build_learned_runtime_aux_vector(
                    current_pose=current_pose,
                    feature_summary=current_feature_summary,
                    step_index=iteration_index,
                    max_steps=max_iterations,
                ),
            )
            raw_vector = np.array(prediction["vector"], dtype=float)
            translation_delta = raw_vector[:3].copy()
            translation_delta[0] = float(np.clip(translation_delta[0], -0.010, 0.010))
            translation_delta[1] = float(np.clip(translation_delta[1], -0.020, 0.020))
            translation_delta[2] = float(np.clip(translation_delta[2], -0.020, 0.010))
            rotation_delta = raw_vector[3:].copy()
            rotation_norm = float(np.linalg.norm(rotation_delta))
            if rotation_norm > 0.14:
                rotation_delta *= 0.14 / rotation_norm

            target_pose = self._apply_pose_residual_base(
                current_pose,
                translation_delta_base=translation_delta,
                rotation_delta_rotvec_base=rotation_delta,
            )
            debug_run.log_command_sample(
                "submission_sfp_learned_teacher_step",
                target_pose,
                self.time_now().nanoseconds / 1e9,
                note=(
                    f"iter={iteration_index} "
                    f"dx={translation_delta[0]:.4f} dy={translation_delta[1]:.4f} dz={translation_delta[2]:.4f} "
                    f"rot_norm={float(np.linalg.norm(rotation_delta)):.4f}"
                ),
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=0.22,
                debug_run=debug_run,
                phase_name="submission_sfp_learned_teacher_step",
            )
            self._hold_pose(
                move_robot,
                target_pose,
                duration_sec=0.10,
                debug_run=debug_run,
                phase_name="submission_sfp_learned_teacher_step_settle",
            )
            next_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if next_observation is not None:
                current_observation = next_observation
                last_stamp_sec = _stamp_to_float(
                    next_observation.center_image.header.stamp
                )
                current_pose = self._observed_tcp_pose(next_observation, target_pose)
            else:
                current_pose = target_pose
            prediction_records.append(
                {
                    "iteration_index": iteration_index,
                    "raw_vector": [float(value) for value in raw_vector],
                    "applied_translation_delta_base": [
                        float(value) for value in translation_delta
                    ],
                    "applied_rotation_delta_rotvec_base": [
                        float(value) for value in rotation_delta
                    ],
                    "feature_summary": current_feature_summary,
                    "target_pose": _pose_to_dict(target_pose),
                }
            )
            if float(np.linalg.norm(translation_delta)) < 0.003 and float(
                np.linalg.norm(rotation_delta)
            ) < 0.04:
                break

        return current_pose, current_observation, {"predictions": prediction_records}

    def _run_sfp_submission_learned_coarse_to_fine(
        self,
        task: Task,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        visual_template = self._sfp_visual_template_for_task(task)
        hover_uvz = self._learned_phase_template_uvz(
            "sfp",
            task,
            "hover",
            (
                float(visual_template["union_center_uv"][0]),
                float(visual_template["union_center_uv"][1]),
                float(visual_template["camera_depth_m"]),
            ),
        )
        mid_uvz = self._learned_phase_template_uvz(
            "sfp",
            task,
            "insert_mid",
            (
                hover_uvz[0],
                hover_uvz[1],
                max(hover_uvz[2] - 0.030, 0.10),
            ),
        )
        final_uvz = self._learned_phase_template_uvz(
            "sfp",
            task,
            "insert_near",
            (
                mid_uvz[0],
                mid_uvz[1],
                max(mid_uvz[2] - 0.020, 0.08),
            ),
        )

        hover_pose, hover_observation, hover_feature = self._run_sfp_submission_learned_servo(
            task,
            initial_observation,
            get_observation,
            move_robot,
            debug_run,
            desired_uv=(hover_uvz[0], hover_uvz[1]),
            desired_depth_m=hover_uvz[2],
            max_steps=16,
            position_gain=0.80,
            max_step_m=0.024,
            align_tol_px=30.0,
            depth_tol_m=0.025,
            phase_name="submission_sfp_learned_hover",
            settle_phase_name="submission_sfp_learned_hover_settle",
        )
        if hover_pose is None:
            return None, hover_observation, {"hover_feature": hover_feature}

        mid_input_observation = hover_observation if hover_observation is not None else initial_observation
        mid_pose, mid_observation, mid_feature = self._run_sfp_submission_learned_servo(
            task,
            mid_input_observation,
            get_observation,
            move_robot,
            debug_run,
            desired_uv=(mid_uvz[0], mid_uvz[1]),
            desired_depth_m=mid_uvz[2],
            max_steps=14,
            position_gain=0.76,
            max_step_m=0.018,
            align_tol_px=24.0,
            depth_tol_m=0.018,
            phase_name="submission_sfp_learned_mid",
            settle_phase_name="submission_sfp_learned_mid_settle",
        )
        if mid_pose is None:
            return hover_pose, hover_observation, {
                "hover_feature": hover_feature,
                "mid_feature": mid_feature,
            }

        fine_input_observation = mid_observation if mid_observation is not None else mid_input_observation
        fine_pose, fine_observation, fine_feature = self._run_sfp_submission_learned_servo(
            task,
            fine_input_observation,
            get_observation,
            move_robot,
            debug_run,
            desired_uv=(final_uvz[0], final_uvz[1]),
            desired_depth_m=final_uvz[2],
            max_steps=12,
            position_gain=0.70,
            max_step_m=0.012,
            align_tol_px=18.0,
            depth_tol_m=0.010,
            phase_name="submission_sfp_learned_final",
            settle_phase_name="submission_sfp_learned_final_settle",
        )
        if fine_pose is None:
            return mid_pose, mid_observation, {
                "hover_feature": hover_feature,
                "mid_feature": mid_feature,
                "fine_feature": fine_feature,
            }
        return fine_pose, fine_observation, {
            "hover_feature": hover_feature,
            "mid_feature": mid_feature,
            "fine_feature": fine_feature,
        }

    def _run_sfp_submission_learned_closed_loop_insertion(
        self,
        task: Task,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        push_schedule_m: tuple[float, ...] = (
            0.0030,
            0.0030,
            0.0030,
            0.0025,
            0.0025,
            0.0020,
        ),
        retreat_distance_m: float = 0.0015,
    ) -> tuple[Pose, Observation | None, dict]:
        visual_template = self._sfp_visual_template_for_task(task)
        hover_uvz = self._learned_phase_template_uvz(
            "sfp",
            task,
            "hover",
            (
                float(visual_template["union_center_uv"][0]),
                float(visual_template["union_center_uv"][1]),
                float(visual_template["camera_depth_m"]),
            ),
        )
        mid_uvz = self._learned_phase_template_uvz(
            "sfp",
            task,
            "insert_mid",
            (
                hover_uvz[0],
                hover_uvz[1],
                max(hover_uvz[2] - 0.030, 0.10),
            ),
        )
        near_uvz = self._learned_phase_template_uvz(
            "sfp",
            task,
            "insert_near",
            (
                mid_uvz[0],
                mid_uvz[1],
                max(mid_uvz[2] - 0.020, 0.08),
            ),
        )

        current_observation: Observation | None = initial_observation
        current_pose = self._observed_tcp_pose(
            initial_observation,
            initial_observation.controller_state.reference_tcp_pose,
        )
        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        baseline_force_norm = self._force_norm(initial_observation)
        stalled_cycles = 0
        cycle_records: list[dict[str, object]] = []

        for cycle_index, push_distance_m in enumerate(push_schedule_m, start=1):
            feature_before = self._sfp_learned_feature(task, current_observation)
            if feature_before is None:
                break

            current_depth_m = float(feature_before["point_center_optical"][2])
            if current_depth_m > near_uvz[2] + 0.020:
                phase_label = "mid"
                desired_uvz = mid_uvz
                servo_kwargs = {
                    "max_steps": 8,
                    "position_gain": 0.68,
                    "max_step_m": 0.010,
                    "align_tol_px": 22.0,
                    "depth_tol_m": 0.012,
                }
            else:
                phase_label = "near"
                desired_uvz = near_uvz
                servo_kwargs = {
                    "max_steps": 6,
                    "position_gain": 0.60,
                    "max_step_m": 0.008,
                    "align_tol_px": 18.0,
                    "depth_tol_m": 0.008,
                }

            aligned_pose, aligned_observation, aligned_feature = (
                self._run_sfp_submission_learned_servo(
                    task,
                    current_observation
                    if current_observation is not None
                    else initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                    desired_uv=(desired_uvz[0], desired_uvz[1]),
                    desired_depth_m=desired_uvz[2],
                    phase_name=f"submission_sfp_cycle_{cycle_index}_{phase_label}_servo",
                    settle_phase_name=(
                        f"submission_sfp_cycle_{cycle_index}_{phase_label}_settle"
                    ),
                    **servo_kwargs,
                )
            )
            if aligned_observation is not None:
                current_observation = aligned_observation
            if aligned_pose is not None:
                current_pose = self._observed_tcp_pose(current_observation, aligned_pose)

            axis_base = self._tool_axis_in_base(current_pose, self._SFP_TOOL_INSERTION_AXIS)
            axis_base /= max(np.linalg.norm(axis_base), 1e-6)
            pre_push_position = self._pose_position(current_pose)
            pushed_pose = self._run_tool_axis_push(
                move_robot,
                current_pose,
                self._SFP_TOOL_INSERTION_AXIS,
                push_distance_m=push_distance_m,
                duration_sec=0.18,
                hold_sec=0.12,
                debug_run=debug_run,
                phase_prefix=f"submission_sfp_cycle_{cycle_index}",
            )
            pushed_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.5,
                newer_than_sec=last_stamp_sec,
            )
            if pushed_observation is not None:
                current_observation = pushed_observation
                last_stamp_sec = _stamp_to_float(
                    pushed_observation.center_image.header.stamp
                )
            current_pose = self._observed_tcp_pose(current_observation, pushed_pose)

            post_push_position = self._pose_position(current_pose)
            actual_advance_m = float(np.dot(post_push_position - pre_push_position, axis_base))
            tracking_error_m = None
            measured_force_norm = None
            excess_force_norm = None
            if current_observation is not None:
                tracking_error_m = float(
                    np.linalg.norm(current_observation.controller_state.tcp_error[:3])
                )
                measured_force_norm = self._force_norm(current_observation)
                excess_force_norm = measured_force_norm - baseline_force_norm

            feature_after = self._sfp_learned_feature(task, current_observation)
            depth_after_m = (
                None
                if feature_after is None
                else float(feature_after["point_center_optical"][2])
            )

            cycle_record: dict[str, object] = {
                "cycle_index": cycle_index,
                "phase_label": phase_label,
                "push_distance_m": float(push_distance_m),
                "actual_advance_m": actual_advance_m,
                "tracking_error_m": tracking_error_m,
                "force_norm": measured_force_norm,
                "excess_force_norm": excess_force_norm,
                "depth_before_m": current_depth_m,
                "depth_after_m": depth_after_m,
                "aligned_feature": aligned_feature,
                "feature_after": feature_after,
                "pose": _pose_to_dict(current_pose),
            }
            cycle_records.append(cycle_record)
            debug_run.log_phase(
                f"submission_sfp_cycle_{cycle_index}_summary",
                self.time_now().nanoseconds / 1e9,
                (
                    f"phase={phase_label} push={push_distance_m:.4f} "
                    f"advance={actual_advance_m:.4f} "
                    f"track={tracking_error_m if tracking_error_m is not None else float('nan'):.4f} "
                    f"force={measured_force_norm if measured_force_norm is not None else float('nan'):.2f} "
                    f"excess={excess_force_norm if excess_force_norm is not None else float('nan'):.2f} "
                    f"depth_before={current_depth_m:.3f} "
                    f"depth_after={depth_after_m if depth_after_m is not None else float('nan'):.3f}"
                ),
            )

            stalled = actual_advance_m < 0.0010
            overloaded = excess_force_norm is not None and excess_force_norm > 8.0
            diverged = tracking_error_m is not None and tracking_error_m > 0.024
            if stalled or overloaded or diverged:
                stalled_cycles += 1
                retreat_pose = self._run_tool_axis_push(
                    move_robot,
                    current_pose,
                    self._SFP_TOOL_INSERTION_AXIS,
                    push_distance_m=-retreat_distance_m,
                    duration_sec=0.12,
                    hold_sec=0.08,
                    debug_run=debug_run,
                    phase_prefix=f"submission_sfp_cycle_{cycle_index}_retreat",
                )
                retreat_observation = self._wait_for_observation(
                    get_observation,
                    timeout_sec=0.5,
                    newer_than_sec=last_stamp_sec,
                )
                if retreat_observation is not None:
                    current_observation = retreat_observation
                    last_stamp_sec = _stamp_to_float(
                        retreat_observation.center_image.header.stamp
                    )
                current_pose = self._observed_tcp_pose(current_observation, retreat_pose)
            else:
                stalled_cycles = 0

            if stalled_cycles >= 2:
                break

        return current_pose, current_observation, {"cycles": cycle_records}

    def _run_sc_submission_servo(
        self,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        target_quat_wxyz: tuple[float, float, float, float] | None = None,
        desired_uv: tuple[float, float] | None = None,
        desired_area_px: float | None = None,
        max_steps: int = 20,
        xy_gains: tuple[float, float] = (0.42, 0.52),
        nominal_depth_m: float = 0.06,
        depth_gain: float = 0.025,
        xy_clip_m: float = 0.008,
        z_clip_m: tuple[float, float] = (-0.008, 0.012),
        align_tol_px: float = 22.0,
        area_ratio_tol: float = 0.90,
        phase_name: str = "submission_sc_servo",
        settle_phase_name: str = "submission_sc_settle",
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        rotation_base_from_optical = self._lookup_camera_rotation_base_from_optical(
            self._CENTER_CAMERA_FRAME
        )
        if rotation_base_from_optical is None:
            return None, None, None

        initial_feature = self._sc_center_feature(initial_observation)
        if initial_feature is None:
            return None, initial_observation, None

        if desired_uv is None:
            desired_u = 0.5 * float(initial_observation.center_image.width)
            desired_v = 0.5 * float(initial_observation.center_image.height)
        else:
            desired_u, desired_v = desired_uv
        if desired_area_px is None:
            desired_area = max(
                float(initial_feature["cyan_area_px"]) * 1.30,
                float(initial_feature["cyan_area_px"]) + 4000.0,
            )
            desired_area = float(np.clip(desired_area, 10000.0, 45000.0))
        else:
            desired_area = float(desired_area_px)
        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)

        observation = initial_observation
        best_pose = initial_observation.controller_state.tcp_pose
        best_feature = initial_feature

        for step in range(max_steps):
            feature = self._sc_center_feature(observation)
            if feature is None:
                break
            best_feature = feature
            current_pose = observation.controller_state.tcp_pose
            fx = float(observation.center_camera_info.k[0])
            fy = float(observation.center_camera_info.k[4])
            union_u, union_v = feature["cyan_centroid_uv"]
            area_px = float(feature["cyan_area_px"])
            du = union_u - desired_u
            dv = union_v - desired_v
            depth_error = (desired_area / max(area_px, 1.0)) - 1.0
            aligned = (
                abs(du) < align_tol_px
                and abs(dv) < align_tol_px
                and area_px >= area_ratio_tol * desired_area
            )
            debug_run.log_command_sample(
                phase_name,
                current_pose,
                self.time_now().nanoseconds / 1e9,
                note=(
                    f"step {step + 1}/{max_steps} du={du:.1f} dv={dv:.1f} "
                    f"area={area_px:.0f} desired_area={desired_area:.0f}"
                ),
            )
            if aligned:
                best_pose = current_pose
                break

            delta_cam = np.array(
                [
                    xy_gains[0] * nominal_depth_m * du / max(fx, 1.0),
                    xy_gains[1] * nominal_depth_m * dv / max(fy, 1.0),
                    float(np.clip(depth_gain * depth_error, z_clip_m[0], z_clip_m[1])),
                ],
                dtype=float,
            )
            delta_cam[0] = float(np.clip(delta_cam[0], -xy_clip_m, xy_clip_m))
            delta_cam[1] = float(np.clip(delta_cam[1], -xy_clip_m, xy_clip_m))
            target_pose = self._make_pose(
                self._pose_position(current_pose) + (rotation_base_from_optical @ delta_cam),
                (
                    self._quat_xyzw_to_wxyz(current_pose.orientation)
                    if target_quat_wxyz is None
                    else target_quat_wxyz
                ),
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=0.18,
                debug_run=debug_run,
                phase_name=phase_name,
            )
            best_pose = target_pose
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if observation is None:
                break
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        self._hold_pose(
            move_robot,
            best_pose,
            duration_sec=0.25,
            debug_run=debug_run,
            phase_name=settle_phase_name,
        )
        final_observation = self._wait_for_observation(
            get_observation,
            timeout_sec=0.8,
            newer_than_sec=last_stamp_sec,
        )
        if final_observation is None:
            final_observation = observation
        return best_pose, final_observation, best_feature

    def _run_sc_submission_servo_coarse_to_fine(
        self,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        coarse_pose, coarse_observation, coarse_feature = self._run_sc_submission_servo(
            initial_observation,
            get_observation,
            move_robot,
            debug_run,
            desired_uv=self._SC_SUBMISSION_COARSE_TEMPLATE["cyan_centroid_uv"],
            desired_area_px=self._SC_SUBMISSION_COARSE_TEMPLATE["desired_area_px"],
            max_steps=24,
            xy_gains=(0.82, 0.95),
            nominal_depth_m=self._SC_SUBMISSION_COARSE_TEMPLATE["nominal_depth_m"],
            depth_gain=0.032,
            xy_clip_m=0.014,
            z_clip_m=(-0.012, 0.018),
            align_tol_px=34.0,
            area_ratio_tol=0.82,
            phase_name="submission_sc_coarse_servo",
            settle_phase_name="submission_sc_coarse_settle",
        )
        if coarse_pose is None:
            return None, coarse_observation, coarse_feature

        fine_input_observation = (
            coarse_observation if coarse_observation is not None else initial_observation
        )
        fine_pose, fine_observation, fine_feature = self._run_sc_submission_servo(
            fine_input_observation,
            get_observation,
            move_robot,
            debug_run,
            desired_uv=self._SC_SUBMISSION_FINE_TEMPLATE["cyan_centroid_uv"],
            desired_area_px=self._SC_SUBMISSION_FINE_TEMPLATE["desired_area_px"],
            max_steps=18,
            xy_gains=(0.68, 0.82),
            nominal_depth_m=self._SC_SUBMISSION_FINE_TEMPLATE["nominal_depth_m"],
            depth_gain=0.028,
            xy_clip_m=0.010,
            z_clip_m=(-0.010, 0.014),
            align_tol_px=28.0,
            area_ratio_tol=0.80,
            phase_name="submission_sc_fine_servo",
            settle_phase_name="submission_sc_fine_settle",
        )
        if fine_pose is None:
            return coarse_pose, coarse_observation, {
                "coarse_feature": coarse_feature,
                "fine_feature": fine_feature,
            }
        return fine_pose, fine_observation, {
            "coarse_feature": coarse_feature,
            "fine_feature": fine_feature,
        }

    def _run_sc_submission_triangulated_servo(
        self,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        desired_uv: tuple[float, float],
        desired_depth_m: float,
        max_steps: int = 16,
        position_gain: float = 0.65,
        max_step_m: float = 0.014,
        align_tol_px: float = 28.0,
        depth_tol_m: float = 0.012,
        rotation_gain: float = 0.45,
        max_rotation_step_deg: float = 3.0,
        phase_name: str = "submission_sc_triangulated_servo",
        settle_phase_name: str = "submission_sc_triangulated_settle",
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        triangulated_feature = self._sc_triangulated_feature(initial_observation)
        if triangulated_feature is None:
            return None, initial_observation, None

        desired_point_optical = self._desired_point_in_camera(
            initial_observation.center_camera_info,
            desired_uv,
            desired_depth_m,
        )
        if desired_point_optical is None:
            return None, initial_observation, triangulated_feature

        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        observation = initial_observation
        best_pose = initial_observation.controller_state.tcp_pose
        best_feature = triangulated_feature

        for step in range(max_steps):
            triangulated_feature = self._sc_triangulated_feature(observation)
            if triangulated_feature is None:
                break
            best_feature = triangulated_feature
            center_feature = triangulated_feature["center_feature"]
            current_point_optical = triangulated_feature["point_center_optical"]
            current_pose = observation.controller_state.tcp_pose
            current_quat_wxyz = self._quat_xyzw_to_wxyz(current_pose.orientation)

            du = float(center_feature["cyan_centroid_uv"][0] - desired_uv[0])
            dv = float(center_feature["cyan_centroid_uv"][1] - desired_uv[1])
            depth_error_m = float(current_point_optical[2] - desired_depth_m)
            angle_error_deg = 0.0
            magenta_angle_deg = center_feature.get("magenta_angle_deg")
            if magenta_angle_deg is not None:
                angle_error_deg = (
                    float(self._SC_MAGENTA_REFERENCE_ANGLE_DEG)
                    - float(magenta_angle_deg)
                )
            aligned = (
                abs(du) < align_tol_px
                and abs(dv) < align_tol_px
                and abs(depth_error_m) < depth_tol_m
            )
            debug_run.log_command_sample(
                phase_name,
                current_pose,
                self.time_now().nanoseconds / 1e9,
                note=(
                    f"step {step + 1}/{max_steps} du={du:.1f} dv={dv:.1f} "
                    f"depth={current_point_optical[2]:.3f} desired_depth={desired_depth_m:.3f} "
                    f"angle_err={angle_error_deg:.2f}"
                ),
            )
            if aligned:
                best_pose = current_pose
                break

            center_pose = self._lookup_camera_pose_base_from_optical(
                self._CENTER_CAMERA_FRAME
            )
            if center_pose is None:
                break
            rotation_base_from_optical, camera_origin_base = center_pose
            port_point_base = triangulated_feature["triangulated_point_base"]
            target_camera_origin_base = (
                port_point_base - rotation_base_from_optical @ desired_point_optical
            )
            delta_base = target_camera_origin_base - camera_origin_base
            delta_norm = float(np.linalg.norm(delta_base))
            if delta_norm > max_step_m:
                delta_base *= max_step_m / delta_norm
            else:
                delta_base *= position_gain

            target_quat_wxyz = current_quat_wxyz
            if rotation_gain > 0.0 and magenta_angle_deg is not None:
                rotation_step_deg = float(
                    np.clip(
                        rotation_gain * angle_error_deg,
                        -max_rotation_step_deg,
                        max_rotation_step_deg,
                    )
                )
                rotated_quat = self._rotate_quat_about_camera_optical(
                    current_quat_wxyz,
                    rotation_step_deg,
                    self._CENTER_CAMERA_FRAME,
                )
                if rotated_quat is not None:
                    target_quat_wxyz = rotated_quat

            target_pose = self._make_pose(
                self._pose_position(current_pose) + delta_base,
                target_quat_wxyz,
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=0.18,
                debug_run=debug_run,
                phase_name=phase_name,
            )
            best_pose = target_pose
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if observation is None:
                break
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        self._hold_pose(
            move_robot,
            best_pose,
            duration_sec=0.25,
            debug_run=debug_run,
            phase_name=settle_phase_name,
        )
        final_observation = self._wait_for_observation(
            get_observation,
            timeout_sec=0.8,
            newer_than_sec=last_stamp_sec,
        )
        if final_observation is None:
            final_observation = observation
        return best_pose, final_observation, best_feature

    def _run_sc_submission_triangulated_coarse_to_fine(
        self,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        coarse_rotation_gain: float = 0.55,
        coarse_max_rotation_step_deg: float = 4.0,
        fine_rotation_gain: float = 0.40,
        fine_max_rotation_step_deg: float = 2.0,
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        coarse_pose, coarse_observation, coarse_feature = (
            self._run_sc_submission_triangulated_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
                desired_uv=self._SC_SUBMISSION_COARSE_TEMPLATE["cyan_centroid_uv"],
                desired_depth_m=self._SC_SUBMISSION_COARSE_TEMPLATE["nominal_depth_m"],
                max_steps=18,
                position_gain=0.75,
                max_step_m=0.016,
                align_tol_px=36.0,
                depth_tol_m=0.016,
                rotation_gain=coarse_rotation_gain,
                max_rotation_step_deg=coarse_max_rotation_step_deg,
                phase_name="submission_sc_triangulated_coarse",
                settle_phase_name="submission_sc_triangulated_coarse_settle",
            )
        )
        if coarse_pose is None:
            return None, coarse_observation, coarse_feature

        fine_input_observation = (
            coarse_observation if coarse_observation is not None else initial_observation
        )
        fine_pose, fine_observation, fine_feature = (
            self._run_sc_submission_triangulated_servo(
                fine_input_observation,
                get_observation,
                move_robot,
                debug_run,
                desired_uv=self._SC_SUBMISSION_FINE_TEMPLATE["cyan_centroid_uv"],
                desired_depth_m=self._SC_SUBMISSION_FINE_TEMPLATE["nominal_depth_m"],
                max_steps=14,
                position_gain=0.68,
                max_step_m=0.012,
                align_tol_px=30.0,
                depth_tol_m=0.010,
                rotation_gain=fine_rotation_gain,
                max_rotation_step_deg=fine_max_rotation_step_deg,
                phase_name="submission_sc_triangulated_fine",
                settle_phase_name="submission_sc_triangulated_fine_settle",
            )
        )
        if fine_pose is None:
            return coarse_pose, coarse_observation, {
                "coarse_feature": coarse_feature,
                "fine_feature": fine_feature,
            }
        return fine_pose, fine_observation, {
            "coarse_feature": coarse_feature,
            "fine_feature": fine_feature,
        }

    def _run_sc_submission_learned_servo(
        self,
        task: Task,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        desired_uv: tuple[float, float],
        desired_depth_m: float,
        max_steps: int = 16,
        position_gain: float = 0.65,
        max_step_m: float = 0.014,
        align_tol_px: float = 28.0,
        depth_tol_m: float = 0.012,
        phase_name: str = "submission_sc_learned_servo",
        settle_phase_name: str = "submission_sc_learned_settle",
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        learned_feature = self._sc_learned_feature(
            task,
            initial_observation,
            step_index=1,
            max_steps=max_steps,
        )
        if learned_feature is None:
            return None, initial_observation, None

        desired_point_optical = self._desired_point_in_camera(
            initial_observation.center_camera_info,
            desired_uv,
            desired_depth_m,
        )
        if desired_point_optical is None:
            return None, initial_observation, learned_feature

        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        observation = initial_observation
        best_pose = initial_observation.controller_state.tcp_pose
        best_feature = learned_feature

        for step in range(max_steps):
            learned_feature = self._sc_learned_feature(
                task,
                observation,
                step_index=step + 1,
                max_steps=max_steps,
            )
            if learned_feature is None:
                break
            best_feature = learned_feature
            center_feature = learned_feature["center_feature"]
            current_point_optical = learned_feature["point_center_optical"]
            current_pose = observation.controller_state.tcp_pose

            du = float(center_feature["cyan_centroid_uv"][0] - desired_uv[0])
            dv = float(center_feature["cyan_centroid_uv"][1] - desired_uv[1])
            depth_error_m = float(current_point_optical[2] - desired_depth_m)
            aligned = (
                abs(du) < align_tol_px
                and abs(dv) < align_tol_px
                and abs(depth_error_m) < depth_tol_m
            )
            debug_run.log_command_sample(
                phase_name,
                current_pose,
                self.time_now().nanoseconds / 1e9,
                note=(
                    f"step {step + 1}/{max_steps} learned_du={du:.1f} "
                    f"learned_dv={dv:.1f} depth={current_point_optical[2]:.3f} "
                    f"desired_depth={desired_depth_m:.3f}"
                ),
            )
            if aligned:
                best_pose = current_pose
                break

            center_pose = self._lookup_camera_pose_base_from_optical(
                self._CENTER_CAMERA_FRAME
            )
            if center_pose is None:
                break
            rotation_base_from_optical, camera_origin_base = center_pose
            port_point_base = learned_feature["triangulated_point_base"]
            target_camera_origin_base = (
                port_point_base - rotation_base_from_optical @ desired_point_optical
            )
            delta_base = target_camera_origin_base - camera_origin_base
            delta_norm = float(np.linalg.norm(delta_base))
            if delta_norm > max_step_m:
                delta_base *= max_step_m / delta_norm
            else:
                delta_base *= position_gain

            target_pose = self._make_pose(
                self._pose_position(current_pose) + delta_base,
                self._quat_xyzw_to_wxyz(current_pose.orientation),
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=0.18,
                debug_run=debug_run,
                phase_name=phase_name,
            )
            best_pose = target_pose
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if observation is None:
                break
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        self._hold_pose(
            move_robot,
            best_pose,
            duration_sec=0.25,
            debug_run=debug_run,
            phase_name=settle_phase_name,
        )
        final_observation = self._wait_for_observation(
            get_observation,
            timeout_sec=0.8,
            newer_than_sec=last_stamp_sec,
        )
        if final_observation is None:
            final_observation = observation
        return best_pose, final_observation, best_feature

    def _run_sc_submission_learned_coarse_to_fine(
        self,
        task: Task,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        hover_pose, hover_observation, hover_feature = (
            self._run_sc_submission_learned_servo(
                task,
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
                desired_uv=self._SC_LEARNED_HOVER_TEMPLATE["cyan_centroid_uv"],
                desired_depth_m=self._SC_LEARNED_HOVER_TEMPLATE["nominal_depth_m"],
                max_steps=18,
                position_gain=0.78,
                max_step_m=0.022,
                align_tol_px=32.0,
                depth_tol_m=0.022,
                phase_name="submission_sc_learned_hover",
                settle_phase_name="submission_sc_learned_hover_settle",
            )
        )
        if hover_pose is None:
            return None, hover_observation, hover_feature

        mid_input_observation = (
            hover_observation if hover_observation is not None else initial_observation
        )
        mid_pose, mid_observation, mid_feature = (
            self._run_sc_submission_learned_servo(
                task,
                mid_input_observation,
                get_observation,
                move_robot,
                debug_run,
                desired_uv=self._SC_LEARNED_INSERT_MID_TEMPLATE["cyan_centroid_uv"],
                desired_depth_m=self._SC_LEARNED_INSERT_MID_TEMPLATE["nominal_depth_m"],
                max_steps=18,
                position_gain=0.74,
                max_step_m=0.018,
                align_tol_px=28.0,
                depth_tol_m=0.018,
                phase_name="submission_sc_learned_insert_mid",
                settle_phase_name="submission_sc_learned_insert_mid_settle",
            )
        )
        if mid_pose is None:
            return hover_pose, hover_observation, {
                "hover_feature": hover_feature,
                "mid_feature": mid_feature,
            }

        fine_input_observation = mid_observation if mid_observation is not None else mid_input_observation
        fine_pose, fine_observation, fine_feature = self._run_sc_submission_learned_servo(
                task,
                fine_input_observation,
                get_observation,
                move_robot,
                debug_run,
                desired_uv=self._SC_LEARNED_INSERT_FINAL_TEMPLATE["cyan_centroid_uv"],
                desired_depth_m=self._SC_LEARNED_INSERT_FINAL_TEMPLATE["nominal_depth_m"],
                max_steps=16,
                position_gain=0.70,
                max_step_m=0.014,
                align_tol_px=24.0,
                depth_tol_m=0.014,
                phase_name="submission_sc_learned_insert_final",
                settle_phase_name="submission_sc_learned_insert_final_settle",
            )
        if fine_pose is None:
            return mid_pose, mid_observation, {
                "hover_feature": hover_feature,
                "mid_feature": mid_feature,
                "fine_feature": fine_feature,
            }
        return fine_pose, fine_observation, {
            "hover_feature": hover_feature,
            "mid_feature": mid_feature,
            "fine_feature": fine_feature,
        }

    def _run_sc_submission_hover_then_primitive(
        self,
        task: Task,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        hover_pose, hover_observation, hover_feature = self._run_sc_submission_learned_servo(
            task,
            initial_observation,
            get_observation,
            move_robot,
            debug_run,
            desired_uv=self._SC_LEARNED_HOVER_TEMPLATE["cyan_centroid_uv"],
            desired_depth_m=self._SC_LEARNED_HOVER_TEMPLATE["nominal_depth_m"],
            max_steps=18,
            position_gain=0.78,
            max_step_m=0.022,
            align_tol_px=32.0,
            depth_tol_m=0.022,
            phase_name="submission_sc_learned_hover",
            settle_phase_name="submission_sc_learned_hover_settle",
        )
        if hover_pose is None:
            return None, hover_observation, {"hover_feature": hover_feature}

        current_pose = hover_pose
        current_observation = (
            hover_observation if hover_observation is not None else initial_observation
        )
        primitive_features: list[dict] = []
        for segment_index, delta_xyz in enumerate(
            self._SC_LEARNED_PRIMITIVE_SEGMENTS_BASE, start=1
        ):
            delta_base = np.array(delta_xyz, dtype=float)
            target_pose = self._make_pose(
                self._pose_position(current_pose) + delta_base,
                self._quat_xyzw_to_wxyz(current_pose.orientation),
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=0.45,
                debug_run=debug_run,
                phase_name=f"submission_sc_primitive_{segment_index}",
            )
            self._hold_pose(
                move_robot,
                target_pose,
                duration_sec=0.20,
                debug_run=debug_run,
                phase_name=f"submission_sc_primitive_{segment_index}_settle",
            )
            newer_than_sec = (
                None
                if current_observation is None
                else _stamp_to_float(current_observation.center_image.header.stamp)
            )
            updated_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=newer_than_sec,
            )
            if updated_observation is not None:
                current_observation = updated_observation
            current_pose = target_pose
            primitive_features.append(
                {
                    "segment_index": segment_index,
                    "delta_base": [float(value) for value in delta_base],
                    "feature": self._sc_learned_feature(task, current_observation),
                    "pose": _pose_to_dict(current_pose),
                }
            )

        return current_pose, current_observation, {
            "hover_feature": hover_feature,
            "primitive_features": primitive_features,
        }

    def _collect_learning_pose_sweep(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        anchor_observation: Observation | None,
        phase_prefix: str,
        debug_run: DebugRun,
    ) -> None:
        if anchor_observation is None:
            return
        anchor_pose = anchor_observation.controller_state.tcp_pose
        rotation_base_from_optical = self._lookup_camera_rotation_base_from_optical(
            self._CENTER_CAMERA_FRAME
        )
        if rotation_base_from_optical is None:
            return
        offsets_optical = [
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([0.010, 0.0, 0.0], dtype=float),
            np.array([-0.010, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.010, 0.0], dtype=float),
            np.array([0.0, -0.010, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.012], dtype=float),
            np.array([0.0, 0.0, -0.012], dtype=float),
            np.array([0.008, 0.008, 0.0], dtype=float),
            np.array([0.008, -0.008, 0.0], dtype=float),
            np.array([-0.008, 0.008, 0.0], dtype=float),
            np.array([-0.008, -0.008, 0.0], dtype=float),
            np.array([0.006, 0.0, 0.010], dtype=float),
            np.array([-0.006, 0.0, 0.010], dtype=float),
        ]
        last_stamp_sec = _stamp_to_float(anchor_observation.center_image.header.stamp)
        for offset_index, offset_optical in enumerate(offsets_optical):
            target_pose = self._make_pose(
                self._pose_position(anchor_pose)
                + (rotation_base_from_optical @ offset_optical),
                self._quat_xyzw_to_wxyz(anchor_pose.orientation),
            )
            self._move_for_duration(
                move_robot,
                anchor_pose,
                target_pose,
                duration_sec=0.20,
                debug_run=debug_run,
                phase_name=f"{phase_prefix}_sweep_{offset_index}",
            )
            self._hold_pose(
                move_robot,
                target_pose,
                duration_sec=0.18,
                debug_run=debug_run,
                phase_name=f"{phase_prefix}_settle_{offset_index}",
            )
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if observation is None:
                continue
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)
            self._capture_gt_port_sample(
                task,
                observation,
                phase=f"{phase_prefix}_sweep",
                extra={
                    "offset_optical": [float(value) for value in offset_optical],
                    "feature_summary": self._extract_feature_summary(task, observation),
                },
            )

    def _run_stage_learn_collect_v0(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        self._capture_gt_port_sample(
            task,
            initial_observation,
            phase="initial",
            extra={"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        self._collect_learning_pose_sweep(
            task,
            get_observation,
            move_robot,
            initial_observation,
            phase_prefix="initial",
            debug_run=debug_run,
        )
        send_feedback("learn_collect_v0: collecting a second, slightly closer sweep anchor")
        rotation_base_from_optical = self._lookup_camera_rotation_base_from_optical(
            self._CENTER_CAMERA_FRAME
        )
        if rotation_base_from_optical is not None:
            closer_pose = self._make_pose(
                self._pose_position(initial_observation.controller_state.tcp_pose)
                + (rotation_base_from_optical @ np.array([0.0, 0.0, 0.020], dtype=float)),
                self._quat_xyzw_to_wxyz(initial_observation.controller_state.tcp_pose.orientation),
            )
            self._move_for_duration(
                move_robot,
                initial_observation.controller_state.tcp_pose,
                closer_pose,
                duration_sec=0.35,
                debug_run=debug_run,
                phase_name="closer_anchor",
            )
            self._hold_pose(
                move_robot,
                closer_pose,
                duration_sec=0.20,
                debug_run=debug_run,
                phase_name="closer_anchor_settle",
            )
            closer_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=_stamp_to_float(initial_observation.center_image.header.stamp),
            )
            debug_run.save_observation_snapshot(
                "closer_anchor",
                closer_observation,
                {
                    "feature_summary": self._extract_feature_summary(
                        task, closer_observation
                    ),
                    "closer_pose": _pose_to_dict(closer_pose),
                },
            )
            if closer_observation is not None:
                self._capture_gt_port_sample(
                    task,
                    closer_observation,
                    phase="closer_anchor",
                    extra={
                        "feature_summary": self._extract_feature_summary(
                            task, closer_observation
                        ),
                        "closer_pose": _pose_to_dict(closer_pose),
                    },
                )
                self._collect_learning_pose_sweep(
                    task,
                    get_observation,
                    move_robot,
                    closer_observation,
                    phase_prefix="closer",
                    debug_run=debug_run,
                )

        final_observation = self._wait_for_observation(get_observation, timeout_sec=1.0)
        debug_run.finalize(
            True,
            "learn_collect_v0 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"learn_collect_v0 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_learn_collect_v1(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if task.plug_type != "sc":
            return self._run_stage_learn_collect_v0(
                task, get_observation, move_robot, send_feedback
            )

        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        port_frame = self._port_frame_name(task)
        baseline_force = self._force_norm(initial_observation)
        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        self._capture_gt_port_sample(
            task,
            initial_observation,
            phase="initial",
            extra={
                "feature_summary": self._extract_feature_summary(task, initial_observation),
                "teacher_mode": "gt_cheatcode_trajectory",
            },
        )

        send_feedback("learn_collect_v1: following GT teacher trajectory for near-port data")
        for step in range(0, 81):
            interp_fraction = step / 80.0
            target_pose = self._gt_teacher_gripper_pose(
                task,
                port_frame,
                slerp_fraction=interp_fraction,
                position_fraction=interp_fraction,
                z_offset=0.18,
                reset_xy_integrator=True,
            )
            if target_pose is None:
                debug_run.finalize(
                    False,
                    "gt teacher pose unavailable during hover interpolation",
                    initial_observation,
                )
                return False
            self.set_pose_target(move_robot=move_robot, pose=target_pose)
            if step == 0 or step == 80 or step % 10 == 0:
                debug_run.log_command_sample(
                    "gt_teacher_hover",
                    target_pose,
                    self.time_now().nanoseconds / 1e9,
                    note=f"hover step {step}/80",
                )
            self._wait_for_sim_progress(0.05)
            if step % 5 != 0:
                continue
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.5,
                newer_than_sec=last_stamp_sec,
            )
            if observation is None:
                continue
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)
            self._capture_gt_port_sample(
                task,
                observation,
                phase="teacher_hover",
                extra={
                    "hover_fraction": float(interp_fraction),
                    "teacher_pose": _pose_to_dict(target_pose),
                    "feature_summary": self._extract_feature_summary(task, observation),
                },
            )

        hover_observation = self._wait_for_observation(
            get_observation,
            timeout_sec=0.8,
            newer_than_sec=last_stamp_sec,
        )
        debug_run.save_observation_snapshot(
            "teacher_hover",
            hover_observation,
            {
                "feature_summary": self._extract_feature_summary(task, hover_observation),
                "baseline_force_norm": baseline_force,
            },
        )
        if hover_observation is not None:
            last_stamp_sec = _stamp_to_float(hover_observation.center_image.header.stamp)

        z_offset = 0.18
        final_observation = hover_observation
        for step in range(90):
            z_offset -= 0.002
            target_pose = self._gt_teacher_gripper_pose(
                task,
                port_frame,
                z_offset=z_offset,
                reset_xy_integrator=False,
            )
            if target_pose is None:
                break
            self.set_pose_target(move_robot=move_robot, pose=target_pose)
            if step == 0 or step == 89 or step % 10 == 0:
                debug_run.log_command_sample(
                    "gt_teacher_insert",
                    target_pose,
                    self.time_now().nanoseconds / 1e9,
                    note=f"insert step {step + 1}/90 z_offset={z_offset:.4f}",
                )
            self._wait_for_sim_progress(0.05)
            if step % 4 != 0:
                continue
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.5,
                newer_than_sec=last_stamp_sec,
            )
            if observation is None:
                continue
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)
            final_observation = observation
            measured_force = self._force_norm(observation)
            self._capture_gt_port_sample(
                task,
                observation,
                phase="teacher_insert",
                extra={
                    "teacher_z_offset": float(z_offset),
                    "measured_force_norm": measured_force,
                    "baseline_force_norm": baseline_force,
                    "feature_summary": self._extract_feature_summary(task, observation),
                },
            )
            if measured_force - baseline_force > 10.0:
                break

        debug_run.save_observation_snapshot(
            "teacher_insert_final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "baseline_force_norm": baseline_force,
            },
        )
        debug_run.finalize(
            True,
            "learn_collect_v1 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"learn_collect_v1 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_learn_collect_v2(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        port_frame = self._port_frame_name(task)

        self._capture_gt_port_sample(
            task,
            initial_observation,
            phase="initial",
            extra={
                "feature_summary": self._extract_feature_summary(task, initial_observation),
                "teacher_mode": "gt_teacher_v2",
            },
        )
        self._collect_learning_pose_sweep(
            task,
            get_observation,
            move_robot,
            initial_observation,
            phase_prefix="initial",
            debug_run=debug_run,
        )

        teacher_anchors = [
            ("teacher_hover", 0.16 if task.plug_type == "sfp" else 0.18),
            ("teacher_insert", 0.08 if task.plug_type == "sfp" else 0.11),
        ]
        current_observation = initial_observation

        send_feedback("learn_collect_v2: sweeping around GT teacher anchors")
        for phase_name, z_offset in teacher_anchors:
            target_pose = self._gt_teacher_gripper_pose(
                task,
                port_frame,
                slerp_fraction=1.0,
                position_fraction=1.0,
                z_offset=z_offset,
                reset_xy_integrator=(phase_name == "teacher_hover"),
            )
            if target_pose is None:
                debug_run.finalize(
                    False, f"gt teacher pose unavailable during {phase_name}", initial_observation
                )
                return False
            start_pose = (
                current_observation.controller_state.tcp_pose
                if current_observation is not None
                else initial_observation.controller_state.tcp_pose
            )
            self._move_for_duration(
                move_robot,
                start_pose,
                target_pose,
                duration_sec=0.40 if task.plug_type == "sfp" else 0.45,
                debug_run=debug_run,
                phase_name=f"{phase_name}_anchor_move",
            )
            self._hold_pose(
                move_robot,
                target_pose,
                duration_sec=0.20,
                debug_run=debug_run,
                phase_name=f"{phase_name}_anchor_settle",
            )
            current_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
            )
            self._capture_gt_port_sample(
                task,
                current_observation,
                phase=phase_name,
                extra={
                    "teacher_z_offset": float(z_offset),
                    "teacher_pose": _pose_to_dict(target_pose),
                    "feature_summary": self._extract_feature_summary(task, current_observation),
                },
            )
            debug_run.save_observation_snapshot(
                phase_name,
                current_observation,
                {
                    "feature_summary": self._extract_feature_summary(task, current_observation),
                    "teacher_pose": _pose_to_dict(target_pose),
                },
            )
            self._collect_learning_pose_sweep(
                task,
                get_observation,
                move_robot,
                current_observation,
                phase_prefix=phase_name,
                debug_run=debug_run,
            )

        final_observation = self._wait_for_observation(get_observation, timeout_sec=1.0)
        debug_run.save_observation_snapshot(
            "teacher_insert_final",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        debug_run.finalize(
            True,
            "learn_collect_v2 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"learn_collect_v2 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_learn_collect_sfp_insert_v0(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if task.plug_type != "sfp":
            return self._run_stage_learn_collect_v2(
                task, get_observation, move_robot, send_feedback
            )

        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        send_feedback("learn_collect_sfp_insert_v0: collecting GT teacher_insert sweeps")
        port_frame = self._port_frame_name(task)
        teacher_insert_pose = self._gt_teacher_gripper_pose(
            task,
            port_frame,
            slerp_fraction=1.0,
            position_fraction=1.0,
            z_offset=0.08,
            reset_xy_integrator=True,
        )
        if teacher_insert_pose is None:
            debug_run.finalize(False, "gt teacher insert pose unavailable", initial_observation)
            return False

        self._move_for_duration(
            move_robot,
            initial_observation.controller_state.tcp_pose,
            teacher_insert_pose,
            duration_sec=0.35,
            debug_run=debug_run,
            phase_name="teacher_insert_anchor_move",
        )
        self._hold_pose(
            move_robot,
            teacher_insert_pose,
            duration_sec=0.16,
            debug_run=debug_run,
            phase_name="teacher_insert_anchor_settle",
        )
        teacher_observation = self._wait_for_observation(
            get_observation,
            timeout_sec=0.8,
        )
        self._capture_gt_port_sample(
            task,
            teacher_observation,
            phase="teacher_insert",
            extra={
                "teacher_pose": _pose_to_dict(teacher_insert_pose),
                "teacher_z_offset": 0.08,
                "feature_summary": self._extract_feature_summary(task, teacher_observation),
            },
        )
        debug_run.save_observation_snapshot(
            "teacher_insert",
            teacher_observation,
            {
                "feature_summary": self._extract_feature_summary(task, teacher_observation),
                "teacher_pose": _pose_to_dict(teacher_insert_pose),
            },
        )
        self._collect_learning_pose_sweep(
            task,
            get_observation,
            move_robot,
            teacher_observation,
            phase_prefix="teacher_insert",
            debug_run=debug_run,
        )

        final_observation = self._wait_for_observation(get_observation, timeout_sec=1.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        debug_run.finalize(
            True,
            "learn_collect_sfp_insert_v0 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(
            f"learn_collect_sfp_insert_v0 artifacts saved to {debug_run.root}"
        )
        return True

    def _run_stage_learn_collect_sfp_servo_residual_v0(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if task.plug_type != "sfp":
            return self._run_stage_learn_collect_v2(
                task, get_observation, move_robot, send_feedback
            )

        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        send_feedback(
            "learn_collect_sfp_servo_residual_v0: collecting GT labels from the legal SFP servo state"
        )
        aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
            initial_observation,
            get_observation,
            move_robot,
            debug_run,
        )
        debug_run.save_observation_snapshot(
            "after_legal_servo",
            aligned_observation,
            {
                "feature_summary": self._extract_feature_summary(task, aligned_observation),
                "feature": aligned_feature,
                "aligned_pose": None if aligned_pose is None else _pose_to_dict(aligned_pose),
            },
        )
        if aligned_observation is None:
            debug_run.finalize(
                False,
                "learn_collect_sfp_servo_residual_v0 failed to get post-servo observation",
                initial_observation,
            )
            return False

        self._capture_gt_port_sample(
            task,
            aligned_observation,
            phase="post_legal_servo",
            extra={
                "feature": aligned_feature,
                "feature_summary": self._extract_feature_summary(task, aligned_observation),
                "aligned_pose": None if aligned_pose is None else _pose_to_dict(aligned_pose),
            },
        )
        self._collect_learning_pose_sweep(
            task,
            get_observation,
            move_robot,
            aligned_observation,
            phase_prefix="post_legal_servo",
            debug_run=debug_run,
        )

        final_observation = self._wait_for_observation(get_observation, timeout_sec=1.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        debug_run.finalize(
            True,
            "learn_collect_sfp_servo_residual_v0 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(
            f"learn_collect_sfp_servo_residual_v0 artifacts saved to {debug_run.root}"
        )
        return True

    def _run_stage_learn_collect_sfp_teacher_step_v0(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if task.plug_type != "sfp":
            return self._run_stage_learn_collect_v2(
                task, get_observation, move_robot, send_feedback
            )

        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        send_feedback(
            "learn_collect_sfp_teacher_step_v0: collecting small GT teacher steps from the legal SFP servo state"
        )
        aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
            initial_observation,
            get_observation,
            move_robot,
            debug_run,
        )
        debug_run.save_observation_snapshot(
            "after_legal_servo",
            aligned_observation,
            {
                "feature_summary": self._extract_feature_summary(task, aligned_observation),
                "feature": aligned_feature,
                "aligned_pose": None if aligned_pose is None else _pose_to_dict(aligned_pose),
            },
        )
        if aligned_pose is None or aligned_observation is None:
            debug_run.finalize(
                False,
                "learn_collect_sfp_teacher_step_v0 failed to get post-servo observation",
                initial_observation,
            )
            return False

        teacher_insert_pose = self._gt_teacher_gripper_pose(
            task,
            self._port_frame_name(task),
            slerp_fraction=1.0,
            position_fraction=1.0,
            z_offset=0.08,
            reset_xy_integrator=True,
        )
        if teacher_insert_pose is None:
            debug_run.finalize(
                False,
                "learn_collect_sfp_teacher_step_v0 teacher insert pose unavailable",
                aligned_observation,
            )
            return False

        current_observation: Observation | None = aligned_observation
        current_pose = self._observed_tcp_pose(aligned_observation, aligned_pose)
        last_stamp_sec = _stamp_to_float(aligned_observation.center_image.header.stamp)
        max_teacher_steps = 8

        for step_index in range(max_teacher_steps):
            current_pose = self._observed_tcp_pose(current_observation, current_pose)
            remaining_translation_delta, remaining_rotation_delta = self._pose_delta_base(
                current_pose,
                teacher_insert_pose,
            )
            if float(np.linalg.norm(remaining_translation_delta)) < 0.003 and float(
                np.linalg.norm(remaining_rotation_delta)
            ) < 0.05:
                break
            applied_translation_delta, applied_rotation_delta = (
                self._clamp_sfp_teacher_step_delta(
                    remaining_translation_delta,
                    remaining_rotation_delta,
                )
            )
            next_pose = self._apply_pose_residual_base(
                current_pose,
                translation_delta_base=applied_translation_delta,
                rotation_delta_rotvec_base=applied_rotation_delta,
            )
            self._capture_gt_port_sample(
                task,
                current_observation,
                phase="post_legal_servo_teacher_step",
                extra={
                    "teacher_step_index": int(step_index + 1),
                    "remaining_translation_norm_m": float(
                        np.linalg.norm(remaining_translation_delta)
                    ),
                    "remaining_rotation_norm_rad": float(
                        np.linalg.norm(remaining_rotation_delta)
                    ),
                    "applied_translation_delta_base": [
                        float(value) for value in applied_translation_delta
                    ],
                    "applied_rotation_delta_rotvec_base": [
                        float(value) for value in applied_rotation_delta
                    ],
                    "feature_summary": self._extract_feature_summary(
                        task, current_observation
                    ),
                },
                label_overrides={"teacher_step_pose": _pose_to_dict(next_pose)},
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                next_pose,
                duration_sec=0.20,
                debug_run=debug_run,
                phase_name=f"teacher_step_move_{step_index + 1}",
            )
            self._hold_pose(
                move_robot,
                next_pose,
                duration_sec=0.08,
                debug_run=debug_run,
                phase_name=f"teacher_step_settle_{step_index + 1}",
            )
            next_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if next_observation is None:
                current_pose = next_pose
                current_observation = current_observation
                continue
            current_observation = next_observation
            last_stamp_sec = _stamp_to_float(next_observation.center_image.header.stamp)
            current_pose = self._observed_tcp_pose(next_observation, next_pose)

        final_observation = self._wait_for_observation(
            get_observation,
            timeout_sec=1.0,
            newer_than_sec=last_stamp_sec,
        )
        if final_observation is None:
            final_observation = current_observation
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        debug_run.finalize(
            True,
            "learn_collect_sfp_teacher_step_v0 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(
            f"learn_collect_sfp_teacher_step_v0 artifacts saved to {debug_run.root}"
        )
        return True

    def _run_sfp_gt_teacher_insert(
        self,
        task: Task,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
    ) -> tuple[Pose | None, Observation | None, dict[str, object]]:
        port_frame = self._port_frame_name(task)
        current_observation: Observation | None = initial_observation
        current_pose = self._observed_tcp_pose(
            initial_observation,
            initial_observation.controller_state.reference_tcp_pose,
        )
        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        phase_records: list[dict[str, object]] = []
        teacher_schedule = (
            ("teacher_hover", 0.16, 0.38, 0.14),
            ("teacher_preinsert_far", 0.10, 0.28, 0.10),
            ("teacher_preinsert_mid", 0.06, 0.24, 0.10),
            ("teacher_preinsert_near", 0.04, 0.20, 0.08),
            ("teacher_insert_1", 0.025, 0.18, 0.08),
            ("teacher_insert_2", 0.015, 0.16, 0.08),
            ("teacher_insert_3", 0.008, 0.14, 0.08),
            ("teacher_insert_4", 0.003, 0.12, 0.10),
        )

        for phase_index, (phase_name, z_offset, move_sec, hold_sec) in enumerate(
            teacher_schedule, start=1
        ):
            target_pose = self._gt_teacher_gripper_pose(
                task,
                port_frame,
                slerp_fraction=1.0,
                position_fraction=1.0,
                z_offset=z_offset,
                reset_xy_integrator=(phase_index == 1),
            )
            if target_pose is None:
                return None, current_observation, {
                    "phase_records": phase_records,
                    "failed_phase": phase_name,
                    "message": "gt teacher pose unavailable",
                }

            remaining_translation_delta, remaining_rotation_delta = self._pose_delta_base(
                current_pose,
                target_pose,
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=move_sec,
                debug_run=debug_run,
                phase_name=f"{phase_name}_move",
            )
            self._hold_pose(
                move_robot,
                target_pose,
                duration_sec=hold_sec,
                debug_run=debug_run,
                phase_name=f"{phase_name}_settle",
            )
            next_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if next_observation is not None:
                current_observation = next_observation
                last_stamp_sec = _stamp_to_float(next_observation.center_image.header.stamp)
                current_pose = self._observed_tcp_pose(next_observation, target_pose)
            else:
                current_pose = target_pose

            phase_records.append(
                {
                    "phase_name": phase_name,
                    "z_offset": float(z_offset),
                    "move_duration_sec": float(move_sec),
                    "hold_duration_sec": float(hold_sec),
                    "remaining_translation_norm_m": float(
                        np.linalg.norm(remaining_translation_delta)
                    ),
                    "remaining_rotation_norm_rad": float(
                        np.linalg.norm(remaining_rotation_delta)
                    ),
                    "target_pose": _pose_to_dict(target_pose),
                }
            )

        return current_pose, current_observation, {
            "port_frame": port_frame,
            "phase_records": phase_records,
        }

    def _run_sfp_gt_teacher_terminal_insert(
        self,
        task: Task,
        initial_pose: Pose,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
    ) -> tuple[Pose, Observation | None, dict[str, object]]:
        current_pose = initial_pose
        current_observation: Observation | None = initial_observation
        last_stamp_sec = _stamp_to_float(initial_observation.center_image.header.stamp)
        phase_records: list[dict[str, object]] = []
        contact_schedule = (
            ("teacher_contact_1", 0.0015, 0.12, 0.12),
            ("teacher_contact_2", 0.0, 0.12, 0.14),
            ("teacher_contact_3", -0.0015, 0.12, 0.16),
            ("teacher_contact_4", -0.003, 0.12, 0.18),
        )

        for phase_name, z_offset, move_sec, hold_sec in contact_schedule:
            target_pose = self._gt_teacher_gripper_pose(
                task,
                self._port_frame_name(task),
                slerp_fraction=1.0,
                position_fraction=1.0,
                z_offset=z_offset,
            )
            if target_pose is None:
                break
            remaining_translation_delta, remaining_rotation_delta = self._pose_delta_base(
                current_pose,
                target_pose,
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=move_sec,
                debug_run=debug_run,
                phase_name=f"{phase_name}_move",
            )
            self._hold_pose(
                move_robot,
                target_pose,
                duration_sec=hold_sec,
                debug_run=debug_run,
                phase_name=f"{phase_name}_settle",
            )
            next_observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if next_observation is not None:
                current_observation = next_observation
                last_stamp_sec = _stamp_to_float(next_observation.center_image.header.stamp)
                current_pose = self._observed_tcp_pose(next_observation, target_pose)
            else:
                current_pose = target_pose
            phase_records.append(
                {
                    "phase_name": phase_name,
                    "z_offset": float(z_offset),
                    "move_duration_sec": float(move_sec),
                    "hold_duration_sec": float(hold_sec),
                    "remaining_translation_norm_m": float(
                        np.linalg.norm(remaining_translation_delta)
                    ),
                    "remaining_rotation_norm_rad": float(
                        np.linalg.norm(remaining_rotation_delta)
                    ),
                    "target_pose": _pose_to_dict(target_pose),
                }
            )

        return current_pose, current_observation, {"phase_records": phase_records}

    def _run_stage_teacher_feasibility_v0(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False
        if task.plug_type != "sfp":
            debug_run.finalize(
                False,
                "teacher_feasibility_v0 only supports SFP tasks",
                initial_observation,
                {"task_key": list(self._task_key(task))},
            )
            return False

        debug_run.save_target_metadata(
            {
                "provider": "ground_truth_teacher_feasibility_only",
                "task_key": list(self._task_key(task)),
                "port_frame": self._port_frame_name(task),
                "notes": [
                    "Uses ground-truth port and plug frames to test controller feasibility only",
                    "This stage is not submission-safe and exists only to gate the teacher route",
                    "Success criterion is insertion-led randomized SFP score, not proximity polish",
                ],
            }
        )

        send_feedback("teacher_feasibility_v0: GT teacher approach + bounded terminal push")
        teacher_pose, teacher_observation, teacher_metadata = self._run_sfp_gt_teacher_insert(
            task,
            initial_observation,
            get_observation,
            move_robot,
            debug_run,
        )
        debug_run.save_observation_snapshot(
            "after_gt_teacher",
            teacher_observation,
            {
                "feature_summary": self._extract_feature_summary(task, teacher_observation),
                "teacher_metadata": teacher_metadata,
                "teacher_pose": None if teacher_pose is None else _pose_to_dict(teacher_pose),
            },
        )
        if teacher_pose is None or teacher_observation is None:
            debug_run.finalize(
                False,
                "teacher_feasibility_v0 teacher approach failed",
                teacher_observation,
                {"teacher_metadata": teacher_metadata},
            )
            return False

        terminal_pose, terminal_observation, terminal_metadata = (
            self._run_sfp_gt_teacher_terminal_insert(
                task,
                teacher_pose,
                teacher_observation,
                get_observation,
                move_robot,
                debug_run,
            )
        )
        debug_run.save_observation_snapshot(
            "after_teacher_terminal",
            terminal_observation,
            {
                "feature_summary": self._extract_feature_summary(task, terminal_observation),
                "teacher_pose": _pose_to_dict(terminal_pose),
                "terminal_metadata": terminal_metadata,
            },
        )

        final_pose, final_observation, insertion_feature = self._run_sfp_incremental_insertion(
            terminal_observation if terminal_observation is not None else teacher_observation,
            get_observation,
            move_robot,
            debug_run,
            aligned_feature=None,
            push_schedule_m=(0.0015, 0.0015, 0.001, 0.001),
        )
        debug_run.save_observation_snapshot(
            "after_teacher_insertion",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "feature": insertion_feature,
                "final_pose": _pose_to_dict(final_pose),
                "teacher_metadata": teacher_metadata,
                "terminal_metadata": terminal_metadata,
            },
        )
        debug_run.finalize(
            True,
            "teacher_feasibility_v0 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"teacher_feasibility_v0 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v17(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "learned_sfp_model_dir": self._learned_sfp_model_dir,
                "notes": [
                    "SFP path reuses the stable submission_safe_v6 legal servo timing",
                    "SFP then runs a learned closed-loop small-step policy trained from GT teacher trajectory snippets",
                    "SC path matches submission_safe_v6",
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback(
                "submission_v17: SFP legal servo + learned teacher-step closed loop"
            )
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None or aligned_observation is None:
                debug_run.finalize(
                    False,
                    "submission_v17 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            learned_pose, learned_observation, learned_metadata = (
                self._run_sfp_submission_learned_teacher_step_policy(
                    task,
                    aligned_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_learned_teacher_step",
                learned_observation,
                {
                    "feature_summary": self._extract_feature_summary(
                        task, learned_observation
                    ),
                    "feature": self._sfp_union_feature(learned_observation),
                    "learned_metadata": learned_metadata,
                    "learned_pose": None if learned_pose is None else _pose_to_dict(learned_pose),
                },
            )
            if learned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v17 learned teacher-step policy unavailable",
                    learned_observation,
                    {"learned_metadata": learned_metadata},
                )
                return False

            final_pose, final_observation, insertion_feature = (
                self._run_sfp_incremental_insertion(
                    learned_observation
                    if learned_observation is not None
                    else aligned_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                    aligned_feature,
                )
            )
            debug_run.save_observation_snapshot(
                "after_insertion_cycles",
                final_observation,
                {
                    "feature_summary": self._extract_feature_summary(task, final_observation),
                    "feature": insertion_feature,
                    "final_pose": _pose_to_dict(final_pose),
                },
            )
        else:
            return self._run_stage_submission_safe_v6(
                task, get_observation, move_robot, send_feedback
            )

        debug_run.finalize(
            True,
            "submission_safe_v17 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v17 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v3(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "notes": [
                    "no public-sample world targets",
                    "no development-only target provider",
                    "SFP path matches submission_safe_v0",
                    "SC path triangulates the cyan port feature from wrist cameras",
                    "SC legal servo couples bounded translation with bounded optical-axis rotation",
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback("submission_v3: SFP center-camera legal servo")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running SFP visual servo using only current observation",
            )
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v3 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            send_feedback("submission_v3: SFP bounded tool-axis push")
            debug_run.log_phase(
                "insert",
                self.time_now().nanoseconds / 1e9,
                "applying short tool-axis insertion push from legal servo pose",
            )
            final_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SFP_TOOL_INSERTION_AXIS,
                push_distance_m=0.012,
                duration_sec=0.45,
                hold_sec=0.8,
                debug_run=debug_run,
                phase_prefix="submission_sfp",
            )
        else:
            send_feedback("submission_v3: SC triangulated legal servo")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running SC triangulated visual servo using only current observation",
            )
            aligned_pose, aligned_observation, aligned_feature = (
                self._run_sc_submission_triangulated_coarse_to_fine(
                    initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v3 SC servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            baseline_force_norm = self._force_norm(initial_observation)
            debug_run.log_phase(
                "insert",
                self.time_now().nanoseconds / 1e9,
                "applying short SC tool-axis push followed by bounded legal refinement",
            )
            pushed_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SC_TOOL_INSERTION_AXIS,
                push_distance_m=0.004,
                duration_sec=0.35,
                hold_sec=0.25,
                debug_run=debug_run,
                phase_prefix="submission_sc",
            )
            refined_pose = self._run_sc_force_refine(
                get_observation,
                move_robot,
                debug_run,
                start_pose=pushed_pose,
                baseline_force_norm=baseline_force_norm,
            )
            refined_observation = self._wait_for_observation(
                get_observation, timeout_sec=0.8
            )
            debug_run.save_observation_snapshot(
                "after_force_refine",
                refined_observation,
                {
                    "feature_summary": self._extract_feature_summary(
                        task, refined_observation
                    ),
                    "feature": self._sc_triangulated_feature(refined_observation),
                    "refined_pose": _pose_to_dict(refined_pose),
                },
            )
            debug_run.log_phase(
                "residual",
                self.time_now().nanoseconds / 1e9,
                "applying legal SC residual correction",
            )
            final_pose = self._run_sc_visual_residual(
                get_observation,
                move_robot,
                debug_run,
                refined_pose,
                baseline_force_norm=baseline_force_norm,
            )

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "final_pose": _pose_to_dict(final_pose),
            },
        )
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            "submission_safe_v3 returned True",
        )
        debug_run.finalize(
            True,
            "submission_safe_v3 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v3 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v4(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "notes": [
                    "no public-sample world targets",
                    "no development-only target provider",
                    "SFP path matches submission_safe_v0",
                    "SC path triangulates the cyan port feature from wrist cameras",
                    "SC legal servo uses bounded translation only",
                    "SC orientation correction is disabled for v3/v4 A-B debugging",
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback("submission_v4: SFP center-camera legal servo")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running SFP visual servo using only current observation",
            )
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v4 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            send_feedback("submission_v4: SFP bounded tool-axis push")
            debug_run.log_phase(
                "insert",
                self.time_now().nanoseconds / 1e9,
                "applying short tool-axis insertion push from legal servo pose",
            )
            final_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SFP_TOOL_INSERTION_AXIS,
                push_distance_m=0.012,
                duration_sec=0.45,
                hold_sec=0.8,
                debug_run=debug_run,
                phase_prefix="submission_sfp",
            )
        else:
            send_feedback("submission_v4: SC triangulated translation-only servo")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running SC triangulated translation-only servo using only current observation",
            )
            aligned_pose, aligned_observation, aligned_feature = (
                self._run_sc_submission_triangulated_coarse_to_fine(
                    initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                    coarse_rotation_gain=0.0,
                    coarse_max_rotation_step_deg=0.0,
                    fine_rotation_gain=0.0,
                    fine_max_rotation_step_deg=0.0,
                )
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v4 SC servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            baseline_force_norm = self._force_norm(initial_observation)
            debug_run.log_phase(
                "insert",
                self.time_now().nanoseconds / 1e9,
                "applying short SC tool-axis push followed by bounded legal refinement",
            )
            pushed_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SC_TOOL_INSERTION_AXIS,
                push_distance_m=0.004,
                duration_sec=0.35,
                hold_sec=0.25,
                debug_run=debug_run,
                phase_prefix="submission_sc",
            )
            refined_pose = self._run_sc_force_refine(
                get_observation,
                move_robot,
                debug_run,
                start_pose=pushed_pose,
                baseline_force_norm=baseline_force_norm,
            )
            refined_observation = self._wait_for_observation(
                get_observation, timeout_sec=0.8
            )
            debug_run.save_observation_snapshot(
                "after_force_refine",
                refined_observation,
                {
                    "feature_summary": self._extract_feature_summary(
                        task, refined_observation
                    ),
                    "feature": self._sc_triangulated_feature(refined_observation),
                    "refined_pose": _pose_to_dict(refined_pose),
                },
            )
            debug_run.log_phase(
                "residual",
                self.time_now().nanoseconds / 1e9,
                "applying legal SC residual correction",
            )
            final_pose = self._run_sc_visual_residual(
                get_observation,
                move_robot,
                debug_run,
                refined_pose,
                baseline_force_norm=baseline_force_norm,
            )

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "final_pose": _pose_to_dict(final_pose),
            },
        )
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            "submission_safe_v4 returned True",
        )
        debug_run.finalize(
            True,
            "submission_safe_v4 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v4 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v6(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "learned_model_dir": self._learned_model_dir,
                "notes": [
                    "SFP path matches submission_safe_v4",
                    "SC path replaces triangulated translation-only servo with learned multi-view center_uvz prediction",
                    (
                        "SC force refine and residual stages are enabled"
                        if self._enable_sc_refinement
                        else "SC refinement is disabled for fail-fast isolation of learned target acquisition"
                    ),
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback("submission_v6: SFP center-camera legal servo")
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v6 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False
            final_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SFP_TOOL_INSERTION_AXIS,
                push_distance_m=0.012,
                duration_sec=0.45,
                hold_sec=0.8,
                debug_run=debug_run,
                phase_prefix="submission_sfp",
            )
        else:
            if self._learned_port_inference_or_none() is None:
                debug_run.finalize(
                    False,
                    "submission_v6 requires AIC_QUAL_LEARNED_SC_MODEL_DIR",
                    initial_observation,
                )
                return False
            send_feedback("submission_v6: SC learned multi-view translation servo")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running learned SC translation-only servo using current observation",
            )
            aligned_pose, aligned_observation, aligned_feature = (
                self._run_sc_submission_learned_coarse_to_fine(
                    task,
                    initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v6 SC learned servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            baseline_force_norm = self._force_norm(initial_observation)
            pushed_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SC_TOOL_INSERTION_AXIS,
                push_distance_m=0.004,
                duration_sec=0.35,
                hold_sec=0.25,
                debug_run=debug_run,
                phase_prefix="submission_sc",
            )
            if self._enable_sc_refinement:
                refined_pose = self._run_sc_force_refine(
                    get_observation,
                    move_robot,
                    debug_run,
                    start_pose=pushed_pose,
                    baseline_force_norm=baseline_force_norm,
                )
                refined_observation = self._wait_for_observation(
                    get_observation, timeout_sec=0.8
                )
                debug_run.save_observation_snapshot(
                    "after_force_refine",
                    refined_observation,
                    {
                        "feature_summary": self._extract_feature_summary(
                            task, refined_observation
                        ),
                        "feature": self._sc_learned_feature(task, refined_observation),
                        "refined_pose": _pose_to_dict(refined_pose),
                    },
                )
                final_pose = self._run_sc_visual_residual(
                    get_observation,
                    move_robot,
                    debug_run,
                    refined_pose,
                    baseline_force_norm=baseline_force_norm,
                )
            else:
                pushed_observation = self._wait_for_observation(
                    get_observation, timeout_sec=0.8
                )
                debug_run.save_observation_snapshot(
                    "after_push",
                    pushed_observation,
                    {
                        "feature_summary": self._extract_feature_summary(
                            task, pushed_observation
                        ),
                        "feature": self._sc_learned_feature(task, pushed_observation),
                        "pushed_pose": _pose_to_dict(pushed_pose),
                    },
                )
                final_pose = pushed_pose

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "final_pose": _pose_to_dict(final_pose),
            },
        )
        debug_run.finalize(
            True,
            "submission_safe_v6 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v6 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v7(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "learned_model_dir": self._learned_model_dir,
                "notes": [
                    "SFP path matches submission_safe_v4",
                    "SC hover uses learned multi-view center_uvz prediction",
                    "SC near-contact path uses a hover-relative primitive extracted from the successful SC-only run",
                    (
                        "SC force refine and residual stages are enabled"
                        if self._enable_sc_refinement
                        else "SC refinement is disabled for fail-fast isolation of hover-relative primitive insertion"
                    ),
                ],
            }
        )

        final_observation: Observation | None = None
        if task.plug_type == "sfp":
            send_feedback("submission_v7: SFP center-camera legal servo")
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v7 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False
            final_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SFP_TOOL_INSERTION_AXIS,
                push_distance_m=0.012,
                duration_sec=0.45,
                hold_sec=0.8,
                debug_run=debug_run,
                phase_prefix="submission_sfp",
            )
        else:
            if self._learned_port_inference_or_none() is None:
                debug_run.finalize(
                    False,
                    "submission_v7 requires AIC_QUAL_LEARNED_SC_MODEL_DIR",
                    initial_observation,
                )
                return False
            send_feedback("submission_v7: SC learned hover plus primitive insertion")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running learned hover alignment followed by hover-relative primitive insertion",
            )
            aligned_pose, aligned_observation, aligned_feature = (
                self._run_sc_submission_hover_then_primitive(
                    task,
                    initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v7 SC hover-plus-primitive failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            baseline_force_norm = self._force_norm(initial_observation)
            pushed_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SC_TOOL_INSERTION_AXIS,
                push_distance_m=0.004,
                duration_sec=0.35,
                hold_sec=0.25,
                debug_run=debug_run,
                phase_prefix="submission_sc",
            )
            if self._enable_sc_refinement:
                refined_pose = self._run_sc_force_refine(
                    get_observation,
                    move_robot,
                    debug_run,
                    start_pose=pushed_pose,
                    baseline_force_norm=baseline_force_norm,
                )
                refined_observation = self._wait_for_observation(
                    get_observation, timeout_sec=0.8
                )
                debug_run.save_observation_snapshot(
                    "after_force_refine",
                    refined_observation,
                    {
                        "feature_summary": self._extract_feature_summary(
                            task, refined_observation
                        ),
                        "feature": self._sc_learned_feature(task, refined_observation),
                        "refined_pose": _pose_to_dict(refined_pose),
                    },
                )
                final_observation = refined_observation
                final_pose = self._run_sc_visual_residual(
                    get_observation,
                    move_robot,
                    debug_run,
                    refined_pose,
                    baseline_force_norm=baseline_force_norm,
                )
            else:
                pushed_observation = aligned_observation
                debug_run.save_observation_snapshot(
                    "after_push",
                    pushed_observation,
                    {
                        "feature_summary": self._extract_feature_summary(
                            task, pushed_observation
                        ),
                        "feature": self._sc_learned_feature(task, pushed_observation),
                        "pushed_pose": _pose_to_dict(pushed_pose),
                    },
                )
                final_observation = pushed_observation
                final_pose = pushed_pose

        if final_observation is None:
            final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "final_pose": _pose_to_dict(final_pose),
            },
        )
        debug_run.finalize(
            True,
            "submission_safe_v7 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v7 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v8(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "learned_model_dir": self._learned_model_dir,
                "notes": [
                    "SFP path matches submission_safe_v4",
                    "SC path uses learned multi-view center_uvz prediction",
                    "SC post-alignment refinement is a legal tool-frame bounded force search only",
                    "No DEV_TARGETS-dependent residual is used in this stage",
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback("submission_v8: SFP center-camera legal servo")
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v8 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False
            final_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SFP_TOOL_INSERTION_AXIS,
                push_distance_m=0.012,
                duration_sec=0.45,
                hold_sec=0.8,
                debug_run=debug_run,
                phase_prefix="submission_sfp",
            )
        else:
            if self._learned_port_inference_or_none() is None:
                debug_run.finalize(
                    False,
                    "submission_v8 requires AIC_QUAL_LEARNED_SC_MODEL_DIR",
                    initial_observation,
                )
                return False
            send_feedback("submission_v8: SC learned servo plus legal tool-frame force search")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running learned SC translation servo followed by legal tool-frame force search",
            )
            aligned_pose, aligned_observation, aligned_feature = (
                self._run_sc_submission_learned_coarse_to_fine(
                    task,
                    initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v8 SC learned servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            baseline_force_norm = self._force_norm(initial_observation)
            pushed_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SC_TOOL_INSERTION_AXIS,
                push_distance_m=0.004,
                duration_sec=0.35,
                hold_sec=0.20,
                debug_run=debug_run,
                phase_prefix="submission_sc",
            )
            refined_pose = self._run_sc_force_refine_tool_frame(
                get_observation,
                move_robot,
                debug_run,
                start_pose=pushed_pose,
                baseline_force_norm=baseline_force_norm,
            )
            refined_observation = self._wait_for_observation(get_observation, timeout_sec=0.8)
            debug_run.save_observation_snapshot(
                "after_force_refine",
                refined_observation,
                {
                    "feature_summary": self._extract_feature_summary(
                        task, refined_observation
                    ),
                    "feature": self._sc_learned_feature(task, refined_observation),
                    "refined_pose": _pose_to_dict(refined_pose),
                },
            )
            final_pose = refined_pose

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "final_pose": _pose_to_dict(final_pose),
            },
        )
        debug_run.finalize(
            True,
            "submission_safe_v8 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v8 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v9(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "learned_model_dir": self._learned_model_dir,
                "notes": [
                    "SFP path replaces single blind push with observed-pose closed-loop insertion cycles",
                    "bag analysis showed the submission_safe_v6 SFP blind push could leave multi-centimeter tracking error",
                    "SC path matches submission_safe_v6",
                    (
                        "SC force refine and residual stages are enabled"
                        if self._enable_sc_refinement
                        else "SC refinement is disabled for fail-fast isolation"
                    ),
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback("submission_v9: SFP closed-loop insertion")
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v9 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False
            final_pose, final_observation, insertion_feature = self._run_sfp_closed_loop_insertion(
                aligned_observation if aligned_observation is not None else initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_insertion_cycles",
                final_observation,
                {
                    "feature_summary": self._extract_feature_summary(task, final_observation),
                    "feature": insertion_feature,
                    "final_pose": _pose_to_dict(final_pose),
                },
            )
        else:
            if self._learned_port_inference_or_none() is None:
                debug_run.finalize(
                    False,
                    "submission_v9 requires AIC_QUAL_LEARNED_SC_MODEL_DIR",
                    initial_observation,
                )
                return False
            send_feedback("submission_v9: SC learned multi-view translation servo")
            aligned_pose, aligned_observation, aligned_feature = (
                self._run_sc_submission_learned_coarse_to_fine(
                    task,
                    initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v9 SC learned servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            baseline_force_norm = self._force_norm(initial_observation)
            pushed_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SC_TOOL_INSERTION_AXIS,
                push_distance_m=0.004,
                duration_sec=0.35,
                hold_sec=0.25,
                debug_run=debug_run,
                phase_prefix="submission_sc",
            )
            if self._enable_sc_refinement:
                refined_pose = self._run_sc_force_refine(
                    get_observation,
                    move_robot,
                    debug_run,
                    start_pose=pushed_pose,
                    baseline_force_norm=baseline_force_norm,
                )
                refined_observation = self._wait_for_observation(
                    get_observation, timeout_sec=0.8
                )
                debug_run.save_observation_snapshot(
                    "after_force_refine",
                    refined_observation,
                    {
                        "feature_summary": self._extract_feature_summary(
                            task, refined_observation
                        ),
                        "feature": self._sc_learned_feature(task, refined_observation),
                        "refined_pose": _pose_to_dict(refined_pose),
                    },
                )
                final_pose = self._run_sc_visual_residual(
                    get_observation,
                    move_robot,
                    debug_run,
                    refined_pose,
                    baseline_force_norm=baseline_force_norm,
                )
            else:
                pushed_observation = self._wait_for_observation(
                    get_observation,
                    timeout_sec=0.8,
                )
                debug_run.save_observation_snapshot(
                    "after_push",
                    pushed_observation,
                    {
                        "feature_summary": self._extract_feature_summary(
                            task, pushed_observation
                        ),
                        "feature": self._sc_learned_feature(task, pushed_observation),
                        "pushed_pose": _pose_to_dict(pushed_pose),
                    },
                )
                final_pose = pushed_pose
            final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "final_pose": _pose_to_dict(final_pose),
            },
        )
        debug_run.finalize(
            True,
            "submission_safe_v9 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v9 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v10(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "learned_model_dir": self._learned_model_dir,
                "notes": [
                    "SFP path uses one legal servo pass followed by observed-tcp incremental insertion cycles",
                    "cycle timing is shortened for fail-fast debugging",
                    "mini recentering runs only when SFP image drift exceeds threshold",
                    "SC path matches submission_safe_v6",
                    (
                        "SC force refine and residual stages are enabled"
                        if self._enable_sc_refinement
                        else "SC refinement is disabled for fail-fast isolation"
                    ),
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback("submission_v10: SFP incremental insertion")
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
                max_steps=12,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v10 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False
            final_pose, final_observation, insertion_feature = (
                self._run_sfp_incremental_insertion(
                    aligned_observation if aligned_observation is not None else initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                    aligned_feature,
                )
            )
            debug_run.save_observation_snapshot(
                "after_insertion_cycles",
                final_observation,
                {
                    "feature_summary": self._extract_feature_summary(task, final_observation),
                    "feature": insertion_feature,
                    "final_pose": _pose_to_dict(final_pose),
                },
            )
        else:
            return self._run_stage_submission_safe_v6(
                task, get_observation, move_robot, send_feedback
            )

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "final_pose": _pose_to_dict(final_pose),
            },
        )
        debug_run.finalize(
            True,
            "submission_safe_v10 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v10 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v11(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "learned_sc_model_dir": self._learned_sc_model_dir,
                "learned_sfp_model_dir": self._learned_sfp_model_dir,
                "notes": [
                    "SFP path replaces green-mask legal servo with learned multi-view center_uvz acquisition",
                    "SFP insertion keeps the proven tool-axis push primitive from submission_safe_v6",
                    "SC path matches submission_safe_v6",
                    (
                        "SC force refine and residual stages are enabled"
                        if self._enable_sc_refinement
                        else "SC refinement is disabled for fail-fast isolation of learned target acquisition"
                    ),
                ],
            }
        )

        if task.plug_type == "sfp":
            if self._learned_port_inference_or_none("sfp") is None:
                debug_run.finalize(
                    False,
                    "submission_v11 requires AIC_QUAL_LEARNED_SFP_MODEL_DIR",
                    initial_observation,
                )
                return False
            send_feedback("submission_v11: SFP learned multi-view acquisition")
            aligned_pose, aligned_observation, aligned_feature = (
                self._run_sfp_submission_learned_coarse_to_fine(
                    task,
                    initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v11 SFP learned servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False
            final_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SFP_TOOL_INSERTION_AXIS,
                push_distance_m=0.012,
                duration_sec=0.45,
                hold_sec=0.8,
                debug_run=debug_run,
                phase_prefix="submission_sfp",
            )
            pushed_observation = self._wait_for_observation(get_observation, timeout_sec=0.8)
            debug_run.save_observation_snapshot(
                "after_push",
                pushed_observation,
                {
                    "feature_summary": self._extract_feature_summary(task, pushed_observation),
                    "feature": self._sfp_learned_feature(task, pushed_observation),
                    "pushed_pose": _pose_to_dict(final_pose),
                },
            )
        else:
            if self._learned_port_inference_or_none("sc") is None:
                debug_run.finalize(
                    False,
                    "submission_v11 requires AIC_QUAL_LEARNED_SC_MODEL_DIR",
                    initial_observation,
                )
                return False
            send_feedback("submission_v11: SC learned multi-view translation servo")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running learned SC translation-only servo using current observation",
            )
            aligned_pose, aligned_observation, aligned_feature = (
                self._run_sc_submission_learned_coarse_to_fine(
                    task,
                    initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v11 SC learned servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            baseline_force_norm = self._force_norm(initial_observation)
            pushed_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SC_TOOL_INSERTION_AXIS,
                push_distance_m=0.004,
                duration_sec=0.35,
                hold_sec=0.25,
                debug_run=debug_run,
                phase_prefix="submission_sc",
            )
            if self._enable_sc_refinement:
                refined_pose = self._run_sc_force_refine(
                    get_observation,
                    move_robot,
                    debug_run,
                    start_pose=pushed_pose,
                    baseline_force_norm=baseline_force_norm,
                )
                refined_observation = self._wait_for_observation(
                    get_observation, timeout_sec=0.8
                )
                debug_run.save_observation_snapshot(
                    "after_force_refine",
                    refined_observation,
                    {
                        "feature_summary": self._extract_feature_summary(
                            task, refined_observation
                        ),
                        "feature": self._sc_learned_feature(task, refined_observation),
                        "refined_pose": _pose_to_dict(refined_pose),
                    },
                )
                final_pose = self._run_sc_visual_residual(
                    get_observation,
                    move_robot,
                    debug_run,
                    refined_pose,
                    baseline_force_norm=baseline_force_norm,
                )
            else:
                pushed_observation = self._wait_for_observation(
                    get_observation, timeout_sec=0.8
                )
                debug_run.save_observation_snapshot(
                    "after_push",
                    pushed_observation,
                    {
                        "feature_summary": self._extract_feature_summary(
                            task, pushed_observation
                        ),
                        "feature": self._sc_learned_feature(task, pushed_observation),
                        "pushed_pose": _pose_to_dict(pushed_pose),
                    },
                )
                final_pose = pushed_pose

        final_observation = self._wait_for_observation(get_observation, timeout_sec=1.0)
        debug_run.finalize(
            True,
            "submission_safe_v11 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v11 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v12(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "learned_sc_model_dir": self._learned_sc_model_dir,
                "learned_sfp_model_dir": self._learned_sfp_model_dir,
                "notes": [
                    "SFP path uses learned multi-view acquisition followed by visual reacquire plus bounded insertion cycles",
                    "SFP insertion retreats on stall, force overload, or tracking divergence instead of holding a blind push",
                    "SC path matches submission_safe_v11",
                ],
            }
        )

        if task.plug_type == "sfp":
            if self._learned_port_inference_or_none("sfp") is None:
                debug_run.finalize(
                    False,
                    "submission_v12 requires AIC_QUAL_LEARNED_SFP_MODEL_DIR",
                    initial_observation,
                )
                return False
            send_feedback("submission_v12: SFP learned acquisition + closed-loop insertion")
            aligned_pose, aligned_observation, aligned_feature = (
                self._run_sfp_submission_learned_coarse_to_fine(
                    task,
                    initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v12 SFP learned servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False
            final_pose, final_observation, insertion_feature = (
                self._run_sfp_submission_learned_closed_loop_insertion(
                    task,
                    aligned_observation if aligned_observation is not None else initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_insertion_cycles",
                final_observation,
                {
                    "feature_summary": self._extract_feature_summary(task, final_observation),
                    "feature": insertion_feature,
                    "final_pose": _pose_to_dict(final_pose),
                },
            )
        else:
            return self._run_stage_submission_safe_v11(
                task, get_observation, move_robot, send_feedback
            )

        final_observation = self._wait_for_observation(get_observation, timeout_sec=1.0)
        debug_run.finalize(
            True,
            "submission_safe_v12 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v12 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v13(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "learned_model_dir": self._learned_model_dir,
                "notes": [
                    "SFP path keeps the stable center-camera legal servo from submission_safe_v6",
                    "SFP insertion replaces the blind push with a bounded tool-frame hole-finding search",
                    "SC path matches submission_safe_v6",
                    (
                        "SC force refine and residual stages are enabled"
                        if self._enable_sc_refinement
                        else "SC refinement is disabled for fail-fast isolation of SFP insertion changes"
                    ),
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback("submission_v13: SFP legal servo + tool-frame insertion search")
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.log_phase(
                "after_legal_servo",
                self.time_now().nanoseconds / 1e9,
                note="submission_safe_v13 SFP legal servo completed",
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v13 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False
            baseline_force_norm = self._force_norm(initial_observation)
            debug_run.log_phase(
                "before_insertion_search",
                self.time_now().nanoseconds / 1e9,
                note=f"baseline_force_norm={baseline_force_norm:.3f}",
            )
            final_pose = self._run_sfp_force_refine_tool_frame(
                get_observation,
                move_robot,
                debug_run,
                start_pose=aligned_pose,
                baseline_force_norm=baseline_force_norm,
            )
            debug_run.log_phase(
                "after_insertion_search",
                self.time_now().nanoseconds / 1e9,
                note="submission_safe_v13 SFP insertion search completed",
            )
            refined_observation = self._wait_for_observation(
                get_observation, timeout_sec=0.8
            )
            debug_run.save_observation_snapshot(
                "after_insertion_search",
                refined_observation,
                {
                    "feature_summary": self._extract_feature_summary(
                        task, refined_observation
                    ),
                    "feature": self._sfp_union_feature(refined_observation),
                    "refined_pose": _pose_to_dict(final_pose),
                },
            )
        else:
            return self._run_stage_submission_safe_v6(
                task, get_observation, move_robot, send_feedback
            )

        debug_run.log_phase(
            "before_final_wait",
            self.time_now().nanoseconds / 1e9,
            note="waiting for final observation",
        )
        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "final_pose": _pose_to_dict(final_pose),
            },
        )
        debug_run.log_phase(
            "after_final_snapshot",
            self.time_now().nanoseconds / 1e9,
            note="final observation snapshot saved",
        )
        debug_run.finalize(
            True,
            "submission_safe_v13 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v13 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v14(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "learned_sc_model_dir": self._learned_sc_model_dir,
                "learned_sfp_model_dir": self._learned_sfp_model_dir,
                "notes": [
                    "SFP path uses the stable legal servo from submission_safe_v6",
                    "A learned teacher-insert residual corrects the near-port SFP pose before insertion",
                    "SFP insertion then uses bounded incremental pushes",
                    "SC path matches submission_safe_v6",
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback("submission_v14: SFP legal servo + learned insert residual")
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
                max_steps=12,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None or aligned_observation is None:
                debug_run.finalize(
                    False,
                    "submission_v14 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False
            residual_pose, residual_observation, residual_metadata = (
                self._run_sfp_submission_learned_teacher_insert_residual(
                    task,
                    aligned_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_learned_residual",
                residual_observation,
                {
                    "feature_summary": self._extract_feature_summary(
                        task, residual_observation
                    ),
                    "feature": self._sfp_union_feature(residual_observation),
                    "residual_metadata": residual_metadata,
                    "residual_pose": _pose_to_dict(residual_pose),
                },
            )
            if residual_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v14 learned residual unavailable",
                    residual_observation,
                    {"residual_metadata": residual_metadata},
                )
                return False
            final_pose, final_observation, insertion_feature = (
                self._run_sfp_incremental_insertion(
                    residual_observation
                    if residual_observation is not None
                    else aligned_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                    aligned_feature,
                )
            )
            debug_run.log_phase(
                "before_after_insertion_snapshot",
                self.time_now().nanoseconds / 1e9,
                note="saving after_insertion_cycles snapshot",
            )
            debug_run.save_observation_snapshot(
                "after_insertion_cycles",
                final_observation,
                {
                    "feature_summary": self._extract_feature_summary(task, final_observation),
                    "feature": insertion_feature,
                    "final_pose": _pose_to_dict(final_pose),
                },
            )
            debug_run.log_phase(
                "after_after_insertion_snapshot",
                self.time_now().nanoseconds / 1e9,
                note="after_insertion_cycles snapshot saved",
            )
        else:
            return self._run_stage_submission_safe_v6(
                task, get_observation, move_robot, send_feedback
            )

        debug_run.log_phase(
            "before_finalize",
            self.time_now().nanoseconds / 1e9,
            note="finalizing with latest insertion observation",
        )
        debug_run.finalize(
            True,
            "submission_safe_v14 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v14 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v15(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "learned_sc_model_dir": self._learned_sc_model_dir,
                "learned_sfp_model_dir": self._learned_sfp_model_dir,
                "notes": [
                    "SFP path reuses the stable submission_safe_v6 legal servo timing",
                    "SFP learned residual is trained from post-legal-servo GT labels to reduce train-deployment mismatch",
                    "SFP applies a single residual step before bounded incremental pushes",
                    "SC path matches submission_safe_v6",
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback(
                "submission_v15: SFP v6 legal servo + post-servo learned residual"
            )
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None or aligned_observation is None:
                debug_run.finalize(
                    False,
                    "submission_v15 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False
            residual_pose, residual_observation, residual_metadata = (
                self._run_sfp_submission_learned_teacher_insert_residual(
                    task,
                    aligned_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                    max_iterations=1,
                )
            )
            debug_run.save_observation_snapshot(
                "after_learned_residual",
                residual_observation,
                {
                    "feature_summary": self._extract_feature_summary(
                        task, residual_observation
                    ),
                    "feature": self._sfp_union_feature(residual_observation),
                    "residual_metadata": residual_metadata,
                    "residual_pose": _pose_to_dict(residual_pose),
                },
            )
            if residual_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v15 learned residual unavailable",
                    residual_observation,
                    {"residual_metadata": residual_metadata},
                )
                return False
            final_pose, final_observation, insertion_feature = (
                self._run_sfp_incremental_insertion(
                    residual_observation
                    if residual_observation is not None
                    else aligned_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                    aligned_feature,
                )
            )
            debug_run.save_observation_snapshot(
                "after_insertion_cycles",
                final_observation,
                {
                    "feature_summary": self._extract_feature_summary(task, final_observation),
                    "feature": insertion_feature,
                    "final_pose": _pose_to_dict(final_pose),
                },
            )
        else:
            return self._run_stage_submission_safe_v6(
                task, get_observation, move_robot, send_feedback
            )

        debug_run.finalize(
            True,
            "submission_safe_v15 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v15 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v16(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "notes": [
                    "SFP path reuses the stable submission_safe_v6 legal servo timing",
                    "SFP then executes a GT-derived module-conditioned canonical pre-insertion primitive",
                    "The primitive is measured from post-legal-servo to teacher_insert poses and applied in staged chunks",
                    "SC path matches submission_safe_v6",
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback(
                "submission_v16: SFP legal servo + structured teacher-derived pre-insertion"
            )
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None
                        if aligned_observation is None
                        else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None or aligned_observation is None:
                debug_run.finalize(
                    False,
                    "submission_v16 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            structured_pose, structured_observation, structured_metadata = (
                self._run_sfp_submission_structured_teacher_insert(
                    task,
                    aligned_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_structured_preinsert",
                structured_observation,
                {
                    "feature_summary": self._extract_feature_summary(
                        task, structured_observation
                    ),
                    "feature": self._sfp_union_feature(structured_observation),
                    "structured_metadata": structured_metadata,
                    "structured_pose": _pose_to_dict(structured_pose),
                },
            )

            final_pose, final_observation, insertion_feature = (
                self._run_sfp_incremental_insertion(
                    structured_observation
                    if structured_observation is not None
                    else aligned_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                    aligned_feature,
                )
            )
            debug_run.save_observation_snapshot(
                "after_insertion_cycles",
                final_observation,
                {
                    "feature_summary": self._extract_feature_summary(task, final_observation),
                    "feature": insertion_feature,
                    "final_pose": _pose_to_dict(final_pose),
                },
            )
        else:
            return self._run_stage_submission_safe_v6(
                task, get_observation, move_robot, send_feedback
            )

        debug_run.finalize(
            True,
            "submission_safe_v16 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v16 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_submission_safe_v2(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        target_metadata = {
            "provider": "official_task_and_observation_only",
            "task_key": list(self._task_key(task)),
            "notes": [
                "no public-sample world targets",
                "no development-only target provider",
                "SFP path matches submission_safe_v0",
                "SC path estimates insertion orientation from the observed magenta frame angle",
            ],
        }
        if task.plug_type == "sc":
            sc_feature = self._sc_center_feature(initial_observation)
            target_metadata["sc_magenta_angle_deg"] = (
                None if sc_feature is None else sc_feature.get("magenta_angle_deg")
            )
            target_metadata["sc_reference_angle_deg"] = self._SC_MAGENTA_REFERENCE_ANGLE_DEG
            target_metadata["sc_target_quat_wxyz"] = list(
                self._estimate_sc_insertion_quat_wxyz(initial_observation)
            )
        debug_run.save_target_metadata(target_metadata)

        if task.plug_type == "sfp":
            send_feedback("submission_v2: SFP center-camera legal servo")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running SFP visual servo using only current observation",
            )
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None if aligned_observation is None else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v2 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            send_feedback("submission_v2: SFP bounded tool-axis push")
            debug_run.log_phase(
                "insert",
                self.time_now().nanoseconds / 1e9,
                "applying short tool-axis insertion push from legal servo pose",
            )
            final_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SFP_TOOL_INSERTION_AXIS,
                push_distance_m=0.012,
                duration_sec=0.45,
                hold_sec=0.8,
                debug_run=debug_run,
                phase_prefix="submission_sfp",
            )
            success = True
        else:
            sc_target_quat = self._estimate_sc_insertion_quat_wxyz(initial_observation)
            send_feedback("submission_v2: SC legal servo with estimated orientation")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running SC visual servo with image-derived insertion orientation",
            )
            aligned_pose, aligned_observation, aligned_feature = self._run_sc_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
                target_quat_wxyz=sc_target_quat,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None if aligned_observation is None else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                    "target_quat_wxyz": list(sc_target_quat),
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v2 SC servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            baseline_force_norm = self._force_norm(initial_observation)
            debug_run.log_phase(
                "insert",
                self.time_now().nanoseconds / 1e9,
                "applying short SC tool-axis push followed by bounded legal refinement",
            )
            pushed_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SC_TOOL_INSERTION_AXIS,
                push_distance_m=0.004,
                duration_sec=0.35,
                hold_sec=0.25,
                debug_run=debug_run,
                phase_prefix="submission_sc",
            )
            refined_pose = self._run_sc_force_refine(
                get_observation,
                move_robot,
                debug_run,
                start_pose=pushed_pose,
                baseline_force_norm=baseline_force_norm,
            )
            debug_run.log_phase(
                "residual",
                self.time_now().nanoseconds / 1e9,
                "applying legal SC residual correction",
            )
            final_pose = self._run_sc_visual_residual(
                get_observation,
                move_robot,
                debug_run,
                refined_pose,
                baseline_force_norm=baseline_force_norm,
            )
            success = True

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "final_pose": _pose_to_dict(final_pose),
            },
        )
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            f"submission_safe_v2 returned {success}",
        )
        debug_run.finalize(
            bool(success),
            "submission_safe_v2 completed" if success else "submission_safe_v2 failed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v2 artifacts saved to {debug_run.root}")
        return bool(success)

    def _replay_sc_trajectory_with_debug(
        self,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        time_scale: float = 1.0,
        final_hold_sec: float = 0.5,
    ) -> Pose:
        replay_poses = [self._pose_from_replay_row(row) for row in SC_REPLAY_TRAJECTORY]
        self.set_pose_target(move_robot=move_robot, pose=replay_poses[0])
        debug_run.log_command_sample(
            "sc_replay",
            replay_poses[0],
            self.time_now().nanoseconds / 1e9,
            note="replay start",
        )
        for idx in range(len(SC_REPLAY_TRAJECTORY) - 1):
            current_time = SC_REPLAY_TRAJECTORY[idx][0]
            next_time = SC_REPLAY_TRAJECTORY[idx + 1][0]
            self._move_for_duration(
                move_robot,
                replay_poses[idx],
                replay_poses[idx + 1],
                duration_sec=max(0.02, time_scale * (next_time - current_time)),
                debug_run=debug_run,
                phase_name="sc_replay",
            )
        self._hold_pose(
            move_robot,
            replay_poses[-1],
            duration_sec=final_hold_sec,
            debug_run=debug_run,
            phase_name="sc_replay_hold",
        )
        return replay_poses[-1]

    def _run_sc_force_refine(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        start_pose: Pose | None = None,
        baseline_force_norm: float | None = None,
    ) -> Pose:
        if start_pose is None:
            start_pose = self._replay_sc_trajectory_with_debug(
                move_robot,
                debug_run,
                time_scale=0.35,
                final_hold_sec=0.2,
            )
        if baseline_force_norm is None:
            baseline_observation = self._wait_for_observation(
                get_observation, timeout_sec=0.5
            )
            baseline_force_norm = self._force_norm(baseline_observation)
            last_stamp_sec = (
                None
                if baseline_observation is None
                else _stamp_to_float(baseline_observation.center_image.header.stamp)
            )
        else:
            baseline_observation = self._wait_for_observation(
                get_observation, timeout_sec=0.5
            )
            last_stamp_sec = (
                None
                if baseline_observation is None
                else _stamp_to_float(baseline_observation.center_image.header.stamp)
            )

        base_position = self._pose_position(start_pose)
        quat_wxyz = self._quat_xyzw_to_wxyz(start_pose.orientation)
        lateral_offsets = [
            (0.0, 0.0),
            (0.0025, 0.0),
            (-0.0025, 0.0),
            (0.0, 0.0025),
            (0.0, -0.0025),
        ]
        insertion_steps_m = [0.002, 0.004, 0.006, 0.008]
        best_pose = start_pose

        for offset_x, offset_y in lateral_offsets:
            lateral_pose = self._make_pose(
                base_position + np.array([offset_x, offset_y, 0.0], dtype=float),
                quat_wxyz,
            )
            self._move_for_duration(
                move_robot,
                best_pose,
                lateral_pose,
                duration_sec=0.35,
                debug_run=debug_run,
                phase_name="force_search_lateral",
            )
            self._hold_pose(
                move_robot,
                lateral_pose,
                duration_sec=0.1,
                debug_run=debug_run,
                phase_name="force_search_settle",
            )
            best_pose = lateral_pose
            if last_stamp_sec is not None:
                settle_observation = self._wait_for_observation(
                    get_observation,
                    timeout_sec=0.5,
                    newer_than_sec=last_stamp_sec,
                )
                if settle_observation is not None:
                    last_stamp_sec = _stamp_to_float(
                        settle_observation.center_image.header.stamp
                    )

            for insertion_depth in insertion_steps_m:
                inserted_pose = self._make_pose(
                    self._pose_position(lateral_pose)
                    + np.array([0.0, 0.0, -insertion_depth], dtype=float),
                    quat_wxyz,
                )
                self._move_for_duration(
                    move_robot,
                    best_pose,
                    inserted_pose,
                    duration_sec=0.25,
                    debug_run=debug_run,
                    phase_name="force_search_insert",
                )
                observation = self._wait_for_observation(
                    get_observation,
                    timeout_sec=0.5,
                    newer_than_sec=last_stamp_sec,
                )
                if observation is not None:
                    last_stamp_sec = _stamp_to_float(
                        observation.center_image.header.stamp
                    )
                measured_force = self._force_norm(observation)
                excess_force = measured_force - baseline_force_norm
                debug_run.log_command_sample(
                    "force_search_insert",
                    inserted_pose,
                    self.time_now().nanoseconds / 1e9,
                    note=(
                        f"offset=({offset_x:.4f},{offset_y:.4f}) depth={insertion_depth:.4f} "
                        f"force={measured_force:.2f} excess={excess_force:.2f}"
                    ),
                )
                if excess_force > 12.0:
                    self._move_for_duration(
                        move_robot,
                        inserted_pose,
                        lateral_pose,
                        duration_sec=0.2,
                        debug_run=debug_run,
                        phase_name="force_search_retreat",
                    )
                    best_pose = lateral_pose
                    break
                best_pose = inserted_pose
            self._hold_pose(
                move_robot,
                best_pose,
                duration_sec=0.2,
                debug_run=debug_run,
                phase_name="force_search_hold",
            )
        return best_pose

    def _run_sc_force_refine_tool_frame(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        start_pose: Pose,
        baseline_force_norm: float | None = None,
    ) -> Pose:
        if baseline_force_norm is None:
            baseline_observation = self._wait_for_observation(
                get_observation, timeout_sec=0.5
            )
            baseline_force_norm = self._force_norm(baseline_observation)
            last_stamp_sec = (
                None
                if baseline_observation is None
                else _stamp_to_float(baseline_observation.center_image.header.stamp)
            )
        else:
            baseline_observation = self._wait_for_observation(
                get_observation, timeout_sec=0.5
            )
            last_stamp_sec = (
                None
                if baseline_observation is None
                else _stamp_to_float(baseline_observation.center_image.header.stamp)
            )

        quat_wxyz = self._quat_xyzw_to_wxyz(start_pose.orientation)
        insertion_axis = self._tool_axis_in_base(start_pose, self._SC_TOOL_INSERTION_AXIS)
        insertion_axis /= max(np.linalg.norm(insertion_axis), 1e-6)

        reference_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(reference_axis, insertion_axis))) > 0.90:
            reference_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        lateral_axis_1 = np.cross(insertion_axis, reference_axis)
        lateral_axis_1 /= max(np.linalg.norm(lateral_axis_1), 1e-6)
        lateral_axis_2 = np.cross(insertion_axis, lateral_axis_1)
        lateral_axis_2 /= max(np.linalg.norm(lateral_axis_2), 1e-6)

        lateral_offsets = [
            np.array([0.0, 0.0, 0.0], dtype=float),
            0.0020 * lateral_axis_1,
            -0.0020 * lateral_axis_1,
            0.0020 * lateral_axis_2,
            -0.0020 * lateral_axis_2,
        ]
        insertion_steps_m = [0.0015, 0.0030, 0.0045, 0.0060]
        best_pose = start_pose

        for lateral_offset in lateral_offsets:
            debug_run.log_phase(
                "sfp_force_tool_offset_start",
                self.time_now().nanoseconds / 1e9,
                note=(
                    "offset=("
                    f"{float(lateral_offset[0]):.4f},"
                    f"{float(lateral_offset[1]):.4f},"
                    f"{float(lateral_offset[2]):.4f})"
                ),
            )
            lateral_pose = self._make_pose(
                self._pose_position(start_pose) + lateral_offset,
                quat_wxyz,
            )
            self._move_for_duration(
                move_robot,
                best_pose,
                lateral_pose,
                duration_sec=0.25,
                debug_run=debug_run,
                phase_name="force_search_tool_lateral",
            )
            self._hold_pose(
                move_robot,
                lateral_pose,
                duration_sec=0.10,
                debug_run=debug_run,
                phase_name="force_search_tool_settle",
            )
            best_pose = lateral_pose
            if last_stamp_sec is not None:
                settle_observation = self._wait_for_observation(
                    get_observation,
                    timeout_sec=0.5,
                    newer_than_sec=last_stamp_sec,
                )
                if settle_observation is not None:
                    last_stamp_sec = _stamp_to_float(
                        settle_observation.center_image.header.stamp
                    )

            for insertion_depth in insertion_steps_m:
                inserted_pose = self._make_pose(
                    self._pose_position(lateral_pose)
                    + insertion_depth * insertion_axis,
                    quat_wxyz,
                )
                self._move_for_duration(
                    move_robot,
                    best_pose,
                    inserted_pose,
                    duration_sec=0.22,
                    debug_run=debug_run,
                    phase_name="force_search_tool_insert",
                )
                observation = self._wait_for_observation(
                    get_observation,
                    timeout_sec=0.5,
                    newer_than_sec=last_stamp_sec,
                )
                if observation is not None:
                    last_stamp_sec = _stamp_to_float(
                        observation.center_image.header.stamp
                    )
                measured_force = self._force_norm(observation)
                excess_force = measured_force - baseline_force_norm
                debug_run.log_command_sample(
                    "force_search_tool_insert",
                    inserted_pose,
                    self.time_now().nanoseconds / 1e9,
                    note=(
                        "offset=("
                        f"{float(lateral_offset[0]):.4f},"
                        f"{float(lateral_offset[1]):.4f},"
                        f"{float(lateral_offset[2]):.4f}) "
                        f"depth={insertion_depth:.4f} "
                        f"force={measured_force:.2f} excess={excess_force:.2f}"
                    ),
                )
                if excess_force > 12.0:
                    self._move_for_duration(
                        move_robot,
                        inserted_pose,
                        lateral_pose,
                        duration_sec=0.18,
                        debug_run=debug_run,
                        phase_name="force_search_tool_retreat",
                    )
                    best_pose = lateral_pose
                    break
                best_pose = inserted_pose
            self._hold_pose(
                move_robot,
                best_pose,
                duration_sec=0.15,
                debug_run=debug_run,
                phase_name="force_search_tool_hold",
            )
        return best_pose

    def _run_sfp_force_refine_tool_frame(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        start_pose: Pose,
        baseline_force_norm: float | None = None,
    ) -> Pose:
        if baseline_force_norm is None:
            baseline_observation = self._wait_for_observation(
                get_observation, timeout_sec=0.5
            )
            baseline_force_norm = self._force_norm(baseline_observation)
            last_stamp_sec = (
                None
                if baseline_observation is None
                else _stamp_to_float(baseline_observation.center_image.header.stamp)
            )
        else:
            baseline_observation = self._wait_for_observation(
                get_observation, timeout_sec=0.5
            )
            last_stamp_sec = (
                None
                if baseline_observation is None
                else _stamp_to_float(baseline_observation.center_image.header.stamp)
            )

        quat_wxyz = self._quat_xyzw_to_wxyz(start_pose.orientation)
        insertion_axis = self._tool_axis_in_base(start_pose, self._SFP_TOOL_INSERTION_AXIS)
        insertion_axis /= max(np.linalg.norm(insertion_axis), 1e-6)

        reference_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(reference_axis, insertion_axis))) > 0.90:
            reference_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        lateral_axis_1 = np.cross(insertion_axis, reference_axis)
        lateral_axis_1 /= max(np.linalg.norm(lateral_axis_1), 1e-6)
        lateral_axis_2 = np.cross(insertion_axis, lateral_axis_1)
        lateral_axis_2 /= max(np.linalg.norm(lateral_axis_2), 1e-6)

        diagonal_axis = lateral_axis_1 + lateral_axis_2
        diagonal_axis /= max(np.linalg.norm(diagonal_axis), 1e-6)
        lateral_offsets = [
            np.array([0.0, 0.0, 0.0], dtype=float),
            0.0010 * lateral_axis_1,
            -0.0010 * lateral_axis_1,
            0.0010 * lateral_axis_2,
            -0.0010 * lateral_axis_2,
            0.0012 * diagonal_axis,
            -0.0012 * diagonal_axis,
        ]
        insertion_steps_m = [0.0020, 0.0040, 0.0060, 0.0080, 0.0100]
        best_pose = start_pose

        for lateral_offset in lateral_offsets:
            lateral_pose = self._make_pose(
                self._pose_position(start_pose) + lateral_offset,
                quat_wxyz,
            )
            self._move_for_duration(
                move_robot,
                best_pose,
                lateral_pose,
                duration_sec=0.18,
                debug_run=debug_run,
                phase_name="sfp_force_tool_lateral",
            )
            self._hold_pose(
                move_robot,
                lateral_pose,
                duration_sec=0.08,
                debug_run=debug_run,
                phase_name="sfp_force_tool_settle",
            )
            best_pose = lateral_pose
            if last_stamp_sec is not None:
                settle_observation = self._wait_for_observation(
                    get_observation,
                    timeout_sec=0.5,
                    newer_than_sec=last_stamp_sec,
                )
                if settle_observation is not None:
                    last_stamp_sec = _stamp_to_float(
                        settle_observation.center_image.header.stamp
                    )
            else:
                settle_observation = None

            observed_lateral_pose = self._observed_tcp_pose(
                settle_observation, lateral_pose
            )
            lateral_position = self._pose_position(observed_lateral_pose)
            for insertion_depth in insertion_steps_m:
                inserted_pose = self._make_pose(
                    lateral_position + insertion_depth * insertion_axis,
                    quat_wxyz,
                )
                debug_run.log_phase(
                    "sfp_force_tool_insert_start",
                    self.time_now().nanoseconds / 1e9,
                    note=(
                        "offset=("
                        f"{float(lateral_offset[0]):.4f},"
                        f"{float(lateral_offset[1]):.4f},"
                        f"{float(lateral_offset[2]):.4f}) "
                        f"depth={insertion_depth:.4f}"
                    ),
                )
                self._move_for_duration(
                    move_robot,
                    best_pose,
                    inserted_pose,
                    duration_sec=0.16,
                    debug_run=debug_run,
                    phase_name="sfp_force_tool_insert",
                )
                self._hold_pose(
                    move_robot,
                    inserted_pose,
                    duration_sec=0.08,
                    debug_run=debug_run,
                    phase_name="sfp_force_tool_hold",
                )
                observation = self._wait_for_observation(
                    get_observation,
                    timeout_sec=0.5,
                    newer_than_sec=last_stamp_sec,
                )
                if observation is not None:
                    last_stamp_sec = _stamp_to_float(
                        observation.center_image.header.stamp
                    )
                measured_force = self._force_norm(observation)
                excess_force = measured_force - baseline_force_norm
                observed_pose = self._observed_tcp_pose(observation, inserted_pose)
                observed_position = self._pose_position(observed_pose)
                actual_advance_m = float(
                    np.dot(observed_position - lateral_position, insertion_axis)
                )
                tracking_error_m = None
                if observation is not None:
                    tracking_error_m = float(
                        np.linalg.norm(observation.controller_state.tcp_error[:3])
                    )
                debug_run.log_command_sample(
                    "sfp_force_tool_insert",
                    inserted_pose,
                    self.time_now().nanoseconds / 1e9,
                    note=(
                        "offset=("
                        f"{float(lateral_offset[0]):.4f},"
                        f"{float(lateral_offset[1]):.4f},"
                        f"{float(lateral_offset[2]):.4f}) "
                        f"depth={insertion_depth:.4f} "
                        f"advance={actual_advance_m:.4f} "
                        f"force={measured_force:.2f} excess={excess_force:.2f} "
                        f"track={tracking_error_m if tracking_error_m is not None else float('nan'):.4f}"
                    ),
                )
                if excess_force > 8.0 or (
                    tracking_error_m is not None and tracking_error_m > 0.025
                ):
                    debug_run.log_phase(
                        "sfp_force_tool_insert_abort",
                        self.time_now().nanoseconds / 1e9,
                        note=(
                            f"depth={insertion_depth:.4f} "
                            f"excess_force={excess_force:.2f} "
                            f"tracking_error={tracking_error_m if tracking_error_m is not None else float('nan'):.4f}"
                        ),
                    )
                    self._move_for_duration(
                        move_robot,
                        inserted_pose,
                        lateral_pose,
                        duration_sec=0.14,
                        debug_run=debug_run,
                        phase_name="sfp_force_tool_retreat",
                    )
                    best_pose = lateral_pose
                    break
                best_pose = self._make_pose(observed_position, quat_wxyz)
            debug_run.log_phase(
                "sfp_force_tool_offset_done",
                self.time_now().nanoseconds / 1e9,
                note=(
                    "offset=("
                    f"{float(lateral_offset[0]):.4f},"
                    f"{float(lateral_offset[1]):.4f},"
                    f"{float(lateral_offset[2]):.4f})"
                ),
            )
        return best_pose

    def _run_sc_center_servo(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        max_steps: int = 12,
    ) -> Pose | None:
        rotation = self._lookup_camera_rotation_base_from_optical(self._CENTER_CAMERA_FRAME)
        if rotation is None:
            return None

        desired_u, desired_v = self._SC_CENTER_RESIDUAL_TEMPLATE["cyan_centroid_uv"]
        nominal_depth = float(self._SC_CENTER_RESIDUAL_TEMPLATE["nominal_depth_m"])
        sc_target = self._DEV_TARGETS.get(
            ("sc", task.target_module_name, task.port_name),
            self._DEV_TARGETS[("sc", "sc_port_1", "sc_port_base")],
        )
        desired_quat = sc_target.port_quat_wxyz

        observation = self._wait_for_observation(get_observation, timeout_sec=1.0)
        if observation is None:
            return None
        last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        current_pose = observation.controller_state.tcp_pose
        oriented_pose = self._make_pose(self._pose_position(current_pose), desired_quat)
        self._move_for_duration(
            move_robot,
            current_pose,
            oriented_pose,
            duration_sec=0.45,
            debug_run=debug_run,
            phase_name="sc_visual_orient",
        )
        best_pose = oriented_pose
        observation = self._wait_for_observation(
            get_observation,
            timeout_sec=0.8,
            newer_than_sec=last_stamp_sec,
        )
        if observation is None:
            return best_pose
        last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        for step in range(max_steps):
            feature = self._sc_center_feature(observation)
            if observation is None or feature is None:
                break
            current_pose = observation.controller_state.tcp_pose
            fx = float(observation.center_camera_info.k[0])
            fy = float(observation.center_camera_info.k[4])
            union_u, union_v = feature["cyan_centroid_uv"]
            du = union_u - desired_u
            dv = union_v - desired_v
            delta_cam = np.array(
                [
                    0.85 * nominal_depth * du / max(fx, 1.0),
                    1.05 * nominal_depth * dv / max(fy, 1.0),
                    0.0,
                ],
                dtype=float,
            )
            delta_cam[0] = float(np.clip(delta_cam[0], -0.012, 0.012))
            delta_cam[1] = float(np.clip(delta_cam[1], -0.012, 0.012))
            delta_base = rotation @ delta_cam
            target_pose = self._make_pose(
                self._pose_position(current_pose) + delta_base,
                desired_quat,
            )
            debug_run.log_command_sample(
                "sc_visual_center",
                target_pose,
                self.time_now().nanoseconds / 1e9,
                note=f"step {step + 1}/{max_steps} du={du:.1f} dv={dv:.1f}",
            )
            self._move_for_duration(
                move_robot,
                current_pose,
                target_pose,
                duration_sec=0.2,
                debug_run=debug_run,
                phase_name="sc_visual_center",
            )
            best_pose = target_pose
            if abs(du) < 28.0 and abs(dv) < 28.0:
                break
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if observation is None:
                break
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        self._hold_pose(
            move_robot,
            best_pose,
            duration_sec=0.2,
            debug_run=debug_run,
            phase_name="sc_visual_center_hold",
        )
        return best_pose

    def _run_stage_hold_only(
        self,
        stage_note: str,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            observation,
            {"feature_summary": self._extract_feature_summary(task, observation)},
        )
        if observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False
        send_feedback(stage_note)
        debug_run.log_phase(
            "hold",
            self.time_now().nanoseconds / 1e9,
            stage_note,
        )
        self._hold_pose(
            move_robot,
            observation.controller_state.tcp_pose,
            2.0,
            debug_run=debug_run,
            phase_name="hold",
        )
        final_observation = get_observation()
        debug_run.log_phase("done", self.time_now().nanoseconds / 1e9, "hold-only stage")
        debug_run.finalize(
            True,
            "hold-only stage completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"Hold-only artifacts saved to {debug_run.root}")
        return True

    def _visual_servo_sfp(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        phase_name: str,
        max_steps: int,
        hold_when_aligned: bool,
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        rotation_base_from_optical = self._lookup_camera_rotation_base_from_optical(
            self._CENTER_CAMERA_FRAME
        )
        if rotation_base_from_optical is None:
            return None, None, None

        template = self._sfp_visual_template_for_task(task)
        desired_u, desired_v = template["union_center_uv"]
        desired_w, desired_h = template["union_size_px"]
        desired_quat = template["nominal_quat_wxyz"]
        nominal_depth = float(template["camera_depth_m"])

        best_observation = None
        best_feature = None
        observation = self._wait_for_observation(get_observation, timeout_sec=1.0)
        if observation is None:
            return None, None, None
        last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        for step in range(max_steps):
            feature = self._sfp_union_feature(observation)
            if feature is None:
                return None, observation, feature

            best_observation = observation
            best_feature = feature

            current_pose = observation.controller_state.tcp_pose
            fx = float(observation.center_camera_info.k[0])
            fy = float(observation.center_camera_info.k[4])
            union_u, union_v = feature["union_center_uv"]
            union_w, union_h = feature["union_size_px"]
            du = union_u - desired_u
            dv = union_v - desired_v
            width_scale = desired_w / max(union_w, 1.0)
            height_scale = desired_h / max(union_h, 1.0)
            scale = 0.5 * (width_scale + height_scale)
            estimated_depth = nominal_depth * scale
            depth_error = estimated_depth - nominal_depth

            aligned = (
                abs(du) < 18.0
                and abs(dv) < 18.0
                and abs(union_w - desired_w) < 26.0
                and abs(union_h - desired_h) < 26.0
            )
            debug_run.log_command_sample(
                phase_name,
                current_pose,
                self.time_now().nanoseconds / 1e9,
                note=(
                    f"servo step {step + 1}/{max_steps} "
                    f"du={du:.1f} dv={dv:.1f} "
                    f"depth_error={depth_error:.3f} "
                    f"size=({union_w:.1f},{union_h:.1f})"
                ),
            )
            if aligned:
                aligned_pose = self._make_pose(self._pose_position(current_pose), desired_quat)
                self.set_pose_target(move_robot=move_robot, pose=aligned_pose)
                if hold_when_aligned:
                    self._hold_pose(
                        move_robot,
                        aligned_pose,
                        duration_sec=0.5,
                        debug_run=debug_run,
                        phase_name=f"{phase_name}_hold",
                    )
                return aligned_pose, observation, feature

            delta_cam = np.array(
                [
                    0.75 * estimated_depth * du / max(fx, 1.0),
                    0.85 * estimated_depth * dv / max(fy, 1.0),
                    np.clip(0.85 * depth_error, -0.03, 0.03),
                ],
                dtype=float,
            )
            delta_cam[0] = float(np.clip(delta_cam[0], -0.02, 0.02))
            delta_cam[1] = float(np.clip(delta_cam[1], -0.025, 0.025))
            delta_base = rotation_base_from_optical @ delta_cam
            target_position = self._pose_position(current_pose) + delta_base
            target_pose = self._make_pose(target_position, desired_quat)
            self.set_pose_target(move_robot=move_robot, pose=target_pose)
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if observation is None:
                break
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        if best_observation is None:
            return None, None, None
        final_pose = self._make_pose(
            self._pose_position(best_observation.controller_state.tcp_pose),
            desired_quat,
        )
        return final_pose, best_observation, best_feature

    def _visual_servo_sfp_multicam(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        phase_name: str,
        max_steps: int,
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        rotations = {
            camera_name: self._lookup_camera_rotation_base_from_optical(camera_frame)
            for camera_name, camera_frame in self._CAMERA_FRAMES.items()
        }
        if rotations["center"] is None:
            return None, None, None

        template = self._sfp_multi_camera_template_for_task(task)
        center_template = self._sfp_visual_template_for_task(task)
        nominal_quat = center_template["nominal_quat_wxyz"]
        nominal_depth = float(center_template["camera_depth_m"])
        weights = {"left": 0.22, "center": 0.56, "right": 0.22}

        best_observation = None
        best_features = None
        observation = self._wait_for_observation(get_observation, timeout_sec=1.0)
        if observation is None:
            return None, None, None
        last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        for step in range(max_steps):
            best_observation = observation
            features = {
                camera_name: self._sfp_union_feature_for_camera(observation, camera_name)
                for camera_name in self._CAMERA_FRAMES
            }
            best_features = features
            center_feature = features["center"]
            if center_feature is None:
                return None, observation, features

            current_pose = observation.controller_state.tcp_pose
            aggregated_delta = np.zeros(3, dtype=float)
            residual_summary = []
            total_weight = 0.0

            for camera_name, feature in features.items():
                rotation = rotations[camera_name]
                camera_template = template.get(camera_name)
                if feature is None or rotation is None or camera_template is None:
                    continue
                _, camera_info = self._camera_view(observation, camera_name)
                if camera_info is None:
                    continue
                fx = float(camera_info.k[0])
                fy = float(camera_info.k[4])
                desired_u, desired_v = camera_template["union_center_uv"]
                desired_w, desired_h = camera_template["union_size_px"]
                union_u, union_v = feature["union_center_uv"]
                union_w, union_h = feature["union_size_px"]
                du = union_u - desired_u
                dv = union_v - desired_v
                scale_w = desired_w / max(union_w, 1.0)
                scale_h = desired_h / max(union_h, 1.0)
                estimated_depth = nominal_depth * 0.5 * (scale_w + scale_h)
                delta_cam = np.array(
                    [
                        0.65 * estimated_depth * du / max(fx, 1.0),
                        0.75 * estimated_depth * dv / max(fy, 1.0),
                        0.0,
                    ],
                    dtype=float,
                )
                if camera_name == "center":
                    depth_error = estimated_depth - nominal_depth
                    delta_cam[2] = float(np.clip(0.8 * depth_error, -0.02, 0.02))
                delta_cam[0] = float(np.clip(delta_cam[0], -0.015, 0.015))
                delta_cam[1] = float(np.clip(delta_cam[1], -0.02, 0.02))
                weight = weights[camera_name]
                aggregated_delta += weight * (rotation @ delta_cam)
                total_weight += weight
                residual_summary.append(
                    f"{camera_name}:du={du:.1f},dv={dv:.1f},size=({union_w:.0f},{union_h:.0f})"
                )

            if total_weight <= 1e-6:
                return None, observation, features
            aggregated_delta /= total_weight
            center_union_w, center_union_h = center_feature["union_size_px"]
            center_desired_w, center_desired_h = template["center"]["union_size_px"]
            aligned = (
                np.linalg.norm(aggregated_delta[:2]) < 0.003
                and abs(center_union_w - center_desired_w) < 20.0
                and abs(center_union_h - center_desired_h) < 20.0
            )
            debug_run.log_command_sample(
                phase_name,
                current_pose,
                self.time_now().nanoseconds / 1e9,
                note=f"servo step {step + 1}/{max_steps} {' | '.join(residual_summary)}",
            )
            if aligned:
                aligned_pose = self._make_pose(
                    self._pose_position(current_pose),
                    nominal_quat,
                )
                self.set_pose_target(move_robot=move_robot, pose=aligned_pose)
                self._hold_pose(
                    move_robot,
                    aligned_pose,
                    duration_sec=0.35,
                    debug_run=debug_run,
                    phase_name=f"{phase_name}_hold",
                )
                return aligned_pose, observation, features

            target_pose = self._make_pose(
                self._pose_position(current_pose) + aggregated_delta,
                nominal_quat,
            )
            self.set_pose_target(move_robot=move_robot, pose=target_pose)
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if observation is None:
                break
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)

        if best_observation is None:
            return None, None, None
        fallback_pose = self._make_pose(
            self._pose_position(best_observation.controller_state.tcp_pose),
            nominal_quat,
        )
        return fallback_pose, best_observation, best_features

    def _run_public_sfp_sequence(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        send_feedback: SendFeedbackCallback,
        use_multicam: bool,
    ) -> bool:
        task_key = self._task_key(task)
        target = self._public_trial_pilot._TARGETS.get(task_key)
        if target is None:
            return False

        current_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        if current_observation is None:
            return False
        current_pose = current_observation.controller_state.tcp_pose
        entrance_pose = self._pose_from_target(
            target.entrance_position, target.entrance_quat_wxyz
        )
        port_pose = self._pose_from_target(target.port_position, target.port_quat_wxyz)
        entrance = np.array(target.entrance_position, dtype=float)
        port = np.array(target.port_position, dtype=float)
        insertion_axis = port - entrance
        insertion_axis /= max(np.linalg.norm(insertion_axis), 1e-6)
        pushed_pose = self._make_pose(
            port + target.push_distance_m * insertion_axis,
            target.port_quat_wxyz,
        )

        send_feedback("M5: public baseline entrance pose")
        debug_run.log_phase(
            "approach",
            self.time_now().nanoseconds / 1e9,
            "moving to public baseline entrance pose",
        )
        self._move_for_duration(
            move_robot,
            current_pose,
            entrance_pose,
            duration_sec=0.8,
            debug_run=debug_run,
            phase_name="m5_approach",
        )
        self._hold_pose(
            move_robot,
            entrance_pose,
            duration_sec=0.1,
            debug_run=debug_run,
            phase_name="m5_approach_hold",
        )

        send_feedback("M5: descending to pre-insertion pose")
        self._move_for_duration(
            move_robot,
            entrance_pose,
            port_pose,
            duration_sec=0.9,
            debug_run=debug_run,
            phase_name="m5_descend",
        )
        self._hold_pose(
            move_robot,
            port_pose,
            duration_sec=0.1,
            debug_run=debug_run,
            phase_name="m5_port_hold",
        )

        aligned_pose = port_pose
        if use_multicam:
            send_feedback("M5: multi-camera late-fusion correction")
            aligned_pose, aligned_observation, aligned_features = (
                self._visual_servo_sfp_multicam(
                    task,
                    get_observation,
                    move_robot,
                    debug_run,
                    phase_name="m5_multicam",
                    max_steps=12,
                )
            )
            debug_run.save_observation_snapshot(
                "after_multicam_refine",
                aligned_observation,
                {"feature_summary": aligned_features},
            )
            if aligned_pose is None:
                aligned_pose = port_pose
        else:
            send_feedback("M4: direct public-target insertion baseline")

        insertion_axis = np.array(self._SFP_INSERTION_VECTOR_BASE, dtype=float)
        insertion_axis /= max(np.linalg.norm(insertion_axis), 1e-6)
        refined_push_pose = self._make_pose(
            self._pose_position(aligned_pose) + (0.015 * insertion_axis),
            self._sfp_visual_template_for_task(task)["nominal_quat_wxyz"],
        )
        blended_push_pose = self._make_pose(
            0.45 * self._pose_position(pushed_pose) + 0.55 * self._pose_position(refined_push_pose),
            self._sfp_visual_template_for_task(task)["nominal_quat_wxyz"],
        )

        send_feedback("M5: applying bounded insertion push")
        self._move_for_duration(
            move_robot,
            aligned_pose,
            blended_push_pose,
            duration_sec=0.45,
            debug_run=debug_run,
            phase_name="m5_push",
        )
        self._hold_pose(
            move_robot,
            blended_push_pose,
            duration_sec=0.8,
            debug_run=debug_run,
            phase_name="m5_push_hold",
        )
        return True

    def _run_sc_visual_residual(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        start_pose: Pose,
        baseline_force_norm: float | None = None,
    ) -> Pose:
        rotation = self._lookup_camera_rotation_base_from_optical(self._CENTER_CAMERA_FRAME)
        if rotation is None:
            return start_pose
        if baseline_force_norm is None:
            baseline_observation = self._wait_for_observation(
                get_observation, timeout_sec=0.5
            )
            baseline_force_norm = self._force_norm(baseline_observation)
        else:
            baseline_observation = self._wait_for_observation(
                get_observation, timeout_sec=0.5
            )

        best_pose = start_pose
        desired_u, desired_v = self._SC_CENTER_RESIDUAL_TEMPLATE["cyan_centroid_uv"]
        nominal_depth = float(self._SC_CENTER_RESIDUAL_TEMPLATE["nominal_depth_m"])
        last_stamp_sec = (
            None
            if baseline_observation is None
            else _stamp_to_float(baseline_observation.center_image.header.stamp)
        )

        for step in range(3):
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            feature = self._sc_center_feature(observation)
            if observation is None or feature is None:
                break
            last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)
            current_pose = observation.controller_state.tcp_pose
            fx = float(observation.center_camera_info.k[0])
            fy = float(observation.center_camera_info.k[4])
            union_u, union_v = feature["cyan_centroid_uv"]
            du = union_u - desired_u
            dv = union_v - desired_v
            delta_cam = np.array(
                [
                    0.45 * nominal_depth * du / max(fx, 1.0),
                    0.55 * nominal_depth * dv / max(fy, 1.0),
                    0.0,
                ],
                dtype=float,
            )
            delta_cam[0] = float(np.clip(delta_cam[0], -0.004, 0.004))
            delta_cam[1] = float(np.clip(delta_cam[1], -0.004, 0.004))
            delta_base = rotation @ delta_cam
            target_pose = self._make_pose(
                self._pose_position(current_pose) + delta_base,
                self._quat_xyzw_to_wxyz(current_pose.orientation),
            )
            debug_run.log_command_sample(
                "sc_visual_residual",
                current_pose,
                self.time_now().nanoseconds / 1e9,
                note=f"step {step + 1}/3 du={du:.1f} dv={dv:.1f}",
            )
            self._move_for_duration(
                move_robot,
                best_pose,
                target_pose,
                duration_sec=0.2,
                debug_run=debug_run,
                phase_name="sc_visual_residual",
            )
            inserted_pose = self._make_pose(
                self._pose_position(target_pose) + np.array([0.0, 0.0, -0.002], dtype=float),
                self._quat_xyzw_to_wxyz(target_pose.orientation),
            )
            self._move_for_duration(
                move_robot,
                target_pose,
                inserted_pose,
                duration_sec=0.2,
                debug_run=debug_run,
                phase_name="sc_residual_push",
            )
            observation = self._wait_for_observation(
                get_observation,
                timeout_sec=0.8,
                newer_than_sec=last_stamp_sec,
            )
            if observation is not None:
                last_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)
            measured_force = self._force_norm(observation)
            excess_force = measured_force - baseline_force_norm
            debug_run.log_command_sample(
                "sc_residual_push",
                inserted_pose,
                self.time_now().nanoseconds / 1e9,
                note=(
                    f"step {step + 1}/3 force={measured_force:.2f} "
                    f"excess={excess_force:.2f}"
                ),
            )
            if excess_force > 10.0:
                self._move_for_duration(
                    move_robot,
                    inserted_pose,
                    target_pose,
                    duration_sec=0.15,
                    debug_run=debug_run,
                    phase_name="sc_residual_retreat",
                )
                best_pose = target_pose
                continue
            best_pose = inserted_pose

        self._hold_pose(
            move_robot,
            best_pose,
            duration_sec=0.6,
            debug_run=debug_run,
            phase_name="sc_residual_hold",
        )
        return best_pose

    def _run_stage_m0(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            observation,
            {"feature_summary": self._extract_feature_summary(task, observation)},
        )
        if observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        send_feedback("M0: capturing observations and holding pose")
        debug_run.log_phase(
            "acquire_target",
            self.time_now().nanoseconds / 1e9,
            "captured initial observation",
        )
        self._hold_pose(
            move_robot,
            observation.controller_state.tcp_pose,
            self._hold_duration_sec,
            debug_run=debug_run,
            phase_name="observe",
        )
        final_observation = get_observation()
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            "completed M0 hold",
        )
        debug_run.finalize(
            True,
            "m0 completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"M0 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_m1_dev(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")

        observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            observation,
            {"feature_summary": self._extract_feature_summary(task, observation)},
        )
        if observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        task_key = self._task_key(task)
        target = self._DEV_TARGETS.get(task_key)
        if target is None:
            debug_run.finalize(
                False,
                f"no development target for task key {task_key}",
                observation,
                {"feature_summary": self._extract_feature_summary(task, observation)},
            )
            return False

        entrance_pose = self._pose_from_target(
            target.entrance_position, target.entrance_quat_wxyz
        )
        port_pose = self._pose_from_target(target.port_position, target.port_quat_wxyz)
        pushed_pose = self._push_pose(target)

        debug_run.save_target_metadata(
            {
                "provider": "development_pose_targets",
                "task_key": list(task_key),
                "target": asdict(target),
                "entrance_pose": _pose_to_dict(entrance_pose),
                "port_pose": _pose_to_dict(port_pose),
                "pushed_pose": _pose_to_dict(pushed_pose),
            }
        )

        current_pose = observation.controller_state.tcp_pose

        send_feedback("M1: using development target provider")
        debug_run.log_phase(
            "acquire_target",
            self.time_now().nanoseconds / 1e9,
            "loaded development target pose",
        )

        send_feedback("M1: approaching development entrance pose")
        debug_run.log_phase(
            "approach",
            self.time_now().nanoseconds / 1e9,
            "moving current pose to entrance pose",
        )
        self._move_for_duration(
            move_robot,
            current_pose,
            entrance_pose,
            duration_sec=self._approach_duration_sec,
            debug_run=debug_run,
            phase_name="approach",
        )
        approach_observation = get_observation()
        debug_run.save_observation_snapshot(
            "after_approach",
            approach_observation,
            {"feature_summary": self._extract_feature_summary(task, approach_observation)},
        )

        self._hold_pose(
            move_robot,
            entrance_pose,
            duration_sec=self._entrance_hold_sec,
            debug_run=debug_run,
            phase_name="approach_hold",
        )

        send_feedback("M1: aligning to development port target")
        debug_run.log_phase(
            "align",
            self.time_now().nanoseconds / 1e9,
            "moving entrance pose to port pose",
        )
        self._move_for_duration(
            move_robot,
            entrance_pose,
            port_pose,
            duration_sec=self._align_duration_sec,
            debug_run=debug_run,
            phase_name="align",
        )
        self._hold_pose(
            move_robot,
            port_pose,
            duration_sec=self._align_hold_sec,
            debug_run=debug_run,
            phase_name="align_hold",
        )
        align_observation = get_observation()
        debug_run.save_observation_snapshot(
            "after_align",
            align_observation,
            {"feature_summary": self._extract_feature_summary(task, align_observation)},
        )

        send_feedback("M1: inserting along development target axis")
        debug_run.log_phase(
            "insert",
            self.time_now().nanoseconds / 1e9,
            "moving port pose to pushed pose",
        )
        self._move_for_duration(
            move_robot,
            port_pose,
            pushed_pose,
            duration_sec=self._insert_duration_sec,
            debug_run=debug_run,
            phase_name="insert",
        )
        self._hold_pose(
            move_robot,
            pushed_pose,
            duration_sec=self._final_hold_sec,
            debug_run=debug_run,
            phase_name="insert_hold",
        )
        insert_observation = get_observation()
        debug_run.save_observation_snapshot(
            "after_insert",
            insert_observation,
            {"feature_summary": self._extract_feature_summary(task, insert_observation)},
        )

        debug_run.log_phase(
            "recover",
            self.time_now().nanoseconds / 1e9,
            "recovery not triggered; keeping last insertion pose",
        )
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            "completed M1 development controller path",
        )
        final_observation = get_observation()
        debug_run.finalize(
            True,
            "m1_dev completed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"M1 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_m2_sfp_center(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if task.plug_type != "sfp":
            return self._run_stage_hold_only(
                "M2: SC not implemented yet; holding pose",
                task,
                get_observation,
                move_robot,
                send_feedback,
            )

        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        initial_feature = self._sfp_union_feature(observation)
        debug_run.save_observation_snapshot(
            "initial",
            observation,
            {
                "feature_summary": (
                    None if initial_feature is None else initial_feature["feature_summary"]
                )
            },
        )
        if observation is None or initial_feature is None:
            debug_run.finalize(
                False,
                "no initial SFP feature detected",
                observation,
            )
            return False

        debug_run.save_target_metadata(
            {
                "provider": "center_camera_sfp_visual_servo",
                "task_key": list(self._task_key(task)),
                "visual_template": self._sfp_visual_template_for_task(task),
                "initial_feature": initial_feature,
            }
        )

        send_feedback("M2: center-camera SFP visual servo")
        debug_run.log_phase(
            "acquire_target",
            self.time_now().nanoseconds / 1e9,
            "detected initial SFP feature in center camera",
        )
        debug_run.log_phase(
            "approach",
            self.time_now().nanoseconds / 1e9,
            "running center-camera visual servo to SFP template",
        )
        aligned_pose, aligned_observation, aligned_feature = self._visual_servo_sfp(
            task,
            get_observation,
            move_robot,
            debug_run,
            phase_name="approach",
            max_steps=35,
            hold_when_aligned=True,
        )
        debug_run.save_observation_snapshot(
            "after_localize",
            aligned_observation,
            {
                "feature_summary": (
                    None if aligned_feature is None else aligned_feature["feature_summary"]
                )
            },
        )
        if aligned_pose is None or aligned_observation is None or aligned_feature is None:
            debug_run.finalize(False, "visual servo failed to localize SFP", aligned_observation)
            return False

        debug_run.log_phase(
            "align",
            self.time_now().nanoseconds / 1e9,
            "localized SFP target and held pose",
        )
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            "completed M2 SFP localization-only stage",
        )
        final_observation = get_observation()
        debug_run.finalize(
            True,
            "m2_sfp_center completed",
            final_observation,
            {
                "feature_summary": (
                    None
                    if final_observation is None
                    else self._sfp_union_feature(final_observation)
                )
            },
        )
        self.get_logger().info(f"M2 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_m3_sfp_insert(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if task.plug_type != "sfp":
            return self._run_stage_hold_only(
                "M3: SC not implemented yet; holding pose",
                task,
                get_observation,
                move_robot,
                send_feedback,
            )

        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        initial_feature = self._sfp_union_feature(observation)
        debug_run.save_observation_snapshot(
            "initial",
            observation,
            {
                "feature_summary": (
                    None if initial_feature is None else initial_feature["feature_summary"]
                )
            },
        )
        if observation is None or initial_feature is None:
            debug_run.finalize(
                False,
                "no initial SFP feature detected",
                observation,
            )
            return False

        debug_run.save_target_metadata(
            {
                "provider": "center_camera_sfp_visual_servo_with_insert",
                "task_key": list(self._task_key(task)),
                "visual_template": self._sfp_visual_template_for_task(task),
                "initial_feature": initial_feature,
            }
        )

        send_feedback("M3: center-camera SFP localization")
        debug_run.log_phase(
            "acquire_target",
            self.time_now().nanoseconds / 1e9,
            "detected initial SFP feature in center camera",
        )
        debug_run.log_phase(
            "approach",
            self.time_now().nanoseconds / 1e9,
            "running center-camera visual servo to SFP template",
        )
        aligned_pose, aligned_observation, aligned_feature = self._visual_servo_sfp(
            task,
            get_observation,
            move_robot,
            debug_run,
            phase_name="approach",
            max_steps=40,
            hold_when_aligned=True,
        )
        debug_run.save_observation_snapshot(
            "after_localize",
            aligned_observation,
            {
                "feature_summary": (
                    None if aligned_feature is None else aligned_feature["feature_summary"]
                )
            },
        )
        if aligned_pose is None or aligned_observation is None:
            debug_run.finalize(False, "visual servo failed to localize SFP", aligned_observation)
            return False

        insertion_axis = np.array(self._SFP_INSERTION_VECTOR_BASE, dtype=float)
        insertion_axis /= max(np.linalg.norm(insertion_axis), 1e-6)
        approach_position = self._pose_position(aligned_pose)
        port_position = approach_position + 0.0458 * insertion_axis
        pushed_position = approach_position + 0.0608 * insertion_axis
        nominal_quat = self._sfp_visual_template_for_task(task)["nominal_quat_wxyz"]
        port_pose = self._make_pose(port_position, nominal_quat)
        pushed_pose = self._make_pose(pushed_position, nominal_quat)

        send_feedback("M3: slow SFP insertion push")
        debug_run.log_phase(
            "insert",
            self.time_now().nanoseconds / 1e9,
            "moving from localized pose along nominal SFP insertion axis",
        )
        self._move_for_duration(
            move_robot,
            aligned_pose,
            port_pose,
            duration_sec=1.8,
            debug_run=debug_run,
            phase_name="insert_approach",
        )
        self._move_for_duration(
            move_robot,
            port_pose,
            pushed_pose,
            duration_sec=1.2,
            debug_run=debug_run,
            phase_name="insert_push",
        )
        self._hold_pose(
            move_robot,
            pushed_pose,
            duration_sec=2.0,
            debug_run=debug_run,
            phase_name="insert_hold",
        )
        inserted_observation = get_observation()
        debug_run.save_observation_snapshot(
            "after_insert",
            inserted_observation,
            {
                "feature_summary": (
                    None
                    if inserted_observation is None
                    else self._sfp_union_feature(inserted_observation)
                )
            },
        )
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            "completed M3 SFP insertion baseline",
        )
        final_observation = get_observation()
        debug_run.finalize(
            True,
            "m3_sfp_insert completed",
            final_observation,
            {
                "feature_summary": (
                    None
                    if final_observation is None
                    else self._sfp_union_feature(final_observation)
                )
            },
        )
        self.get_logger().info(f"M3 artifacts saved to {debug_run.root}")
        return True

    def _run_stage_m4_public_baseline(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "public_trial_pose_baseline",
                "task_key": list(self._task_key(task)),
                "note": "Development-only public-sample baseline for full-pipeline debugging.",
            }
        )
        send_feedback("M4: running public-sample baseline with debug capture")
        if task.plug_type == "sfp":
            debug_run.log_phase(
                "execute",
                self.time_now().nanoseconds / 1e9,
                "running shortened public-target SFP baseline",
            )
            success = self._run_public_sfp_sequence(
                task,
                get_observation,
                move_robot,
                debug_run,
                send_feedback,
                use_multicam=False,
            )
        else:
            debug_run.log_phase(
                "execute",
                self.time_now().nanoseconds / 1e9,
                "running shortened SC replay baseline",
            )
            self._replay_sc_trajectory_with_debug(
                move_robot,
                debug_run,
                time_scale=0.35,
                final_hold_sec=0.4,
            )
            success = True
        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            f"public baseline returned {success}",
        )
        debug_run.finalize(
            bool(success),
            "m4_public_baseline completed" if success else "m4_public_baseline failed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"M4 artifacts saved to {debug_run.root}")
        return bool(success)

    def _run_stage_m6_sc_force_refine(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "public_sample_plus_sc_force_refine",
                "task_key": list(self._task_key(task)),
            }
        )

        if task.plug_type != "sc":
            send_feedback("M6: using M5 SFP baseline")
            debug_run.log_phase(
                "execute",
                self.time_now().nanoseconds / 1e9,
                "running SFP path with late-fusion correction",
            )
            success = self._run_public_sfp_sequence(
                task,
                get_observation,
                move_robot,
                debug_run,
                send_feedback,
                use_multicam=True,
            )
        else:
            send_feedback("M6: SC center-camera alignment then bounded force-guided refine")
            baseline_force_norm = self._force_norm(initial_observation)
            debug_run.log_phase(
                "align",
                self.time_now().nanoseconds / 1e9,
                "running SC center-camera coarse alignment",
            )
            centered_pose = self._run_sc_center_servo(
                task,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.log_phase(
                "force_refine",
                self.time_now().nanoseconds / 1e9,
                "running SC bounded force-guided refine",
            )
            self._run_sc_force_refine(
                get_observation,
                move_robot,
                debug_run,
                start_pose=centered_pose,
                baseline_force_norm=baseline_force_norm,
            )
            success = True

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "after_refine",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            f"m6 stage returned {success}",
        )
        debug_run.finalize(
            bool(success),
            "m6_sc_force_refine completed" if success else "m6_sc_force_refine failed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"M6 artifacts saved to {debug_run.root}")
        return bool(success)

    def _run_stage_m5_multi_camera_late_fusion(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "public_sfp_plus_multicam_or_sc_replay",
                "task_key": list(self._task_key(task)),
                "sfp_template": (
                    self._sfp_multi_camera_template_for_task(task)
                    if task.plug_type == "sfp"
                    else None
                ),
            }
        )

        if task.plug_type == "sfp":
            success = self._run_public_sfp_sequence(
                task,
                get_observation,
                move_robot,
                debug_run,
                send_feedback,
                use_multicam=True,
            )
        else:
            send_feedback("M5: SC still on shortened replay baseline")
            debug_run.log_phase(
                "execute",
                self.time_now().nanoseconds / 1e9,
                "running shortened SC replay baseline",
            )
            self._replay_sc_trajectory_with_debug(
                move_robot,
                debug_run,
                time_scale=0.35,
                final_hold_sec=0.4,
            )
            success = True

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            f"m5 stage returned {success}",
        )
        debug_run.finalize(
            bool(success),
            "m5_multi_camera_late_fusion completed"
            if success
            else "m5_multi_camera_late_fusion failed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"M5 artifacts saved to {debug_run.root}")
        return bool(success)

    def _run_stage_m7_residual_refine(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "m5_sfp_plus_sc_force_and_visual_residual",
                "task_key": list(self._task_key(task)),
                "sc_center_template": self._SC_CENTER_RESIDUAL_TEMPLATE,
            }
        )

        if task.plug_type == "sfp":
            success = self._run_public_sfp_sequence(
                task,
                get_observation,
                move_robot,
                debug_run,
                send_feedback,
                use_multicam=True,
            )
        else:
            send_feedback("M7: SC center-camera alignment -> force refine -> visual residual")
            baseline_force_norm = self._force_norm(initial_observation)
            debug_run.log_phase(
                "align",
                self.time_now().nanoseconds / 1e9,
                "running SC center-camera coarse alignment",
            )
            centered_pose = self._run_sc_center_servo(
                task,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.log_phase(
                "force_refine",
                self.time_now().nanoseconds / 1e9,
                "running SC bounded force-guided refine",
            )
            refined_pose = self._run_sc_force_refine(
                get_observation,
                move_robot,
                debug_run,
                start_pose=centered_pose,
                baseline_force_norm=baseline_force_norm,
            )
            debug_run.log_phase(
                "residual",
                self.time_now().nanoseconds / 1e9,
                "applying center-camera SC residual correction",
            )
            self._run_sc_visual_residual(
                get_observation,
                move_robot,
                debug_run,
                refined_pose,
                baseline_force_norm=baseline_force_norm,
            )
            success = True

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "after_residual",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            f"m7 stage returned {success}",
        )
        debug_run.finalize(
            bool(success),
            "m7_residual_refine completed" if success else "m7_residual_refine failed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"M7 artifacts saved to {debug_run.root}")
        return bool(success)

    def _run_stage_submission_safe_v0(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "notes": [
                    "no public-sample world targets",
                    "no development-only target provider",
                    "tool-axis push derived from grasp-relative TCP motion",
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback("submission_v0: SFP center-camera legal servo")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running SFP visual servo using only current observation",
            )
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None if aligned_observation is None else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v0 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            send_feedback("submission_v0: SFP bounded tool-axis push")
            debug_run.log_phase(
                "insert",
                self.time_now().nanoseconds / 1e9,
                "applying short tool-axis insertion push from legal servo pose",
            )
            final_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SFP_TOOL_INSERTION_AXIS,
                push_distance_m=0.012,
                duration_sec=0.45,
                hold_sec=0.8,
                debug_run=debug_run,
                phase_prefix="submission_sfp",
            )
            success = True
        else:
            send_feedback("submission_v0: SC center-camera legal servo")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running SC visual servo using only current observation",
            )
            aligned_pose, aligned_observation, aligned_feature = self._run_sc_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None if aligned_observation is None else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v0 SC servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            baseline_force_norm = self._force_norm(initial_observation)
            debug_run.log_phase(
                "insert",
                self.time_now().nanoseconds / 1e9,
                "applying short SC tool-axis push followed by bounded legal refinement",
            )
            pushed_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SC_TOOL_INSERTION_AXIS,
                push_distance_m=0.004,
                duration_sec=0.35,
                hold_sec=0.25,
                debug_run=debug_run,
                phase_prefix="submission_sc",
            )
            refined_pose = self._run_sc_force_refine(
                get_observation,
                move_robot,
                debug_run,
                start_pose=pushed_pose,
                baseline_force_norm=baseline_force_norm,
            )
            debug_run.log_phase(
                "residual",
                self.time_now().nanoseconds / 1e9,
                "applying legal SC residual correction",
            )
            final_pose = self._run_sc_visual_residual(
                get_observation,
                move_robot,
                debug_run,
                refined_pose,
                baseline_force_norm=baseline_force_norm,
            )
            success = True

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "final_pose": _pose_to_dict(final_pose),
            },
        )
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            f"submission_safe_v0 returned {success}",
        )
        debug_run.finalize(
            bool(success),
            "submission_safe_v0 completed" if success else "submission_safe_v0 failed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v0 artifacts saved to {debug_run.root}")
        return bool(success)

    def _run_stage_submission_safe_v1(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        debug_run = DebugRun(self._stage, task)
        debug_run.log_phase("idle", self.time_now().nanoseconds / 1e9, "task accepted")
        initial_observation = self._wait_for_observation(get_observation, timeout_sec=5.0)
        debug_run.save_observation_snapshot(
            "initial",
            initial_observation,
            {"feature_summary": self._extract_feature_summary(task, initial_observation)},
        )
        if initial_observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        debug_run.save_target_metadata(
            {
                "provider": "official_task_and_observation_only",
                "task_key": list(self._task_key(task)),
                "notes": [
                    "no public-sample world targets",
                    "no development-only target provider",
                    "SFP path matches submission_safe_v0",
                    "SC path uses coarse-to-fine task-local sight-picture templates",
                ],
            }
        )

        if task.plug_type == "sfp":
            send_feedback("submission_v1: SFP center-camera legal servo")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running SFP visual servo using only current observation",
            )
            aligned_pose, aligned_observation, aligned_feature = self._run_sfp_submission_servo(
                initial_observation,
                get_observation,
                move_robot,
                debug_run,
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None if aligned_observation is None else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v1 SFP servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            send_feedback("submission_v1: SFP bounded tool-axis push")
            debug_run.log_phase(
                "insert",
                self.time_now().nanoseconds / 1e9,
                "applying short tool-axis insertion push from legal servo pose",
            )
            final_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SFP_TOOL_INSERTION_AXIS,
                push_distance_m=0.012,
                duration_sec=0.45,
                hold_sec=0.8,
                debug_run=debug_run,
                phase_prefix="submission_sfp",
            )
            success = True
        else:
            send_feedback("submission_v1: SC coarse-to-fine legal servo")
            debug_run.log_phase(
                "acquire_target",
                self.time_now().nanoseconds / 1e9,
                "running SC coarse-to-fine visual servo using only current observation",
            )
            aligned_pose, aligned_observation, aligned_feature = (
                self._run_sc_submission_servo_coarse_to_fine(
                    initial_observation,
                    get_observation,
                    move_robot,
                    debug_run,
                )
            )
            debug_run.save_observation_snapshot(
                "after_legal_servo",
                aligned_observation,
                {
                    "feature_summary": (
                        None if aligned_observation is None else self._extract_feature_summary(task, aligned_observation)
                    ),
                    "feature": aligned_feature,
                },
            )
            if aligned_pose is None:
                debug_run.finalize(
                    False,
                    "submission_v1 SC servo failed",
                    aligned_observation,
                    {"feature_summary": self._extract_feature_summary(task, aligned_observation)},
                )
                return False

            baseline_force_norm = self._force_norm(initial_observation)
            debug_run.log_phase(
                "insert",
                self.time_now().nanoseconds / 1e9,
                "applying short SC tool-axis push followed by bounded legal refinement",
            )
            pushed_pose = self._run_tool_axis_push(
                move_robot,
                aligned_pose,
                self._SC_TOOL_INSERTION_AXIS,
                push_distance_m=0.004,
                duration_sec=0.35,
                hold_sec=0.25,
                debug_run=debug_run,
                phase_prefix="submission_sc",
            )
            refined_pose = self._run_sc_force_refine(
                get_observation,
                move_robot,
                debug_run,
                start_pose=pushed_pose,
                baseline_force_norm=baseline_force_norm,
            )
            debug_run.log_phase(
                "residual",
                self.time_now().nanoseconds / 1e9,
                "applying legal SC residual correction",
            )
            final_pose = self._run_sc_visual_residual(
                get_observation,
                move_robot,
                debug_run,
                refined_pose,
                baseline_force_norm=baseline_force_norm,
            )
            success = True

        final_observation = self._wait_for_observation(get_observation, timeout_sec=2.0)
        debug_run.save_observation_snapshot(
            "final",
            final_observation,
            {
                "feature_summary": self._extract_feature_summary(task, final_observation),
                "final_pose": _pose_to_dict(final_pose),
            },
        )
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            f"submission_safe_v1 returned {success}",
        )
        debug_run.finalize(
            bool(success),
            "submission_safe_v1 completed" if success else "submission_safe_v1 failed",
            final_observation,
            {"feature_summary": self._extract_feature_summary(task, final_observation)},
        )
        self.get_logger().info(f"submission_safe_v1 artifacts saved to {debug_run.root}")
        return bool(success)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(
            f"QualPhasePilot.insert_cable() stage={self._stage} task={task}"
        )
        if self._stage == "m0":
            return self._run_stage_m0(task, get_observation, move_robot, send_feedback)
        if self._stage == "m1_dev":
            return self._run_stage_m1_dev(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "m2_sfp_center":
            return self._run_stage_m2_sfp_center(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "m3_sfp_insert":
            return self._run_stage_m3_sfp_insert(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "m4_public_baseline":
            return self._run_stage_m4_public_baseline(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "m5_multi_camera_late_fusion":
            return self._run_stage_m5_multi_camera_late_fusion(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "m6_sc_force_refine":
            return self._run_stage_m6_sc_force_refine(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "m7_residual_refine":
            return self._run_stage_m7_residual_refine(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v0":
            return self._run_stage_submission_safe_v0(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v1":
            return self._run_stage_submission_safe_v1(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v2":
            return self._run_stage_submission_safe_v2(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v3":
            return self._run_stage_submission_safe_v3(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v4":
            return self._run_stage_submission_safe_v4(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v6":
            return self._run_stage_submission_safe_v6(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v7":
            return self._run_stage_submission_safe_v7(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v8":
            return self._run_stage_submission_safe_v8(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v9":
            return self._run_stage_submission_safe_v9(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v10":
            return self._run_stage_submission_safe_v10(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v11":
            return self._run_stage_submission_safe_v11(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v12":
            return self._run_stage_submission_safe_v12(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v13":
            return self._run_stage_submission_safe_v13(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v14":
            return self._run_stage_submission_safe_v14(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v15":
            return self._run_stage_submission_safe_v15(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v16":
            return self._run_stage_submission_safe_v16(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "submission_safe_v17":
            return self._run_stage_submission_safe_v17(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "learn_collect_v0":
            return self._run_stage_learn_collect_v0(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "learn_collect_v1":
            return self._run_stage_learn_collect_v1(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "learn_collect_v2":
            return self._run_stage_learn_collect_v2(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "learn_collect_sfp_insert_v0":
            return self._run_stage_learn_collect_sfp_insert_v0(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "learn_collect_sfp_servo_residual_v0":
            return self._run_stage_learn_collect_sfp_servo_residual_v0(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "learn_collect_sfp_teacher_step_v0":
            return self._run_stage_learn_collect_sfp_teacher_step_v0(
                task, get_observation, move_robot, send_feedback
            )
        if self._stage == "teacher_feasibility_v0":
            return self._run_stage_teacher_feasibility_v0(
                task, get_observation, move_robot, send_feedback
            )

        self.get_logger().error(f"Stage '{self._stage}' is not implemented yet.")
        return False
