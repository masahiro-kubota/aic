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
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from transforms3d._gohlketransforms import quaternion_slerp
from transforms3d.quaternions import quat2mat

from aic_example_policies.ros.PublicTrialPosePilot import (
    PublicTrialPosePilot,
    SC_REPLAY_TRAJECTORY,
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
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

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
    _SFP_TOOL_INSERTION_AXIS = (
        0.000159431701,
        -0.350186974,
        0.936679805,
    )
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
        deadline = self.time_now() + Duration(seconds=timeout_sec)
        while self.time_now() < deadline:
            observation = get_observation()
            if observation is None:
                self.sleep_for(0.05)
                continue
            if newer_than_sec is None:
                return observation
            observation_stamp_sec = _stamp_to_float(observation.center_image.header.stamp)
            if observation_stamp_sec > newer_than_sec + 1e-6:
                return observation
            self.sleep_for(0.05)
        return None

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
            self.sleep_for(self._command_period_sec)

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
            self.sleep_for(self._command_period_sec)

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
        return quat2mat(quat_wxyz)

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
        if observation is None:
            return None
        image, _ = self._camera_view(observation, "center")
        if image is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(
            hsv,
            np.array([80, 80, 80], dtype=np.uint8),
            np.array([110, 255, 255], dtype=np.uint8),
        )
        cyan_components = _mask_components(cyan_mask, min_area_px=80)
        if not cyan_components:
            return None
        largest = max(cyan_components, key=lambda item: item["area_px"])
        return {
            "cyan_components": cyan_components,
            "cyan_centroid_uv": tuple(largest["centroid_uv"]),
            "cyan_bbox_xywh": tuple(largest["bbox_xywh"]),
            "cyan_area_px": int(largest["area_px"]),
        }

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

    def _run_sc_submission_servo(
        self,
        initial_observation: Observation,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        max_steps: int = 20,
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        rotation_base_from_optical = self._lookup_camera_rotation_base_from_optical(
            self._CENTER_CAMERA_FRAME
        )
        if rotation_base_from_optical is None:
            return None, None, None

        initial_feature = self._sc_center_feature(initial_observation)
        if initial_feature is None:
            return None, initial_observation, None

        desired_u = 0.5 * float(initial_observation.center_image.width)
        desired_v = 0.5 * float(initial_observation.center_image.height)
        desired_area = max(
            float(initial_feature["cyan_area_px"]) * 1.30,
            float(initial_feature["cyan_area_px"]) + 4000.0,
        )
        desired_area = float(np.clip(desired_area, 10000.0, 45000.0))
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
                abs(du) < 22.0
                and abs(dv) < 22.0
                and area_px >= 0.90 * desired_area
            )
            debug_run.log_command_sample(
                "submission_sc_servo",
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
                    0.42 * 0.06 * du / max(fx, 1.0),
                    0.52 * 0.06 * dv / max(fy, 1.0),
                    float(np.clip(0.025 * depth_error, -0.008, 0.012)),
                ],
                dtype=float,
            )
            delta_cam[0] = float(np.clip(delta_cam[0], -0.008, 0.008))
            delta_cam[1] = float(np.clip(delta_cam[1], -0.008, 0.008))
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
                phase_name="submission_sc_servo",
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
            phase_name="submission_sc_settle",
        )
        final_observation = self._wait_for_observation(
            get_observation,
            timeout_sec=0.8,
            newer_than_sec=last_stamp_sec,
        )
        if final_observation is None:
            final_observation = observation
        return best_pose, final_observation, best_feature

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

        self.get_logger().error(f"Stage '{self._stage}' is not implemented yet.")
        return False
