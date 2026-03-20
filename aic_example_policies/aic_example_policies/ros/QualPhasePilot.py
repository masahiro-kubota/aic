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
from tf2_ros import TransformException


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
    _SFP_VISUAL_TEMPLATE = {
        "union_center_uv": (538.0, 565.0),
        "union_size_px": (306.0, 332.0),
        "camera_depth_m": 0.18,
        "nominal_quat_wxyz": (0.184036, 0.9825145, -0.0277635, -0.0049785),
    }
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
        self.get_logger().info(f"QualPhasePilot.__init__() stage={self._stage}")

    def _task_key(self, task: Task) -> tuple[str, str, str]:
        return (task.plug_type, task.target_module_name, task.port_name)

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
        return Pose(
            position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
            orientation=self._quat_wxyz_to_xyzw(q),
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
        self, get_observation: GetObservationCallback, timeout_sec: float = 5.0
    ) -> Observation | None:
        deadline = self.time_now() + Duration(seconds=timeout_sec)
        while self.time_now() < deadline:
            observation = get_observation()
            if observation is not None:
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
        steps = max(1, int(duration_sec / 0.05))
        for step in range(steps):
            self.set_pose_target(move_robot=move_robot, pose=pose)
            if debug_run is not None and (step == 0 or step == steps - 1):
                debug_run.log_command_sample(
                    phase_name,
                    pose,
                    self.time_now().nanoseconds / 1e9,
                    note=f"hold step {step + 1}/{steps}",
                )
            self.sleep_for(0.05)

    def _move_for_duration(
        self,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
        end_pose: Pose,
        duration_sec: float,
        debug_run: DebugRun | None = None,
        phase_name: str = "move",
    ) -> None:
        steps = max(1, int(duration_sec / 0.05))
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
            self.sleep_for(0.05)

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

    def _lookup_camera_rotation_base_from_optical(self) -> np.ndarray | None:
        try:
            transform = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                self._CENTER_CAMERA_FRAME,
                Time(),
            )
        except TransformException as ex:
            self.get_logger().warn(f"center camera TF lookup failed: {ex}")
            return None
        quat_wxyz = (
            transform.transform.rotation.w,
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
        )
        return quat2mat(quat_wxyz)

    def _sfp_union_feature(self, observation: Observation | None) -> dict | None:
        if observation is None:
            return None
        center = _image_to_bgr(observation.center_image)
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
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
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        debug_run: DebugRun,
        phase_name: str,
        max_steps: int,
        hold_when_aligned: bool,
    ) -> tuple[Pose | None, Observation | None, dict | None]:
        rotation_base_from_optical = self._lookup_camera_rotation_base_from_optical()
        if rotation_base_from_optical is None:
            return None, None, None

        desired_u, desired_v = self._SFP_VISUAL_TEMPLATE["union_center_uv"]
        desired_w, desired_h = self._SFP_VISUAL_TEMPLATE["union_size_px"]
        desired_quat = self._SFP_VISUAL_TEMPLATE["nominal_quat_wxyz"]
        nominal_depth = float(self._SFP_VISUAL_TEMPLATE["camera_depth_m"])

        best_observation = None
        best_feature = None
        for step in range(max_steps):
            observation = self._wait_for_observation(get_observation, timeout_sec=1.0)
            feature = self._sfp_union_feature(observation)
            if observation is None or feature is None:
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
            dh = desired_h - union_h

            aligned = abs(du) < 18.0 and abs(dv) < 18.0 and abs(dh) < 24.0
            debug_run.log_command_sample(
                phase_name,
                current_pose,
                self.time_now().nanoseconds / 1e9,
                note=(
                    f"servo step {step + 1}/{max_steps} "
                    f"du={du:.1f} dv={dv:.1f} dh={dh:.1f}"
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
                    nominal_depth * du / max(fx, 1.0),
                    nominal_depth * dv / max(fy, 1.0),
                    np.clip(0.00035 * dh, -0.012, 0.012),
                ],
                dtype=float,
            )
            delta_cam[0] = float(np.clip(delta_cam[0], -0.02, 0.02))
            delta_cam[1] = float(np.clip(delta_cam[1], -0.02, 0.02))
            delta_base = rotation_base_from_optical @ delta_cam
            target_position = self._pose_position(current_pose) + delta_base
            target_pose = self._make_pose(target_position, desired_quat)
            self.set_pose_target(move_robot=move_robot, pose=target_pose)
            self.sleep_for(0.1)

        if best_observation is None:
            return None, None, None
        final_pose = self._make_pose(
            self._pose_position(best_observation.controller_state.tcp_pose),
            self._SFP_VISUAL_TEMPLATE["nominal_quat_wxyz"],
        )
        return final_pose, best_observation, best_feature

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
                "visual_template": self._SFP_VISUAL_TEMPLATE,
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
            get_observation,
            move_robot,
            debug_run,
            phase_name="approach",
            max_steps=25,
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
                "visual_template": self._SFP_VISUAL_TEMPLATE,
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
            get_observation,
            move_robot,
            debug_run,
            phase_name="approach",
            max_steps=25,
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
        nominal_quat = self._SFP_VISUAL_TEMPLATE["nominal_quat_wxyz"]
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

        self.get_logger().error(f"Stage '{self._stage}' is not implemented yet.")
        return False
