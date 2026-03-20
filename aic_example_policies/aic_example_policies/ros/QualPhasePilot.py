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

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose
from rclpy.duration import Duration


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

    def _write_json(self, path: Path, payload: dict) -> None:
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

    def finalize(self, success: bool, note: str, final_observation: Observation | None) -> None:
        if final_observation is not None:
            self.save_observation_snapshot("final", final_observation, {"note": note})
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
            },
        )


class QualPhasePilot(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._stage = os.environ.get("AIC_QUAL_STAGE", "m0")
        self._hold_duration_sec = float(os.environ.get("AIC_QUAL_M0_HOLD_SEC", "3.0"))
        self.get_logger().info(f"QualPhasePilot.__init__() stage={self._stage}")

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

    def _hold_current_pose(
        self,
        observation: Observation,
        move_robot: MoveRobotCallback,
        duration_sec: float,
    ) -> None:
        pose = observation.controller_state.tcp_pose
        steps = max(1, int(duration_sec / 0.05))
        for _ in range(steps):
            self.set_pose_target(move_robot=move_robot, pose=pose)
            self.sleep_for(0.05)

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
        debug_run.save_observation_snapshot("initial", observation)
        if observation is None:
            debug_run.finalize(False, "no observation received", None)
            return False

        send_feedback("M0: capturing observations and holding pose")
        debug_run.log_phase(
            "acquire_target",
            self.time_now().nanoseconds / 1e9,
            "captured initial observation",
        )
        self._hold_current_pose(observation, move_robot, self._hold_duration_sec)
        final_observation = get_observation()
        debug_run.log_phase(
            "done",
            self.time_now().nanoseconds / 1e9,
            "completed M0 hold",
        )
        debug_run.finalize(True, "m0 completed", final_observation)
        self.get_logger().info(f"M0 artifacts saved to {debug_run.root}")
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

        self.get_logger().error(f"Stage '{self._stage}' is not implemented yet.")
        return False
