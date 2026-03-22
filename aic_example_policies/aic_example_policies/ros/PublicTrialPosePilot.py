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

from dataclasses import dataclass
import time

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from transforms3d._gohlketransforms import quaternion_slerp

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


@dataclass(frozen=True)
class PhaseTarget:
    entrance_position: tuple[float, float, float]
    entrance_quat_wxyz: tuple[float, float, float, float]
    port_position: tuple[float, float, float]
    port_quat_wxyz: tuple[float, float, float, float]
    push_distance_m: float


SC_REPLAY_TRAJECTORY = [
    (0.0, -0.372294, 0.194088, 0.31999, 1.0, 3.5e-05, 6.9e-05, -3.3e-05),
    (0.148, -0.376018, 0.196575, 0.315539, 0.99997, -0.001993, -0.000324, 0.007534),
    (0.302, -0.380163, 0.198971, 0.310088, 0.999844, -0.004599, -0.00102, 0.017018),
    (0.502, -0.386542, 0.203018, 0.301689, 0.99943, -0.008844, -0.002233, 0.032509),
    (0.702, -0.393862, 0.208161, 0.292371, 0.998614, -0.01377, -0.003571, 0.050682),
    (0.907, -0.401975, 0.214235, 0.282354, 0.997306, -0.019128, -0.004899, 0.070643),
    (1.107, -0.410655, 0.221009, 0.272, 0.995483, -0.024663, -0.006162, 0.091474),
    (1.307, -0.419667, 0.228217, 0.26158, 0.9932, -0.030134, -0.007287, 0.112219),
    (1.507, -0.428751, 0.235651, 0.251414, 0.990568, -0.035343, -0.008245, 0.132129),
    (1.707, -0.43763, 0.243073, 0.241817, 0.987742, -0.040124, -0.008988, 0.150584),
    (1.936, -0.446671, 0.250689, 0.232202, 0.984728, -0.044587, -0.009545, 0.168022),
    (2.144, -0.454837, 0.257626, 0.223974, 0.981928, -0.048321, -0.009908, 0.182714),
    (2.345, -0.461824, 0.263539, 0.217138, 0.979509, -0.051298, -0.010156, 0.194493),
    (2.548, -0.467919, 0.268828, 0.211358, 0.977425, -0.053707, -0.010328, 0.20408),
    (2.747, -0.47306, 0.273293, 0.206659, 0.97572, -0.055589, -0.010437, 0.211593),
    (2.948, -0.477276, 0.276949, 0.202836, 0.974371, -0.05703, -0.010512, 0.217344),
    (3.147, -0.480413, 0.279527, 0.199988, 0.973344, -0.05806, -0.010527, 0.221627),
    (3.348, -0.482743, 0.281507, 0.197706, 0.972614, -0.058833, -0.010639, 0.224604),
    (3.503, -0.484068, 0.282647, 0.196339, 0.972231, -0.059259, -0.010741, 0.226142),
    (3.713, -0.48551, 0.283867, 0.194983, 0.971836, -0.059714, -0.01087, 0.227707),
    (3.916, -0.486521, 0.28481, 0.194175, 0.971595, -0.06002, -0.010984, 0.228647),
    (4.138, -0.48725, 0.285611, 0.193682, 0.971446, -0.060225, -0.011072, 0.229224),
    (4.345, -0.487726, 0.286295, 0.193501, 0.971364, -0.060349, -0.011132, 0.229536),
    (4.547, -0.488058, 0.286909, 0.193517, 0.971318, -0.060425, -0.011172, 0.229709),
    (4.747, -0.488295, 0.287486, 0.193642, 0.971293, -0.06047, -0.011198, 0.229798),
    (4.947, -0.488473, 0.288033, 0.193839, 0.971279, -0.060495, -0.011209, 0.22985),
    (5.105, -0.489096, 0.289722, 0.193516, 0.971273, -0.060505, -0.01121, 0.229875),
    (5.312, -0.491052, 0.294759, 0.191517, 0.971273, -0.060505, -0.01121, 0.229875),
    (5.545, -0.492763, 0.295929, 0.189505, 0.971273, -0.060505, -0.011208, 0.229876),
    (5.746, -0.494129, 0.295929, 0.187519, 0.971273, -0.060505, -0.01121, 0.229875),
    (5.948, -0.495087, 0.295929, 0.185519, 0.971273, -0.060505, -0.01121, 0.229875),
    (6.104, -0.495448, 0.295929, 0.184019, 0.971273, -0.060505, -0.01121, 0.229875),
    (6.312, -0.495461, 0.295929, 0.182019, 0.971273, -0.060505, -0.01121, 0.229875),
    (6.535, -0.495002, 0.295929, 0.180003, 0.971273, -0.060505, -0.011211, 0.229875),
    (6.748, -0.49416, 0.295929, 0.178018, 0.971273, -0.060505, -0.01121, 0.229875),
    (6.95, -0.493149, 0.295929, 0.176018, 0.971273, -0.060505, -0.01121, 0.229875),
    (7.105, -0.492379, 0.295929, 0.174517, 0.971273, -0.060505, -0.01121, 0.229875),
    (7.309, -0.491467, 0.295929, 0.172517, 0.971273, -0.060505, -0.01121, 0.229875),
    (7.521, -0.49081, 0.295929, 0.170516, 0.971273, -0.060505, -0.01121, 0.229875),
    (7.753, -0.490388, 0.295929, 0.168515, 0.971273, -0.060505, -0.01121, 0.229875),
    (7.904, -0.490243, 0.295929, 0.167015, 0.971273, -0.060505, -0.01121, 0.229875),
    (8.114, -0.490238, 0.295929, 0.165015, 0.971273, -0.060505, -0.01121, 0.229875),
    (8.322, -0.490452, 0.295929, 0.163015, 0.971273, -0.060505, -0.01121, 0.229875),
    (8.528, -0.490807, 0.295929, 0.161015, 0.971273, -0.060505, -0.01121, 0.229875),
    (8.731, -0.491201, 0.295861, 0.159014, 0.971273, -0.060506, -0.01121, 0.229875),
    (8.932, -0.491515, 0.295666, 0.157014, 0.971273, -0.060505, -0.01121, 0.229875),
    (9.144, -0.491692, 0.295381, 0.155014, 0.971273, -0.060506, -0.01121, 0.229875),
    (9.317, -0.491735, 0.295115, 0.153496, 0.971273, -0.060505, -0.01121, 0.229874),
    (9.536, -0.491621, 0.294725, 0.151514, 0.971273, -0.060505, -0.01121, 0.229875),
    (9.701, -0.491452, 0.294409, 0.150014, 0.971273, -0.060505, -0.01121, 0.229875),
    (9.903, -0.491158, 0.293967, 0.148014, 0.971273, -0.060505, -0.01121, 0.229875),
    (10.108, -0.490797, 0.293522, 0.146013, 0.971273, -0.060505, -0.01121, 0.229875),
    (10.32, -0.490412, 0.293083, 0.144013, 0.971273, -0.060505, -0.01121, 0.229875),
    (10.547, -0.490031, 0.29267, 0.142013, 0.971273, -0.060505, -0.01121, 0.229875),
    (10.707, -0.489797, 0.292384, 0.140512, 0.971273, -0.060505, -0.01121, 0.229874),
    (10.915, -0.489547, 0.292048, 0.138512, 0.971273, -0.060505, -0.01121, 0.229875),
    (11.124, -0.489393, 0.291769, 0.136512, 0.971273, -0.060505, -0.01121, 0.229874),
    (11.337, -0.489333, 0.291581, 0.134512, 0.971273, -0.060505, -0.01121, 0.229874),
    (11.504, -0.489348, 0.291517, 0.133012, 0.971273, -0.060505, -0.01121, 0.229875),
    (11.733, -0.489436, 0.291555, 0.131012, 0.971273, -0.060506, -0.01121, 0.229875),
    (11.938, -0.489558, 0.29176, 0.129013, 0.971273, -0.060505, -0.01121, 0.229875),
    (12.105, -0.489646, 0.292045, 0.127513, 0.971273, -0.060505, -0.01121, 0.229875),
    (12.316, -0.489753, 0.292612, 0.125514, 0.971273, -0.060505, -0.01121, 0.229875),
    (12.549, -0.48992, 0.293393, 0.123498, 0.971273, -0.060506, -0.01121, 0.229876),
    (12.714, -0.490103, 0.294125, 0.122015, 0.971273, -0.060505, -0.01121, 0.229875),
    (12.919, -0.490415, 0.295285, 0.120016, 0.971273, -0.060505, -0.01121, 0.229875),
    (13.122, -0.490792, 0.295929, 0.118018, 0.971273, -0.060505, -0.01121, 0.229875),
    (13.333, -0.49125, 0.295929, 0.116005, 0.971273, -0.060505, -0.011209, 0.229876),
    (13.517, -0.491619, 0.295929, 0.11452, 0.971273, -0.060505, -0.01121, 0.229875),
    (13.73, -0.492127, 0.295929, 0.112521, 0.971273, -0.060505, -0.01121, 0.229875),
    (13.935, -0.492605, 0.295929, 0.110521, 0.971273, -0.060505, -0.01121, 0.229875),
    (14.144, -0.493016, 0.295929, 0.108522, 0.971273, -0.060505, -0.01121, 0.229875),
    (14.318, -0.493274, 0.295929, 0.107023, 0.971273, -0.060505, -0.01121, 0.229875),
    (14.538, -0.493487, 0.295929, 0.105024, 0.971273, -0.060505, -0.01121, 0.229875),
    (14.704, -0.493579, 0.295929, 0.103525, 0.971273, -0.060505, -0.01121, 0.229875),
    (14.935, -0.493676, 0.295929, 0.101525, 0.971273, -0.060505, -0.01121, 0.229875),
    (15.14, -0.493818, 0.295929, 0.099526, 0.971273, -0.060505, -0.01121, 0.229875),
    (15.351, -0.493967, 0.295929, 0.097527, 0.971273, -0.060505, -0.01121, 0.229875),
    (15.508, -0.494072, 0.295929, 0.096028, 0.971273, -0.060505, -0.01121, 0.229875),
    (15.718, -0.494224, 0.295929, 0.094016, 0.971273, -0.060505, -0.01121, 0.229875),
    (15.945, -0.49445, 0.295929, 0.092018, 0.971273, -0.060505, -0.01121, 0.229875),
    (16.102, -0.494688, 0.295929, 0.090531, 0.971273, -0.060505, -0.01121, 0.229875),
    (16.303, -0.49507, 0.295929, 0.088531, 0.971273, -0.060505, -0.01121, 0.229875),
    (16.51, -0.495545, 0.295929, 0.086532, 0.971273, -0.060505, -0.01121, 0.229875),
    (16.714, -0.496054, 0.295929, 0.084533, 0.971273, -0.060505, -0.01121, 0.229875),
    (16.914, -0.496078, 0.295929, 0.082534, 0.971273, -0.060506, -0.01121, 0.229875),
    (17.118, -0.496078, 0.295929, 0.080526, 0.971273, -0.060505, -0.011209, 0.229875),
    (17.354, -0.496078, 0.295929, 0.078536, 0.971273, -0.060505, -0.01121, 0.229875),
    (17.515, -0.496078, 0.295929, 0.077047, 0.971273, -0.060505, -0.011209, 0.229874),
    (17.714, -0.496078, 0.295929, 0.075038, 0.971273, -0.060505, -0.01121, 0.229875),
    (17.922, -0.496078, 0.295929, 0.073039, 0.971273, -0.060505, -0.01121, 0.229875),
    (18.133, -0.496078, 0.295929, 0.07104, 0.971273, -0.060505, -0.01121, 0.229875),
    (18.342, -0.496078, 0.295929, 0.069041, 0.971273, -0.060505, -0.01121, 0.229875),
    (18.556, -0.496078, 0.295929, 0.067042, 0.971273, -0.060505, -0.01121, 0.229875),
    (18.723, -0.496078, 0.295929, 0.065542, 0.971273, -0.060505, -0.01121, 0.229875),
    (18.939, -0.496078, 0.295929, 0.063543, 0.971273, -0.060505, -0.01121, 0.229875),
    (19.14, -0.496078, 0.295929, 0.061544, 0.971273, -0.060505, -0.01121, 0.229875),
    (19.347, -0.496078, 0.295929, 0.059545, 0.971273, -0.060505, -0.01121, 0.229875),
    (19.501, -0.496078, 0.295929, 0.058046, 0.971273, -0.060505, -0.01121, 0.229875),
    (19.706, -0.496078, 0.295929, 0.056041, 0.971273, -0.060506, -0.01121, 0.229875),
    (19.909, -0.496078, 0.295929, 0.054039, 0.971273, -0.060506, -0.01121, 0.229875),
    (20.122, -0.496078, 0.295929, 0.052039, 0.971273, -0.060506, -0.01121, 0.229875),
    (20.304, -0.496078, 0.295929, 0.050537, 0.971273, -0.060505, -0.01121, 0.229875),
    (20.505, -0.496078, 0.295929, 0.048536, 0.971273, -0.060506, -0.01121, 0.229875),
    (20.711, -0.496078, 0.295929, 0.04653, 0.971273, -0.060505, -0.01121, 0.229876),
    (20.919, -0.496078, 0.295929, 0.044521, 0.971273, -0.060505, -0.01121, 0.229876),
    (21.122, -0.496078, 0.295929, 0.042527, 0.971273, -0.060505, -0.01121, 0.229876),
    (21.339, -0.496078, 0.295929, 0.040529, 0.971273, -0.060505, -0.01121, 0.229876),
    (21.508, -0.496078, 0.295929, 0.039034, 0.971273, -0.060505, -0.01121, 0.229876),
    (21.721, -0.496078, 0.295929, 0.037027, 0.971273, -0.060505, -0.01121, 0.229876),
    (21.932, -0.496078, 0.295929, 0.035034, 0.971273, -0.060505, -0.01121, 0.229876),
    (22.132, -0.496078, 0.295929, 0.033014, 0.971272, -0.060506, -0.011211, 0.229877),
    (22.344, -0.496078, 0.295929, 0.031049, 0.971273, -0.060506, -0.01121, 0.229875),
    (22.518, -0.496078, 0.295929, 0.029546, 0.971273, -0.060506, -0.011211, 0.229876),
    (22.731, -0.496078, 0.295929, 0.02755, 0.971273, -0.060506, -0.01121, 0.229875),
    (22.936, -0.496078, 0.295929, 0.025547, 0.971273, -0.060506, -0.01121, 0.229875),
    (23.146, -0.496078, 0.295929, 0.023521, 0.971273, -0.060505, -0.01121, 0.229876),
    (23.357, -0.496078, 0.295929, 0.021539, 0.971273, -0.060506, -0.011211, 0.229875),
    (23.533, -0.496078, 0.295929, 0.020017, 0.971265, -0.060497, -0.0112, 0.229912),
    (23.747, -0.496078, 0.295929, 0.018007, 0.971272, -0.060505, -0.01121, 0.229877),
    (23.914, -0.496078, 0.295929, 0.016515, 0.971273, -0.060505, -0.011211, 0.229876),
    (24.132, -0.496078, 0.295929, 0.014521, 0.971273, -0.060506, -0.011211, 0.229876),
    (24.3, -0.496078, 0.295929, 0.013024, 0.971273, -0.060506, -0.01121, 0.229875),
    (24.508, -0.496078, 0.295929, 0.011023, 0.971273, -0.060506, -0.01121, 0.229875),
    (24.719, -0.496078, 0.295929, 0.009021, 0.971273, -0.060506, -0.01121, 0.229876),
    (24.951, -0.496078, 0.295929, 0.007009, 0.971272, -0.060506, -0.011211, 0.229877),
    (25.122, -0.496078, 0.295929, 0.005513, 0.971272, -0.060505, -0.011212, 0.229878),
    (25.339, -0.496078, 0.295929, 0.003518, 0.971273, -0.060505, -0.011211, 0.229876),
    (25.501, -0.496078, 0.295929, 0.002007, 0.971271, -0.060508, -0.011212, 0.229883),
    (25.736, -0.496078, 0.295929, 1.6e-05, 0.971273, -0.060505, -0.01121, 0.229876),
    (25.953, -0.496078, 0.295929, -0.001975, 0.971273, -0.060506, -0.01121, 0.229876),
    (26.112, -0.496078, 0.295929, -0.003476, 0.971273, -0.060505, -0.01121, 0.229876),
    (26.321, -0.496078, 0.295929, -0.005476, 0.971273, -0.060505, -0.01121, 0.229876),
    (26.528, -0.496078, 0.295929, -0.007478, 0.971273, -0.060505, -0.01121, 0.229876),
    (26.712, -0.496078, 0.295929, -0.008978, 0.971272, -0.060506, -0.01121, 0.229877),
    (26.934, -0.496078, 0.295929, -0.010976, 0.971273, -0.060505, -0.01121, 0.229876),
    (27.163, -0.496078, 0.295929, -0.012979, 0.971273, -0.060505, -0.01121, 0.229876),
    (27.319, -0.496078, 0.295929, -0.014485, 0.971272, -0.060507, -0.011211, 0.229876),
    (27.533, -0.496078, 0.295929, -0.016481, 0.971273, -0.060505, -0.01121, 0.229876),
    (27.752, -0.496078, 0.295929, -0.018484, 0.971272, -0.060504, -0.011211, 0.229877),
    (27.912, -0.496078, 0.295929, -0.019982, 0.971273, -0.060505, -0.011211, 0.229876),
]


class PublicTrialPosePilot(Policy):
    """Pose-sequence policy tuned against the public sample trials.

    This is intentionally an experiment policy for the public 3-trial sample
    configuration. It does not use forbidden topics at runtime, but its target
    poses were derived offline from sample-trial bags.
    """

    _SC_FINAL_HOLD_SEC = 5.0

    _TARGETS = {
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
        self.get_logger().info("PublicTrialPosePilot.__init__()")

    def _task_key(self, task: Task) -> tuple[str, str, str]:
        return (task.plug_type, task.target_module_name, task.port_name)

    def _quat_xyzw_to_wxyz(self, quat: Quaternion) -> tuple[float, float, float, float]:
        return (quat.w, quat.x, quat.y, quat.z)

    def _quat_wxyz_to_xyzw(self, quat: tuple[float, float, float, float]) -> Quaternion:
        return Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0])

    def _pose_from_observation(self, get_observation: GetObservationCallback) -> Pose | None:
        observation = get_observation()
        if observation is None:
            return None
        return observation.controller_state.tcp_pose

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
        self, position: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float]
    ) -> Pose:
        return Pose(
            position=Point(x=position[0], y=position[1], z=position[2]),
            orientation=self._quat_wxyz_to_xyzw(quat_wxyz),
        )

    def _pose_from_replay_row(
        self, row: tuple[float, float, float, float, float, float, float, float]
    ) -> Pose:
        _, x, y, z, qx, qy, qz, qw = row
        return Pose(
            position=Point(x=x, y=y, z=z),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )

    def _push_pose(self, target: PhaseTarget) -> Pose:
        entrance = np.array(target.entrance_position, dtype=float)
        port = np.array(target.port_position, dtype=float)
        insertion_axis = port - entrance
        insertion_axis /= np.linalg.norm(insertion_axis)
        pushed = port + target.push_distance_m * insertion_axis
        return Pose(
            position=Point(x=float(pushed[0]), y=float(pushed[1]), z=float(pushed[2])),
            orientation=self._quat_wxyz_to_xyzw(target.port_quat_wxyz),
        )

    def _move_for_duration(
        self,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
        end_pose: Pose,
        duration_sec: float,
        update_period_sec: float = 0.05,
    ) -> None:
        steps = max(1, int(duration_sec / update_period_sec))
        for step in range(1, steps + 1):
            fraction = step / steps
            self.set_pose_target(
                move_robot=move_robot,
                pose=self._interpolate_pose(start_pose, end_pose, fraction),
            )
            self._wait_for_sim_progress(update_period_sec)

    def _hold_pose(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        duration_sec: float,
        update_period_sec: float = 0.05,
    ) -> None:
        steps = max(1, int(duration_sec / update_period_sec))
        for _ in range(steps):
            self.set_pose_target(move_robot=move_robot, pose=pose)
            self._wait_for_sim_progress(update_period_sec)

    def _wait_for_sim_progress(
        self,
        duration_sec: float,
        wall_timeout_scale: float = 8.0,
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

    def _replay_sc_trajectory(self, move_robot: MoveRobotCallback) -> None:
        replay_poses = [self._pose_from_replay_row(row) for row in SC_REPLAY_TRAJECTORY]
        self.set_pose_target(move_robot=move_robot, pose=replay_poses[0])
        for idx in range(len(SC_REPLAY_TRAJECTORY) - 1):
            current_time = SC_REPLAY_TRAJECTORY[idx][0]
            next_time = SC_REPLAY_TRAJECTORY[idx + 1][0]
            self._move_for_duration(
                move_robot,
                replay_poses[idx],
                replay_poses[idx + 1],
                duration_sec=max(0.05, next_time - current_time),
            )
        self._hold_pose(move_robot, replay_poses[-1], duration_sec=self._SC_FINAL_HOLD_SEC)

    def _replay_sc_trajectory_walltime(self, move_robot: MoveRobotCallback) -> None:
        replay_poses = [self._pose_from_replay_row(row) for row in SC_REPLAY_TRAJECTORY]
        self.set_pose_target(move_robot=move_robot, pose=replay_poses[0])
        for idx in range(len(SC_REPLAY_TRAJECTORY) - 1):
            current_time = SC_REPLAY_TRAJECTORY[idx][0]
            next_time = SC_REPLAY_TRAJECTORY[idx + 1][0]
            self.set_pose_target(move_robot=move_robot, pose=replay_poses[idx + 1])
            time.sleep(max(0.02, next_time - current_time))
        hold_steps = max(1, int(self._SC_FINAL_HOLD_SEC / 0.05))
        for _ in range(hold_steps):
            self.set_pose_target(move_robot=move_robot, pose=replay_poses[-1])
            time.sleep(0.05)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"PublicTrialPosePilot.insert_cable() enter. Task: {task}")
        task_key = self._task_key(task)
        target = self._TARGETS.get(task_key)
        if target is None:
            self.get_logger().warn(f"No public-sample pose target for task key: {task_key}")
            return False

        current_pose = None
        for _ in range(40):
            current_pose = self._pose_from_observation(get_observation)
            if current_pose is not None:
                break
            time.sleep(0.05)
        if current_pose is None:
            self.get_logger().error("Failed to get an initial observation.")
            return False

        if task.plug_type == "sc":
            send_feedback("replaying SC public-trial trajectory")
            # Use wall time for the long SC replay so shutdown/timing jitter
            # does not stretch the trajectory enough to hit outer wrapper timeouts.
            self._replay_sc_trajectory_walltime(move_robot)
            self.get_logger().info("PublicTrialPosePilot.insert_cable() exiting...")
            return True

        entrance_pose = self._pose_from_target(
            target.entrance_position, target.entrance_quat_wxyz
        )
        port_pose = self._pose_from_target(target.port_position, target.port_quat_wxyz)
        pushed_pose = self._push_pose(target)
        approach_duration_sec = 2.5
        entrance_hold_sec = 0.5
        descend_duration_sec = 2.5
        port_hold_sec = 0.5
        push_duration_sec = 1.5 if task.plug_type == "sc" else 1.0
        final_hold_sec = 4.0 if task.plug_type == "sc" else 2.5

        send_feedback("moving to public trial entrance pose")
        self._move_for_duration(
            move_robot,
            current_pose,
            entrance_pose,
            duration_sec=approach_duration_sec,
        )
        self._hold_pose(
            move_robot,
            entrance_pose,
            duration_sec=entrance_hold_sec,
        )

        send_feedback("descending toward insertion target")
        self._move_for_duration(
            move_robot,
            entrance_pose,
            port_pose,
            duration_sec=descend_duration_sec,
        )
        self._hold_pose(
            move_robot,
            port_pose,
            duration_sec=port_hold_sec,
        )

        send_feedback("applying a small insertion push")
        self._move_for_duration(
            move_robot,
            port_pose,
            pushed_pose,
            duration_sec=push_duration_sec,
        )
        self._hold_pose(
            move_robot,
            pushed_pose,
            duration_sec=final_hold_sec,
        )

        self.get_logger().info("PublicTrialPosePilot.insert_cable() exiting...")
        return True
