#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
import unittest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_randomized_qualification_config import (
    DEFAULT_PROFILE,
    build_randomized_config,
    render_randomized_config,
)
from qualification_randomization import load_yaml


class GenerateRandomizedQualificationConfigTest(unittest.TestCase):
    def test_same_seed_is_byte_identical(self) -> None:
        first = render_randomized_config(seed=91, profile_path=DEFAULT_PROFILE)
        second = render_randomized_config(seed=91, profile_path=DEFAULT_PROFILE)
        self.assertEqual(first, second)

    def test_different_seed_changes_output(self) -> None:
        first = render_randomized_config(seed=91, profile_path=DEFAULT_PROFILE)
        second = render_randomized_config(seed=92, profile_path=DEFAULT_PROFILE)
        self.assertNotEqual(first, second)

    def test_generated_trials_match_public_profile_constraints(self) -> None:
        config = build_randomized_config(seed=7, profile_path=DEFAULT_PROFILE)
        profile = load_yaml(DEFAULT_PROFILE)
        limits = config["task_board_limits"]
        sfp_trials = ("trial_1", "trial_2")
        for trial_id in sfp_trials:
            trial = config["trials"][trial_id]
            task_board = trial["scene"]["task_board"]
            task = trial["tasks"]["task_1"]
            pose = task_board["pose"]
            target_rail = int(task["target_module_name"].rsplit("_", maxsplit=1)[-1])
            self.assertEqual(trial["metadata"]["trial_kind"], "sfp")
            self.assertTrue(task_board[f"nic_rail_{target_rail}"]["entity_present"])
            self.assertIn(task["port_name"], ("sfp_port_0", "sfp_port_1"))
            self.assertGreaterEqual(
                pose["x"], profile["trial_1_2"]["board_pose"]["x"]["min"]
            )
            self.assertLessEqual(pose["x"], profile["trial_1_2"]["board_pose"]["x"]["max"])
            self.assertGreaterEqual(
                pose["y"], profile["trial_1_2"]["board_pose"]["y"]["min"]
            )
            self.assertLessEqual(pose["y"], profile["trial_1_2"]["board_pose"]["y"]["max"])
            self.assertGreaterEqual(
                pose["yaw"], profile["trial_1_2"]["board_pose"]["yaw"]["min"]
            )
            self.assertLessEqual(
                pose["yaw"], profile["trial_1_2"]["board_pose"]["yaw"]["max"]
            )
            occupied_rails = trial["metadata"]["occupied_nic_rails"]
            self.assertIn(target_rail, occupied_rails)
            for rail_idx in occupied_rails:
                rail_pose = task_board[f"nic_rail_{rail_idx}"]["entity_pose"]
                self.assertGreaterEqual(
                    rail_pose["translation"], limits["nic_rail"]["min_translation"]
                )
                self.assertLessEqual(
                    rail_pose["translation"], limits["nic_rail"]["max_translation"]
                )
                self.assertGreaterEqual(rail_pose["yaw"], -0.17453292519943295)
                self.assertLessEqual(rail_pose["yaw"], 0.17453292519943295)

        sc_trial = config["trials"]["trial_3"]
        sc_task_board = sc_trial["scene"]["task_board"]
        sc_task = sc_trial["tasks"]["task_1"]
        sc_pose = sc_task_board["pose"]
        target_port = sc_task["target_module_name"]
        self.assertEqual(sc_trial["metadata"]["trial_kind"], "sc")
        self.assertIn(target_port, ("sc_port_0", "sc_port_1"))
        target_idx = int(target_port.rsplit("_", maxsplit=1)[-1])
        self.assertTrue(sc_task_board[f"sc_rail_{target_idx}"]["entity_present"])
        self.assertEqual(sc_task["port_name"], "sc_port_base")
        self.assertGreaterEqual(sc_pose["x"], profile["trial_3"]["board_pose"]["x"]["min"])
        self.assertLessEqual(sc_pose["x"], profile["trial_3"]["board_pose"]["x"]["max"])
        self.assertGreaterEqual(sc_pose["y"], profile["trial_3"]["board_pose"]["y"]["min"])
        self.assertLessEqual(sc_pose["y"], profile["trial_3"]["board_pose"]["y"]["max"])
        self.assertGreaterEqual(
            sc_pose["yaw"], profile["trial_3"]["board_pose"]["yaw"]["min"]
        )
        self.assertLessEqual(
            sc_pose["yaw"], profile["trial_3"]["board_pose"]["yaw"]["max"]
        )
        for rail_idx in range(5):
            self.assertFalse(sc_task_board[f"nic_rail_{rail_idx}"]["entity_present"])
        for rail_idx in sc_trial["metadata"]["present_sc_rails"]:
            rail_pose = sc_task_board[f"sc_rail_{rail_idx}"]["entity_pose"]
            self.assertGreaterEqual(
                rail_pose["translation"], limits["sc_rail"]["min_translation"]
            )
            self.assertLessEqual(
                rail_pose["translation"], limits["sc_rail"]["max_translation"]
            )
            self.assertEqual(rail_pose["yaw"], 0.0)


if __name__ == "__main__":
    unittest.main()
