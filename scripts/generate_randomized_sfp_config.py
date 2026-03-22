#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping YAML at {path}")
    return data


def _uniform(rng: random.Random, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _set_false_rails(task_board: dict[str, Any], prefix: str, count: int) -> None:
    for idx in range(count):
        key = f"{prefix}_{idx}"
        if key in task_board:
            task_board[key]["entity_present"] = False


def _randomize_entity_translation(
    rng: random.Random,
    rail_node: dict[str, Any],
    lo: float,
    hi: float,
) -> None:
    rail_node.setdefault(
        "entity_pose",
        {
            "translation": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },
    )
    rail_node["entity_pose"]["translation"] = _uniform(rng, lo, hi)


def _build_trial(
    rng: random.Random,
    base_trial: dict[str, Any],
    limits: dict[str, Any],
    trial_index: int,
    forced_target_rail: int | None,
    board_x_range: tuple[float, float],
    board_y_range: tuple[float, float],
    board_yaw_range: tuple[float, float],
    nic_yaw_abs_max: float,
    extra_nics_max: int,
    grasp_pos_noise: float,
    grasp_rot_noise: float,
) -> dict[str, Any]:
    trial = copy.deepcopy(base_trial)
    task_board = trial["scene"]["task_board"]
    task = trial["tasks"]["task_1"]
    cable_pose = trial["scene"]["cables"]["cable_0"]["pose"]

    task_board["pose"]["x"] = _uniform(rng, *board_x_range)
    task_board["pose"]["y"] = _uniform(rng, *board_y_range)
    task_board["pose"]["yaw"] = _uniform(rng, *board_yaw_range)

    nic_limits = limits["nic_rail"]
    sc_limits = limits["sc_rail"]
    mount_limits = limits["mount_rail"]

    _set_false_rails(task_board, "nic_rail", 5)
    target_rail = forced_target_rail if forced_target_rail is not None else rng.randrange(5)
    remaining_rails = [idx for idx in range(5) if idx != target_rail]
    extra_count = min(extra_nics_max, len(remaining_rails))
    extra_rails = (
        rng.sample(remaining_rails, rng.randint(0, extra_count)) if extra_count > 0 else []
    )
    occupied_rails = [target_rail, *sorted(extra_rails)]

    for rail_idx in occupied_rails:
        rail_name = f"nic_rail_{rail_idx}"
        rail = task_board[rail_name]
        rail["entity_present"] = True
        rail["entity_name"] = f"nic_card_{rail_idx}"
        rail.setdefault(
            "entity_pose",
            {
                "translation": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        )
        rail["entity_pose"]["translation"] = _uniform(
            rng,
            nic_limits["min_translation"],
            nic_limits["max_translation"],
        )
        rail["entity_pose"]["roll"] = 0.0
        rail["entity_pose"]["pitch"] = 0.0
        rail["entity_pose"]["yaw"] = _uniform(rng, -nic_yaw_abs_max, nic_yaw_abs_max)

    task["target_module_name"] = f"nic_card_mount_{target_rail}"

    for sc_key in ("sc_rail_0", "sc_rail_1"):
        if sc_key in task_board and task_board[sc_key]["entity_present"]:
            _randomize_entity_translation(
                rng,
                task_board[sc_key],
                sc_limits["min_translation"],
                sc_limits["max_translation"],
            )
            task_board[sc_key]["entity_pose"]["yaw"] = 0.0

    for mount_key in (
        "lc_mount_rail_0",
        "sfp_mount_rail_0",
        "sc_mount_rail_0",
        "lc_mount_rail_1",
        "sfp_mount_rail_1",
        "sc_mount_rail_1",
    ):
        if mount_key in task_board and task_board[mount_key]["entity_present"]:
            _randomize_entity_translation(
                rng,
                task_board[mount_key],
                mount_limits["min_translation"],
                mount_limits["max_translation"],
            )

    base_z = rng.choice((0.04245, 0.04545))
    cable_pose["gripper_offset"]["x"] = _uniform(rng, -grasp_pos_noise, grasp_pos_noise)
    cable_pose["gripper_offset"]["y"] = 0.015385 + _uniform(
        rng, -grasp_pos_noise, grasp_pos_noise
    )
    cable_pose["gripper_offset"]["z"] = base_z + _uniform(
        rng, -grasp_pos_noise, grasp_pos_noise
    )
    cable_pose["roll"] = 0.4432 + _uniform(rng, -grasp_rot_noise, grasp_rot_noise)
    cable_pose["pitch"] = -0.4838 + _uniform(rng, -grasp_rot_noise, grasp_rot_noise)
    cable_pose["yaw"] = 1.3303 + _uniform(rng, -grasp_rot_noise, grasp_rot_noise)

    trial["metadata"] = {
        "generated_by": "generate_randomized_sfp_config.py",
        "trial_index": trial_index,
        "target_rail": target_rail,
        "occupied_nic_rails": occupied_rails,
    }
    return trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an SFP-only engine config with qualification-style randomization "
            "for teacher data collection."
        )
    )
    parser.add_argument("output", type=Path, help="Output YAML path")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("aic_engine/config/sfp_trials_only.yaml"),
        help="Base SFP config to copy scoring and non-randomized scene structure from",
    )
    parser.add_argument("--num-trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--board-x-min", type=float, default=0.145)
    parser.add_argument("--board-x-max", type=float, default=0.175)
    parser.add_argument("--board-y-min", type=float, default=-0.20)
    parser.add_argument("--board-y-max", type=float, default=-0.02)
    parser.add_argument("--board-yaw-min", type=float, default=3.00)
    parser.add_argument("--board-yaw-max", type=float, default=3.15)
    parser.add_argument("--nic-yaw-abs-max", type=float, default=0.12)
    parser.add_argument("--extra-nics-max", type=int, default=2)
    parser.add_argument("--grasp-pos-noise", type=float, default=0.002)
    parser.add_argument("--grasp-rot-noise", type=float, default=0.04)
    parser.add_argument(
        "--target-rail-sequence",
        type=str,
        default="",
        help="Optional comma-separated target NIC rail sequence, e.g. 0,1,2,3,4",
    )
    args = parser.parse_args()

    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")

    rng = random.Random(args.seed)
    template = _load_yaml(args.template)
    limits = template["task_board_limits"]
    base_trial = template["trials"]["trial_1"]

    output = {
        "scoring": copy.deepcopy(template["scoring"]),
        "task_board_limits": copy.deepcopy(limits),
        "trials": {},
        "robot": copy.deepcopy(template["robot"]),
    }

    board_x_range = (args.board_x_min, args.board_x_max)
    board_y_range = (args.board_y_min, args.board_y_max)
    board_yaw_range = (args.board_yaw_min, args.board_yaw_max)
    target_rail_sequence = [
        int(token.strip())
        for token in args.target_rail_sequence.split(",")
        if token.strip()
    ]
    for rail in target_rail_sequence:
        if rail < 0 or rail > 4:
            raise ValueError("--target-rail-sequence entries must be in [0, 4]")

    for trial_index in range(1, args.num_trials + 1):
        forced_target_rail = None
        if target_rail_sequence:
            forced_target_rail = target_rail_sequence[
                (trial_index - 1) % len(target_rail_sequence)
            ]
        output["trials"][f"trial_{trial_index}"] = _build_trial(
            rng=rng,
            base_trial=base_trial,
            limits=limits,
            trial_index=trial_index,
            forced_target_rail=forced_target_rail,
            board_x_range=board_x_range,
            board_y_range=board_y_range,
            board_yaw_range=board_yaw_range,
            nic_yaw_abs_max=args.nic_yaw_abs_max,
            extra_nics_max=args.extra_nics_max,
            grasp_pos_noise=args.grasp_pos_noise,
            grasp_rot_noise=args.grasp_rot_noise,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, sort_keys=False)

    print(f"Wrote {args.num_trials} randomized SFP trials to {args.output}")


if __name__ == "__main__":
    main()
