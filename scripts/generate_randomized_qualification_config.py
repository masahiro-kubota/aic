#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

from qualification_randomization import (
    dump_yaml,
    ensure_entity_pose,
    load_yaml,
    randomize_entity_translation,
    resolve_path,
    set_false_rails,
    uniform,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = REPO_ROOT / "aic_engine/config/qualification_randomization_public_v1.yaml"


def _range_from_profile(node: dict[str, Any], key: str) -> tuple[float, float]:
    value = node[key]
    return (float(value["min"]), float(value["max"]))


def _randomize_board_pose(
    rng: random.Random,
    pose: dict[str, Any],
    board_pose_profile: dict[str, Any],
) -> None:
    pose["x"] = uniform(rng, *_range_from_profile(board_pose_profile, "x"))
    pose["y"] = uniform(rng, *_range_from_profile(board_pose_profile, "y"))
    pose["yaw"] = uniform(rng, *_range_from_profile(board_pose_profile, "yaw"))


def _randomize_mount_translations(
    rng: random.Random,
    task_board: dict[str, Any],
    mount_limits: dict[str, Any],
) -> None:
    for mount_key in (
        "lc_mount_rail_0",
        "sfp_mount_rail_0",
        "sc_mount_rail_0",
        "lc_mount_rail_1",
        "sfp_mount_rail_1",
        "sc_mount_rail_1",
    ):
        mount = task_board.get(mount_key)
        if mount and mount.get("entity_present"):
            randomize_entity_translation(
                rng,
                mount,
                float(mount_limits["min_translation"]),
                float(mount_limits["max_translation"]),
            )


def _randomize_grasp_pose(
    rng: random.Random,
    cable_pose: dict[str, Any],
    position_noise_abs_max: float,
    rotation_noise_abs_max: float,
) -> None:
    gripper_offset = cable_pose["gripper_offset"]
    gripper_offset["x"] = float(gripper_offset["x"]) + uniform(
        rng, -position_noise_abs_max, position_noise_abs_max
    )
    gripper_offset["y"] = float(gripper_offset["y"]) + uniform(
        rng, -position_noise_abs_max, position_noise_abs_max
    )
    gripper_offset["z"] = float(gripper_offset["z"]) + uniform(
        rng, -position_noise_abs_max, position_noise_abs_max
    )
    cable_pose["roll"] = float(cable_pose["roll"]) + uniform(
        rng, -rotation_noise_abs_max, rotation_noise_abs_max
    )
    cable_pose["pitch"] = float(cable_pose["pitch"]) + uniform(
        rng, -rotation_noise_abs_max, rotation_noise_abs_max
    )
    cable_pose["yaw"] = float(cable_pose["yaw"]) + uniform(
        rng, -rotation_noise_abs_max, rotation_noise_abs_max
    )


def _load_profile(profile_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_profile = profile_path.resolve()
    profile = load_yaml(resolved_profile)
    template_setting = profile.get("template")
    if not isinstance(template_setting, str) or not template_setting:
        raise ValueError("Profile must define a non-empty 'template' path")
    template_config_path = Path(template_setting)
    if template_config_path.is_absolute():
        template_path = template_config_path
    else:
        repo_candidate = (REPO_ROOT / template_config_path).resolve()
        profile_candidate = resolve_path(
            template_config_path,
            resolved_profile.parent,
        )
        template_path = repo_candidate if repo_candidate.exists() else profile_candidate
    template = load_yaml(template_path)
    return profile, template


def _build_sfp_trial(
    rng: random.Random,
    trial_id: str,
    base_trial: dict[str, Any],
    limits: dict[str, Any],
    profile: dict[str, Any],
    grasp_noise: dict[str, Any],
) -> dict[str, Any]:
    trial = copy.deepcopy(base_trial)
    task_board = trial["scene"]["task_board"]
    task = trial["tasks"]["task_1"]

    _randomize_board_pose(rng, task_board["pose"], profile["board_pose"])
    _randomize_mount_translations(rng, task_board, limits["mount_rail"])

    nic_limits = limits["nic_rail"]
    sc_limits = limits["sc_rail"]
    nic_yaw_abs_max = float(profile["nic_yaw_abs_max"])

    set_false_rails(task_board, "nic_rail", 5)
    target_rail = rng.randrange(5)
    remaining_rails = [rail for rail in range(5) if rail != target_rail]
    extra_nics_max = min(int(profile["extra_nics_max"]), len(remaining_rails))
    extra_nics = []
    if extra_nics_max > 0:
        extra_nic_count = rng.randint(0, extra_nics_max)
        extra_nics = rng.sample(remaining_rails, extra_nic_count)
    occupied_nic_rails = [target_rail, *sorted(extra_nics)]

    for rail_idx in occupied_nic_rails:
        rail_key = f"nic_rail_{rail_idx}"
        rail = task_board[rail_key]
        rail["entity_present"] = True
        rail["entity_name"] = f"nic_card_{rail_idx}"
        entity_pose = ensure_entity_pose(rail)
        entity_pose["translation"] = uniform(
            rng,
            float(nic_limits["min_translation"]),
            float(nic_limits["max_translation"]),
        )
        entity_pose["roll"] = 0.0
        entity_pose["pitch"] = 0.0
        entity_pose["yaw"] = uniform(rng, -nic_yaw_abs_max, nic_yaw_abs_max)

    for sc_key in ("sc_rail_0", "sc_rail_1"):
        sc_rail = task_board.get(sc_key)
        if sc_rail and sc_rail.get("entity_present"):
            randomize_entity_translation(
                rng,
                sc_rail,
                float(sc_limits["min_translation"]),
                float(sc_limits["max_translation"]),
            )
            entity_pose = ensure_entity_pose(sc_rail)
            entity_pose["roll"] = 0.0
            entity_pose["pitch"] = 0.0
            entity_pose["yaw"] = 0.0

    target_port = rng.choice(list(profile["target_ports"]))
    task["target_module_name"] = f"nic_card_mount_{target_rail}"
    task["port_name"] = target_port

    cable_name = task["cable_name"]
    _randomize_grasp_pose(
        rng,
        trial["scene"]["cables"][cable_name]["pose"],
        float(grasp_noise["position_abs_max"]),
        float(grasp_noise["rotation_abs_max"]),
    )

    trial["metadata"] = {
        "generated_by": "scripts/generate_randomized_qualification_config.py",
        "trial_id": trial_id,
        "trial_kind": "sfp",
        "target_nic_rail": target_rail,
        "target_port_name": target_port,
        "occupied_nic_rails": occupied_nic_rails,
    }
    return trial


def _build_sc_trial(
    rng: random.Random,
    trial_id: str,
    base_trial: dict[str, Any],
    limits: dict[str, Any],
    profile: dict[str, Any],
    grasp_noise: dict[str, Any],
) -> dict[str, Any]:
    trial = copy.deepcopy(base_trial)
    task_board = trial["scene"]["task_board"]
    task = trial["tasks"]["task_1"]

    _randomize_board_pose(rng, task_board["pose"], profile["board_pose"])
    _randomize_mount_translations(rng, task_board, limits["mount_rail"])

    set_false_rails(task_board, "nic_rail", 5)
    set_false_rails(task_board, "sc_rail", 2)

    target_port = rng.choice(list(profile["target_ports"]))
    target_idx = int(target_port.rsplit("_", maxsplit=1)[-1])
    other_idx = 1 - target_idx
    other_present_probability = float(profile["other_sc_port_present_probability"])
    other_present = rng.random() < other_present_probability
    present_sc_rails = [target_idx]
    if other_present:
        present_sc_rails.append(other_idx)
    present_sc_rails.sort()

    sc_limits = limits["sc_rail"]
    for rail_idx in present_sc_rails:
        rail_key = f"sc_rail_{rail_idx}"
        rail = task_board[rail_key]
        rail["entity_present"] = True
        rail["entity_name"] = f"sc_port_{rail_idx}"
        entity_pose = ensure_entity_pose(rail)
        entity_pose["translation"] = uniform(
            rng,
            float(sc_limits["min_translation"]),
            float(sc_limits["max_translation"]),
        )
        entity_pose["roll"] = 0.0
        entity_pose["pitch"] = 0.0
        entity_pose["yaw"] = 0.0

    task["target_module_name"] = target_port
    task["port_name"] = "sc_port_base"
    task["cable_name"] = "cable_1"
    task["plug_type"] = "sc"
    task["plug_name"] = "sc_tip"
    task["port_type"] = "sc"

    cable_name = task["cable_name"]
    _randomize_grasp_pose(
        rng,
        trial["scene"]["cables"][cable_name]["pose"],
        float(grasp_noise["position_abs_max"]),
        float(grasp_noise["rotation_abs_max"]),
    )

    trial["metadata"] = {
        "generated_by": "scripts/generate_randomized_qualification_config.py",
        "trial_id": trial_id,
        "trial_kind": "sc",
        "target_sc_port_name": target_port,
        "present_sc_rails": present_sc_rails,
    }
    return trial


def build_randomized_config(
    seed: int,
    profile_path: Path = DEFAULT_PROFILE,
) -> dict[str, Any]:
    profile, template = _load_profile(profile_path)

    required_trials = ("trial_1", "trial_2", "trial_3")
    missing_trials = [
        trial_id for trial_id in required_trials if trial_id not in template["trials"]
    ]
    if missing_trials:
        raise ValueError(f"Template missing required trials: {', '.join(missing_trials)}")

    if not profile.get("mount_randomization", {}).get("translation_only", False):
        raise ValueError(
            "Only translation-only mount randomization is currently supported"
        )

    rng = random.Random(seed)
    limits = copy.deepcopy(template["task_board_limits"])
    grasp_noise = profile["grasp_noise"]
    output = {
        "metadata": {
            "generated_by": "scripts/generate_randomized_qualification_config.py",
            "seed": seed,
            "profile_name": profile["profile_name"],
            "template": profile["template"],
            "fidelity": "public-spec approximation",
        },
        "scoring": copy.deepcopy(template["scoring"]),
        "task_board_limits": limits,
        "trials": {},
        "robot": copy.deepcopy(template["robot"]),
    }

    output["trials"]["trial_1"] = _build_sfp_trial(
        rng=rng,
        trial_id="trial_1",
        base_trial=template["trials"]["trial_1"],
        limits=limits,
        profile=profile["trial_1_2"],
        grasp_noise=grasp_noise,
    )
    output["trials"]["trial_2"] = _build_sfp_trial(
        rng=rng,
        trial_id="trial_2",
        base_trial=template["trials"]["trial_2"],
        limits=limits,
        profile=profile["trial_1_2"],
        grasp_noise=grasp_noise,
    )
    output["trials"]["trial_3"] = _build_sc_trial(
        rng=rng,
        trial_id="trial_3",
        base_trial=template["trials"]["trial_3"],
        limits=limits,
        profile=profile["trial_3"],
        grasp_noise=grasp_noise,
    )
    return output


def render_randomized_config(seed: int, profile_path: Path = DEFAULT_PROFILE) -> str:
    return dump_yaml(build_randomized_config(seed=seed, profile_path=profile_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a seed-reproducible, public-spec qualification config that "
            "can be passed to scripts/run_qualification_stage.sh via "
            "AIC_ENGINE_CONFIG_FILE."
        )
    )
    parser.add_argument("output", type=Path, help="Output YAML path")
    parser.add_argument("--seed", type=int, required=True, help="Deterministic RNG seed")
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="Randomization profile YAML",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_randomized_config(seed=args.seed, profile_path=args.profile),
        encoding="utf-8",
    )
    print(f"Wrote randomized qualification config to {args.output}")


if __name__ == "__main__":
    main()
