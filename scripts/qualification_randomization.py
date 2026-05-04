from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping YAML at {path}")
    return data


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False)


def uniform(rng: random.Random, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def ensure_entity_pose(rail_node: dict[str, Any]) -> dict[str, Any]:
    entity_pose = rail_node.setdefault(
        "entity_pose",
        {
            "translation": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },
    )
    entity_pose.setdefault("translation", 0.0)
    entity_pose.setdefault("roll", 0.0)
    entity_pose.setdefault("pitch", 0.0)
    entity_pose.setdefault("yaw", 0.0)
    return entity_pose


def set_false_rails(task_board: dict[str, Any], prefix: str, count: int) -> None:
    for idx in range(count):
        key = f"{prefix}_{idx}"
        if key in task_board:
            task_board[key]["entity_present"] = False


def randomize_entity_translation(
    rng: random.Random,
    rail_node: dict[str, Any],
    lo: float,
    hi: float,
) -> None:
    entity_pose = ensure_entity_pose(rail_node)
    entity_pose["translation"] = uniform(rng, lo, hi)


def resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
