#!/usr/bin/env python3
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "aic_example_policies"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from aic_example_policies.ros import learned_port_pipeline as learned_port_pipeline_module


def build_aux_target_local(record: dict[str, Any]) -> np.ndarray:
    labels = record["labels"]
    extra = record.get("extra", {})
    current_pose = labels["current_tcp_pose"]
    position = current_pose["position"]
    orientation = current_pose["orientation"]
    feature_summary = extra.get("feature_summary", {})
    image_shape = list(feature_summary.get("image_shape", [1152, 1024]))
    image_width = max(float(image_shape[0]), 1.0)
    image_height = max(float(image_shape[1]), 1.0)
    center_camera = feature_summary.get("center_camera", {})
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
    step_index = int(extra.get("teacher_step_index", 1))
    step_fraction = float(max(step_index - 1, 0)) / 7.0
    return np.array(
        [
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
            float(orientation["x"]),
            float(orientation["y"]),
            float(orientation["z"]),
            float(orientation["w"]),
            *feature_values.tolist(),
            step_fraction,
        ],
        dtype=np.float32,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_split_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--plug-type", default="sc")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--target-kind",
        default="center_uvz",
        choices=("center_uvz", "teacher_insert_delta6", "teacher_step_delta6"),
    )
    parser.add_argument(
        "--phase-prefixes",
        nargs="*",
        default=None,
        help="Optional record phase prefixes to keep, e.g. teacher_insert teacher_insert_sweep",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_records(
    dataset_split_dir: Path,
    plug_type: str,
    target_kind: str,
    phase_prefixes: list[str] | None,
) -> list[dict]:
    index_path = dataset_split_dir / "samples.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    records = []
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("task", {}).get("plug_type") != plug_type:
            continue
        if phase_prefixes:
            phase = str(record.get("phase", ""))
            if not any(phase.startswith(prefix) for prefix in phase_prefixes):
                continue
        try:
            learned_port_pipeline_module.build_target(record, target_kind)
        except Exception:
            continue
        records.append(record)
    return records


def split_records(
    records: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, []
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    if not train_records:
        train_records, val_records = shuffled[:-1], shuffled[-1:]
    return train_records, val_records


def make_dataloader(dataset: Any, batch_size: int, shuffle: bool) -> DataLoader:
    def collate_fn(samples: list[dict]) -> dict[str, object]:
        return {
            "left": torch.stack([sample["left"] for sample in samples], dim=0),
            "center": torch.stack([sample["center"] for sample in samples], dim=0),
            "right": torch.stack([sample["right"] for sample in samples], dim=0),
            "task_indices": {
                "plug_type": torch.stack(
                    [sample["task_indices"]["plug_type"] for sample in samples], dim=0
                ),
                "target_module_name": torch.stack(
                    [sample["task_indices"]["target_module_name"] for sample in samples],
                    dim=0,
                ),
                "port_name": torch.stack(
                    [sample["task_indices"]["port_name"] for sample in samples], dim=0
                ),
            },
            "target": torch.stack([sample["target"] for sample in samples], dim=0),
            "raw_target": torch.stack(
                [sample["raw_target"] for sample in samples], dim=0
            ),
            "aux": torch.stack([sample["aux"] for sample in samples], dim=0),
            "raw_aux": torch.stack([sample["raw_aux"] for sample in samples], dim=0),
            "record": [sample["record"] for sample in samples],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )


def summarize_template(records: list[dict]) -> dict[str, object] | None:
    if not records:
        return None
    targets = np.stack(
        [learned_port_pipeline_module.build_center_uvz_target(record) for record in records],
        axis=0,
    )
    median = np.median(targets, axis=0)
    return {
        "center_uvz_median": [float(value) for value in median],
        "count": len(records),
    }


def build_phase_templates(records: list[dict]) -> dict[str, dict[str, object]]:
    task_groups: dict[str, list[dict]] = {}
    for record in records:
        task = record.get("task", {})
        task_key = "|".join(
            [
                str(task.get("plug_type", "")),
                str(task.get("target_module_name", "")),
                str(task.get("port_name", "")),
            ]
        )
        task_groups.setdefault(task_key, []).append(record)

    phase_templates: dict[str, dict[str, object]] = {}
    for task_key, task_records in task_groups.items():
        hover_records = [
            record for record in task_records if record.get("phase") == "teacher_hover"
        ]
        insert_records = [
            record for record in task_records if record.get("phase") == "teacher_insert"
        ]
        initial_records = [
            record for record in task_records if record.get("phase") == "initial"
        ]
        insert_sorted = sorted(
            insert_records,
            key=lambda record: float(
                learned_port_pipeline_module.build_center_uvz_target(record)[2]
            ),
        )
        near_count = max(1, len(insert_sorted) // 3) if insert_sorted else 0
        mid_start = len(insert_sorted) // 3
        mid_end = max(mid_start + 1, (2 * len(insert_sorted)) // 3) if insert_sorted else 0
        templates = {
            "initial": summarize_template(initial_records),
            "hover": summarize_template(hover_records),
            "insert_all": summarize_template(insert_records),
            "insert_mid": summarize_template(insert_sorted[mid_start:mid_end]),
            "insert_near": summarize_template(insert_sorted[:near_count]),
        }
        phase_templates[task_key] = {
            key: value for key, value in templates.items() if value is not None
        }
    return phase_templates


def batch_target_mae(
    normalized_pred: torch.Tensor,
    normalized_target: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    pred = normalized_pred * std + mean
    target = normalized_target * std + mean
    return (pred - target).abs().mean(dim=0)


def target_metric_names(target_kind: str) -> list[str]:
    if target_kind == "center_uvz":
        return ["u_px", "v_px", "depth_m"]
    if target_kind in ("teacher_insert_delta6", "teacher_step_delta6"):
        return [
            "dx_m",
            "dy_m",
            "dz_m",
            "rotvec_x_rad",
            "rotvec_y_rad",
            "rotvec_z_rad",
        ]
    raise ValueError(f"unsupported target_kind: {target_kind}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_split_dir = args.dataset_split_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(
        dataset_split_dir,
        args.plug_type,
        args.target_kind,
        args.phase_prefixes,
    )
    if len(records) < 8:
        raise RuntimeError(
            f"need at least 8 {args.plug_type} samples for fail-fast training, got {len(records)}"
        )

    train_records, val_records = split_records(records, args.val_fraction, args.seed)
    vocabulary = learned_port_pipeline_module.TaskVocabulary.build(train_records)
    train_targets = np.stack(
        [
            learned_port_pipeline_module.build_target(record, args.target_kind)
            for record in train_records
        ],
        axis=0,
    )
    normalizer = learned_port_pipeline_module.TargetNormalizer.fit(train_targets)
    train_aux = np.stack([build_aux_target_local(record) for record in train_records], axis=0)
    aux_normalizer = learned_port_pipeline_module.TargetNormalizer.fit(train_aux)

    train_dataset = cast(Any, learned_port_pipeline_module.PortPointDataset)(
        dataset_root=dataset_split_dir,
        vocabulary=vocabulary,
        normalizer=normalizer,
        aux_normalizer=aux_normalizer,
        image_size=args.image_size,
        records=train_records,
        target_kind=args.target_kind,
    )
    val_dataset = cast(Any, learned_port_pipeline_module.PortPointDataset)(
        dataset_root=dataset_split_dir,
        vocabulary=vocabulary,
        normalizer=normalizer,
        aux_normalizer=aux_normalizer,
        image_size=args.image_size,
        records=val_records,
        target_kind=args.target_kind,
    )
    train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_dataset, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_names = target_metric_names(args.target_kind)
    model = cast(Any, learned_port_pipeline_module.build_model_from_vocabulary)(
        vocabulary,
        output_dim=len(metric_names),
        aux_dim=int(train_aux.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    mean = torch.tensor(normalizer.mean, device=device, dtype=torch.float32)
    std = torch.tensor(normalizer.std, device=device, dtype=torch.float32)

    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_maes = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                left=batch["left"].to(device),
                center=batch["center"].to(device),
                right=batch["right"].to(device),
                plug_type=batch["task_indices"]["plug_type"].to(device),
                target_module_name=batch["task_indices"]["target_module_name"].to(device),
                port_name=batch["task_indices"]["port_name"].to(device),
                aux=batch["aux"].to(device),
            )
            target = batch["target"].to(device)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_maes.append(
                batch_target_mae(prediction.detach(), target, mean, std)
                .cpu()
                .numpy()
            )

        model.eval()
        val_losses = []
        val_maes = []
        with torch.no_grad():
            for batch in val_loader:
                prediction = model(
                    left=batch["left"].to(device),
                    center=batch["center"].to(device),
                    right=batch["right"].to(device),
                    plug_type=batch["task_indices"]["plug_type"].to(device),
                    target_module_name=batch["task_indices"]["target_module_name"].to(device),
                    port_name=batch["task_indices"]["port_name"].to(device),
                    aux=batch["aux"].to(device),
                )
                target = batch["target"].to(device)
                loss = criterion(prediction, target)
                val_losses.append(float(loss.detach().cpu()))
                val_maes.append(
                    batch_target_mae(prediction, target, mean, std).cpu().numpy()
                )

        train_mae = np.mean(np.stack(train_maes, axis=0), axis=0)
        val_mae = (
            np.mean(np.stack(val_maes, axis=0), axis=0)
            if val_maes
            else np.zeros(len(metric_names))
        )
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        for index, metric_name in enumerate(metric_names):
            epoch_summary[f"train_mae_{metric_name}"] = float(train_mae[index])
            epoch_summary[f"val_mae_{metric_name}"] = float(val_mae[index])
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, sort_keys=True))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    manifest = {
        "dataset_split_dir": str(dataset_split_dir),
        "plug_type_filter": args.plug_type,
        "target_kind": args.target_kind,
        "output_dim": len(metric_names),
        "aux_dim": int(train_aux.shape[1]),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "phase_prefixes": args.phase_prefixes or [],
        "record_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "all": len(records),
        },
        "phase_templates": (
            build_phase_templates(records) if args.target_kind == "center_uvz" else {}
        ),
        "vocabulary": vocabulary.to_dict(),
        "target_normalizer": normalizer.to_dict(),
        "aux_normalizer": aux_normalizer.to_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
