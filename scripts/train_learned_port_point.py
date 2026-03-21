#!/usr/bin/env python3
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "aic_example_policies"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from aic_example_policies.ros.learned_port_pipeline import (
    PortPointDataset,
    TargetNormalizer,
    TaskVocabulary,
    build_center_uvz_target,
    build_model_from_vocabulary,
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_records(dataset_split_dir: Path, plug_type: str) -> list[dict]:
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


def make_dataloader(dataset: PortPointDataset, batch_size: int, shuffle: bool) -> DataLoader:
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
            "record": [sample["record"] for sample in samples],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )


def batch_target_mae(
    normalized_pred: torch.Tensor,
    normalized_target: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    pred = normalized_pred * std + mean
    target = normalized_target * std + mean
    return (pred - target).abs().mean(dim=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_split_dir = args.dataset_split_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(dataset_split_dir, args.plug_type)
    if len(records) < 8:
        raise RuntimeError(
            f"need at least 8 {args.plug_type} samples for fail-fast training, got {len(records)}"
        )

    train_records, val_records = split_records(records, args.val_fraction, args.seed)
    vocabulary = TaskVocabulary.build(train_records)
    train_targets = np.stack(
        [build_center_uvz_target(record) for record in train_records], axis=0
    )
    normalizer = TargetNormalizer.fit(train_targets)

    train_dataset = PortPointDataset(
        dataset_root=dataset_split_dir,
        vocabulary=vocabulary,
        normalizer=normalizer,
        image_size=args.image_size,
        records=train_records,
    )
    val_dataset = PortPointDataset(
        dataset_root=dataset_split_dir,
        vocabulary=vocabulary,
        normalizer=normalizer,
        image_size=args.image_size,
        records=val_records,
    )
    train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_dataset, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_vocabulary(vocabulary).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    mean = torch.tensor(normalizer.mean, device=device, dtype=torch.float32)
    std = torch.tensor(normalizer.std, device=device, dtype=torch.float32)

    best_val_loss = float("inf")
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
                )
                target = batch["target"].to(device)
                loss = criterion(prediction, target)
                val_losses.append(float(loss.detach().cpu()))
                val_maes.append(
                    batch_target_mae(prediction, target, mean, std).cpu().numpy()
                )

        train_mae = np.mean(np.stack(train_maes, axis=0), axis=0)
        val_mae = np.mean(np.stack(val_maes, axis=0), axis=0) if val_maes else np.zeros(3)
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae_u_px": float(train_mae[0]),
            "train_mae_v_px": float(train_mae[1]),
            "train_mae_depth_m": float(train_mae[2]),
            "val_mae_u_px": float(val_mae[0]),
            "val_mae_v_px": float(val_mae[1]),
            "val_mae_depth_m": float(val_mae[2]),
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, sort_keys=True))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    manifest = {
        "dataset_split_dir": str(dataset_split_dir),
        "plug_type_filter": args.plug_type,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "record_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "all": len(records),
        },
        "vocabulary": vocabulary.to_dict(),
        "target_normalizer": normalizer.to_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
