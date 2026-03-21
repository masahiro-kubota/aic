from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


TASK_FIELD_NAMES = ("plug_type", "target_module_name", "port_name")


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = _as_path(path)
    if not resolved.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in resolved.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = _as_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _resize_bgr(image_bgr: np.ndarray, image_size: int) -> np.ndarray:
    resized = cv2.resize(
        image_bgr,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float32) / 255.0


def _encode_image(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    resized = _resize_bgr(image_bgr, image_size)
    chw = np.transpose(resized[:, :, ::-1], (2, 0, 1))
    return torch.from_numpy(chw.copy())


def _task_field_dict(task: Any) -> dict[str, str]:
    if isinstance(task, dict):
        return {name: str(task.get(name, "")) for name in TASK_FIELD_NAMES}
    return {name: str(getattr(task, name, "")) for name in TASK_FIELD_NAMES}


def build_center_uvz_target(record: dict[str, Any]) -> np.ndarray:
    labels = record["labels"]
    center = labels["per_camera"]["center"]
    u, v = center["uv"]
    _, _, z = center["point_optical"]
    return np.array([float(u), float(v), float(z)], dtype=np.float32)


def uvz_to_point_optical(uvz: np.ndarray, camera_info_k: list[float]) -> np.ndarray:
    fx = float(camera_info_k[0])
    fy = float(camera_info_k[4])
    cx = float(camera_info_k[2])
    cy = float(camera_info_k[5])
    u, v, depth = [float(value) for value in uvz]
    x = (u - cx) * depth / max(fx, 1e-6)
    y = (v - cy) * depth / max(fy, 1e-6)
    return np.array([x, y, depth], dtype=np.float32)


@dataclass
class TaskVocabulary:
    plug_type_to_idx: dict[str, int]
    target_module_name_to_idx: dict[str, int]
    port_name_to_idx: dict[str, int]

    @classmethod
    def build(cls, records: list[dict[str, Any]]) -> "TaskVocabulary":
        def make_index(field_name: str) -> dict[str, int]:
            values = sorted(
                {
                    str(record.get("task", {}).get(field_name, ""))
                    for record in records
                }
            )
            return {"<unk>": 0, **{value: idx + 1 for idx, value in enumerate(values)}}

        return cls(
            plug_type_to_idx=make_index("plug_type"),
            target_module_name_to_idx=make_index("target_module_name"),
            port_name_to_idx=make_index("port_name"),
        )

    def encode(self, task_fields: dict[str, str]) -> dict[str, int]:
        return {
            "plug_type": self.plug_type_to_idx.get(task_fields["plug_type"], 0),
            "target_module_name": self.target_module_name_to_idx.get(
                task_fields["target_module_name"], 0
            ),
            "port_name": self.port_name_to_idx.get(task_fields["port_name"], 0),
        }

    def to_dict(self) -> dict[str, dict[str, int]]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, dict[str, int]]) -> "TaskVocabulary":
        return cls(
            plug_type_to_idx={str(k): int(v) for k, v in payload["plug_type_to_idx"].items()},
            target_module_name_to_idx={
                str(k): int(v) for k, v in payload["target_module_name_to_idx"].items()
            },
            port_name_to_idx={str(k): int(v) for k, v in payload["port_name_to_idx"].items()},
        )


@dataclass
class TargetNormalizer:
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, values: np.ndarray) -> "TargetNormalizer":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(float).tolist(), std=std.astype(float).tolist())

    def normalize(self, values: np.ndarray) -> np.ndarray:
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        return (values - mean) / std

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        return values * std + mean

    def to_dict(self) -> dict[str, list[float]]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "TargetNormalizer":
        return cls(
            mean=[float(value) for value in payload["mean"]],
            std=[float(value) for value in payload["std"]],
        )


class GroundTruthPortDatasetWriter:
    def __init__(self, root: str | Path, split_name: str = "train") -> None:
        base = _as_path(root)
        self.root = base / split_name
        self.images_dir = self.root / "images"
        self.index_path = self.root / "samples.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def append_sample(
        self,
        task: Any,
        stage: str,
        phase: str,
        observation_stamp_sec: float,
        images_bgr: dict[str, np.ndarray],
        labels: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task_fields = _task_field_dict(task)
        sample_id = (
            f"{time.strftime('%Y%m%d_%H%M%S')}_{task_fields['plug_type']}_"
            f"{task_fields['target_module_name']}_{task_fields['port_name']}_"
            f"{int(observation_stamp_sec * 1000.0)}"
        )
        image_paths: dict[str, str] = {}
        for camera_name, image_bgr in images_bgr.items():
            rel_path = Path("images") / f"{sample_id}_{camera_name}.png"
            cv2.imwrite(str(self.root / rel_path), image_bgr)
            image_paths[camera_name] = rel_path.as_posix()

        record = {
            "sample_id": sample_id,
            "stage": stage,
            "phase": phase,
            "observation_stamp_sec": float(observation_stamp_sec),
            "task": task_fields,
            "paths": image_paths,
            "labels": labels,
            "extra": extra or {},
        }
        _append_jsonl(self.index_path, record)
        return record


class PortPointDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        vocabulary: TaskVocabulary,
        normalizer: TargetNormalizer,
        image_size: int = 224,
        records: list[dict[str, Any]] | None = None,
        plug_type_filter: str | None = None,
    ) -> None:
        self.root = _as_path(dataset_root)
        loaded_records = (
            list(records)
            if records is not None
            else _read_jsonl(self.root / "samples.jsonl")
        )
        if plug_type_filter is not None:
            loaded_records = [
                record
                for record in loaded_records
                if record.get("task", {}).get("plug_type") == plug_type_filter
            ]
        self.records = loaded_records
        self.vocabulary = vocabulary
        self.normalizer = normalizer
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        images = {}
        for camera_name in ("left", "center", "right"):
            path = self.root / record["paths"][camera_name]
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(path)
            images[camera_name] = _encode_image(image_bgr, self.image_size)

        task_fields = _task_field_dict(record["task"])
        task_indices = self.vocabulary.encode(task_fields)
        target_uvz = build_center_uvz_target(record)
        normalized_target = self.normalizer.normalize(target_uvz)
        return {
            "left": images["left"],
            "center": images["center"],
            "right": images["right"],
            "task_indices": {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in task_indices.items()
            },
            "target": torch.tensor(normalized_target, dtype=torch.float32),
            "raw_target": torch.tensor(target_uvz, dtype=torch.float32),
            "record": record,
        }


class _SharedImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoded = self.layers(image)
        return encoded.flatten(start_dim=1)


class MultiViewPortPointRegressor(nn.Module):
    def __init__(
        self,
        plug_vocab_size: int,
        target_module_vocab_size: int,
        port_name_vocab_size: int,
        embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = _SharedImageEncoder()
        self.plug_embedding = nn.Embedding(plug_vocab_size, embedding_dim)
        self.target_module_embedding = nn.Embedding(
            target_module_vocab_size, embedding_dim
        )
        self.port_name_embedding = nn.Embedding(port_name_vocab_size, embedding_dim)
        feature_dim = 3 * 128 + 3 * embedding_dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        left: torch.Tensor,
        center: torch.Tensor,
        right: torch.Tensor,
        plug_type: torch.Tensor,
        target_module_name: torch.Tensor,
        port_name: torch.Tensor,
    ) -> torch.Tensor:
        left_features = self.encoder(left)
        center_features = self.encoder(center)
        right_features = self.encoder(right)
        task_features = torch.cat(
            [
                self.plug_embedding(plug_type),
                self.target_module_embedding(target_module_name),
                self.port_name_embedding(port_name),
            ],
            dim=1,
        )
        return self.head(
            torch.cat(
                [left_features, center_features, right_features, task_features], dim=1
            )
        )


def build_model_from_vocabulary(vocabulary: TaskVocabulary) -> MultiViewPortPointRegressor:
    return MultiViewPortPointRegressor(
        plug_vocab_size=len(vocabulary.plug_type_to_idx),
        target_module_vocab_size=len(vocabulary.target_module_name_to_idx),
        port_name_vocab_size=len(vocabulary.port_name_to_idx),
    )


class LearnedPortInference:
    def __init__(
        self,
        model: MultiViewPortPointRegressor,
        vocabulary: TaskVocabulary,
        normalizer: TargetNormalizer,
        image_size: int,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.vocabulary = vocabulary
        self.normalizer = normalizer
        self.image_size = int(image_size)
        self.device = device

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        device: str | torch.device | None = None,
    ) -> "LearnedPortInference":
        artifact_root = _as_path(artifact_dir)
        manifest = json.loads((artifact_root / "manifest.json").read_text())
        vocabulary = TaskVocabulary.from_dict(manifest["vocabulary"])
        normalizer = TargetNormalizer.from_dict(manifest["target_normalizer"])
        resolved_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = build_model_from_vocabulary(vocabulary)
        state_dict = torch.load(
            artifact_root / "best_model.pt",
            map_location=resolved_device,
            weights_only=False,
        )
        model.load_state_dict(state_dict)
        model.to(resolved_device)
        return cls(
            model=model,
            vocabulary=vocabulary,
            normalizer=normalizer,
            image_size=int(manifest["image_size"]),
            device=resolved_device,
        )

    def predict_center_uvz(
        self,
        task: Any,
        images_bgr: dict[str, np.ndarray],
        center_camera_k: list[float],
    ) -> dict[str, Any]:
        task_fields = _task_field_dict(task)
        task_indices = self.vocabulary.encode(task_fields)
        with torch.no_grad():
            left = (
                _encode_image(images_bgr["left"], self.image_size)
                .unsqueeze(0)
                .to(self.device)
            )
            center = (
                _encode_image(images_bgr["center"], self.image_size)
                .unsqueeze(0)
                .to(self.device)
            )
            right = (
                _encode_image(images_bgr["right"], self.image_size)
                .unsqueeze(0)
                .to(self.device)
            )
            output = self.model(
                left=left,
                center=center,
                right=right,
                plug_type=torch.tensor(
                    [task_indices["plug_type"]], dtype=torch.long, device=self.device
                ),
                target_module_name=torch.tensor(
                    [task_indices["target_module_name"]],
                    dtype=torch.long,
                    device=self.device,
                ),
                port_name=torch.tensor(
                    [task_indices["port_name"]], dtype=torch.long, device=self.device
                ),
            )
        normalized = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
        uvz = self.normalizer.denormalize(normalized)
        point_optical = uvz_to_point_optical(uvz, center_camera_k)
        return {
            "task_fields": task_fields,
            "center_uvz": [float(value) for value in uvz],
            "point_center_optical": [float(value) for value in point_optical],
        }
