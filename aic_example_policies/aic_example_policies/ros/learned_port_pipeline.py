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


def _pose_dict_to_arrays(pose_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = pose_dict["position"]
    orientation = pose_dict["orientation"]
    return (
        np.array(
            [float(position["x"]), float(position["y"]), float(position["z"])],
            dtype=np.float32,
        ),
        np.array(
            [
                float(orientation["x"]),
                float(orientation["y"]),
                float(orientation["z"]),
                float(orientation["w"]),
            ],
            dtype=np.float32,
        ),
    )


def _pose_like_to_arrays(pose: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(pose, dict):
        return _pose_dict_to_arrays(pose)
    position = pose.position
    orientation = pose.orientation
    return (
        np.array(
            [float(position.x), float(position.y), float(position.z)],
            dtype=np.float32,
        ),
        np.array(
            [
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ],
            dtype=np.float32,
        ),
    )


def _quat_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(value) for value in quat_xyzw]
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-9:
        return np.eye(3, dtype=np.float32)
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    angle = float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float32,
    )
    axis /= max(2.0 * np.sin(angle), 1e-6)
    return axis * angle


def build_teacher_insert_delta_target(record: dict[str, Any]) -> np.ndarray:
    labels = record["labels"]
    teacher_pose = labels.get("teacher_insert_pose")
    current_pose = labels.get("current_tcp_pose")
    if teacher_pose is None or current_pose is None:
        raise KeyError("teacher_insert_pose/current_tcp_pose")
    teacher_position, teacher_quat = _pose_dict_to_arrays(teacher_pose)
    current_position, current_quat = _pose_dict_to_arrays(current_pose)
    translation_delta = teacher_position - current_position
    teacher_rotation = _quat_xyzw_to_rotation_matrix(teacher_quat)
    current_rotation = _quat_xyzw_to_rotation_matrix(current_quat)
    rotation_delta = teacher_rotation @ current_rotation.T
    rotvec = _rotation_matrix_to_rotvec(rotation_delta)
    return np.concatenate([translation_delta, rotvec], axis=0).astype(np.float32)


def build_teacher_step_delta_target(record: dict[str, Any]) -> np.ndarray:
    labels = record["labels"]
    teacher_pose = labels.get("teacher_step_pose")
    current_pose = labels.get("current_tcp_pose")
    if teacher_pose is None or current_pose is None:
        raise KeyError("teacher_step_pose/current_tcp_pose")
    teacher_position, teacher_quat = _pose_dict_to_arrays(teacher_pose)
    current_position, current_quat = _pose_dict_to_arrays(current_pose)
    translation_delta = teacher_position - current_position
    teacher_rotation = _quat_xyzw_to_rotation_matrix(teacher_quat)
    current_rotation = _quat_xyzw_to_rotation_matrix(current_quat)
    rotation_delta = teacher_rotation @ current_rotation.T
    rotvec = _rotation_matrix_to_rotvec(rotation_delta)
    return np.concatenate([translation_delta, rotvec], axis=0).astype(np.float32)


def build_target(record: dict[str, Any], target_kind: str) -> np.ndarray:
    if target_kind == "center_uvz":
        return build_center_uvz_target(record)
    if target_kind == "teacher_insert_delta6":
        return build_teacher_insert_delta_target(record)
    if target_kind == "teacher_step_delta6":
        return build_teacher_step_delta_target(record)
    raise ValueError(f"unsupported target_kind: {target_kind}")


def target_dim(target_kind: str) -> int:
    return int(build_target_dummy(target_kind).shape[0])


def build_target_dummy(target_kind: str) -> np.ndarray:
    if target_kind == "center_uvz":
        return np.zeros(3, dtype=np.float32)
    if target_kind == "teacher_insert_delta6":
        return np.zeros(6, dtype=np.float32)
    if target_kind == "teacher_step_delta6":
        return np.zeros(6, dtype=np.float32)
    raise ValueError(f"unsupported target_kind: {target_kind}")


def uvz_to_point_optical(uvz: np.ndarray, camera_info_k: list[float]) -> np.ndarray:
    fx = float(camera_info_k[0])
    fy = float(camera_info_k[4])
    cx = float(camera_info_k[2])
    cy = float(camera_info_k[5])
    u, v, depth = [float(value) for value in uvz]
    x = (u - cx) * depth / max(fx, 1e-6)
    y = (v - cy) * depth / max(fy, 1e-6)
    return np.array([x, y, depth], dtype=np.float32)


def build_runtime_aux_vector(
    current_pose: Any,
    feature_summary: dict[str, Any] | None,
    step_index: int = 1,
    max_steps: int = 8,
) -> np.ndarray:
    position, quat_xyzw = _pose_like_to_arrays(current_pose)
    image_shape = (
        list(feature_summary.get("image_shape", [1152, 1024]))
        if isinstance(feature_summary, dict)
        else [1152, 1024]
    )
    image_width = max(float(image_shape[0]), 1.0)
    image_height = max(float(image_shape[1]), 1.0)
    center_camera = (
        feature_summary.get("center_camera", {})
        if isinstance(feature_summary, dict)
        else {}
    )
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
    step_fraction = float(max(step_index - 1, 0)) / float(max(max_steps - 1, 1))
    return np.concatenate(
        [position.astype(np.float32), quat_xyzw.astype(np.float32), feature_values, np.array([step_fraction], dtype=np.float32)],
        axis=0,
    ).astype(np.float32)


def build_aux_target(record: dict[str, Any]) -> np.ndarray:
    labels = record["labels"]
    extra = record.get("extra", {})
    current_pose = labels.get("current_tcp_pose")
    if current_pose is None:
        raise KeyError("current_tcp_pose")
    step_index = int(extra.get("teacher_step_index", 1))
    return build_runtime_aux_vector(
        current_pose=current_pose,
        feature_summary=extra.get("feature_summary"),
        step_index=step_index,
    )


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
        aux_normalizer: TargetNormalizer | None = None,
        image_size: int = 224,
        records: list[dict[str, Any]] | None = None,
        plug_type_filter: str | None = None,
        target_kind: str = "center_uvz",
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
        self.aux_normalizer = aux_normalizer
        self.image_size = int(image_size)
        self.target_kind = target_kind

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
        target_value = build_target(record, self.target_kind)
        normalized_target = self.normalizer.normalize(target_value)
        aux_value = build_aux_target(record)
        normalized_aux = (
            self.aux_normalizer.normalize(aux_value)
            if self.aux_normalizer is not None
            else aux_value
        )
        return {
            "left": images["left"],
            "center": images["center"],
            "right": images["right"],
            "task_indices": {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in task_indices.items()
            },
            "target": torch.tensor(normalized_target, dtype=torch.float32),
            "raw_target": torch.tensor(target_value, dtype=torch.float32),
            "aux": torch.tensor(normalized_aux, dtype=torch.float32),
            "raw_aux": torch.tensor(aux_value, dtype=torch.float32),
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
        output_dim: int = 3,
        aux_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = _SharedImageEncoder()
        self.aux_dim = int(aux_dim)
        self.plug_embedding = nn.Embedding(plug_vocab_size, embedding_dim)
        self.target_module_embedding = nn.Embedding(
            target_module_vocab_size, embedding_dim
        )
        self.port_name_embedding = nn.Embedding(port_name_vocab_size, embedding_dim)
        feature_dim = 3 * 128 + 3 * embedding_dim + self.aux_dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )

    def forward(
        self,
        left: torch.Tensor,
        center: torch.Tensor,
        right: torch.Tensor,
        plug_type: torch.Tensor,
        target_module_name: torch.Tensor,
        port_name: torch.Tensor,
        aux: torch.Tensor | None = None,
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
        aux_features = (
            aux
            if self.aux_dim > 0 and aux is not None
            else left_features.new_zeros((left_features.shape[0], self.aux_dim))
        )
        return self.head(
            torch.cat(
                [left_features, center_features, right_features, task_features, aux_features], dim=1
            )
        )


def build_model_from_vocabulary(
    vocabulary: TaskVocabulary, output_dim: int = 3, aux_dim: int = 0
) -> MultiViewPortPointRegressor:
    return MultiViewPortPointRegressor(
        plug_vocab_size=len(vocabulary.plug_type_to_idx),
        target_module_vocab_size=len(vocabulary.target_module_name_to_idx),
        port_name_vocab_size=len(vocabulary.port_name_to_idx),
        output_dim=output_dim,
        aux_dim=aux_dim,
    )


class LearnedPortInference:
    manifest: dict[str, Any]

    def __init__(
        self,
        model: MultiViewPortPointRegressor,
        vocabulary: TaskVocabulary,
        normalizer: TargetNormalizer,
        aux_normalizer: TargetNormalizer | None,
        image_size: int,
        device: torch.device,
        manifest: dict[str, Any],
    ) -> None:
        self.model = model.eval()
        self.vocabulary = vocabulary
        self.normalizer = normalizer
        self.aux_normalizer = aux_normalizer
        self.image_size = int(image_size)
        self.device = device
        self.manifest: dict[str, Any] = manifest
        self.target_kind = str(manifest.get("target_kind", "center_uvz"))
        self.aux_dim = int(manifest.get("aux_dim", 0))

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
        aux_normalizer = (
            TargetNormalizer.from_dict(manifest["aux_normalizer"])
            if manifest.get("aux_normalizer") is not None
            else None
        )
        resolved_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        output_dim = int(manifest.get("output_dim", 3))
        model = build_model_from_vocabulary(
            vocabulary,
            output_dim=output_dim,
            aux_dim=int(manifest.get("aux_dim", 0)),
        )
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
            aux_normalizer=aux_normalizer,
            image_size=int(manifest["image_size"]),
            device=resolved_device,
            manifest=manifest,
        )

    def predict_target_vector(
        self,
        task: Any,
        images_bgr: dict[str, np.ndarray],
        aux_vector: np.ndarray | None = None,
    ) -> dict[str, Any]:
        task_fields = _task_field_dict(task)
        task_indices = self.vocabulary.encode(task_fields)
        if self.aux_dim > 0:
            if aux_vector is None:
                aux_vector = np.zeros(self.aux_dim, dtype=np.float32)
            normalized_aux = (
                self.aux_normalizer.normalize(aux_vector)
                if self.aux_normalizer is not None
                else aux_vector
            )
            aux_tensor = (
                torch.from_numpy(np.asarray(normalized_aux, dtype=np.float32))
                .unsqueeze(0)
                .to(self.device)
            )
        else:
            aux_tensor = None
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
                aux=aux_tensor,
            )
        normalized = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return {
            "task_fields": task_fields,
            "target_kind": self.target_kind,
            "vector": [
                float(value) for value in self.normalizer.denormalize(normalized)
            ],
        }

    def predict_center_uvz(
        self,
        task: Any,
        images_bgr: dict[str, np.ndarray],
        center_camera_k: list[float],
        aux_vector: np.ndarray | None = None,
    ) -> dict[str, Any]:
        prediction = self.predict_target_vector(
            task, images_bgr, aux_vector=aux_vector
        )
        if prediction["target_kind"] != "center_uvz":
            raise ValueError(
                f"model target_kind is {prediction['target_kind']}, not center_uvz"
            )
        uvz = np.array(prediction["vector"], dtype=np.float32)
        point_optical = uvz_to_point_optical(uvz, center_camera_k)
        return {
            "task_fields": prediction["task_fields"],
            "center_uvz": [float(value) for value in uvz],
            "point_center_optical": [float(value) for value in point_optical],
        }
