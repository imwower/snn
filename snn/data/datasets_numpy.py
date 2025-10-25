from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

DATASET_IMAGE_SHAPES = {
    "MNIST": (28, 28, 1),
    "FASHION": (28, 28, 1),
    "FASHION-MNIST": (28, 28, 1),
    "CIFAR10": (32, 32, 3),
}


def ensure_feature_stats(train_x: np.ndarray, stats_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load or compute per-feature mean/std and persist them under the dataset directory."""
    stats_dir.mkdir(parents=True, exist_ok=True)
    mean_path = stats_dir / "mean.npy"
    std_path = stats_dir / "std.npy"
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    if mean_path.exists() and std_path.exists():
        try:
            mean = np.load(mean_path)
            std = np.load(std_path)
        except Exception:
            mean = None
            std = None
    if mean is None or std is None or mean.shape[0] != train_x.shape[1] or std.shape[0] != train_x.shape[1]:
        mean = np.mean(train_x, axis=0, dtype=np.float64).astype(np.float32)
        var = np.var(train_x, axis=0, dtype=np.float64)
        std = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)
        np.save(mean_path, mean)
        np.save(std_path, std)
    return mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)


def standardize_batch(batch: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if batch.shape[1] != mean.shape[0] or batch.shape[1] != std.shape[0]:
        raise ValueError("batch feature dimension must match mean/std length")
    denom = np.clip(std, 1e-6, None)
    return (batch - mean) / denom


def augment_flat_batch(
    flat_batch: np.ndarray,
    image_shape: Optional[Tuple[int, ...]],
    rng: Optional[np.random.Generator],
    *,
    max_translate: int = 2,
    max_crop: int = 2,
) -> np.ndarray:
    if image_shape is None or flat_batch.size == 0:
        return flat_batch
    if rng is None:
        raise ValueError("rng must be provided when applying augmentation")
    normalized_shape = _normalize_image_shape(image_shape)
    batch_size = flat_batch.shape[0]
    h, w, c = normalized_shape
    feature_dim = flat_batch.shape[1]
    if feature_dim != h * w * c:
        raise ValueError(f"flat batch dim {feature_dim} does not match image shape {normalized_shape}")
    images = flat_batch.reshape(batch_size, h, w, c)
    augmented = np.empty_like(images)
    pad_width = (
        (0, 0),
        (max_translate, max_translate),
        (max_translate, max_translate),
        (0, 0),
    )
    padded = np.pad(images, pad_width, mode="constant")
    for idx in range(batch_size):
        y_offset = rng.integers(0, max_translate * 2 + 1)
        x_offset = rng.integers(0, max_translate * 2 + 1)
        translated = padded[idx, y_offset:y_offset + h, x_offset:x_offset + w, :]
        if rng.random() < 0.5:
            translated = translated[:, ::-1, :]
        crop_top = rng.integers(0, max_crop + 1)
        crop_bottom = rng.integers(0, max_crop + 1)
        crop_left = rng.integers(0, max_crop + 1)
        crop_right = rng.integers(0, max_crop + 1)
        y_start = min(crop_top, h - 1)
        y_end = max(y_start + 1, h - crop_bottom)
        x_start = min(crop_left, w - 1)
        x_end = max(x_start + 1, w - crop_right)
        cropped = translated[y_start:y_end, x_start:x_end, :]
        augmented[idx] = _pad_to_shape(cropped, (h, w, c), crop_top, crop_bottom, crop_left, crop_right)
    return augmented.reshape(flat_batch.shape[0], -1)


def _normalize_image_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(shape) == 2:
        return shape[0], shape[1], 1
    if len(shape) == 3:
        return shape
    raise ValueError(f"Unsupported image shape: {shape}")


def _pad_to_shape(
    image: np.ndarray,
    target_shape: Tuple[int, int, int],
    crop_top: int,
    crop_bottom: int,
    crop_left: int,
    crop_right: int,
) -> np.ndarray:
    h, w, _ = target_shape
    pad_top = crop_top
    pad_bottom = crop_bottom
    pad_left = crop_left
    pad_right = crop_right
    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
    )
    return padded[:h, :w, :]
