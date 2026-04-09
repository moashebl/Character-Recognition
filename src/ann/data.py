from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def _resize_images(images: np.ndarray, target_size: int) -> np.ndarray:
    if images.shape[1] == target_size and images.shape[2] == target_size:
        return images

    resized = np.zeros((images.shape[0], target_size, target_size), dtype=np.uint8)
    for i, image in enumerate(images):
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((target_size, target_size), Image.Resampling.BILINEAR)
        resized[i] = np.asarray(pil_image, dtype=np.uint8)
    return resized


def load_from_npz(
    npz_path: str | Path,
    target_size: int = 28,
    flatten: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    images = data["images"]
    labels = data["labels"].astype(np.int64)

    images = _resize_images(images, target_size)

    x = images.astype(np.float32)
    if normalize:
        x /= 255.0
    if flatten:
        x = x.reshape(x.shape[0], -1)

    return x, labels


def get_npz_class_names(npz_path: str | Path) -> list[str] | None:
    data = np.load(npz_path, allow_pickle=False)
    if "class_names" not in data.files:
        return None

    names = data["class_names"]
    return [str(name) for name in names.tolist()]


def make_labels_contiguous(
    labels: np.ndarray,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    unique_labels = np.unique(labels)
    remapped = np.searchsorted(unique_labels, labels).astype(np.int64)

    if class_names is not None and len(class_names) == len(unique_labels):
        names = class_names
    elif len(unique_labels) == 26 and np.array_equal(unique_labels, np.arange(26)):
        names = [chr(i + ord("A")) for i in range(26)]
    else:
        names = [str(int(label)) for label in unique_labels]

    return remapped, names


def load_from_image_folders(
    root_dir: str | Path,
    target_size: int = 28,
    flatten: bool = True,
    normalize: bool = True,
    max_per_class: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    root_path = Path(root_dir)
    class_folders = sorted([p for p in root_path.iterdir() if p.is_dir()])

    images = []
    labels = []

    class_names = [folder.name for folder in class_folders]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_folder in class_folders:
        class_name = class_folder.name

        image_files = sorted([p for p in class_folder.iterdir() if p.is_file()])
        if max_per_class is not None:
            image_files = image_files[:max_per_class]

        label = class_to_idx[class_name]
        for image_path in image_files:
            image = Image.open(image_path).convert("L")
            image = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.uint8))
            labels.append(label)

    x = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    if normalize:
        x /= 255.0
    if flatten:
        x = x.reshape(x.shape[0], -1)

    return x, y


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if not stratify:
        indices = np.arange(len(y))
        rng.shuffle(indices)
        split = int(len(indices) * (1.0 - test_size))
        train_idx = indices[:split]
        test_idx = indices[split:]
        return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

    train_parts = []
    test_parts = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        split = int(len(cls_idx) * (1.0 - test_size))
        train_parts.append(cls_idx[:split])
        test_parts.append(cls_idx[split:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]
