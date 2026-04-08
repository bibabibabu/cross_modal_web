"""Utility helpers for image validation, preprocessing and rendering."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch


Detection = Tuple[int, int, int, int, float, str]


def allowed_file(filename: str, allowed_extensions: Iterable[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in set(allowed_extensions)


def make_filename(prefix: str, original_name: str) -> str:
    ext = Path(original_name).suffix.lower() or ".png"
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"


def read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_ir_grayscale(path: str | Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Unable to read infrared image: {path}")
    return gray


def prepare_fusion_tensor(
    rgb_image: np.ndarray,
    ir_image: np.ndarray,
    image_size: int = 640,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build [R, G, B, IR] tensor with normalized values for custom fusion inference.

    Returns shape [1, 4, H, W].
    """

    rgb_resized = cv2.resize(rgb_image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    ir_resized = cv2.resize(ir_image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    ir_channel = np.expand_dims(ir_resized, axis=-1)
    stacked = np.concatenate([rgb_resized, ir_channel], axis=-1).astype(np.float32) / 255.0
    chw = np.transpose(stacked, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    return tensor


def draw_detections(image: np.ndarray, detections: List[Detection], color=(52, 123, 246)) -> np.ndarray:
    out = image.copy()
    for x1, y1, x2, y2, conf, label in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {conf:.2f}"
        cv2.putText(out, caption, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def mock_detections(image: np.ndarray, label: str = "object") -> List[Detection]:
    h, w = image.shape[:2]
    x1, y1 = int(w * 0.15), int(h * 0.2)
    x2, y2 = int(w * 0.7), int(h * 0.8)
    return [(x1, y1, x2, y2, 0.75, label)]


def save_image(path: str | Path, image_rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
