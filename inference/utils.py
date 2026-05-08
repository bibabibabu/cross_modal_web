"""Utility helpers for image validation, preprocessing, rendering and metrics."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch

Detection = tuple[int, int, int, int, float, str]


def allowed_file(filename: str, allowed_extensions: Iterable[str] | None = None) -> bool:
    """Return whether *filename* has an allowed image extension."""
    extensions = set(allowed_extensions or {"jpg", "jpeg", "png", "bmp", "tif", "tiff"})
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def make_filename(prefix: str, original_name: str) -> str:
    """Create a collision-resistant filename while preserving the extension."""
    ext = Path(original_name).suffix.lower() or ".png"
    safe_prefix = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in prefix)
    return f"{safe_prefix}_{int(time.time())}_{uuid.uuid4().hex[:10]}{ext}"


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as RGB uint8 ndarray."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取 RGB 图像：{path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_ir_grayscale(path: str | Path) -> np.ndarray:
    """Read an infrared image as a single-channel grayscale uint8 ndarray."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"无法读取 IR 图像：{path}")
    return gray


def _resize_pair(rgb_image: np.ndarray, ir_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb_image.shape[:2]
    if ir_image.shape[:2] != (h, w):
        ir_image = cv2.resize(ir_image, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb_image, ir_image


def prepare_fusion_source(rgb_image: np.ndarray, ir_image: np.ndarray) -> np.ndarray:
    """Build an HWC four-channel [R, G, B, IR] uint8 source image."""
    rgb_image, ir_image = _resize_pair(rgb_image, ir_image)
    if ir_image.ndim == 3:
        ir_image = cv2.cvtColor(ir_image, cv2.COLOR_RGB2GRAY)
    ir_channel = np.expand_dims(ir_image.astype(np.uint8), axis=-1)
    return np.concatenate([rgb_image.astype(np.uint8), ir_channel], axis=-1)


def prepare_fusion_tensor(
    rgb_image: np.ndarray,
    ir_image: np.ndarray,
    image_size: int = 640,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a normalized [1, 4, image_size, image_size] tensor as [R,G,B,IR]."""
    fusion_source = prepare_fusion_source(rgb_image, ir_image)
    resized = cv2.resize(fusion_source, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    chw = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).to(device)


def _clip_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_detections(image: np.ndarray, detections: list[Detection], color: tuple[int, int, int] = (52, 123, 246)) -> np.ndarray:
    """Draw bounding boxes and labels on an RGB image."""
    out = image.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2, conf, label in detections:
        x1, y1, x2, y2 = _clip_box((x1, y1, x2, y2), w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(0, y1 - th - 10)
        cv2.rectangle(out, (x1, top), (min(w - 1, x1 + tw + 8), top + th + 8), color, -1)
        cv2.putText(out, caption, (x1 + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def extract_detections_from_ultralytics(result: Any, image_shape: tuple[int, int] | tuple[int, int, int]) -> list[Detection]:
    """Normalize an Ultralytics result object into Detection tuples."""
    height, width = image_shape[:2]
    if result is None or getattr(result, "boxes", None) is None:
        return []

    names = getattr(result, "names", {}) or {}
    detections: list[Detection] = []
    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
        x1, y1, x2, y2 = _clip_box(xyxy, width, height)
        conf = float(box.conf[0].detach().cpu().item()) if getattr(box, "conf", None) is not None else 0.0
        cls_id = int(box.cls[0].detach().cpu().item()) if getattr(box, "cls", None) is not None else 0
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        detections.append((x1, y1, x2, y2, conf, label))
    return detections


def summarize_detections(detections: list[Detection], image_shape: tuple[int, int] | tuple[int, int, int]) -> dict[str, Any]:
    """Compute thesis-demo friendly detection statistics."""
    height, width = image_shape[:2]
    count = len(detections)
    confidences = [det[4] for det in detections]
    total_area = 0
    top_items = []
    for x1, y1, x2, y2, conf, label in detections:
        x1, y1, x2, y2 = _clip_box((x1, y1, x2, y2), width, height)
        total_area += max(0, x2 - x1) * max(0, y2 - y1)
        top_items.append({"label": label, "confidence": round(conf, 2)})

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    max_conf = float(np.max(confidences)) if confidences else 0.0
    coverage = min(total_area / float(max(1, width * height)), 1.0)
    top_items = sorted(top_items, key=lambda item: item["confidence"], reverse=True)[:5]

    return {
        "count": count,
        "avg_confidence": avg_conf,
        "max_confidence": max_conf,
        "coverage": coverage,
        "avg_confidence_text": f"{avg_conf:.2f}",
        "max_confidence_text": f"{max_conf:.2f}",
        "coverage_text": f"{coverage * 100:.1f}%",
        "top_items": top_items,
    }


def mock_detections(image: np.ndarray, label: str = "object", count: int = 2) -> list[Detection]:
    """Generate deterministic demo detections when model weights are unavailable."""
    h, w = image.shape[:2]
    boxes = [
        (int(w * 0.12), int(h * 0.20), int(w * 0.42), int(h * 0.62), 0.82, label),
        (int(w * 0.55), int(h * 0.28), int(w * 0.82), int(h * 0.72), 0.68, label),
        (int(w * 0.32), int(h * 0.55), int(w * 0.58), int(h * 0.86), 0.59, label),
    ]
    return boxes[: max(1, min(count, len(boxes)))]


def save_image(path: str | Path, image_rgb: np.ndarray) -> None:
    """Save an RGB or grayscale ndarray to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if image_rgb.ndim == 2:
        cv2.imwrite(str(path), image_rgb)
        return
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
