"""Detector orchestration for RGB, IR and RGB-IR fusion branches."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .fusion_adapter import FusionDetectorAdapter
from .utils import draw_detections, extract_detections_from_ultralytics, mock_detections, summarize_detections

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None


class SingleModalDetector:
    """Wrapper for one Ultralytics YOLO single-modal detector."""

    def __init__(self, model_path: str | Path | None, branch_name: str, image_size: int = 640):
        self.model_path = Path(model_path) if model_path else None
        self.branch_name = branch_name
        self.image_size = image_size
        self.model = self._load_model()

    def _load_model(self) -> Any | None:
        if YOLO is None or self.model_path is None or not self.model_path.exists():
            return None
        try:
            return YOLO(str(self.model_path))
        except Exception:
            return None

    @property
    def status(self) -> str:
        return "已使用真实模型" if self.model is not None else "演示模式（未加载权重）"

    @property
    def backend(self) -> str:
        return "Ultralytics YOLO" if self.model is not None else "Demo fallback"

    def infer(self, image_rgb: np.ndarray) -> dict[str, Any]:
        start = time.perf_counter()
        detections = self._predict_or_fallback(image_rgb)
        color = (46, 111, 237) if self.branch_name == "rgb" else (16, 185, 129)
        rendered = draw_detections(image_rgb, detections, color=color)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        summary = summarize_detections(detections, image_rgb.shape)

        return {
            "rendered_image": rendered,
            "detections": detections,
            "summary": summary,
            "time_ms": elapsed_ms,
            "time_text": f"{elapsed_ms:.2f} ms",
            "status": self.status,
            "backend": self.backend,
            "model_path": str(self.model_path) if self.model_path else "未选择模型",
        }

    def _predict_or_fallback(self, image_rgb: np.ndarray) -> list:
        if self.model is None:
            label = "Car" if self.branch_name == "rgb" else "People"
            count = 2 if self.branch_name == "rgb" else 1
            return mock_detections(image_rgb, label=label, count=count)

        try:
            results = self.model.predict(source=image_rgb, imgsz=self.image_size, verbose=False)
            if not results:
                return []
            return extract_detections_from_ultralytics(results[0], image_rgb.shape)
        except Exception:
            return mock_detections(image_rgb, label=f"{self.branch_name}-fallback", count=1)


class MultiBranchDetector:
    """Run RGB, IR and RGB-IR fusion detection branches."""

    def __init__(
        self,
        single_model_path: str | Path | None,
        fusion_model_path: str | Path | None,
        image_size: int = 640,
    ):
        self.rgb_detector = SingleModalDetector(single_model_path, branch_name="rgb", image_size=image_size)
        self.ir_detector = SingleModalDetector(single_model_path, branch_name="ir", image_size=image_size)
        self.fusion_detector = FusionDetectorAdapter(fusion_model_path, image_size=image_size)

    @staticmethod
    def _normalize_ir_for_single_modal(ir_gray: np.ndarray) -> np.ndarray:
        if ir_gray.ndim == 2:
            return cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2RGB)
        return ir_gray

    def run_all(self, rgb_image: np.ndarray, ir_gray_image: np.ndarray) -> dict[str, dict[str, Any]]:
        ir_rgb = self._normalize_ir_for_single_modal(ir_gray_image)
        rgb_result = self.rgb_detector.infer(rgb_image)
        ir_result = self.ir_detector.infer(ir_rgb)
        fusion_result = self.fusion_detector.infer(rgb_image, ir_gray_image)
        return {"rgb": rgb_result, "ir": ir_result, "fusion": fusion_result}
