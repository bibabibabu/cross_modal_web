"""Detector orchestration for RGB, IR and fusion branches."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .fusion_adapter import FusionDetectorAdapter
from .utils import draw_detections, mock_detections

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None


class SingleModalDetector:
    def __init__(self, model_path: str | Path, branch_name: str):
        self.model_path = Path(model_path)
        self.branch_name = branch_name
        self.model = self._load_model()

    def _load_model(self):
        if YOLO is None or not self.model_path.exists():
            return None
        try:
            return YOLO(str(self.model_path))
        except Exception:
            return None

    def infer(self, image_rgb: np.ndarray) -> Dict:
        start = time.perf_counter()
        detections = self._predict_or_fallback(image_rgb)
        color = (52, 123, 246) if self.branch_name == "rgb" else (16, 185, 129)
        rendered = draw_detections(image_rgb, detections, color=color)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        status = "model_loaded" if self.model is not None else "fallback_mock"

        return {
            "rendered_image": rendered,
            "detections": detections,
            "time_ms": elapsed_ms,
            "status": status,
        }

    def _predict_or_fallback(self, image_rgb: np.ndarray) -> List:
        if self.model is None:
            return mock_detections(image_rgb, label=f"{self.branch_name}-object")

        try:
            results = self.model.predict(source=image_rgb, verbose=False)
            if not results:
                return []
            res = results[0]
            if res.boxes is None:
                return []

            names = res.names
            det = []
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                det.append((x1, y1, x2, y2, conf, label))
            return det
        except Exception:
            return mock_detections(image_rgb, label=f"{self.branch_name}-fallback")


class MultiBranchDetector:
    """Run all three detection branches and return rendered results."""

    def __init__(self, rgb_model_path: str | Path, ir_model_path: str | Path, fusion_model_path: str | Path, image_size: int = 640):
        self.rgb_detector = SingleModalDetector(rgb_model_path, branch_name="rgb")
        self.ir_detector = SingleModalDetector(ir_model_path, branch_name="ir")
        self.fusion_detector = FusionDetectorAdapter(fusion_model_path, image_size=image_size)

    @staticmethod
    def _normalize_ir_for_single_modal(ir_gray: np.ndarray) -> np.ndarray:
        if len(ir_gray.shape) == 2:
            return cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2RGB)
        return ir_gray

    def run_all(self, rgb_image: np.ndarray, ir_gray_image: np.ndarray) -> Dict[str, Dict]:
        ir_rgb = self._normalize_ir_for_single_modal(ir_gray_image)

        rgb_result = self.rgb_detector.infer(rgb_image)
        ir_result = self.ir_detector.infer(ir_rgb)
        fusion_result = self.fusion_detector.infer(rgb_image, ir_gray_image)

        return {
            "rgb": rgb_result,
            "ir": ir_result,
            "fusion": fusion_result,
        }
