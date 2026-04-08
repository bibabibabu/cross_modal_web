"""Adapter for custom cross-modal (RGB+IR) fusion model inference."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .utils import draw_detections, mock_detections, prepare_fusion_tensor


class FusionDetectorAdapter:
    """Adapter around your custom Ultralytics YOLO11-style fusion model.

    This file intentionally keeps custom integration points explicit.
    Replace placeholder logic with your model's real loading and forward APIs.
    """

    def __init__(self, model_path: str | Path, image_size: int = 640, device: str | None = None):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_custom_model()

    def _load_custom_model(self):
        """Load fusion model with optional custom module registration."""
        # TODO: register CrossModalAttention and AGF/EFC custom modules here
        # TODO: if your project also requires Silence / SilenceChannel, register them too
        # TODO: replace with actual custom Ultralytics model loading
        # Example integration point:
        # from your_custom_repo.nn.modules import CrossModalAttention, AGF
        # torch.serialization.add_safe_globals([CrossModalAttention, AGF])
        # model = torch.load(self.model_path, map_location=self.device)
        # return model.eval()
        if self.model_path.exists():
            try:
                model = torch.load(self.model_path, map_location=self.device)
                if hasattr(model, "eval"):
                    model.eval()
                return model
            except Exception:
                return None
        return None

    def infer(self, rgb_image: np.ndarray, ir_image: np.ndarray) -> Dict:
        start = time.perf_counter()

        input_tensor = prepare_fusion_tensor(
            rgb_image=rgb_image,
            ir_image=ir_image,
            image_size=self.image_size,
            device=self.device,
        )

        detections = self._run_model_or_fallback(input_tensor, rgb_image)
        rendered = draw_detections(rgb_image, detections, color=(244, 114, 40))

        return {
            "rendered_image": rendered,
            "detections": detections,
            "time_ms": (time.perf_counter() - start) * 1000.0,
            "status": "model_loaded" if self.model is not None else "fallback_mock",
        }

    def _run_model_or_fallback(self, input_tensor: torch.Tensor, rgb_reference: np.ndarray) -> List:
        """Run actual custom fusion inference or fallback.

        TODO: adapt preprocessing to my 4-channel RGB+IR input pipeline
        TODO: replace this block with your custom model's predict/forward and NMS postprocess
        """
        if self.model is None:
            return mock_detections(rgb_reference, label="fusion-object")

        try:
            with torch.no_grad():
                # TODO: adapt based on your model's forward signature.
                _ = self.model(input_tensor)
            # TODO: decode model outputs into [(x1,y1,x2,y2,conf,label), ...]
            return mock_detections(rgb_reference, label="fusion-pred")
        except Exception:
            return mock_detections(rgb_reference, label="fusion-fallback")
