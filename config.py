"""Configuration for the Cross-Modal Visible-Infrared Object Detection System."""
from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Base application configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "cross-modal-demo-secret")

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}

    UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
    OUTPUT_FOLDER = BASE_DIR / "static" / "outputs"
    MODEL_FOLDER = BASE_DIR / "models"

    RGB_MODEL_PATH = MODEL_FOLDER / "rgb_model.pt"
    IR_MODEL_PATH = MODEL_FOLDER / "ir_model.pt"
    FUSION_MODEL_PATH = MODEL_FOLDER / "fusion_model.pt"

    # Shared input size for placeholder and standard YOLO inference.
    DETECTION_IMAGE_SIZE = 640


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
