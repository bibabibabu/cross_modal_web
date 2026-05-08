"""Configuration for the Cross-Modal RGB-IR Object Detection System."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Base application configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "cross-modal-rgb-ir-thesis-demo")
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

    UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
    OUTPUT_FOLDER = BASE_DIR / "static" / "outputs"
    MODEL_FOLDER = BASE_DIR / "models"
    SINGLE_MODEL_FOLDER = MODEL_FOLDER / "single"
    FUSION_MODEL_FOLDER = MODEL_FOLDER / "fusion"

    DETECTION_IMAGE_SIZE = int(os.environ.get("DETECTION_IMAGE_SIZE", "640"))


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {"development": DevelopmentConfig, "production": ProductionConfig}
