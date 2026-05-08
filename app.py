"""Flask entrypoint for the cross-modal RGB-IR target detection web system."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from config import DevelopmentConfig
from inference import MultiBranchDetector
from inference.utils import allowed_file, make_filename, read_ir_grayscale, read_rgb, save_image

MODEL_EXTENSIONS = {".pt", ".pth"}


def _ensure_directories(app: Flask) -> None:
    for key in ["UPLOAD_FOLDER", "OUTPUT_FOLDER", "MODEL_FOLDER", "SINGLE_MODEL_FOLDER", "FUSION_MODEL_FOLDER"]:
        Path(app.config[key]).mkdir(parents=True, exist_ok=True)


def _relative_to_base(path: Path) -> str:
    try:
        return str(path.relative_to(Path(__file__).resolve().parent)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def scan_models(directory: str | Path) -> list[dict[str, str]]:
    """Scan model weight files in one directory for the model selector UI."""
    root = Path(directory)
    if not root.exists():
        return []
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS)
    return [{"name": path.name, "path": _relative_to_base(path)} for path in files]


def resolve_model_path(selected: str | None, manual: str | None, model_root: Path) -> Path | None:
    """Resolve a dropdown or manually typed model path safely within the project."""
    raw_value = (manual or "").strip() or (selected or "").strip()
    if not raw_value:
        return None

    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parent / candidate
    candidate = candidate.resolve()

    project_root = Path(__file__).resolve().parent
    allowed_roots = [project_root.resolve(), model_root.resolve()]
    if not any(candidate == root or root in candidate.parents for root in allowed_roots):
        return None
    return candidate


def build_advantage_analysis(results: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """Create a concise qualitative fusion advantage analysis."""
    rgb = results["rgb"]["summary"]
    ir = results["ir"]["summary"]
    fusion = results["fusion"]["summary"]
    best_single_count = max(rgb["count"], ir["count"])
    best_single_conf = max(rgb["avg_confidence"], ir["avg_confidence"])

    observations: list[str] = []
    if fusion["count"] > best_single_count:
        observations.append(f"融合分支检测到 {fusion['count']} 个目标，高于最佳单模态的 {best_single_count} 个。")
    else:
        observations.append(f"融合分支检测到 {fusion['count']} 个目标，可与单模态结果进行互补对照。")

    if fusion["avg_confidence"] >= best_single_conf:
        observations.append(f"融合分支平均置信度 {fusion['avg_confidence_text']}，不低于最佳单模态的 {best_single_conf:.2f}。")
    else:
        observations.append("融合分支可补充 RGB 与 IR 分支的漏检区域，适合结合可视化结果进行定性分析。")

    if fusion["coverage"] >= max(rgb["coverage"], ir["coverage"]):
        observations.append(f"融合结果框覆盖率为 {fusion['coverage_text']}，体现更充分的候选目标区域响应。")

    if fusion["count"] > best_single_count and fusion["avg_confidence"] >= best_single_conf:
        headline = "本次样本中，融合分支在检测数量与平均置信度方面优于最佳单模态模型，体现出跨模态互补优势。"
    elif fusion["count"] > best_single_count:
        headline = "本次样本中，融合分支在检测数量方面更突出，说明 RGB 与 IR 信息融合能够缓解单一模态漏检。"
    elif fusion["avg_confidence"] >= best_single_conf:
        headline = "本次样本中，融合分支在置信度稳定性方面表现较好，体现跨模态特征互补价值。"
    else:
        headline = "本次样本中，三路结果存在差异，可用于展示 RGB、IR 与融合检测在复杂场景下的互补关系。"
    return headline, observations


def create_app(config_object=DevelopmentConfig) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)
    _ensure_directories(app)

    @app.route("/", methods=["GET"])
    def index():
        single_models = scan_models(app.config["SINGLE_MODEL_FOLDER"])
        fusion_models = scan_models(app.config["FUSION_MODEL_FOLDER"])
        return render_template(
            "index.html",
            single_models=single_models,
            fusion_models=fusion_models,
            model_count=len(single_models) + len(fusion_models),
        )

    @app.route("/detect", methods=["POST"])
    def detect():
        rgb_file = request.files.get("rgb_image")
        ir_file = request.files.get("ir_image")

        if not rgb_file or rgb_file.filename == "":
            flash("请上传 RGB 可见光图像。", "error")
            return redirect(url_for("index"))
        if not ir_file or ir_file.filename == "":
            flash("请上传 IR 红外图像。", "error")
            return redirect(url_for("index"))
        if not allowed_file(rgb_file.filename, app.config["ALLOWED_EXTENSIONS"]):
            flash("RGB 图像格式不支持，请使用 jpg/png/jpeg/bmp/tif。", "error")
            return redirect(url_for("index"))
        if not allowed_file(ir_file.filename, app.config["ALLOWED_EXTENSIONS"]):
            flash("IR 图像格式不支持，请使用 jpg/png/jpeg/bmp/tif。", "error")
            return redirect(url_for("index"))

        rgb_upload_name = make_filename("rgb", secure_filename(rgb_file.filename))
        ir_upload_name = make_filename("ir", secure_filename(ir_file.filename))
        rgb_upload_path = Path(app.config["UPLOAD_FOLDER"]) / rgb_upload_name
        ir_upload_path = Path(app.config["UPLOAD_FOLDER"]) / ir_upload_name
        rgb_file.save(rgb_upload_path)
        ir_file.save(ir_upload_path)

        try:
            rgb_image = read_rgb(rgb_upload_path)
            ir_gray = read_ir_grayscale(ir_upload_path)
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("index"))

        single_model_path = resolve_model_path(
            request.form.get("single_model_select"),
            request.form.get("single_model_manual"),
            Path(app.config["SINGLE_MODEL_FOLDER"]),
        )
        fusion_model_path = resolve_model_path(
            request.form.get("fusion_model_select"),
            request.form.get("fusion_model_manual"),
            Path(app.config["FUSION_MODEL_FOLDER"]),
        )

        detector = MultiBranchDetector(
            single_model_path=single_model_path,
            fusion_model_path=fusion_model_path,
            image_size=app.config["DETECTION_IMAGE_SIZE"],
        )
        results = detector.run_all(rgb_image, ir_gray)

        output_names = {
            "rgb": make_filename("rgb_result", "result.png"),
            "ir": make_filename("ir_result", "result.png"),
            "fusion": make_filename("fusion_result", "result.png"),
        }
        for branch, filename in output_names.items():
            save_image(Path(app.config["OUTPUT_FOLDER"]) / filename, results[branch]["rendered_image"])

        headline, observations = build_advantage_analysis(results)
        return render_template(
            "result.html",
            original_rgb=url_for("static", filename=f"uploads/{rgb_upload_name}"),
            original_ir=url_for("static", filename=f"uploads/{ir_upload_name}"),
            result_images={branch: url_for("static", filename=f"outputs/{name}") for branch, name in output_names.items()},
            results=results,
            advantage_headline=headline,
            observations=observations,
            single_model_display=str(single_model_path) if single_model_path else "未选择模型（演示模式）",
            fusion_model_display=str(fusion_model_path) if fusion_model_path else "未选择模型（演示模式）",
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
