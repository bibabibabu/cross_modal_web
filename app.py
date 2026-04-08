"""Flask app entrypoint.

Run locally:
    1) pip install -r requirements.txt
    2) python app.py
    3) Open http://127.0.0.1:5000
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from config import DevelopmentConfig
from inference import MultiBranchDetector
from inference.utils import allowed_file, make_filename, read_ir_grayscale, read_rgb, save_image


def create_app(config_object=DevelopmentConfig) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)

    for directory in [app.config["UPLOAD_FOLDER"], app.config["OUTPUT_FOLDER"], app.config["MODEL_FOLDER"]]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    detector = MultiBranchDetector(
        rgb_model_path=app.config["RGB_MODEL_PATH"],
        ir_model_path=app.config["IR_MODEL_PATH"],
        fusion_model_path=app.config["FUSION_MODEL_PATH"],
        image_size=app.config["DETECTION_IMAGE_SIZE"],
    )

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/detect", methods=["POST"])
    def detect():
        rgb_file = request.files.get("rgb_image")
        ir_file = request.files.get("ir_image")

        if not rgb_file or rgb_file.filename == "":
            flash("Please upload a visible RGB image.", "error")
            return redirect(url_for("index"))

        if not ir_file or ir_file.filename == "":
            flash("Please upload an infrared image.", "error")
            return redirect(url_for("index"))

        if not allowed_file(rgb_file.filename, app.config["ALLOWED_EXTENSIONS"]):
            flash("RGB image format is not supported.", "error")
            return redirect(url_for("index"))

        if not allowed_file(ir_file.filename, app.config["ALLOWED_EXTENSIONS"]):
            flash("Infrared image format is not supported.", "error")
            return redirect(url_for("index"))

        rgb_upload_name = make_filename("rgb_upload", secure_filename(rgb_file.filename))
        ir_upload_name = make_filename("ir_upload", secure_filename(ir_file.filename))

        rgb_upload_path = Path(app.config["UPLOAD_FOLDER"]) / rgb_upload_name
        ir_upload_path = Path(app.config["UPLOAD_FOLDER"]) / ir_upload_name

        rgb_file.save(rgb_upload_path)
        ir_file.save(ir_upload_path)

        try:
            rgb_image = read_rgb(rgb_upload_path)
            ir_gray = read_ir_grayscale(ir_upload_path)
        except ValueError as error:
            flash(f"Failed to read uploaded images: {error}", "error")
            return redirect(url_for("index"))

        results = detector.run_all(rgb_image, ir_gray)

        rgb_output_name = make_filename("rgb_result", "result.png")
        ir_output_name = make_filename("ir_result", "result.png")
        fusion_output_name = make_filename("fusion_result", "result.png")

        rgb_output_path = Path(app.config["OUTPUT_FOLDER"]) / rgb_output_name
        ir_output_path = Path(app.config["OUTPUT_FOLDER"]) / ir_output_name
        fusion_output_path = Path(app.config["OUTPUT_FOLDER"]) / fusion_output_name

        save_image(rgb_output_path, results["rgb"]["rendered_image"])
        save_image(ir_output_path, results["ir"]["rendered_image"])
        save_image(fusion_output_path, results["fusion"]["rendered_image"])

        context = {
            "original_rgb": url_for("static", filename=f"uploads/{rgb_upload_name}"),
            "original_ir": url_for("static", filename=f"uploads/{ir_upload_name}"),
            "rgb_result": url_for("static", filename=f"outputs/{rgb_output_name}"),
            "ir_result": url_for("static", filename=f"outputs/{ir_output_name}"),
            "fusion_result": url_for("static", filename=f"outputs/{fusion_output_name}"),
            "timings": {
                "rgb": f"{results['rgb']['time_ms']:.2f} ms",
                "ir": f"{results['ir']['time_ms']:.2f} ms",
                "fusion": f"{results['fusion']['time_ms']:.2f} ms",
            },
            "statuses": {
                "rgb": results["rgb"]["status"],
                "ir": results["ir"]["status"],
                "fusion": results["fusion"]["status"],
            },
        }
        return render_template("result.html", **context)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
