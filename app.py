import os
import uuid

from ml_service import predict_video, VideoProcessingError
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Maximum upload size: 100 MB
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Enable frontend-backend communication
CORS(app)

# Upload configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

ALLOWED_EXTENSIONS = {
    "mp4", "avi", "mov", "mkv", "mpeg", "mpg"
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# Home endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ASD Video Screening Backend is running!"
    }), 200


# Health-check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "service": "ASD Video Screening Backend",
        "status": "healthy"
    }), 200


# Video-upload endpoint
@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({
            "error": "No video file provided"
        }), 400

    file = request.files["video"]

    if not file.filename:
        return jsonify({
            "error": "No file selected"
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported video format"
        }), 400

    original_filename = secure_filename(file.filename)

    if not original_filename:
        return jsonify({
            "error": "Invalid filename"
        }), 400

    extension = original_filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:
        file.save(file_path)
    except OSError:
        app.logger.exception("Failed to save uploaded video")
        return jsonify({
            "error": "Could not save the uploaded video"
        }), 500

    return jsonify({
        "message": "Video uploaded successfully",
        "filename": filename,
        "original_filename": original_filename
    }), 201


# Handle files exceeding the upload-size limit
@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Video file exceeds the 100 MB limit"
    }), 413


# Handle unexpected server errors
@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "error": "An internal server error occurred"
    }), 500


# Video-prediction endpoint
@app.route("/predict", methods=["POST"])
def predict_uploaded_video():
    if "video" not in request.files:
        return jsonify({
            "error": "No video file provided"
        }), 400

    file = request.files["video"]

    if not file.filename:
        return jsonify({
            "error": "No file selected"
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported video format"
        }), 400

    original_filename = secure_filename(file.filename)

    if not original_filename or not allowed_file(original_filename):
        return jsonify({
            "error": "Invalid filename"
        }), 400

    extension = original_filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:
        try:
            file.save(file_path)
        except OSError:
            app.logger.exception("Failed to save prediction upload")
            return jsonify({
                "error": "Could not save the uploaded video"
            }), 500

        if os.path.getsize(file_path) == 0:
            return jsonify({
                "error": "Uploaded file is empty"
            }), 400

        try:
            result = predict_video(file_path)
        except VideoProcessingError:
            app.logger.exception("Video processing failed")
            return jsonify({
                "error": "Could not process the video. "
                         "Check that it is a readable video file."
            }), 422

        return jsonify(result), 200

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the backend terminal."
        }), 500

    finally:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except OSError:
            app.logger.exception(
                "Could not delete temporary prediction upload"
            )


# Start the Flask development server
if __name__ == "__main__":
    app.run(debug=True)