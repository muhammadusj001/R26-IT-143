from flask import Flask, render_template, request, Response, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

MODEL_PATH = "model/swimming_pool_garbage_yolo.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

model = YOLO(MODEL_PATH)

GARBAGE_CLASSES = ["aluminium foil", "bottle", "juice", "thermocol"]


def build_notification(detected, media_kind):
    if detected:
        return {
            "notification_level": "alert",
            "notification_title": "Garbage detected in the pool",
            "notification_message": (
                f"Garbage has been detected in the uploaded {media_kind}. "
                "The swimming pool needs to be cleaned immediately."
            ),
        }

    return {
        "notification_level": "safe",
        "notification_title": "Pool is clean",
        "notification_message": (
            f"No garbage was detected in the uploaded {media_kind}. "
            "The swimming pool appears clean and safe."
        ),
    }


def render_dashboard(**context):
    base_context = {
        "detected": None,
        "media_result": None,
        "media_result_type": None,
        "garbage_classes": GARBAGE_CLASSES,
        "model_path": MODEL_PATH,
        "camera_source": "Laptop webcam / default CCTV input",
        "notification_level": None,
        "notification_title": None,
        "notification_message": None,
        "processed_media_kind": None,
    }
    base_context.update(context)
    return render_template("index.html", **base_context)


def render_landing():
    return render_template(
        "landing.html",
        garbage_classes=GARBAGE_CLASSES,
        model_path=MODEL_PATH,
        supported_uploads="Images and videos",
    )


def detection_contains_garbage(results):
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name in GARBAGE_CLASSES:
                return True

    return False


def save_upload(file):
    original_name = secure_filename(file.filename)
    extension = os.path.splitext(original_name)[1].lower()
    filename = f"{uuid.uuid4()}_{original_name}"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)
    return filename, upload_path, extension


def process_image(upload_path, filename):
    results = model(upload_path, conf=0.25)
    detected = detection_contains_garbage(results)

    result_image = results[0].plot()
    output_filename = "detected_" + filename
    output_path = os.path.join(RESULT_FOLDER, output_filename)
    cv2.imwrite(output_path, result_image)

    return {
        "detected": detected,
        "media_result": url_for("static", filename=f"results/{output_filename}"),
        "media_result_type": "image",
        "processed_media_kind": "image",
        **build_notification(detected, "image"),
    }


def create_video_writer(output_path, fps, width, height):
    candidates = [
        ("mp4v", output_path),
        ("avc1", output_path),
        ("H264", output_path),
        ("XVID", output_path.replace(".mp4", ".avi")),
    ]

    for codec, candidate_path in candidates:
        writer = cv2.VideoWriter(
            candidate_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, candidate_path
        writer.release()

    return None, None


def process_video(upload_path, filename):
    capture = cv2.VideoCapture(upload_path)

    if not capture.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    if fps <= 0:
        fps = 20.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Could not read the uploaded video dimensions.")

    output_filename = "detected_" + os.path.splitext(filename)[0] + ".mp4"
    output_path = os.path.join(RESULT_FOLDER, output_filename)
    writer, actual_output_path = create_video_writer(output_path, fps, width, height)

    if writer is None or actual_output_path is None:
        capture.release()
        raise ValueError("Could not create the output video.")

    detected = False

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            results = model(frame, conf=0.25)
            frame_detected = detection_contains_garbage(results)
            detected = detected or frame_detected

            annotated_frame = results[0].plot()

            overlay_text = (
                "ALERT: Garbage detected - clean immediately"
                if frame_detected
                else "Pool clean - no garbage detected"
            )
            overlay_color = (0, 0, 255) if frame_detected else (40, 140, 40)

            cv2.putText(
                annotated_frame,
                overlay_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                overlay_color,
                2,
            )

            writer.write(annotated_frame)
    finally:
        capture.release()
        writer.release()

    result_filename = os.path.basename(actual_output_path)
    return {
        "detected": detected,
        "media_result": url_for("static", filename=f"results/{result_filename}"),
        "media_result_type": "video",
        "processed_media_kind": "video",
        **build_notification(detected, "video"),
    }


@app.route("/")
def home():
    return render_landing()


@app.route("/detect-media", methods=["GET", "POST"])
@app.route("/detect-image", methods=["POST"])
def detect_media():
    if request.method == "GET":
        return render_dashboard()

    file = request.files.get("media") or request.files.get("image")

    if file is None or not file.filename:
        return render_dashboard(
            notification_level="alert",
            notification_title="Upload required",
            notification_message="Please upload a swimming pool image or video before starting detection.",
        )

    filename, upload_path, extension = save_upload(file)

    try:
        if extension in IMAGE_EXTENSIONS:
            context = process_image(upload_path, filename)
        elif extension in VIDEO_EXTENSIONS:
            context = process_video(upload_path, filename)
        else:
            return render_dashboard(
                notification_level="alert",
                notification_title="Unsupported file type",
                notification_message="Please upload a supported image or video file.",
            )
    except ValueError as exc:
        return render_dashboard(
            notification_level="alert",
            notification_title="Processing failed",
            notification_message=str(exc),
        )

    return render_dashboard(**context)


def generate_frames():
    camera = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            results = model(frame, conf=0.25)

            detected = detection_contains_garbage(results)
            annotated_frame = results[0].plot()

            if detected:
                cv2.putText(
                    annotated_frame,
                    "ALERT: Garbage Detected in Swimming Pool!",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        camera.release()


@app.route("/video-feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
