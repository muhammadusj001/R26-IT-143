# ═══════════════════════════════════════════════════
# AI SWIMMING POOL MONITORING SYSTEM
# Component 3: Drowning Detection Dashboard
# Backend: Flask + SocketIO
# Webcam Mode: Iriun Webcam
# ═══════════════════════════════════════════════════

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import threading
import time
import base64
from ultralytics import YOLO
from datetime import datetime

# ═══════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════
app = Flask(
    __name__,
    static_folder='static',
    template_folder='templates'
)

app.config['SECRET_KEY'] = 'drowning_detection_2026'

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading'
)

# ═══════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════

MODEL_PATH = r'C:\Project\R26-IT-143\component3_drowning_detection\models\best.pt'

# CAMERA INDEX
# 0 = default webcam
# 1 = Iriun Webcam (most common)
# 2 = external webcam
CAMERA_INDEX = 1

CONF_THRESH = 0.35
DROWNING_FRAMES = 3

# ═══════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════

detection_state = {
    'running': False,
    'swimming': 0,
    'drowning': 0,
    'out_of_water': 0,
    'total_alerts': 0,
    'fps': 0.0,
    'status': 'SAFE',
    'session_time': '00:00',
    'alerts': [],
    'frame_num': 0
}

COLORS = {
    'Drowning': (0, 0, 255),
    'Person out of water': (0, 165, 255),
    'Swimming': (0, 255, 0),
}

model = None
camera_thread = None
cap = None

# ═══════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════

def load_model():
    global model

    print("Loading YOLOv8 model...")

    try:
        model = YOLO(MODEL_PATH)
        print("✓ Model loaded successfully")
        return True

    except Exception as e:
        print(f"✗ Model error: {e}")
        return False

# ═══════════════════════════════════════════════════
# DETECTION THREAD
# ═══════════════════════════════════════════════════

def detection_loop():

    global cap, detection_state

    print(f"Connecting to webcam index: {CAMERA_INDEX}")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Reduce lag
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)

    if not cap.isOpened():

        print("✗ Cannot connect to webcam")

        detection_state['running'] = False

        socketio.emit('camera_error', {
            'message': f'Cannot connect to webcam index {CAMERA_INDEX}'
        })

        return

    print("✓ Webcam connected")

    detection_state['running'] = True

    start_time = time.time()
    consec_drown = 0
    frame_num = 0

    while detection_state['running']:

        ret, frame = cap.read()

        if not ret:

            print("Reconnecting webcam...")

            cap.release()
            time.sleep(2)

            cap = cv2.VideoCapture(CAMERA_INDEX)

            continue

        frame_num += 1

        detection_state['frame_num'] = frame_num

        # Skip frames for better FPS
        if frame_num % 2 != 0:
            continue

        # ═══════════════════════════════════════
        # YOLO DETECTION
        # ═══════════════════════════════════════

        results = model(
            frame,
            imgsz=416,
            conf=CONF_THRESH,
            verbose=False
        )

        counts = {
            'Drowning': 0,
            'Person out of water': 0,
            'Swimming': 0
        }

        # ═══════════════════════════════════════
        # DRAW DETECTIONS
        # ═══════════════════════════════════════

        for result in results:

            for box in result.boxes:

                cls_id = int(box.cls[0])

                cls_name = model.names[cls_id]

                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                color = COLORS.get(
                    cls_name,
                    (255, 255, 255)
                )

                thick = 4 if cls_name == 'Drowning' else 2

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thick
                )

                label = f'{cls_name} {conf:.0%}'

                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2
                )

                cv2.rectangle(
                    frame,
                    (x1, y1 - th - 12),
                    (x1 + tw + 12, y1),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                if cls_name in counts:
                    counts[cls_name] += 1

        # ═══════════════════════════════════════
        # DROWNING ALERT LOGIC
        # ═══════════════════════════════════════

        if counts['Drowning'] > 0:

            consec_drown += 1

            if consec_drown >= DROWNING_FRAMES:

                detection_state['total_alerts'] += 1

                ts = datetime.now().strftime('%H:%M:%S')

                alert = {
                    'time': ts,
                    'count': counts['Drowning'],
                    'frame': frame_num
                }

                detection_state['alerts'].insert(0, alert)

                detection_state['alerts'] = \
                    detection_state['alerts'][:10]

                detection_state['status'] = 'DANGER'

                # Alert banner
                h, w = frame.shape[:2]

                overlay = frame.copy()

                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (w, 90),
                    (0, 0, 180),
                    -1
                )

                cv2.addWeighted(
                    overlay,
                    0.75,
                    frame,
                    0.25,
                    0,
                    frame
                )

                cv2.putText(
                    frame,
                    'DROWNING DETECTED!',
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.8,
                    (255, 255, 255),
                    3
                )

                socketio.emit(
                    'drowning_alert',
                    alert
                )

        else:

            consec_drown = 0

            if detection_state['status'] == 'DANGER':
                detection_state['status'] = 'SAFE'

        # ═══════════════════════════════════════
        # UPDATE DASHBOARD STATE
        # ═══════════════════════════════════════

        elapsed = time.time() - start_time

        fps = frame_num / elapsed if elapsed > 0 else 0

        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        detection_state.update({
            'swimming': counts['Swimming'],
            'drowning': counts['Drowning'],
            'out_of_water': counts['Person out of water'],
            'fps': round(fps, 1),
            'session_time': f'{mins:02d}:{secs:02d}',
        })

        # ═══════════════════════════════════════
        # ENCODE FRAME
        # ═══════════════════════════════════════

        _, buffer = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        frame_b64 = base64.b64encode(
            buffer
        ).decode('utf-8')

        # ═══════════════════════════════════════
        # SEND TO DASHBOARD
        # ═══════════════════════════════════════

        socketio.emit('frame_update', {
            'frame': frame_b64,
            'state': detection_state.copy()
        })

        time.sleep(0.02)

    if cap:
        cap.release()

    print("Detection stopped")

# ═══════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():

    return {
        'model_loaded': model is not None,
        'running': detection_state['running'],
        'camera_index': CAMERA_INDEX
    }

# ═══════════════════════════════════════════════════
# SOCKETIO EVENTS
# ═══════════════════════════════════════════════════

@socketio.on('connect')
def on_connect():

    print("Dashboard connected")

    emit(
        'initial_state',
        detection_state.copy()
    )

@socketio.on('start_detection')
def start_detection():

    global camera_thread

    if not detection_state['running']:

        camera_thread = threading.Thread(
            target=detection_loop,
            daemon=True
        )

        camera_thread.start()

        emit('detection_started', {
            'message': 'Detection started'
        })

@socketio.on('stop_detection')
def stop_detection():

    detection_state['running'] = False

    emit('detection_stopped', {
        'message': 'Detection stopped'
    })

@socketio.on('disconnect')
def on_disconnect():
    print("Dashboard disconnected")

# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':

    print("=" * 60)
    print(" AI POOL MONITORING — DASHBOARD ")
    print("=" * 60)

    print(f" Model:  {MODEL_PATH}")
    print(f" Webcam Index: {CAMERA_INDEX}")

    print("=" * 60)

    if load_model():

        print("\n✓ Starting dashboard...")
        print(" Open browser: http://localhost:5000")
        print(" Press Ctrl+C to stop")

        print("=" * 60)

        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )

    else:

        print("✗ Failed to load model")
        print("Check MODEL_PATH in app.py")