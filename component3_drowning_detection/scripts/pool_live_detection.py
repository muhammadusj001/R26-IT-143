# ═══════════════════════════════════════════════════════
# AI-BASED SWIMMING POOL MONITORING SYSTEM
# Module: Swimmer Behaviour and Drowning Detection
# Mode: Real-Time Live Pool Detection
# Project: R26-IT-143
# ═══════════════════════════════════════════════════════

import cv2
import time
import os
from ultralytics import YOLO
from datetime import datetime

# ══════════════════════════════════════════════════════
# CONFIGURATION — UPDATE PHONE IP BEFORE RUNNING
# ══════════════════════════════════════════════════════
MODEL_PATH = r'C:\Project\R26-IT-143\component3_drowning_detection\models\best.pt'
OUTPUT_DIR = r'C:\Project\R26-IT-143\component3_drowning_detection\results\live_tests'
CONF_THRESH     = 0.5
DROWNING_FRAMES = 3

# !! CHANGE THIS TO YOUR PHONE IP !!
PHONE_IP   = '192.168.1.5'
PHONE_PORT = '8080'
STREAM_URL = f'http://{PHONE_IP}:{PHONE_PORT}/video'
# ══════════════════════════════════════════════════════

COLORS = {
    'Drowning':            (0, 0, 255),
    'Person out of water': (0, 165, 255),
    'Swimming':            (0, 255, 0),
}

alert_log  = []
start_time = time.time()
fps        = 0

def log_alert(frame_num, count, timestamp):
    alert_log.append({
        'frame': frame_num,
        'count': count,
        'time':  timestamp
    })
    print(f"DROWNING ALERT | "
          f"Time: {timestamp} | "
          f"Frame: {frame_num} | "
          f"Count: {count}")

def save_alert_log():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(OUTPUT_DIR,
                            f'alert_log_{ts}.txt')
    with open(log_path, 'w') as f:
        f.write("DROWNING DETECTION ALERT LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Total Alerts: {len(alert_log)}\n")
        f.write("=" * 50 + "\n\n")
        for a in alert_log:
            f.write(f"Time: {a['time']} | "
                   f"Frame: {a['frame']} | "
                   f"Count: {a['count']}\n")
    print(f"Alert log saved: {log_path}")

def play_beep():
    try:
        import winsound
        winsound.Beep(1000, 500)
    except:
        pass

def draw_detections(frame, results, model):
    counts = {
        'Drowning': 0,
        'Person out of water': 0,
        'Swimming': 0
    }
    for result in results:
        for box in result.boxes:
            cls_id   = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist())
            color = COLORS.get(cls_name,
                               (255, 255, 255))
            thick = 4 if cls_name == 'Drowning' \
                    else 2
            cv2.rectangle(frame,
                         (x1, y1), (x2, y2),
                         color, thick)
            label = f'{cls_name} {conf:.0%}'
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, 2)
            cv2.rectangle(frame,
                         (x1, y1-th-12),
                         (x1+tw+12, y1),
                         color, -1)
            cv2.putText(frame, label,
                       (x1+6, y1-6),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
            if cls_name in counts:
                counts[cls_name] += 1
    return frame, counts

def draw_alert_banner(frame, counts):
    if counts['Drowning'] > 0:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0),
                     (w, 100), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.75,
                       frame, 0.25, 0, frame)
        cv2.putText(
            frame,
            f'DROWNING DETECTED — '
            f'{counts["Drowning"]} PERSON(S)',
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3, (255, 255, 255), 3)
        if int(time.time() * 2) % 2 == 0:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0),
                         (w-1, h-1),
                         (0, 0, 255), 10)
    return frame

def draw_info_panel(frame, fps, frame_num,
                    counts, recording,
                    total_alerts, elapsed):
    h, w = frame.shape[:2]
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    session = f'{mins:02d}:{secs:02d}'

    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-150),
                 (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8,
                   frame, 0.2, 0, frame)

    # Title
    cv2.putText(frame,
               'AI POOL MONITORING SYSTEM',
               (10, h-130),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (0, 200, 255), 2)

    cv2.line(frame, (0, h-115),
             (w, h-115), (80, 80, 80), 1)

    # Left column
    left = [
        f'FPS:     {fps:.1f}',
        f'Frame:   {frame_num}',
        f'Session: {session}',
        f'Alerts:  {total_alerts}',
    ]
    for i, s in enumerate(left):
        cv2.putText(frame, s,
                   (10, h-95+i*24),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 255), 2)

    # Right column
    right = [
        f'Swimming:  {counts["Swimming"]}',
        f'Drowning:  {counts["Drowning"]}',
        f'Out water: {counts["Person out of water"]}',
        f'{"REC" if recording else "R=Record"}',
    ]
    for i, s in enumerate(right):
        c = (0, 0, 255) \
            if 'Drowning' in s \
            and counts.get('Drowning', 0) > 0 \
            else (0, 255, 0) \
            if 'REC' in s \
            else (200, 200, 200)
        cv2.putText(frame, s,
                   (w//2, h-95+i*24),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, c, 2)

    # Controls
    cv2.putText(frame,
               'Q=Quit  S=Screenshot  R=Record',
               (10, h-8),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (120, 120, 120), 1)
    return frame

def run():
    print("=" * 60)
    print("  AI SWIMMING POOL MONITORING SYSTEM")
    print("  Swimmer Behaviour & Drowning Detection")
    print("=" * 60)
    print(f"  Camera: {STREAM_URL}")
    print(f"  Model:  {MODEL_PATH}")
    print("=" * 60)

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Make sure best.pt is in models folder")
        return

    # Load model
    print("\nLoading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully")

    # Connect to phone camera
    print(f"\nConnecting to phone camera...")
    print(f"URL: {STREAM_URL}")
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("\nCANNOT CONNECT TO PHONE CAMERA")
        print("\nTroubleshooting:")
        print("1. Install IP Webcam from Play Store")
        print("2. Open app and tap Start server")
        print(f"3. Note IP address on phone screen")
        print(f"4. Update PHONE_IP in script")
        print(f"5. Test in browser: {STREAM_URL}")
        print("\nBoth phone and PC must be on same WiFi")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Connected! Resolution: {w}x{h}")
    print("\nControls:")
    print("  Q = Quit")
    print("  S = Screenshot")
    print("  R = Start/Stop Recording")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    writer       = None
    recording    = False
    frame_num    = 0
    start_time   = time.time()
    total_alerts = 0
    consec_drown = 0
    last_beep    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Connection lost, reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        frame_num += 1

        # Run detection
        results = model(frame,
                       conf=CONF_THRESH,
                       verbose=False)

        # Draw detections
        frame, counts = draw_detections(
            frame, results, model)

        # Drowning logic
        if counts['Drowning'] > 0:
            consec_drown += 1
            if consec_drown >= DROWNING_FRAMES:
                total_alerts += 1
                ts = datetime.now().strftime(
                    '%H:%M:%S')
                log_alert(frame_num,
                         counts['Drowning'], ts)

                # Beep every 2 seconds
                now = time.time()
                if now - last_beep > 2:
                    play_beep()
                    last_beep = now

                # Auto screenshot
                ss = os.path.join(
                    OUTPUT_DIR,
                    f'drowning_{frame_num}.jpg')
                cv2.imwrite(ss, frame.copy())
        else:
            consec_drown = 0

        # Draw overlays
        frame = draw_alert_banner(frame, counts)
        elapsed = time.time() - start_time
        fps = frame_num / elapsed \
              if elapsed > 0 else 0
        frame = draw_info_panel(
            frame, fps, frame_num,
            counts, recording,
            total_alerts, elapsed)

        # Save if recording
        if recording and writer:
            writer.write(frame)

        # Show
        cv2.imshow(
            'AI Pool Monitoring System', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nStopping...")
            break

        elif key == ord('s'):
            ts = datetime.now().strftime('%H%M%S')
            path = os.path.join(
                OUTPUT_DIR,
                f'screenshot_{ts}.jpg')
            cv2.imwrite(path, frame)
            print(f"Screenshot: {path}")

        elif key == ord('r'):
            if not recording:
                ts = datetime.now().strftime(
                    '%Y%m%d_%H%M%S')
                rec = os.path.join(
                    OUTPUT_DIR,
                    f'session_{ts}.mp4')
                fourcc = cv2.VideoWriter_fourcc(
                    *'mp4v')
                writer = cv2.VideoWriter(
                    rec, fourcc, 20, (w, h))
                recording = True
                print(f"Recording: {rec}")
            else:
                recording = False
                if writer:
                    writer.release()
                    writer = None
                print("Recording stopped")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if alert_log:
        save_alert_log()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  SESSION COMPLETE")
    print("=" * 60)
    print(f"  Duration:  {elapsed/60:.1f} minutes")
    print(f"  Frames:    {frame_num}")
    print(f"  Avg FPS:   {fps:.1f}")
    print(f"  Alerts:    {total_alerts}")
    print(f"  Saved to:  {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    run()