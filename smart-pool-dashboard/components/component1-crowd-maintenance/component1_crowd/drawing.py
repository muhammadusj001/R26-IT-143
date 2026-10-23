"""Bounding-box drawing helper (deliberately duplicated in each vision
component repo so every repo is fully standalone for its own viva demo).
Visual style preserved from Component 3's original scripts."""

import cv2


def draw_box(frame, x1, y1, x2, y2, label, color, thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 12, y1), color, -1)
    cv2.putText(frame, label, (x1 + 6, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_alert_banner(frame, text, bgr=(0, 0, 180)):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), bgr, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (255, 255, 255), 3)
    return frame
