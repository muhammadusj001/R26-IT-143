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


def draw_hud(frame, status_lines, event_lines=()):
    """Semi-transparent status panel (top-left) + recent events (bottom)."""
    h, w = frame.shape[:2]
    scale = max(0.45, min(0.6, w / 2200))
    line_h = int(26 * scale / 0.6)
    panel_h = 10 + line_h * len(status_lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    if event_lines:
        ev_h = 10 + line_h * len(event_lines)
        cv2.rectangle(overlay, (0, h - ev_h), (w, h), (0, 0, 120), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, text in enumerate(status_lines):
        cv2.putText(frame, text, (10, 6 + line_h * (i + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
    if event_lines:
        y0 = h - (10 + line_h * len(event_lines))
        for i, text in enumerate(event_lines):
            cv2.putText(frame, text, (10, y0 + line_h * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_alert_banner(frame, text, bgr=(0, 0, 180)):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), bgr, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (255, 255, 255), 3)
    return frame
