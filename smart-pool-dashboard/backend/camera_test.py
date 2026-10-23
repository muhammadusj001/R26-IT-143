"""
Camera helper — find the right CAMERA_SOURCE and preview it live.

Run from smart-pool-dashboard/backend:
    python camera_test.py                         # list cameras + probe indices 0-3
    python camera_test.py 3                       # preview index 3 (e.g. OBS Virtual Camera)
    python camera_test.py droidcam:172.28.11.95   # preview DroidCam over Wi-Fi
    python camera_test.py http://172.28.11.95:4747/video

The preview opens the source through the same CameraManager the dashboard
uses (same backends, reconnect and resize). Press q to quit.
"""

import platform
import sys

import cv2

from core.camera import CameraManager

BACKENDS = [
    ("DSHOW  (DirectShow — best for DroidCam/OBS virtual cams)", cv2.CAP_DSHOW),
    ("MSMF   (Media Foundation — OpenCV default)", cv2.CAP_MSMF),
    ("ANY    (let OpenCV choose)", cv2.CAP_ANY),
]


def list_dshow_devices():
    """Print DirectShow device names with their indices (Windows only)."""
    if platform.system() != "Windows":
        return
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        print("  (Install pygrabber to see camera names: pip install pygrabber)")
        return
    devices = FilterGraph().get_input_devices()
    if not devices:
        print("  No DirectShow cameras found.")
        return
    print("  DirectShow cameras (use the number as CAMERA_SOURCE):")
    for index, name in enumerate(devices):
        print(f"    {index}  {name}")


def probe_indices():
    """Try indices 0-3 on each backend and report which give real frames."""
    working = []
    for index in range(4):
        for name, backend in BACKENDS:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                print(f"  index {index} | {name:<56} : cannot open")
                cap.release()
                continue
            # Opening is not enough — a virtual cam can open but give no frames.
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                print(f"  index {index} | {name:<56} : WORKS  ({w}x{h})")
                working.append((index, name, w, h))
            else:
                print(f"  index {index} | {name:<56} : opens but NO FRAMES")
            cap.release()

    print("=" * 64)
    if working:
        print("  WORKING CAMERAS FOUND:")
        for index, name, w, h in working:
            print(f"    CAMERA_SOURCE={index}   via {name.split('(')[0].strip()}  {w}x{h}")
        print("\n  Preview one with: python camera_test.py <index>")
    else:
        print("  NO WORKING CAMERA FOUND.")
        print("  Checklist:")
        print("    1. Is the DroidCam app open ON THE PHONE?")
        print("    2. For a webcam index: is DroidCam Client / OBS Virtual Camera")
        print("       running ON THE LAPTOP and showing the phone's video?")
        print("    3. Phone and laptop on the SAME Wi-Fi?")
        print("       Or skip the client: python camera_test.py droidcam:<phone-ip>")
        print("    4. Close Zoom / Teams / Camera app (they lock the camera).")


def preview(source):
    camera = CameraManager(source)
    print(f"  Source: {source} -> {camera.source} ({camera.type})")
    if not camera.open():
        sys.exit(1)

    printed_size = False
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                camera.reconnect()
                continue
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if not printed_size:
                h, w = frame.shape[:2]
                native = camera.native_size
                print(f"  Resolution: native {native[0]}x{native[1]}, shown {w}x{h}")
                printed_size = True

            status = camera.status()
            label = f"{status['type']} | {status['backend']} | {status['fps']:.1f} fps"
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Camera test (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    print("=" * 64)
    print("  CAMERA HELPER")
    print("=" * 64)
    if len(sys.argv) > 1:
        preview(sys.argv[1])
        return
    list_dshow_devices()
    print("-" * 64)
    probe_indices()
    print("=" * 64)


if __name__ == "__main__":
    main()
