"""
Camera diagnostic — finds which camera index + backend actually works.

Run this from smart-pool-dashboard/backend:
    python camera_test.py

It tests indices 0-3 with both Windows backends and tells you exactly
which CAMERA_SOURCE value to use.
"""

import cv2

BACKENDS = [
    ("DSHOW  (DirectShow — best for Iriun/OBS virtual cams)", cv2.CAP_DSHOW),
    ("MSMF   (Media Foundation — OpenCV default)", cv2.CAP_MSMF),
    ("ANY    (let OpenCV choose)", cv2.CAP_ANY),
]

print("=" * 64)
print("  CAMERA DIAGNOSTIC")
print("=" * 64)

working = []

for index in range(4):
    for name, backend in BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            print(f"  index {index} | {name:<52} : cannot open")
            cap.release()
            continue

        # Opening is not enough — a virtual cam can open but give no frames.
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  index {index} | {name:<52} : WORKS  ({w}x{h})")
            working.append((index, name, w, h))
        else:
            print(f"  index {index} | {name:<52} : opens but NO FRAMES")
        cap.release()

print("=" * 64)
if working:
    print("  WORKING CAMERAS FOUND:")
    for index, name, w, h in working:
        print(f"    CAMERA_SOURCE=\"{index}\"   via {name.split('(')[0].strip()}  {w}x{h}")
    print()
    print("  Use the lowest-numbered DSHOW result that shows your phone.")
    print("  If several work, try each and see which one is the phone.")
else:
    print("  NO WORKING CAMERA FOUND.")
    print("  Checklist:")
    print("    1. Is the Iriun app open ON THE PHONE?")
    print("    2. Is the Iriun Webcam program open ON THE LAPTOP,")
    print("       and does IT show your phone's video?")
    print("    3. Phone and laptop on the SAME WiFi?")
    print("    4. Close Zoom / Teams / Camera app (they lock the camera).")
print("=" * 64)
