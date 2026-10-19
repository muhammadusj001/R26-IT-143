# Plugging Component 4 v2 into smart-pool-dashboard

Team folder (from the project chat):
`C:\Users\muham\Desktop\R26-IT-143-multirepo-system`

The dashboard runs the copy under
`smart-pool-dashboard\components\component4-garbage-detection\` — that is
the one that matters. Keep the top-level `component4-garbage-detection\`
repo identical (same files) so the member repo and the dashboard never drift.

v2 keeps the same API (`GarbageDetector(...)`, `.load()`,
`.process(frame) -> (frame, result, alert)`), so the dashboard keeps working
with ZERO edits. The 4 small edits below make it work *well*.

## 0. Safety branch
```powershell
cd C:\Users\muham\Desktop\R26-IT-143-multirepo-system
git add -A
git commit -m "checkpoint before C4 v2"
git checkout -b c4-v2
```

## 1. Copy the files (both places)
Unzip `component4-garbage-detection-v2.zip` and copy its contents over:
- `component4-garbage-detection\`
- `smart-pool-dashboard\components\component4-garbage-detection\`

Do NOT delete `models\swimming_pool_garbage_yolo.pt`. In the dashboard copy,
if the pose file is `models\pose_landmarker_lite.task.disabled`, rename it
back to `pose_landmarker_lite.task` (Windows). On a Mac where MediaPipe
crashes, keep it and set `DISABLE_INTENT_ANALYSIS=1` instead.

## 2. backend\app.py — four edits

**a) Construction** (where `garbage = GarbageDetector(...)` is):
```python
garbage = GarbageDetector(model_path=settings.GARBAGE_MODEL_PATH,
                          conf_threshold=settings.GARBAGE_CONF_THRESHOLD)
```
Leave it as it is. If it passes `imgsz=settings.DETECTION_IMGSZ`, that's fine:
v2 never goes below 640 for garbage. The people model (`models\yolov8n.pt`)
and the pose model are found automatically inside the component folder, so
source + intent now work in the dashboard without passing anything.
Optionally print `garbage.status_summary()` in the start-up banner.

**b) Clean frame** — in `vision_loop`, right after the frame is read and the
frame-skip check passes, add ONE line:
```python
raw = frame.copy()   # untouched frame for Component 4 (C1/C3 draw on `frame`)
```
and change the garbage call to:
```python
frame, garbage_result, garbage_alert = garbage.process(frame, clean_frame=raw)
```
Why: crowd and drowning draw boxes and text on `frame` first. Without the
clean copy, C4's pose, splash and pool-floor checks would analyse those
drawings.

**c) Alert severity** — replace
`state.add_alert("garbage", garbage_alert, "warning")` with:
```python
level = "danger" if garbage_alert.startswith("[HIGH]") else "warning"
state.add_alert("garbage", garbage_alert, level)
```
v2 sends an alert ONCE per event (not every frame). HIGH = thrown on
purpose, or an object sank with a splash.

**d) Shutdown** — after the line that creates `garbage`, add:
```python
import atexit
atexit.register(garbage.close)   # releases MediaPipe cleanly
```

## 3. Stagger (CPU optimisation) and intent
The dashboard runs crowd, drowning and garbage on alternate processed frames.
Garbage detection is fine with that. Throw detection needs the arm seen
about 8+ times per second; if C4's turn comes less often, intent becomes
less reliable. Option: let garbage run on EVERY processed frame when
`C4_FULL_RATE=1` (see the Claude Code prompt) and use it for the intent demo.

## 4. requirements.txt (smart-pool-dashboard)
Add:
```
mediapipe>=0.10.14
```
(scikit-learn and joblib are already there.)

## 5. Optional dashboard fields
`state.py` already has `source`, `intent`, `intent_score`,
`intent_confidence`, `risk_level`. v2 also returns `intent_reason`,
`events`, `people_count`, `intent_status`, `fps`. Showing `intent_reason`
under the intent line on the garbage card ("fast arm swing, object launched
sideways/up") is a good viva touch. Never rename existing element ids.

### v2.3 fields (panel feedback: "what is an event?", tiny + sunk garbage)
| Field | Meaning |
|---|---|
| `surface_count` / `underwater_count` | garbage now floating / now under the surface |
| `underwater_labels` | labels of the submerged objects |
| `human_action` | live text: `THROWING (person #2)`, `arm moving`, `standing / still`, `swimming`, `no person in view` |
| `actions` | per person: `{person_id, action, score}` |
| `last_throw_sec_ago` | seconds since the last throwing motion (or null) |
| `event_feed` | last 10 incidents in plain language: `time, title, explanation, risk, object, position (surface/underwater), source, intent, intent_text, reason, message, snapshot` |
| `event_legend` | event type -> short title (for an "i" legend) |

Backend routes added in app.py: `/api/garbage/events` (feed as JSON) and
`/api/garbage/snapshot/<name>` (the photo saved for an incident).

## 6. Pool area for C4 (same camera)
Close the dashboard first (the camera can only be open in one program), then:
```powershell
cd smart-pool-dashboard\components\component4-garbage-detection
python tools\draw_pool_roi.py --source 1 --width 640 --height 480
```
640x480 = the dashboard's FRAME_WIDTH/HEIGHT. C4 prints a warning if the
sizes don't match.

## 7. Check, then commit
```powershell
cd smart-pool-dashboard\components\component4-garbage-detection
python tests\test_logic.py                  # 5 x PASS
cd ..\..\backend
python app.py                               # banner shows C4 status
```
Then run the team's 27-check system test again. If all pass:
```powershell
git add -A
git commit -m "feat: Component 4 v2 - live tracking, continuous intent, sinking detection"
git checkout development
git merge c4-v2
git push origin development
```
If anything breaks: `git checkout development && git branch -D c4-v2`.

---

## Claude Code prompt (does steps 1-5 on your laptop)

Unzip the v2 zip to `C:\Users\muham\Downloads\c4v2\` first, then paste:

```
I have Component 4 v2 unzipped at C:\Users\muham\Downloads\c4v2\component4-garbage-detection\.
Work in C:\Users\muham\Desktop\R26-IT-143-multirepo-system on a new branch c4-v2.

1. Copy every file from the v2 folder over BOTH:
   - component4-garbage-detection\
   - smart-pool-dashboard\components\component4-garbage-detection\
   Never delete models\swimming_pool_garbage_yolo.pt. If the dashboard copy has
   models\pose_landmarker_lite.task.disabled, rename it to pose_landmarker_lite.task.
   Confirm both copies are byte-identical afterwards (ignore models\, ml\runs\, events\).

2. In smart-pool-dashboard\backend\app.py, inside vision_loop:
   a) right after the frame is read and the frame-skip check passes, add
      raw = frame.copy()  with a comment that it is the untouched frame for Component 4
   b) change the garbage call to garbage.process(frame, clean_frame=raw)
   c) when garbage_alert is set, use level "danger" if garbage_alert starts with "[HIGH]",
      else "warning", in state.add_alert("garbage", garbage_alert, level)
   d) in the stagger logic, if os.getenv("C4_FULL_RATE") == "1", call garbage.process on
      EVERY processed frame (crowd and drowning keep alternating). Default stays as now.
   e) after garbage is created: import atexit; atexit.register(garbage.close)
   f) print garbage.status_summary() in the start-up banner
   Do not change crowd, drowning, water quality, thresholds or the decision engine.

3. Add mediapipe>=0.10.14 to smart-pool-dashboard\requirements.txt.

4. In frontend-vanilla: on the garbage card, under the existing intent line, add a small
   line with a NEW id "garbageReason" showing state.garbage.intent_reason when garbage is
   detected. Update dashboard.js to fill it. Do not rename or remove any existing id.

5. Run component4-garbage-detection\tests\test_logic.py (expect 5 PASS) and the 27-check
   system test. Report results. Do not commit.
```
