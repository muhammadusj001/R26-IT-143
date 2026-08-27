"""
R26-IT-143 — Full system verification script (viva prep).

Runs a battery of checks against the actual running system — real
imports, real model loading, a real 10-second pipeline run, a real PDF
generation — and prints PASS/FAIL for each with details, plus a final
summary table. This script is READ-ONLY with respect to the rest of the
codebase: it never edits any project file. It does write one temporary
PDF (deleted at the end) as part of the report-generator check.

Run from anywhere:
    python smart-pool-dashboard/backend/system_check.py
(Python adds this script's own directory to sys.path[0], so the
existing `from config import settings` / `from core... import ...`
style imports used by app.py resolve exactly the same way here.)

Exits 0 if every check passed, 1 otherwise.
"""

import os
import re
import sys
import tempfile
import threading
import time

# Windows consoles often default to a legacy codepage (cp1252) that can't
# encode the PASS/FAIL marks below; force UTF-8 output so this runs the
# same everywhere instead of crashing on print().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import traceback

# ── Result tracking ─────────────────────────────────────────
results = []  # list of dicts: section, check, status, notes


def run_check(section, name, fn):
    """Run one check function, catch anything it raises, record + print
    the result immediately, and never let a single failing check take
    down the rest of the script."""
    try:
        notes = fn()
        status = "PASS"
        notes = notes or ""
    except AssertionError as e:
        status = "FAIL"
        notes = str(e) or "assertion failed"
    except Exception as e:  # noqa: BLE001
        status = "FAIL"
        notes = f"{type(e).__name__}: {e}"
    results.append({"section": section, "check": name, "status": status, "notes": notes})
    mark = "✅ PASS" if status == "PASS" else "❌ FAIL"
    print(f"  [{mark}] {name}")
    if notes:
        for line in str(notes).splitlines():
            print(f"           {line}")
    return status == "PASS"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & STRUCTURE
# ══════════════════════════════════════════════════════════════
section("SECTION 1 — IMPORTS & STRUCTURE")

app = None  # populated by check #2, reused by every later section


def check_app_import():
    global app
    import app as app_module
    app = app_module
    return f"app.py imported OK (Flask app: {app.app.name})"


run_check("1", "2. app.py imports without error", check_app_import)


def check_component_import(pkg_name):
    def _fn():
        mod = __import__(pkg_name)
        return f"{pkg_name} imported OK from {mod.__file__}"
    return _fn


for pkg in ("component1_crowd", "component2_water", "component3_drowning", "component4_garbage"):
    run_check("1", f"1. {pkg} imports cleanly", check_component_import(pkg))


def check_files_exist():
    base = app.settings.BASE_DIR  # smart-pool-dashboard/
    root = base.parent            # project root
    expected = [
        # Component detectors (the mirrored copies app.py actually imports)
        base / "components" / "component1-crowd-maintenance" / "component1_crowd" / "detector.py",
        base / "components" / "component2-water-quality" / "component2_water" / "predictor.py",
        base / "components" / "component3-drowning-detection" / "component3_drowning" / "detector.py",
        base / "components" / "component4-garbage-detection" / "component4_garbage" / "detector.py",
        # Original per-member standalone copies
        root / "component1-crowd-maintenance" / "component1_crowd" / "detector.py",
        root / "component2-water-quality" / "component2_water" / "predictor.py",
        root / "component3-drowning-detection" / "component3_drowning" / "detector.py",
        root / "component4-garbage-detection" / "component4_garbage" / "detector.py",
        # Shared dashboard backend
        base / "backend" / "app.py",
        base / "backend" / "config" / "settings.py",
        base / "backend" / "core" / "state.py",
        base / "backend" / "core" / "camera.py",
        base / "backend" / "core" / "decision_engine.py",
        base / "backend" / "core" / "water_monitor.py",
        base / "backend" / "core" / "session_tracker.py",
        base / "backend" / "core" / "report_generator.py",
        # Frontend
        base / "frontend-vanilla" / "templates" / "index.html",
        base / "frontend-vanilla" / "static" / "css" / "style.css",
        base / "frontend-vanilla" / "static" / "js" / "dashboard.js",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    assert not missing, "Missing files:\n" + "\n".join(missing)
    return f"all {len(expected)} expected files present"


run_check("1", "3. All expected files exist", check_files_exist)


# ══════════════════════════════════════════════════════════════
# SECTION 2 — MODEL FILES
# ══════════════════════════════════════════════════════════════
section("SECTION 2 — MODEL FILES")


def _size_mb(path):
    return f"{path.stat().st_size / (1024 * 1024):.2f} MB"


def check_crowd_model_files():
    s = app.settings
    lines = []
    configured = s.CROWD_MODEL_PATH
    assert configured.exists(), f"configured CROWD_MODEL_PATH missing: {configured}"
    lines.append(f"configured (best_swimmer_model.pt): {configured} — {_size_mb(configured)}")

    yolo11 = configured.parent / "yolo11m.pt"
    if yolo11.exists():
        lines.append(f"yolo11m.pt (stock COCO alt.): {yolo11} — {_size_mb(yolo11)}")
    else:
        lines.append("yolo11m.pt: not present (optional stock-COCO alternative model)")
    return "\n".join(lines)


run_check("2", "4a. Crowd model file(s) exist", check_crowd_model_files)


def check_drowning_model_file():
    p = app.settings.DROWNING_MODEL_PATH
    assert p.exists(), f"missing: {p}"
    return f"{p} — {_size_mb(p)}"


run_check("2", "4b. Drowning model file exists", check_drowning_model_file)


def check_garbage_model_file():
    p = app.settings.GARBAGE_MODEL_PATH
    assert p.exists(), f"missing: {p}"
    return f"{p} — {_size_mb(p)}"


run_check("2", "4c. Garbage model file exists", check_garbage_model_file)


def check_water_model_files():
    d = app.settings.WATER_MODEL_DIR
    files = {
        "model": d / "water_quality_model.pkl",
        "scaler": d / "scaler.pkl",
        "label_encoder": d / "label_encoder.pkl",
    }
    missing = [f"{k}: {v}" for k, v in files.items() if not v.exists()]
    assert not missing, "missing:\n" + "\n".join(missing)
    return "\n".join(f"{k}: {v} — {_size_mb(v)}" for k, v in files.items())


run_check("2", "4d. Water quality model files exist", check_water_model_files)


# ══════════════════════════════════════════════════════════════
# SECTION 3 — MODEL LOADING
# ══════════════════════════════════════════════════════════════
section("SECTION 3 — MODEL LOADING")


def check_crowd_load():
    ok = app.crowd.load()
    assert ok, f"load() returned False, model_status={app.crowd.model_status!r}"
    names = app.crowd.model.names
    sample = list(names.items())[:8]
    return (f"model_status={app.crowd.model_status!r}  is_coco_model={app.crowd.is_coco_model}\n"
            f"class count={len(names)}  sample classes={sample}")


run_check("3", "5a. Crowd model loads", check_crowd_load)


def check_drowning_load():
    ok = app.drowning.load()
    assert ok, f"load() returned False, model_status={app.drowning.model_status!r}"
    names = app.drowning.model.names
    return f"model_status={app.drowning.model_status!r}\nclasses={list(names.values())}"


run_check("3", "5b. Drowning model loads", check_drowning_load)


def check_garbage_load():
    ok = app.garbage.load()
    assert ok, f"load() returned False, model_status={app.garbage.model_status!r}"
    names = app.garbage.model.names
    return f"model_status={app.garbage.model_status!r}\nclasses={list(names.values())}"


run_check("3", "5c. Garbage model loads", check_garbage_load)


def check_water_load():
    ok = app.water.predictor.load()
    assert ok, f"load() returned False, model_status={app.water.predictor.model_status!r}"
    classes = list(app.water.predictor.label_encoder.classes_)
    return f"model_status={app.water.predictor.model_status!r}\nclasses={classes}"


run_check("3", "5d. Water quality model loads", check_water_load)


# ══════════════════════════════════════════════════════════════
# SECTION 4 — LOGIC UNIT TESTS
# ══════════════════════════════════════════════════════════════
section("SECTION 4 — LOGIC UNIT TESTS")


def check_bather_load():
    from component1_crowd.bather_load import BatherLoadCalculator
    calc = BatherLoadCalculator()
    for _ in range(720):
        calc.add_reading(5, interval_seconds=5)
    load = calc.get_current_load()
    assert abs(load - 5.0) < 1e-9, f"expected 5.0 person-hours, got {load}"
    return f"720 readings x 5 swimmers x 5s -> {load} person-hours"


run_check("4", "6. BatherLoadCalculator (720x5 swimmers @5s = 5.0 ph)", check_bather_load)


def check_scheduler_thresholds():
    from component1_crowd.scheduler import MaintenanceScheduler
    sched = MaintenanceScheduler()
    lines = []
    tiers_seen = set()

    sched.update_load(25)
    recs25 = {r["action"]: r["priority"] for r in sched.get_recommendations()}
    assert "Chlorine Dose" in recs25 and recs25["Chlorine Dose"].startswith("MEDIUM"), recs25
    assert "Filter Backwash" not in recs25 and "Skimmer Clean" not in recs25, recs25
    tiers_seen.update(p.split()[0] for p in recs25.values())
    lines.append(f"load=25  -> {recs25}")

    sched.update_load(45)
    recs45 = {r["action"]: r["priority"] for r in sched.get_recommendations()}
    assert recs45["Chlorine Dose"].startswith("CRITICAL"), recs45
    assert recs45["Skimmer Clean"].startswith("HIGH"), recs45
    assert "Filter Backwash" not in recs45, recs45
    tiers_seen.update(p.split()[0] for p in recs45.values())
    lines.append(f"load=45  -> {recs45}")

    sched.update_load(105)
    recs105 = {r["action"]: r["priority"] for r in sched.get_recommendations()}
    for action in ("Chlorine Dose", "Filter Backwash", "Skimmer Clean"):
        assert recs105[action].startswith("CRITICAL"), recs105
    assert recs105["Shock Treatment"].startswith("MEDIUM"), recs105
    assert "Deep Clean" not in recs105, recs105
    tiers_seen.update(p.split()[0] for p in recs105.values())
    lines.append(f"load=105 -> {recs105}")

    assert tiers_seen == {"MEDIUM", "HIGH", "CRITICAL"}, f"tiers seen: {tiers_seen}"
    lines.append("all 3 priority tiers (MEDIUM/HIGH/CRITICAL) observed across the 3 loads")
    return "\n".join(lines)


run_check("4", "7. MaintenanceScheduler thresholds @ 25/45/105", check_scheduler_thresholds)


def check_density_boundaries():
    from component1_crowd.detector import CrowdDetector
    det = CrowdDetector(pool_area_m2=100, area_per_bather_m2=10)  # max_capacity = 10
    assert det.max_capacity == 10, f"expected max_capacity=10, got {det.max_capacity}"
    cases = {4: "LOW", 5: "MODERATE", 7: "MODERATE", 8: "HIGH", 9: "HIGH",
             10: "OVER CAPACITY", 12: "OVER CAPACITY"}
    for count, expected in cases.items():
        actual = det.density_level(count)
        assert actual == expected, f"count={count}: expected {expected}, got {actual}"
    return f"max_capacity=10; verified {cases}"


run_check("4", "8. CrowdDetector.density_level boundaries", check_density_boundaries)


def check_pool_roi():
    from component1_crowd.detector import CrowdDetector
    polygon = [(0.2, 0.3), (0.85, 0.3), (0.9, 0.95), (0.1, 0.95)]
    det = CrowdDetector(pool_polygon=polygon)
    W, H = 640, 480
    in_box = (300, 200, 340, 260)   # bottom-centre (320,260) -> relative (0.50,0.54) -> inside
    out_box = (10, 5, 50, 40)       # bottom-centre (30,40)   -> relative (0.05,0.08) -> outside
    assert det.is_in_pool(in_box, W, H) is True, "expected inside-polygon point to be True"
    assert det.is_in_pool(out_box, W, H) is False, "expected outside-polygon point to be False"
    det_none = CrowdDetector()
    assert det_none.is_in_pool(out_box, W, H) is True, "no polygon set should default to True"
    return "in-pool point -> True, out-of-pool point -> False, no-polygon default -> True"


run_check("4", "9. Pool ROI is_in_pool()", check_pool_roi)


def check_garbage_classes():
    from component4_garbage.detector import GARBAGE_CLASSES, ALL_CLASSES
    assert "ball" not in GARBAGE_CLASSES, "ball should be excluded from garbage alerts"
    assert "leaf" not in GARBAGE_CLASSES, "leaf should be excluded from garbage alerts"
    assert "ball" in ALL_CLASSES and "leaf" in ALL_CLASSES, "ball/leaf should still be detectable classes"
    return f"ALL_CLASSES={ALL_CLASSES}\nGARBAGE_CLASSES={GARBAGE_CLASSES}"


run_check("4", "10. Garbage class disambiguation (ball/leaf excluded)", check_garbage_classes)


def check_water_predictor():
    predictor = app.water.predictor
    assert predictor.model_status == "loaded", f"predictor not loaded: {predictor.model_status}"
    good = {"ph": 7.4, "temperature": 27.5, "chlorine": 1.8, "turbidity": 1.0, "tds": 380.0}
    bad = {"ph": 4.5, "temperature": 36.0, "chlorine": 0.05, "turbidity": 9.0, "tds": 950.0}
    good_result = predictor.predict(good)
    bad_result = predictor.predict(bad)
    valid = {"SAFE", "WARNING", "CRITICAL"}
    assert good_result in valid, f"good reading -> unexpected class {good_result!r}"
    assert bad_result in valid, f"bad reading -> unexpected class {bad_result!r}"
    return f"good reading {good} -> {good_result}\nbad reading  {bad} -> {bad_result}"


run_check("4", "11. WaterQualityPredictor returns a valid class", check_water_predictor)


def check_decision_engine():
    from core.decision_engine import decision_engine

    emergency_snapshot = {
        "crowd": {"density_level": "LOW", "pending_actions": 0, "swimmer_count": 0, "max_capacity": 10},
        "drowning": {"status": "DANGER"},
        "garbage": {"alert_status": "CLEAR"},
        "water_quality": {"status": "SAFE"},
    }
    r1 = decision_engine.evaluate(emergency_snapshot)
    assert r1["overall_risk"] == "EMERGENCY", f"expected EMERGENCY, got {r1['overall_risk']}"

    crowd_water_snapshot = {
        "crowd": {"density_level": "HIGH", "pending_actions": 1, "swimmer_count": 8, "max_capacity": 10},
        "drowning": {"status": "SAFE"},
        "garbage": {"alert_status": "CLEAR"},
        "water_quality": {"status": "WARNING"},
    }
    r2 = decision_engine.evaluate(crowd_water_snapshot)
    assert r2["overall_risk"] == "ELEVATED", f"expected ELEVATED risk, got {r2['overall_risk']}"
    assert r2["maintenance_urgency"] in ("MEDIUM", "HIGH"), (
        f"expected elevated (MEDIUM/HIGH) urgency, got {r2['maintenance_urgency']}"
    )
    return (f"drowning DANGER -> overall_risk={r1['overall_risk']!r}\n"
            f"HIGH crowd + WARNING water -> overall_risk={r2['overall_risk']!r}, "
            f"maintenance_urgency={r2['maintenance_urgency']!r}")


run_check("4", "12. DecisionEngine (drowning->EMERGENCY, crowd+water->elevated)", check_decision_engine)


# ══════════════════════════════════════════════════════════════
# SECTION 5 — PIPELINE
# ══════════════════════════════════════════════════════════════
section("SECTION 5 — PIPELINE (10-second live run, simulate mode)")

_pipeline_snapshot = {}


def check_pipeline_run():
    assert app.settings.CAMERA_SOURCE == "simulate", (
        f"expected simulate mode, CAMERA_SOURCE={app.settings.CAMERA_SOURCE!r} "
        "(pipeline check assumes no real hardware)"
    )
    app.state.update_root({"running": False})
    if not app.water._running:
        app.water.start()

    thread = threading.Thread(target=app.vision_loop, daemon=True)
    thread.start()
    print("           running vision_loop for 10 seconds ...")
    time.sleep(10)
    app.state.update_root({"running": False})
    thread.join(timeout=5)
    assert not thread.is_alive(), "vision_loop thread did not stop within 5s of being told to"

    snap = app.state.snapshot()
    _pipeline_snapshot.update(snap)

    assert snap["frame_num"] > 0, "no frames were processed"
    assert snap["fps"] > 0, f"FPS was not positive: {snap['fps']}"
    for module in ("crowd", "drowning", "garbage"):
        status = snap[module]["model_status"]
        assert status != "not_loaded", f"{module}.model_status still 'not_loaded'"

    return (f"frames_processed={snap['frame_num']}  fps={snap['fps']}  "
            f"camera_connected={snap['camera_connected']}\n"
            f"crowd.model_status={snap['crowd']['model_status']!r}  "
            f"drowning.model_status={snap['drowning']['model_status']!r}  "
            f"garbage.model_status={snap['garbage']['model_status']!r}")


run_check("5", "13. Vision loop runs, frames processed, FPS>0, 3 modules wrote state", check_pipeline_run)


def check_water_thread():
    snap = _pipeline_snapshot or app.state.snapshot()
    wq = snap.get("water_quality", {})
    assert wq.get("model_status") != "not_loaded", "water predictor never reported a status"
    assert wq.get("status") != "UNKNOWN", f"water status still UNKNOWN: {wq}"
    has_reading = any(wq.get(k) is not None for k in ("ph", "temperature", "chlorine", "turbidity", "tds"))
    assert has_reading, f"no sensor reading present in state: {wq}"
    return (f"sensor_connected={wq.get('sensor_connected')}  status={wq.get('status')}  "
            f"ph={wq.get('ph')} temperature={wq.get('temperature')} chlorine={wq.get('chlorine')} "
            f"turbidity={wq.get('turbidity')} tds={wq.get('tds')}")


run_check("5", "14. Water monitor thread produced a reading + prediction", check_water_thread)


def check_measured_fps():
    snap = _pipeline_snapshot or app.state.snapshot()
    fps = snap.get("fps", 0)
    assert fps > 0, f"FPS was not positive: {fps}"
    return f"measured FPS = {fps}  (frame_num={snap.get('frame_num')}, session_time={snap.get('session_time')})"


run_check("5", "15. Report measured FPS", check_measured_fps)


# ══════════════════════════════════════════════════════════════
# SECTION 6 — API & FRONTEND
# ══════════════════════════════════════════════════════════════
section("SECTION 6 — API & FRONTEND")


def check_routes():
    rules = {r.rule for r in app.app.url_map.iter_rules()}
    expected = ["/", "/api/status", "/api/report", "/api/session/summary"]
    missing = [p for p in expected if p not in rules]
    assert not missing, f"missing routes: {missing} (found: {sorted(rules)})"
    return f"found all expected routes: {expected}"


run_check("6", "16. Expected Flask routes exist", check_routes)


def check_html_ids():
    base = app.settings.BASE_DIR
    js = (base / "frontend-vanilla" / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")
    html = (base / "frontend-vanilla" / "templates" / "index.html").read_text(encoding="utf-8")

    ids = set(re.findall(r"getElementById\(['\"]([a-zA-Z0-9_]+)['\"]\)", js))
    for pat in (r"flashCard\(['\"]([a-zA-Z0-9_]+)['\"]\)",
                r"setText\(['\"]([a-zA-Z0-9_]+)['\"]",
                r"setPill\(['\"]([a-zA-Z0-9_]+)['\"]"):
        ids.update(re.findall(pat, js))

    problems = []
    for i in sorted(ids):
        count = len(re.findall(r'id="' + re.escape(i) + r'"', html))
        if count != 1:
            problems.append(f"id={i!r} occurrences={count}")
    assert not problems, "\n".join(problems)
    return f"checked {len(ids)} ids referenced by dashboard.js — each exists exactly once in index.html"


run_check("6", "17. Every dashboard.js id exists exactly once in index.html", check_html_ids)


def check_pdf_report():
    from core.report_generator import generate_session_report
    summary = app.session.get_summary()
    snap = app.state.snapshot()
    summary["frames_processed"] = snap.get("frame_num", 0)
    summary["model_status"] = {
        "crowd": snap.get("crowd", {}).get("model_status"),
        "drowning": snap.get("drowning", {}).get("model_status"),
        "garbage": snap.get("garbage", {}).get("model_status"),
        "water_quality": snap.get("water_quality", {}).get("model_status"),
    }
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        generate_session_report(summary, tmp_path)
        assert os.path.exists(tmp_path), "generate_session_report did not create a file"
        size = os.path.getsize(tmp_path)
        assert size > 0, "generated PDF is empty"
        with open(tmp_path, "rb") as f:
            head = f.read(4)
        assert head == b"%PDF", f"file does not look like a PDF (starts with {head!r})"
        return f"generated {size} bytes at {tmp_path} (deleted after check)"
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


run_check("6", "18. PDF report generator produces a non-empty file", check_pdf_report)


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

col_widths = (8, 58, 6)
header = f"{'Section':<{col_widths[0]}} {'Check':<{col_widths[1]}} {'Result':<{col_widths[2]}}"
print(header)
print("-" * len(header))
for r in results:
    check_label = r["check"]
    if len(check_label) > col_widths[1]:
        check_label = check_label[: col_widths[1] - 1] + "…"
    mark = "PASS" if r["status"] == "PASS" else "FAIL"
    print(f"{r['section']:<{col_widths[0]}} {check_label:<{col_widths[1]}} {mark:<{col_widths[2]}}")

total = len(results)
passed = sum(1 for r in results if r["status"] == "PASS")
failed = total - passed
print("-" * len(header))
print(f"TOTAL: {passed}/{total} passed, {failed} failed")

if failed:
    print("\nFAILED CHECKS — details:")
    for r in results:
        if r["status"] == "FAIL":
            print(f"\n  [{r['section']}] {r['check']}")
            for line in r["notes"].splitlines():
                print(f"      {line}")

sys.exit(0 if failed == 0 else 1)
