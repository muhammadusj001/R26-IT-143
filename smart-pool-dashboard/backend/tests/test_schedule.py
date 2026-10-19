"""
Vision-loop module schedule follows the connected viewers' plans while
running (no Stop/Start after an upgrade).

Run from smart-pool-dashboard/backend:
    python -m unittest discover -s tests -v

Camera and detectors are fakes, so no model is loaded and no camera opened.
"""

import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

_TMP = tempfile.mkdtemp(prefix="schedule-test-")
os.environ.update(DB_PATH=os.path.join(_TMP, "app.db"), CAMERA_SOURCE="simulate",
                  DEMO_MODE="0", SECRET_KEY="test-secret")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as A  # noqa: E402
from core import auth, db  # noqa: E402


def setUpModule():
    # In a full `discover` run `app` was already imported by another test
    # module whose temp database may be gone by now — recreate the schema.
    db.init_db()


def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)


class FakeCamera:
    source = "fake"

    def open(self):
        return True

    def read(self):
        time.sleep(0.005)
        return True, np.zeros((48, 64, 3), np.uint8)

    def reconnect(self):
        return True

    def status(self):
        return {"source": "fake"}

    def release(self):
        pass


class FakeDetectors:
    """Counts calls; drowning reports no model_status so 'waiting' stays visible."""

    def __init__(self):
        self.calls = {"crowd": 0, "drowning": 0, "garbage": 0}

    def crowd(self, frame):
        self.calls["crowd"] += 1
        return frame, {"swimmer_count": 0, "density_level": "LOW", "bather_load": 0.0,
                       "model_status": "loaded"}

    def drowning(self, frame):
        self.calls["drowning"] += 1
        return frame, {"count": 0}, None

    def garbage(self, frame, clean_frame=None):
        self.calls["garbage"] += 1
        return frame, {"model_status": "loaded"}, None


def wait_for(predicate, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeDetectors()
        patches = [
            mock.patch.object(A, "camera", FakeCamera()),
            mock.patch.object(A.crowd, "process", self.fake.crowd),
            mock.patch.object(A.drowning, "process", self.fake.drowning),
            mock.patch.object(A.drowning, "draw_last", lambda f: f),
            mock.patch.object(A.garbage, "process", self.fake.garbage),
            mock.patch.object(A.socketio, "emit"),
            mock.patch.object(A.settings, "PROCESS_EVERY_N_FRAMES", 1),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        for m in ("crowd", "drowning", "garbage"):
            A.state.update_module(m, {"model_status": "not_loaded"})
        A._viewers.clear()
        self.addCleanup(A._viewers.clear)
        self.thread = None

    def tearDown(self):
        A.state.update_root({"running": False})
        if self.thread:
            self.thread.join(5)
            self.assertFalse(self.thread.is_alive(), "vision_loop did not stop")

    def start(self, start_plan):
        self.thread = threading.Thread(target=A.vision_loop, args=(start_plan,), daemon=True)
        self.thread.start()

    def add_viewer(self, sid, plan, user_id=None):
        with A._viewers_lock:
            A._viewers[sid] = {"user_id": user_id, "plan": plan}
        A._schedule_dirty.set()

    def test_upgrade_while_running_starts_drowning_and_garbage(self):
        uid = db.create_user("Basic User", f"basic{time.time_ns()}@example.com", "x")
        self.add_viewer("sid-basic", "basic", uid)
        self.start("basic")
        self.assertTrue(wait_for(lambda: self.fake.calls["crowd"] > 5, 3))
        self.assertEqual((self.fake.calls["drowning"], self.fake.calls["garbage"]), (0, 0))
        self.assertEqual(A.state.data["drowning"]["model_status"], "not_loaded")

        # Upgrade through the same hook billing/admin use
        db.update_user(uid, plan="plus", plan_status="active", plan_renews_at="2999-01-01")
        with self.assertLogs("vision", "INFO") as logs:
            auth._notify_plan_change(uid)
            started = wait_for(lambda: self.fake.calls["drowning"] and self.fake.calls["garbage"], 3)
        self.assertTrue(started, "drowning/garbage did not start within 3 s of the upgrade")
        self.assertIn("Module schedule: crowd, drowning, garbage (plan=plus)", "\n".join(logs.output))
        self.assertEqual(A.state.data["garbage"]["model_status"], "loaded")
        # enabled, ran, but never reported a model status -> "waiting", not "not_loaded"
        self.assertEqual(A.state.data["drowning"]["model_status"], "waiting")
        # the viewer's own entry follows the new plan (filter_state per viewer)
        self.assertEqual(A._viewers["sid-basic"]["plan"], "plus")

    def test_plus_viewer_joining_enables_modules_without_restart(self):
        self.add_viewer("sid-basic", "basic")
        self.start("basic")
        self.assertTrue(wait_for(lambda: self.fake.calls["crowd"] > 5, 3))
        self.assertEqual(self.fake.calls["garbage"], 0)
        self.add_viewer("sid-plus", "plus")
        self.assertTrue(wait_for(lambda: self.fake.calls["garbage"] > 0, 3))
        # Plus viewer leaves: garbage stops again; basic viewer stays locked out
        with A._viewers_lock:
            A._viewers.pop("sid-plus")
        A._schedule_dirty.set()
        time.sleep(0.5)
        before = self.fake.calls["garbage"]
        time.sleep(0.5)
        self.assertEqual(self.fake.calls["garbage"], before)

    def test_pro_viewer_runs_drowning_not_garbage(self):
        self.add_viewer("sid-pro", "pro")
        self.start("basic")
        self.assertTrue(wait_for(lambda: self.fake.calls["drowning"] > 3, 3))
        self.assertEqual(self.fake.calls["garbage"], 0)
        self.assertEqual(A.state.data["garbage"]["model_status"], "not_loaded")

    def test_effective_plan_is_highest_viewer_or_fallback(self):
        self.assertEqual(A._effective_plan("pro"), "pro")
        self.add_viewer("a", "basic")
        self.add_viewer("b", "plus")
        self.add_viewer("c", "pro")
        self.assertEqual(A._effective_plan("basic"), "plus")


if __name__ == "__main__":
    unittest.main()
