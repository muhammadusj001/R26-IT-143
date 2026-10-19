"""
127.0.0.1 -> localhost redirect. Run from smart-pool-dashboard/backend:
    python -m unittest discover -s tests -v
"""

import os
import shutil
import sys
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="redirect-test-")
os.environ.update(DB_PATH=os.path.join(_TMP, "app.db"), CAMERA_SOURCE="simulate",
                  DEMO_MODE="0", SECRET_KEY="test-secret")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as A  # noqa: E402


def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)


class LocalhostRedirectTests(unittest.TestCase):
    def get(self, base_url, path, method="GET"):
        client = A.app.test_client()
        return client.open(path, base_url=base_url, method=method)

    def test_redirects_path_query_and_port(self):
        r = self.get("http://127.0.0.1:5000", "/pricing?plan=pro&x=1")
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r.headers["Location"], "http://localhost:5000/pricing?plan=pro&x=1")

    def test_redirects_root_and_other_ports(self):
        r = self.get("http://127.0.0.1:8080", "/")
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "http://localhost:8080/"))
        r = self.get("http://127.0.0.1", "/login")
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "http://localhost/login"))

    def test_head_redirected(self):
        self.assertEqual(self.get("http://127.0.0.1:5000", "/pricing", "HEAD").status_code, 302)

    def test_post_not_redirected(self):
        # A 302 would turn the POST into a GET and lose the form body
        r = self.get("http://127.0.0.1:5000", "/login", "POST")
        self.assertEqual(r.status_code, 400)  # reached the view (missing CSRF token)

    def test_socketio_not_redirected(self):
        r = self.get("http://127.0.0.1:5000", "/socket.io/?EIO=4&transport=polling")
        self.assertNotEqual(r.status_code, 302)

    def test_other_hosts_pass_through(self):
        for base in ("http://localhost:5000", "https://abc123.ngrok-free.app",
                     "https://x1y2-5000.inc1.devtunnels.ms", "http://192.168.1.20:5000",
                     "http://127.0.0.10:5000"):
            with self.subTest(base=base):
                r = self.get(base, "/pricing")
                self.assertEqual(r.status_code, 200, base)
                r = self.get(base, "/api/status")
                self.assertEqual(r.status_code, 401, base)  # login required, not redirected


if __name__ == "__main__":
    unittest.main()
