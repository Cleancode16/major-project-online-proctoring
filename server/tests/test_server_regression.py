"""
Server smoke test: server.py imports (incl. aiortc) and the REST API answers (Flask test client; no port
is opened, no camera/microphone is used).

Run (from server/):  ..\\.venv\\Scripts\\python.exe -m tests.test_server_regression
"""

import sys

from tests import paths


def main():
    sys.path.insert(0, str(paths.ROOT))
    import server
    c = server.app.test_client()
    h = c.get("/api/health")
    p = c.get("/api/enrollment/passage")
    e = c.get("/api/enrollment/__nobody__")
    o = c.post("/api/webrtc/offer", json={"candidate": "__nobody__", "sdp": "x", "type": "offer"})
    f = c.post("/api/enrollment/face", json={"candidate": "x", "consent": False, "image": ""})
    ok = (h.status_code == 200 and p.status_code == 200 and "text" in p.get_json()
          and e.get_json()["face"] is False and o.status_code == 400 and f.status_code == 400)
    print(f"health {h.status_code}, passage {p.status_code}, unknown enrollment {e.get_json()}, "
          f"offer without enrollment {o.status_code}, face without consent {f.status_code}")
    print("SERVER SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
