"""
DET2: person / object / gaze-calibration rules added after the live session 75_20261015_001107.

  (a) an arm (face-less person box overlapping the candidate's box) is not a second person;
      a second person beside the candidate still counts
  (b) a confident phone held in front of the face is kept; a weak box inside the face is still ignored
  (c) forward-gaze calibration gate: implausible calibration gaze (-48 deg) is rejected

Run (from server/):  ..\\.venv\\Scripts\\python.exe -m tests.test_det2_persons_objects
"""

import sys

import config as C
from video import detectors

FAILS = []


def check(name, ok, info=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAILS.append(name)


def box(x1, y1, x2, y2, conf=0.8, cls=0):
    return {"cls": cls, "name": "person" if cls == 0 else "cell phone", "conf": conf, "xyxy": [x1, y1, x2, y2]}


def face(x, y, w, h):
    return {"bbox": (x, y, w, h)}


def a_persons():
    print("(a) persons")
    pc = C.get_config("v2")["persons"]
    shape = (360, 640, 3)
    cand = box(200, 60, 460, 360)                 # candidate's body, face inside
    f = [face(290, 120, 80, 100)]
    arm = box(60, 100, 300, 360, conf=0.6)        # arm + phone reaching left: 42% of it inside the candidate box
    other = box(500, 80, 640, 360, conf=0.6)      # second person at the right edge, no visible face
    check("candidate + arm -> 1 person", detectors.count_persons([cand, arm], f, shape, pc)[0] == 1)
    check("candidate + person at edge -> 2 persons", detectors.count_persons([cand, other], f, shape, pc)[0] == 2)
    check("rule off (attach_frac None) -> arm counted (old behaviour)",
          detectors.count_persons([cand, arm], f, shape, dict(pc, attach_frac=None))[0] == 2)


def b_objects():
    print("(b) objects inside the face box")
    oc = C.get_config("v2")["objects"]
    fb = (290, 120, 80, 100)
    strong = box(300, 130, 360, 210, conf=0.7, cls=67)
    weak = box(300, 130, 360, 210, conf=0.4, cls=67)
    check("confident phone in front of the face kept", "Cell Phone" in detectors.objects_v2([strong], oc, fb)[0])
    check("weak box inside the face ignored", not detectors.objects_v2([weak], oc, fb)[0])


def c_gaze_gate():
    print("(c) gaze calibration gate")
    lc = C.get_config("v2")["gaze"]["l2cs"]
    ok = lambda h, v: abs(h) <= lc["cal_max_abs_h"] and abs(v) <= lc["cal_max_abs_v"]  # noqa: E731
    check("-48 deg rejected (the bad session's baseline)", not ok(-48.3, 10.8))
    check("screen look accepted (h 5, v 15 down)", ok(5.0, 15.0))


def main():
    a_persons()
    b_objects()
    c_gaze_gate()
    print("DET2:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
