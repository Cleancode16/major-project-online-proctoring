"""
V2 sampling, V3 calibration, V4 head-pose smoothing/persistence.

V2: time-based sampling on the real sample clip (count, spacing, coverage).
V3: calibration on a synthetic stream with outliers: v2 (gated median) vs v1 (mean of first 3).
V4: synthetic relative-angle stream through VideoAnalyzer._flag_pose + finalize():
    single-sample spikes must NOT be counted, sustained turns (>= 1 s) must be.

Run:  .venv\\Scripts\\python.exe -m tests.test_v2_v4_sampling_calibration_pose
"""

import sys

import numpy as np

import config as C
from tests import paths
from video import pipeline


def v2_sampling():
    info = pipeline.video_info(paths.SAMPLE_VIDEO)
    ok = True
    for profile in ("legacy", "v2"):
        cfg = C.get_config(profile)
        ts = [t for _, t, _ in pipeline.iter_sampled_frames(paths.SAMPLE_VIDEO, cfg["sampling"])]
        if cfg["sampling"]["max_samples"]:
            ts = ts[:cfg["sampling"]["max_samples"]]
        gaps = np.diff(ts)
        print(f"V2 [{profile}] samples={len(ts)} first={ts[0]:.3f}s last={ts[-1]:.3f}s "
              f"spacing mean={gaps.mean():.3f}s min={gaps.min():.3f} max={gaps.max():.3f} "
              f"coverage={ts[-1] / info['duration_s'] * 100:.0f}% of {info['duration_s']:.2f}s")
        if profile == "v2":
            # expectations follow the configured rate (FPS10: 10 fps, was 3): samples land on the video's frame grid,
            # so one spacing may be up to one native frame shorter or longer than 1 / fps
            step, frame = 1.0 / cfg["sampling"]["fps"], 1.0 / info["fps"]
            expected = int(np.ceil(info["duration_s"] * cfg["sampling"]["fps"]))
            ok = (abs(len(ts) - expected) <= 1 and abs(gaps.mean() - step) < 0.06 * step
                  and gaps.min() >= step - frame - 1e-6 and gaps.max() <= step + frame + 1e-6)
            print(f"V2 [v2] expected {expected} samples every {step:.3f}s "
                  f"(+-{frame:.3f}s frame grid) -> {'ok' if ok else 'FAIL'}")
    return ok


def _analyzer(profile):
    cfg = C.get_config(profile)
    rate = 3.0 if profile == "v2" else 2.978
    return pipeline.VideoAnalyzer(np.zeros(512), cfg, rate), cfg


def v3_calibration():
    rng = np.random.default_rng(0)
    true = (-14.0, 11.0, 1.0)
    stream = []  # (t, pose, identity_ok) at 3 fps
    for i in range(15):
        pose = tuple(v + rng.normal(0, 1.0) for v in true)
        ident = True
        if i == 0:
            pose = (45.0, -30.0, 5.0)   # candidate still settling, looking away
        if i == 1:
            pose = (true[0], true[1] + 22.0, true[2])  # chin down reading notes (within gate)
        if i == 2:
            pose, ident = (5.0, 0.0, 0.0), False      # someone else in front of the camera
        stream.append((i / 3.0, pose, ident))

    res = {}
    for profile in ("legacy", "v2"):
        va, _ = _analyzer(profile)
        for t, pose, ident in stream:
            if va.baseline is not None:
                break
            va._calibration_add(t, pose, {"h": 0.5, "v": 0.5, "valid": True}, ident)
        b = va.baseline
        err = max(abs(b["yaw"] - true[0]), abs(b["pitch"] - true[1]), abs(b["roll"] - true[2]))
        res[profile] = err
        print(f"V3 [{profile}] baseline yaw={b['yaw']:.1f} pitch={b['pitch']:.1f} roll={b['roll']:.1f} "
              f"(true {true}) accepted={b['n_samples']}/{b['n_seen']} max abs error={err:.1f} deg")

    # fallback: every frame fails the frontal gate (side camera)
    va, _ = _analyzer("v2")
    for i in range(40):
        va._calibration_add(i / 3.0, (40.0, 5.0, 0.0), None, True)
        if va.baseline is not None:
            break
    fb = va.baseline
    print(f"V3 [v2 side-camera] baseline set={fb is not None} fallback={fb and fb['fallback']} "
          f"yaw={fb and round(fb['yaw'], 1)} after t={fb and round(fb['t_end'], 2)}s")
    return res["v2"] < 1.5 and res["v2"] < res["legacy"] and fb is not None and fb["fallback"]


def v4_pose_persistence():
    """Sequence at 3 fps (rel yaw, deg). Ground truth: two real turns, three one-sample spikes."""
    seq = [0] * 6 + [25] + [0] * 6 + [30] * 6 + [0] * 6 + [0, 22, 0, 0, -26, 0] + [0] * 3 + [-35] * 4 + [0] * 4
    truth = [False] * len(seq)
    for a, b in ((13, 19), (34, 38)):  # 30 deg at idx 13-18, -35 deg at idx 34-37
        for i in range(a, b):
            truth[i] = True
    out = {}
    for profile in ("legacy", "v2"):
        va, cfg = _analyzer(profile)
        va.baseline = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        for i, y in enumerate(seq):
            s = {"t": i / 3, "phase": "analysis", "raw": {}}
            va._flag_pose(s, (float(y), 0.0, 0.0))
            va.samples.append(s)
        flags = [s["flags"][C.HEAD_POSE_DEVIATION] for s in va.finalize()["samples"]]
        tp = sum(f and t for f, t in zip(flags, truth))
        fp = sum(f and not t for f, t in zip(flags, truth))
        fn = sum(t and not f for f, t in zip(flags, truth))
        out[profile] = (tp, fp, fn)
        print(f"V4 [{profile}] per-sample TP={tp} FP={fp} FN={fn} (truth positives={sum(truth)})")
    tp, fp, fn = out["v2"]
    return fp == 0 and fn <= 2 and out["legacy"][1] > 0


def main():
    results = {"V2": v2_sampling(), "V3": v3_calibration(), "V4": v4_pose_persistence()}
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
