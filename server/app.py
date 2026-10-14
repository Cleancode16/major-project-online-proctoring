"""
Online Proctoring System - command line entry point (v2).

The original v1 script is preserved unchanged in legacy/app_v1.py.
All thresholds live in config.py and are documented in flow.txt PART II.

Examples
  python app.py                                   # default sample video + reference (as v1)
  python app.py --video exam.mp4 --reference me.jpg --candidate alice
  python app.py --candidate alice --video exam.mp4          # use alice's stored enrollment (UI)
  python app.py --enroll-audio reading.wav --passage 0 ...  # build a voiceprint from a recording
  python app.py --audio-source none                         # video only
  python app.py --profile legacy                            # v1 behaviour, for comparison
"""

import argparse
import sys
import time

print(">>> Script started")

import config as C  # noqa: E402  (sets TF env vars first)


def main():
    ap = argparse.ArgumentParser(description="Offline proctoring analysis of a recorded exam video.")
    ap.add_argument("--video", default=r"C:\Users\bhara\OneDrive\Pictures\Camera Roll\WIN_20260207_13_29_05_Pro.mp4")
    ap.add_argument("--reference", help="reference face image (default: the candidate's stored enrollment, "
                                        "else the v1 default image)")
    ap.add_argument("--candidate", default="candidate", help="candidate id (enrollment folder / report name)")
    ap.add_argument("--audio-source", default="video", help='"video" (default), "none", or an audio file path')
    ap.add_argument("--enroll-audio", help="audio file of the candidate reading a passage -> voiceprint")
    ap.add_argument("--passage", type=int, default=0, help="index of the passage read in --enroll-audio")
    ap.add_argument("--profile", default="v2", choices=sorted(C.PROFILES))
    ap.add_argument("--out", help="output folder for report.json + evidence")
    a = ap.parse_args()

    import enrollment_store
    import proctor
    from pathlib import Path

    cfg = C.get_config(a.profile)
    stored = enrollment_store.load(a.candidate) if a.candidate in enrollment_store.list_candidates() else None
    ref = a.reference or (stored["face_embedding"] if stored is not None and stored["face_embedding"] is not None
                          else r"C:\Users\bhara\OneDrive\Pictures\Camera Roll\WIN_20260206_21_26_58_Pro.jpg")
    voiceprint = stored["voiceprint"] if stored is not None else None

    if a.enroll_audio:
        from audio import enrollment
        from audio import io as aio
        from audio.passages import PASSAGES
        r = enrollment.evaluate_recording(aio.load_file(a.enroll_audio), PASSAGES[a.passage], cfg)
        print(f">>> Voice enrollment: {'OK' if r['ok'] else 'REJECTED'} {r['metrics']}")
        for p in r["problems"] + r["warnings"]:
            print("    -", p)
        if r["ok"]:
            voiceprint = r["voiceprint"]

    t0 = time.time()
    last = [0.0]

    def progress(t, dur, s):
        if t - last[0] >= 2.0 or t >= dur - 0.4:
            last[0] = t
            print(f"\r>>> Analysing video {t:6.1f} / {dur:.1f} s", end="", flush=True)

    report, path, _, _ = proctor.analyze_files(a.video, ref, cfg, voiceprint, a.audio_source, a.candidate,
                                               progress=progress, out_dir=Path(a.out) if a.out else None)
    print(f"\n>>> Done in {time.time() - t0:.1f} s")
    print(proctor.console_summary(report))
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
