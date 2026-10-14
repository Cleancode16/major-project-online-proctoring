"""Voice enrollment: VAD + quality + optional ASR + voiceprint (flow.txt §B.3)."""

from audio import asr, quality, speaker
from audio import io as aio
from audio.vad import VAD, speech_seconds


def evaluate_recording(x, passage, cfg):
    """
    Judge one enrollment recording. Returns a dict:
      ok, problems (user-facing), warnings, metrics {...}, voiceprint (np.ndarray | None), asr {...}
    """
    ec = cfg["enroll"]
    status = aio.source_status(x, cfg["audio"]["silent_rms_dbfs"])
    if status != "ok":
        msg = ("No audio was captured." if status == "unavailable" else
               "The microphone is silent: it may be muted or blocked by Windows privacy settings "
               "(Settings > Privacy & security > Microphone > allow desktop apps).")
        return {"ok": False, "problems": [msg], "warnings": [], "metrics": {"source_status": status},
                "voiceprint": None, "asr": None}

    v = VAD(cfg["vad"])
    v.feed(x)
    segs = v.segments()
    voiced = speech_seconds(segs)
    q = quality.measure(x, segs)
    q.update(voiced_s=round(voiced, 2), vad_engine=v.engine, source_status=status)

    problems, warnings = [], []
    if voiced < ec["min_voiced_s"]:
        problems.append(f"Only {voiced:.1f} s of speech was detected (need {ec['min_voiced_s']:.0f} s). "
                        "Please read the whole passage aloud.")
    if voiced >= 1.0:  # level/SNR are meaningless without any speech
        _, q_problems, q_warn = quality.judge(q, ec)
        problems += q_problems
        warnings += q_warn

    asr_res = asr.check(x, passage, ec)
    if asr_res["status"] == "fail":
        problems.append(f"The recording does not match the passage (word match {asr_res['match']:.0%}). "
                        "Please read the passage shown on screen.")

    vp, consistency, n_chunks = (None, 0.0, 0)
    if voiced >= 1.0:
        vp, consistency, n_chunks = speaker.voiceprint(x, segs, ec["chunk_s"])
    q.update(consistency=round(consistency, 3), n_chunks=n_chunks)
    if vp is not None and n_chunks > 1 and consistency < ec["min_consistency"]:
        problems.append("The voice in the recording is not consistent (another voice or heavy noise). "
                        "Please record again alone in a quiet room.")

    return {"ok": not problems and vp is not None, "problems": problems, "warnings": warnings,
            "metrics": q, "voiceprint": vp, "asr": asr_res}


def best_attempt(attempts):
    """Pick the best passing attempt (highest SNR), or None."""
    good = [a for a in attempts if a["ok"]]
    return max(good, key=lambda a: a["metrics"]["snr_db"]) if good else None
