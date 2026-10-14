"""
VOICE2 end-to-end: the real AudioAnalyzer on simulated live sessions, old rules vs VOICE2 rules.

Sessions per held-out LibriSpeech speaker (all audio through tests.calibrate_speaker_live.live_channel, voiceprint
from the enrollment channel as in calibrate_speaker_live):
  solo          the candidate alone, ~60 s of speech with pauses        -> any VOICE_MISMATCH / MULTIPLE_SPEAKERS
                                                                           window is a false positive
  conversation  candidate + another speaker (3 turns of ~6 s)           -> the other speaker's windows should be
                                                                           flagged (MULTIPLE_SPEAKERS or VOICE_MISMATCH)
  impostor      another speaker only                                    -> VOICE_MISMATCH should cover the speech

Run (from server/):  ..\\.venv\\Scripts\\python.exe -m tests.eval_voice2 [n_speakers]
"""

import copy
import json
import sys

import numpy as np

import config as C
from audio import speaker
from audio.monitor import AudioAnalyzer
from tests import paths
from tests.calibrate_speaker_live import ENROLL_S, ROOT, enroll_channel, live_channel, speaker_audio, voiced

SR = 16000


def gap(rng, lo=0.5, hi=1.5, level=0.002):
    return (rng.standard_normal(int(rng.uniform(lo, hi) * SR)) * level).astype(np.float32)


def speech_of(utts, seconds, rng):
    out, got = [], 0.0
    for u in utts:
        if got >= seconds:
            break
        out += [u, gap(rng)]
        got += len(u) / SR
    return out


def build_sessions(spk, other, cfg, rng):
    utts = speaker_audio(ROOT / spk)
    enroll, rest, got = [], [], 0.0
    for u in utts:
        if got < ENROLL_S:
            enroll.append(u)
            got += len(u) / SR
        else:
            rest.append(u)
    ex = enroll_channel(np.concatenate(enroll), rng)
    vp, _, _ = speaker.voiceprint(ex, voiced(ex, cfg["vad"]), 3.0)
    o_utts = speaker_audio(ROOT / other)
    solo = live_channel(np.concatenate([gap(rng, 2, 2)] + speech_of(rest, 60, rng)), rng)
    # conversation: candidate 15 s, other ~6 s, candidate 15 s, other ~6 s, ...
    parts, labels, t = [gap(rng, 2, 2)], [], 2.0
    cand_parts, oth_parts = speech_of(rest[::-1], 45, rng), speech_of(o_utts, 18, rng)
    ci = oi = 0
    for turn in range(6):
        src, is_other = ((oth_parts, True) if turn % 2 else (cand_parts, False))
        idx = oi if is_other else ci
        target = 6.0 if is_other else 15.0
        acc = 0.0
        while idx < len(src) and acc < target:
            seg = src[idx]
            if is_other and len(seg) > 100 and float(np.std(seg)) > 0.01:
                labels.append((t, t + len(seg) / SR))
            parts.append(seg)
            acc += len(seg) / SR
            t += len(seg) / SR
            idx += 1
        if is_other:
            oi = idx
        else:
            ci = idx
    conv = live_channel(np.concatenate(parts), rng)
    imp = live_channel(np.concatenate([gap(rng, 2, 2)] + speech_of(o_utts[::-1], 40, rng)), rng)
    return vp, {"solo": (solo, []), "conversation": (conv, labels), "impostor": (imp, None)}


def flagged_frac(res, types, where=None):
    """Share of speech windows (optionally only those overlapping `where` intervals) flagged with any of types."""
    ws = [w for w in res["windows"] if w["is_speech"]]
    if where is not None:
        ws = [w for w in ws if any(a < w["t_end"] and w["t_start"] < b for a, b in where)]
    if not ws:
        return None
    return sum(any(w.get("flags", {}).get(t) for t in types) for w in ws) / len(ws)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rng = np.random.default_rng(11)
    new = C.get_config("v2")
    old = copy.deepcopy(new)
    old["speaker"].update(multi_rule="pairwise", adapt_thr=None)
    spks = sorted(p.name for p in ROOT.iterdir() if p.is_dir())[1::2][:n]      # held-out half
    VM, MS = C.VOICE_MISMATCH, C.MULTIPLE_SPEAKERS
    rows = {k: {"old": [], "new": []} for k in ("solo_fp", "conv_detect", "conv_cand_fp", "impostor_detect")}
    for i, spk in enumerate(spks):
        other = spks[(i + 1) % len(spks)]
        vp, sess = build_sessions(spk, other, new, rng)
        for name, (x, labels) in sess.items():
            aa = AudioAnalyzer(new, vp)
            for k in range(0, len(x), 1600):
                aa.feed(x[k:k + 1600])
            for tag, cfg in (("new", new), ("old", old)):
                aa.cfg = cfg
                res = aa.finalize()
                if name == "solo":
                    rows["solo_fp"][tag].append(flagged_frac(res, (VM, MS)))
                elif name == "conversation":
                    rows["conv_detect"][tag].append(flagged_frac(res, (VM, MS), labels))
                    cand = [(a, b) for a, b in zip([0.0] + [e for _, e in labels], [s for s, _ in labels] + [1e9])]
                    rows["conv_cand_fp"][tag].append(flagged_frac(res, (VM, MS), cand))
                else:
                    rows["impostor_detect"][tag].append(flagged_frac(res, (VM,)))
        print(f"  {spk} (other {other}): " + ", ".join(
            f"{k} old {rows[k]['old'][-1] if rows[k]['old'][-1] is not None else float('nan'):.2f} / "
            f"new {rows[k]['new'][-1] if rows[k]['new'][-1] is not None else float('nan'):.2f}" for k in rows))
    summary = {k: {t: (round(float(np.mean([v for v in vals if v is not None])), 3) if any(v is not None for v in vals)
                       else None) for t, vals in r.items()} for k, r in rows.items()}
    print("MEAN over speakers (share of speech windows):")
    for k, v in summary.items():
        print(f"  {k:16s} old {v['old']}   new {v['new']}")
    (paths.RESULTS / "VOICE2_eval.json").write_text(json.dumps({"speakers": spks, "summary": summary,
                                                                 "rows": rows}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
