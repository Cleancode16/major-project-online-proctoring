"""
A5: choose speaker-verification / speaker-change thresholds from public real speech (flow.txt §B.5).

Data: Mini LibriSpeech dev-clean-2 (openslr.org/31), 26 speakers, 16 kHz read speech.
Per speaker: voiceprint from the first ~20 s of voiced speech (same code path as enrollment), test chunks
from the remaining utterances. Speakers are split by sorted id: even index = calibration, odd = held out.

Reports for chunk lengths 1.5 / 2.0 / 3.0 s:
  verification EER threshold (calibration half), FRR/FAR on held-out half, raw and median-of-3 smoothed;
  same-speaker vs different-speaker chunk-pair similarity -> diff_thr at 1% same-speaker false alarms.

Run:  .venv\\Scripts\\python.exe -m tests.calibrate_speaker_threshold
"""

import json
import sys
from pathlib import Path

import numpy as np

import config as C
from audio import io as aio
from audio import speaker
from audio.vad import VAD
from tests import paths

ROOT = paths.DATA / "librispeech" / "LibriSpeech" / "dev-clean-2"
ENROLL_S = 20.0
MAX_TEST_S = 60.0


def speaker_audio(spk_dir):
    files = sorted(Path(spk_dir).rglob("*.flac"))
    return [aio.load_file(f) for f in files]


def voiced(x, vcfg):
    v = VAD(vcfg)
    v.feed(x)
    return v.segments()


def build(vcfg, chunk_lengths):
    data = {}
    for spk in sorted(p.name for p in ROOT.iterdir() if p.is_dir()):
        utts = speaker_audio(ROOT / spk)
        enroll, rest, got = [], [], 0.0
        for u in utts:
            if got < ENROLL_S:
                enroll.append(u)
                got += len(u) / aio.SR
            else:
                rest.append(u)
        ex = np.concatenate(enroll)
        vp, cons, n = speaker.voiceprint(ex, voiced(ex, vcfg), 3.0)
        chunks = {L: [] for L in chunk_lengths}
        used = 0.0
        for u in rest:
            if used >= MAX_TEST_S:
                break
            segs = voiced(u, vcfg)
            vu = speaker.voiced_audio(u, segs)
            used += len(vu) / aio.SR
            for L in chunk_lengths:
                # consecutive chunks of one utterance (keeps order for median-of-3 smoothing)
                chunks[L].append(np.stack([speaker.embed(c) for c in speaker.chunk(vu, L)]) if len(vu) >= 0.6 * L * aio.SR
                                 else np.zeros((0, 256)))
        data[spk] = {"vp": vp, "consistency": cons, "chunks": chunks}
        print(f"  speaker {spk}: enroll {got:.1f}s ({n} chunks, consistency {cons:.2f}), test voiced {used:.1f}s")
    return data


def med3(x):
    if len(x) < 3:
        return x
    p = np.pad(x, 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def scores(data, spks, L, smooth):
    tgt, non = [], []
    for s in spks:
        for t in data:
            for utt in data[s]["chunks"][L]:
                if len(utt) == 0:
                    continue
                sims = utt @ data[t]["vp"]
                sims = med3(sims) if smooth else sims
                (tgt if s == t else non).extend(sims.tolist())
    return np.array(tgt), np.array(non)


def eer_threshold(tgt, non):
    cands = np.linspace(0, 1, 2001)
    frr = np.array([np.mean(tgt < c) for c in cands])
    far = np.array([np.mean(non >= c) for c in cands])
    i = int(np.argmin(np.abs(frr - far)))
    return float(cands[i]), float(frr[i]), float(far[i])


def main():
    vcfg = C.get_config("v2")["vad"]
    Ls = (1.5, 2.0, 3.0)
    print("Embedding Mini LibriSpeech dev-clean-2 ...")
    data = build(vcfg, Ls)
    spks = sorted(data)
    cal, held = spks[0::2], spks[1::2]
    out = {"n_speakers": len(spks), "calibration_speakers": cal, "heldout_speakers": held, "by_chunk": {}}
    for L in Ls:
        row = {}
        for smooth in (False, True):
            tgt_c, non_c = scores(data, cal, L, smooth)
            thr, _, _ = eer_threshold(tgt_c, non_c)
            tgt_h, non_h = scores(data, held, L, smooth)
            row["smoothed" if smooth else "raw"] = {
                "eer_threshold_cal": round(thr, 3),
                "heldout_FRR": round(float(np.mean(tgt_h < thr)), 4),
                "heldout_FAR": round(float(np.mean(non_h >= thr)), 4),
                "heldout_n_target": int(len(tgt_h)), "heldout_n_nontarget": int(len(non_h)),
                "target_mean": round(float(tgt_h.mean()), 3), "nontarget_mean": round(float(non_h.mean()), 3)}
        # speaker change: chunk-vs-chunk similarity, same vs different speaker (calibration half)
        same, diff = [], []
        for i, s in enumerate(cal):
            es = np.concatenate([u for u in data[s]["chunks"][L] if len(u)])
            sm = es @ es.T
            same.extend(sm[np.triu_indices(len(es), 1)].tolist())
            for t in cal[i + 1:]:
                et = np.concatenate([u for u in data[t]["chunks"][L] if len(u)])
                diff.extend((es @ et.T).ravel().tolist())
        same, diff = np.array(same), np.array(diff)
        diff_thr = float(np.percentile(same, 1))  # 1% of same-speaker pairs fall below
        row["speaker_change"] = {"diff_thr_same1pct": round(diff_thr, 3),
                                 "diff_pairs_detected_at_thr": round(float(np.mean(diff < diff_thr)), 4),
                                 "same_mean": round(float(same.mean()), 3), "diff_mean": round(float(diff.mean()), 3)}
        out["by_chunk"][str(L)] = row
        r, s = row["raw"], row["smoothed"]
        print(f"chunk {L}s: raw thr {r['eer_threshold_cal']} FRR {r['heldout_FRR']:.3f} FAR {r['heldout_FAR']:.3f} | "
              f"median-of-3 thr {s['eer_threshold_cal']} FRR {s['heldout_FRR']:.3f} FAR {s['heldout_FAR']:.3f} | "
              f"change thr {row['speaker_change']['diff_thr_same1pct']} catches "
              f"{row['speaker_change']['diff_pairs_detected_at_thr']:.3f} of different-speaker pairs")
    (paths.RESULTS / "A5_speaker_calibration.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
