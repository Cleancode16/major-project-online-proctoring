"""
VOICE2: speaker thresholds under LIVE channel conditions (flow.txt §B.5 "VOICE2").

A5 (calibrate_speaker_threshold.py) calibrated on clean LibriSpeech. Live sessions go through the browser
(noise suppression / AGC) and WebRTC Opus at ~24-32 kbps plus room noise; the user's sessions showed the
candidate's own voice at 0.60-0.69 against verify_thr 0.72 and MULTIPLE_SPEAKERS on 25-40% of the session.

Simulation (real speech, simulated channel):
  enrollment channel: Opus 64 kbps + noise at 40 dB SNR (quiet read-aloud recording)
  live channel:       gain 0.5-1.5, noise at 12-30 dB SNR, Opus 24 or 32 kbps (WebRTC voice bitrates)
Data: Mini LibriSpeech dev-clean-2, 26 speakers; even index = calibration, odd = held out (as A5).

Reports
  (1) verification, 2 s chunks, median-of-3: enrollment print only vs enrollment + session-adapted print
  (2) old MULTIPLE_SPEAKERS rule (any chunk pair < diff_thr within 10 s): false alarms on single-speaker windows
  (3) new rule: a 'foreign' run (>= 2 chunks with candidate similarity < other_thr) next to candidate speech

Run (from server/):  ..\\.venv\\Scripts\\python.exe -m tests.calibrate_speaker_live
"""

import json
import sys
from fractions import Fraction

import numpy as np

import config as C
from audio import io as aio
from audio import speaker
from tests import paths
from tests.calibrate_speaker_threshold import ENROLL_S, ROOT, eer_threshold, med3, speaker_audio, voiced

SR = aio.SR
MAX_TEST_S = 50.0
CHUNK_S = 2.0


# ------------------------------------------------------------------ channel simulation
def opus_roundtrip(x, kbps):
    """16 kHz float -> 48 kHz Opus at `kbps` -> decoded -> 16 kHz float (same codec as WebRTC / MediaRecorder)."""
    import av
    enc = av.CodecContext.create("libopus", "w")
    enc.sample_rate, enc.layout, enc.format = 48000, "mono", "s16"
    enc.bit_rate = int(kbps * 1000)
    enc.time_base = Fraction(1, 48000)
    up = av.AudioResampler(format="s16", layout="mono", rate=48000)
    down = av.AudioResampler(format="flt", layout="mono", rate=SR)
    dec = av.CodecContext.create("libopus", "r")
    dec.sample_rate, dec.layout = 48000, "mono"
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)[None, :]
    frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
    frame.sample_rate = SR
    out = []
    pts = 0
    frames = up.resample(frame) + up.resample(None)
    for fr in frames:
        fr.pts = pts
        pts += fr.samples
        for pkt in enc.encode(fr):
            for d in dec.decode(pkt):
                out += [r.to_ndarray().reshape(-1) for r in down.resample(d)]
    for pkt in enc.encode(None):
        for d in dec.decode(pkt):
            out += [r.to_ndarray().reshape(-1) for r in down.resample(d)]
    y = np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def add_noise(x, snr_db, rng):
    p = float(np.mean(x ** 2)) + 1e-12
    noise = rng.standard_normal(len(x)).astype(np.float32)
    noise = np.convolve(noise, np.ones(4) / 4, mode="same")          # slightly coloured (room / fan)
    noise *= np.sqrt(p / 10 ** (snr_db / 10) / (np.mean(noise ** 2) + 1e-12))
    return (x + noise).astype(np.float32)


def enroll_channel(x, rng):
    return opus_roundtrip(add_noise(x, 40.0, rng), 64)


def live_channel(x, rng):
    y = add_noise(x * rng.uniform(0.5, 1.5), rng.uniform(12.0, 30.0), rng)
    return opus_roundtrip(y, int(rng.choice([24, 32])))


# ------------------------------------------------------------------ data
def build(vcfg, rng):
    data = {}
    for spk in sorted(p.name for p in ROOT.iterdir() if p.is_dir()):
        utts = speaker_audio(ROOT / spk)
        enroll, rest, got = [], [], 0.0
        for u in utts:
            if got < ENROLL_S:
                enroll.append(u)
                got += len(u) / SR
            else:
                rest.append(u)
        ex = enroll_channel(np.concatenate(enroll), rng)
        vp, cons, _ = speaker.voiceprint(ex, voiced(ex, vcfg), 3.0)
        utt_embs, used = [], 0.0
        for u in rest:
            if used >= MAX_TEST_S:
                break
            y = live_channel(u, rng)
            vu = speaker.voiced_audio(y, voiced(y, vcfg))
            if len(vu) < 0.75 * CHUNK_S * SR:
                continue
            used += len(vu) / SR
            utt_embs.append(np.stack([speaker.embed(c) for c in speaker.chunk(vu, CHUNK_S, min_frac=0.75)]))
        data[spk] = {"vp": vp, "utts": utt_embs}
        print(f"  {spk}: enroll consistency {cons:.2f}, live test {used:.0f} s in {len(utt_embs)} utterances")
    return data


def session_print(embs, vp, adapt_thr):
    """Session-adapted candidate print: mean of this session's chunks that clearly match the enrollment."""
    good = [e for e in embs if float(e @ vp) >= adapt_thr]
    if len(good) < 3:
        return None
    m = np.mean(good, axis=0)
    return m / (np.linalg.norm(m) + 1e-12)


def cand_sim(e, vp, sp):
    s = float(e @ vp)
    return max(s, float(e @ sp)) if sp is not None else s


# ------------------------------------------------------------------ reports
def verification(data, spks, adapt_thr=None, thr=None):
    tgt, non = [], []
    for t in spks:
        half = len(data[t]["utts"]) // 2
        sess = np.concatenate(data[t]["utts"][:half]) if half else np.zeros((0, 256))
        sp = session_print(sess, data[t]["vp"], adapt_thr) if adapt_thr is not None else None
        for s in data:
            for utt in (data[s]["utts"][half:] if s == t else data[s]["utts"]):
                sims = med3(np.array([cand_sim(e, data[t]["vp"], sp) for e in utt]))
                (tgt if s == t else non).extend(sims.tolist())
    tgt, non = np.array(tgt), np.array(non)
    if thr is None:
        thr = eer_threshold(tgt, non)[0]
    return thr, float(np.mean(tgt < thr)), float(np.mean(non >= thr)), float(np.median(tgt)), float(np.median(non))


def windows_of(utts, n=8):
    seq = np.concatenate(utts)
    return [seq[i:i + n] for i in range(0, max(1, len(seq) - n + 1), 2)]


def old_rule_fa(data, spks, diff_thr):
    fa = tot = 0
    for t in spks:
        for w in windows_of(data[t]["utts"]):
            if len(w) < 2:
                continue
            sm = w @ w.T
            tot += 1
            fa += float(sm[np.triu_indices(len(w), 1)].min()) < diff_thr
    return fa / max(tot, 1)


def new_rule(window_sims, other_thr, run=2):
    low = window_sims < other_thr
    has_run = any(low[i:i + run].all() for i in range(len(low) - run + 1))
    return bool(has_run)


def new_rule_rates(data, spks, other_thr, adapt_thr, rng):
    """FA on single-speaker windows; detection on windows with a 2nd speaker inserted (4 chunks = ~8 s)."""
    fa = tot_fa = det = tot_det = 0
    for t in spks:
        half = len(data[t]["utts"]) // 2
        sp = session_print(np.concatenate(data[t]["utts"][:half]), data[t]["vp"], adapt_thr) if half else None
        for w in windows_of(data[t]["utts"][half:]):
            sims = med3(np.array([cand_sim(e, data[t]["vp"], sp) for e in w]))
            tot_fa += 1
            fa += new_rule(sims, other_thr)
            o = rng.choice([s for s in spks if s != t])
            oe = np.concatenate(data[o]["utts"])[:4]
            mixed = np.concatenate([w[:4], oe])
            msims = med3(np.array([cand_sim(e, data[t]["vp"], sp) for e in mixed]))
            tot_det += 1
            det += new_rule(msims, other_thr)
    return fa / max(tot_fa, 1), det / max(tot_det, 1)


def main():
    rng = np.random.default_rng(7)
    cfg = C.get_config("v2")
    vcfg, sc = cfg["vad"], cfg["speaker"]
    print("Embedding Mini LibriSpeech dev-clean-2 through the simulated live channel ...")
    data = build(vcfg, rng)
    spks = sorted(data)
    cal, held = spks[0::2], spks[1::2]
    out = {}

    thr_now = sc["verify_thr"]
    _, frr, far, mt, mn = verification(data, held, thr=thr_now)
    out["current_verify_thr"] = {"thr": thr_now, "heldout_FRR": frr, "heldout_FAR": far, "median_target": mt,
                                 "median_nontarget": mn}
    print(f"(1) current verify_thr {thr_now}: held-out FRR {frr:.3f} FAR {far:.3f} (median target {mt:.2f}, "
          f"non-target {mn:.2f})")
    best = None
    for adapt in (None, 0.70, 0.75, 0.80):
        thr, _, _, _, _ = verification(data, cal, adapt_thr=adapt)
        _, frr, far, mt, mn = verification(data, held, adapt_thr=adapt, thr=thr)
        out[f"verify_adapt_{adapt}"] = {"thr_cal_eer": thr, "heldout_FRR": frr, "heldout_FAR": far,
                                        "median_target": mt, "median_nontarget": mn}
        print(f"    adapt {adapt}: EER thr {thr:.3f} -> held-out FRR {frr:.3f} FAR {far:.3f} "
              f"(median target {mt:.2f}, non-target {mn:.2f})")
        if best is None or frr + far < best[1]:
            best = (adapt, frr + far, thr)

    fa_old = old_rule_fa(data, held, sc["diff_thr"])
    out["old_multi_rule_FA"] = fa_old
    print(f"(2) old MULTIPLE_SPEAKERS rule (diff_thr {sc['diff_thr']}): false alarm on {100 * fa_old:.1f}% of "
          f"single-speaker 10 s windows")

    adapt, _, thr = best
    print(f"(3) new rule with adapt {adapt}:")
    rows = []
    for other in np.round(np.arange(thr - 0.15, thr + 0.01, 0.025), 3):
        fa_c, det_c = new_rule_rates(data, cal, other, adapt, rng)
        rows.append((float(other), fa_c, det_c))
        print(f"    other_thr {other:.3f}: cal FA {100 * fa_c:.1f}%  detection {100 * det_c:.1f}%")
    pick = max((r for r in rows if r[1] <= 0.01), key=lambda r: r[2], default=min(rows, key=lambda r: r[1]))
    fa_h, det_h = new_rule_rates(data, held, pick[0], adapt, rng)
    out["new_multi_rule"] = {"adapt_thr": adapt, "other_thr": pick[0], "cal_FA": pick[1], "cal_detection": pick[2],
                             "heldout_FA": fa_h, "heldout_detection": det_h, "verify_thr": thr}
    print(f"    picked other_thr {pick[0]:.3f} (cal FA <= 1%): held-out FA {100 * fa_h:.1f}%, "
          f"detection {100 * det_h:.1f}%")
    (paths.RESULTS / "VOICE2_speaker_live_calibration.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
