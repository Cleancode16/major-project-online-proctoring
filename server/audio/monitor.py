"""
AudioAnalyzer: VAD + YAMNet + speaker verification -> per-window flags + events (flow.txt §B.2-§B.6).

Offline:  res = analyze_audio(x, cfg, voiceprint)
Live:     aa = AudioAnalyzer(cfg, voiceprint); aa.feed(chunk) ...; aa.live_status(); aa.finalize()
"""

import numpy as np

import config as C
from aggregation.events import events_from_flags
from audio import io as aio
from audio import speaker, yamnet
from audio.vad import VAD, overlap
from video.temporal import filter_runs, samples_for

SR = aio.SR
WIN, HOP = yamnet.WIN, yamnet.HOP
WIN_S, HOP_S = WIN / SR, HOP / SR


def voicing_fraction(x, segs, t0, t1, corr_thr=0.5):
    """Fraction of 32 ms frames inside speech within [t0, t1) that show pitch (80-400 Hz autocorr peak)."""
    frames = []
    for s, e in segs:
        a, b = max(s, t0), min(e, t1)
        if b - a < 0.032:
            continue
        seg = x[int(a * SR):int(b * SR)]
        n = len(seg) // 512
        if n:
            frames.append(seg[: n * 512].reshape(n, 512))
    if not frames:
        return None
    f = np.concatenate(frames).astype(np.float64)
    f = f - f.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(f, n=1024, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :512]
    peak = ac[:, 40:201].max(axis=1) / (ac[:, 0] + 1e-12)
    return float(np.mean(peak >= corr_thr))


def merge_close(segs, gap_s):
    """VAD3: join segments separated by <= gap_s into utterances."""
    out = []
    for s, e in segs:
        if out and s - out[-1][1] <= gap_s:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def bandpass(x, band):
    """VAD3: zero-phase band-pass (offline) to the pitch region before the voicing test; None = unchanged."""
    if not band or len(x) < 64:
        return x
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, band, btype="bandpass", fs=SR, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def validate_segments(segs, x, win_scores, cfg):
    """
    VAD2 segment validation (flow.txt §B.2 "Accuracy upgrade"):
      energy gate   segment RMS >= noise floor (10th percentile of 32 ms frame RMS) + vad.min_seg_snr_db
      short check   segments < vad.validate_below_s need YAMNet speech >= vad.yamnet_confirm (or whisper)
                    in an overlapping window (coughs, clicks, knocks are short and not speech for YAMNet)
    Returns (kept segments, {"energy": n, "not_speech": n}).
    """
    vc = cfg["vad"]
    rejected = {"energy": 0, "not_speech": 0}
    if not segs:
        return segs, rejected
    n = len(x) // 512
    floor_db = None
    if vc.get("min_seg_snr_db") is not None and n:
        frame_db = 20 * np.log10(np.sqrt(np.mean(x[:n * 512].reshape(n, 512).astype(np.float64) ** 2, axis=1)) + 1e-12)
        floor_db = float(np.percentile(frame_db, 10))
    th = cfg["yamnet"]["thresholds"]
    kept = []
    for s, e in segs:
        seg = x[int(s * SR):int(e * SR)]
        if floor_db is not None and len(seg):
            if aio.rms_dbfs(seg) < floor_db + vc["min_seg_snr_db"]:
                rejected["energy"] += 1
                continue
        if vc.get("validate_below_s") and (e - s) < vc["validate_below_s"] and win_scores:
            k0, k1 = max(0, int(s / HOP_S) - 1), min(len(win_scores) - 1, int(e / HOP_S))
            confirmed = False
            for k in range(k0, k1 + 1):
                if k * HOP_S < e and k * HOP_S + WIN_S > s:
                    cats, _ = yamnet.categorize(win_scores[k])
                    if cats["speech"] >= vc["yamnet_confirm"] or cats["whisper"] >= th["whisper"]:
                        confirmed = True
                        break
            if not confirmed:
                rejected["not_speech"] += 1
                continue
        kept.append((s, e))
    return kept, rejected


class AudioAnalyzer:
    def __init__(self, cfg, voiceprint=None):
        self.cfg = cfg
        self.vp = None if voiceprint is None else np.asarray(voiceprint, np.float32)
        self.vad = VAD(cfg["vad"])
        self._buf = []          # int16 chunks (whole session; ~115 MB per hour)
        self.n = 0
        self._tail = np.zeros(0, np.float32)   # last <= 30 s as float, for live windows
        self._tail_start = 0                   # absolute sample index of _tail[0]
        self.win_scores = []    # per window: 521 YAMNet scores
        self._live = True
        self._live_embs = []    # VOICE2: live chunks that clearly match the enrollment (session print)
        self._live_sp = None

    # ------------------------------------------------------------ input
    def feed(self, chunk):
        chunk = np.asarray(chunk, np.float32)
        self._buf.append((np.clip(chunk, -1, 1) * 32767).astype(np.int16))
        self.n += len(chunk)
        self.vad.feed(chunk)
        if self._live:
            self._tail = np.concatenate([self._tail, chunk])
            drop = max(0, len(self._tail) - 30 * SR)
            self._tail, self._tail_start = self._tail[drop:], self._tail_start + drop
            while len(self.win_scores) * HOP + yamnet.CTX <= self.n:
                k = len(self.win_scores)
                self.win_scores.append(yamnet.window_scores(self._slice(k * HOP, k * HOP + yamnet.CTX)))

    def _audio(self):
        return np.concatenate(self._buf).astype(np.float32) / 32767 if self._buf else np.zeros(0, np.float32)

    def _slice(self, a, b):
        return self._tail[a - self._tail_start:b - self._tail_start]

    def live_status(self):
        """Cheap status for the UI: last window's categories, recent speech probability, level."""
        if not self.win_scores:
            return {"speech_prob": 0.0, "categories": {}, "top": []}
        cats, top = yamnet.categorize(self.win_scores[-1])
        recent = self.vad.probs[-30:]
        st = {"speech_prob": float(max(recent)) if recent else 0.0, "categories": cats, "top": top,
              "t": round(self.n / SR, 2)}
        sc = self.cfg["speaker"]
        n_fr = max(1, int(sc["chunk_s"] / self.vad.frame_s))
        last = self.vad.probs[-n_fr:]
        speech_frac = (sum(p >= self.cfg["vad"]["start_threshold"] for p in last) / len(last)) if last else 0.0
        # VOICE2: only judge the voice when the last chunk is mostly speech (silence / noise lowered the similarity)
        if self.vp is not None and len(last) >= n_fr and speech_frac >= sc.get("live_min_speech_frac", 0.0):
            tail = self._tail[-int(sc["chunk_s"] * SR):]
            if len(tail) >= SR:
                e = speaker.embed(tail)
                sim = speaker.similarity(e, self.vp)
                if sc.get("adapt_thr") is not None and sim >= sc["adapt_thr"]:
                    self._live_embs.append(e)
                    if len(self._live_embs) >= sc.get("adapt_min_chunks", 3):
                        m = np.mean(self._live_embs[-50:], axis=0)
                        self._live_sp = m / (np.linalg.norm(m) + 1e-12)
                if self._live_sp is not None:
                    sim = max(sim, speaker.similarity(e, self._live_sp))
                st["speaker_sim"] = round(sim, 3)
        return st

    # ------------------------------------------------------------ analysis
    def finalize(self):
        cfg, yc, sc = self.cfg, self.cfg["yamnet"], self.cfg["speaker"]
        x = self._audio()
        dur = len(x) / SR
        status = aio.source_status(x, cfg["audio"]["silent_rms_dbfs"])
        base = {"status": status, "duration_s": round(dur, 2), "vad_engine": self.vad.engine,
                "voiceprint": self.vp is not None}
        if status != "ok":
            return {**base, "windows": [], "segments": [], "chunks": [], "speech_s": 0.0, "events": []}

        if len(self.win_scores) == 0 or not self._live:
            self.win_scores = list(yamnet.scores(x))
        n_win = max(0, min(len(self.win_scores), int(np.floor((len(x) - WIN) / HOP)) + 1)) if len(x) >= WIN else 0
        segs, rejected = validate_segments(self.vad.segments(), x, self.win_scores, cfg)   # VAD2
        base["vad_rejected"] = rejected
        merge_gap = cfg["vad"].get("merge_gap_s") or 0.0
        utterances = merge_close(segs, merge_gap)                                          # VAD3
        base["utterances"] = len(utterances)
        xb = bandpass(x, yc.get("voicing_band_hz"))      # pitch region for the whisper voicing cue (VAD3)
        chunks = self._speaker_chunks(x, utterances) if utterances else []

        windows = []
        for k in range(n_win):
            t0, t1 = k * HOP_S, k * HOP_S + WIN_S
            slot = (0.0 if k == 0 else t0 + HOP_S / 2, t0 + HOP_S / 2 + HOP_S)
            cats, top = yamnet.categorize(self.win_scores[k])
            sfrac = overlap(segs, t0, t1) / WIN_S
            w = {"k": k, "t_start": round(t0, 2), "t_end": round(t1, 2), "slot": slot,
                 "speech_frac": round(sfrac, 3), "is_speech": sfrac >= cfg["vad"]["window_speech_frac"],
                 "cats": {c: round(v, 3) for c, v in cats.items()}, "top": top, "raw": {}}
            if w["is_speech"]:
                w["voiced_frac"] = voicing_fraction(xb, segs, t0, t1, yc["voicing_corr"])
            mid = (t0 + t1) / 2
            near = [c for c in chunks if c["t0"] <= mid <= c["t1"]]
            if near:
                sims = [c["sim_s"] for c in near if c.get("sim_s") is not None]
                w["speaker_sim"] = round(min(sims), 3) if sims else None
            windows.append(w)

        self._raw_flags(windows, chunks)
        rate = 1.0 / HOP_S
        persist = {
            C.SPEECH_DETECTED: (1, 0),
            C.VOICE_MISMATCH: (samples_for(sc["persist_s"], rate), 0),
            C.MULTIPLE_SPEAKERS: (yc["persist_k"], yc["persist_n"] - yc["persist_k"]),
            C.WHISPERING: (yc.get("whisper_persist", yc["persist_k"]), 0),        # VAD3: consecutive windows
            C.SUSPICIOUS_SOUND: (yc["persist_k"], yc["persist_n"] - yc["persist_k"]),
        }
        speech_types = {C.SPEECH_DETECTED, C.VOICE_MISMATCH, C.MULTIPLE_SPEAKERS, C.WHISPERING}
        sus_gap = samples_for(yc["suspicious_gap_s"], rate)
        events = []
        slots = [w["slot"] for w in windows]
        for vtype, (k_min, gap) in persist.items():
            if vtype == C.VOICE_MISMATCH and self.vp is None:
                continue
            raw = [w["raw"].get(vtype, False) for w in windows]
            if vtype == C.SUSPICIOUS_SOUND:  # join hits across ring pauses first, then minimum 2 windows
                final = filter_runs(raw, yc["persist_k"], sus_gap)
            else:
                final = filter_runs(raw, k_min, gap)
            for w, f in zip(windows, final):
                w.setdefault("flags", {})[vtype] = bool(f)
            peaks = [self._peak(w, vtype) for w in windows]
            # VAD3: events of a speech-related type closer than merge_gap_s are one event (pauses between phrases)
            gap_s = max(0.05, merge_gap) if vtype in speech_types else 0.05
            events += events_from_flags(slots, final, vtype, "audio", peaks,
                                        details=lambda idx, vt=vtype: self._details(windows, idx, vt), gap_s=gap_s)
        return {**base, "windows": windows, "segments": [(round(a, 2), round(b, 2)) for a, b in segs],
                "chunks": [{k: v for k, v in c.items() if k != "emb"} for c in chunks],
                "speech_s": round(sum(b - a for a, b in segs), 2), "events": sorted(events, key=lambda e: e["start_s"]),
                "_x": x}  # waveform for evidence clips (flow.txt §C.5); never written to the JSON report

    def _speaker_chunks(self, x, segs):
        sc = self.cfg["speaker"]
        L, hop = sc["chunk_s"], sc["chunk_s"] / 2
        per_seg = []
        for si, (s, e) in enumerate(segs):
            seg_chunks = []
            t = s
            while t < e:
                t1 = min(e, t + L)
                if t1 - t >= sc.get("min_chunk_s", 0.6 * L):   # VAD3: no decision on < 1.5 s of speech
                    emb = speaker.embed(x[int(t * SR):int(t1 * SR)])
                    seg_chunks.append({"seg": si, "t0": round(t, 2), "t1": round(t1, 2), "emb": emb})
                if t1 >= e:
                    break
                t += hop
            per_seg.append(seg_chunks)
        all_chunks = [c for sc_ in per_seg for c in sc_]
        ref, sp = self._candidate_prints(all_chunks)
        self.session_print_chunks = 0 if sp is None else int(sum(float(c["emb"] @ ref) >= sc["adapt_thr"]
                                                                  for c in all_chunks))
        if ref is not None:
            for seg_chunks in per_seg:
                if not seg_chunks:
                    continue
                raw = np.array([float(c["emb"] @ ref) for c in seg_chunks])
                cand = np.maximum(raw, [float(c["emb"] @ sp) for c in seg_chunks]) if sp is not None else raw
                p = np.pad(cand, 1, mode="edge") if len(cand) >= 3 and sc["smooth"] >= 3 else None
                sm = np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0) if p is not None else cand
                for c, r, cs, m in zip(seg_chunks, raw, cand, sm):
                    c["sim_enroll"], c["sim"], c["sim_s"] = round(float(r), 3), round(float(cs), 3), round(float(m), 3)
        return all_chunks

    def _candidate_prints(self, chunks):
        """
        VOICE2: (reference, session print). Reference = enrollment voiceprint, or without one the session's
        dominant voice (mean of all chunks). Session print = mean of the chunks that clearly match the reference
        (>= adapt_thr, at least adapt_min_chunks of them): the candidate's voice as heard through the live channel.
        """
        sc = self.cfg["speaker"]
        if not chunks:
            return self.vp, None
        embs = np.stack([c["emb"] for c in chunks])
        ref = self.vp
        if ref is None:
            m = embs.mean(axis=0)
            ref = m / (np.linalg.norm(m) + 1e-12)
        if sc.get("adapt_thr") is None:
            return ref, None
        good = embs[embs @ ref >= sc["adapt_thr"]]
        if len(good) < sc.get("adapt_min_chunks", 3):
            return ref, None
        m = good.mean(axis=0)
        return ref, m / (np.linalg.norm(m) + 1e-12)

    def _raw_flags(self, windows, chunks):
        yc, sc, pol = self.cfg["yamnet"], self.cfg["speaker"], self.cfg["audio_policy"]
        th = yc["thresholds"]
        cum = 0.0
        for w in windows:
            r = w["raw"]
            if w["is_speech"]:
                cum += HOP_S
                r[C.SPEECH_DETECTED] = pol["speech"] == "flag_all" or (
                    pol["speech"] == "flag_after" and cum > pol["allow_speech_s"])
                vf = w.get("voiced_frac")
                # VAD3: the voicing cue only for windows well inside speech (not at its edges)
                unvoiced = (vf is not None and vf < yc["voiced_frac_whisper"]
                            and w["speech_frac"] >= yc.get("whisper_min_speech_frac", 0.0))
                r[C.WHISPERING] = w["cats"]["whisper"] >= th["whisper"] or unvoiced
            else:
                r[C.WHISPERING] = w["cats"]["whisper"] >= th["whisper"]
            r[C.SUSPICIOUS_SOUND] = any(w["cats"][c] >= th[c] for c in yc["suspicious"])
            if self.vp is not None and w["is_speech"] and w.get("speaker_sim") is not None:
                r[C.VOICE_MISMATCH] = w["speaker_sim"] < sc["verify_thr"]
            # multiple speakers: conflicting chunks within +-window/2 of this speech window
            if w["is_speech"]:
                mid = (w["t_start"] + w["t_end"]) / 2
                near = [c for c in chunks if abs((c["t0"] + c["t1"]) / 2 - mid) <= sc["multi_window_s"] / 2]
                if sc.get("multi_rule", "pairwise") == "foreign_run":
                    r[C.MULTIPLE_SPEAKERS] = self._foreign_run(near, mid)
                else:
                    r[C.MULTIPLE_SPEAKERS] = self._conflict(near, mid)

    def _foreign_run(self, near, mid):
        """
        VOICE2: another voice next to the candidate's = a run of >= other_run consecutive chunks whose candidate
        similarity (sim_s: enrollment or session print, median-of-3) is below other_thr, plus at least one
        candidate chunk (>= verify_thr) in the same window, and this window lies on / next to the foreign run.
        """
        sc = self.cfg["speaker"]
        cs = sorted((c for c in near if c.get("sim_s") is not None), key=lambda c: c["t0"])
        if len(cs) < sc["other_run"] + 1 or not any(c["sim_s"] >= sc["verify_thr"] for c in cs):
            return False
        run = []
        for c in cs:
            run = run + [c] if c["sim_s"] < sc["other_thr"] else []
            if len(run) >= sc["other_run"] and run[0]["t0"] - 1.0 <= mid <= run[-1]["t1"] + 1.0:
                return True
        return False

    def _conflict(self, near, mid):
        sc = self.cfg["speaker"]
        if len(near) < 2:
            return False
        here = [c for c in near if c["t0"] - 0.5 <= mid <= c["t1"] + 0.5]
        if not here:
            return False
        embs = np.stack([c["emb"] for c in near])
        for c in here:
            if np.min(embs @ c["emb"]) < sc["diff_thr"]:
                return True
        if self.vp is not None:
            sims = [c.get("sim_s") for c in near if c.get("sim_s") is not None]
            m = sc.get("mix_margin", 0.0)       # VAD3: clear accept and clear reject, not values around thr
            if sims and min(sims) < sc["verify_thr"] - m and max(sims) >= sc["verify_thr"] + m:
                return True
        return False

    def _peak(self, w, vtype):
        if vtype == C.SUSPICIOUS_SOUND:
            return max(w["cats"][c] for c in self.cfg["yamnet"]["suspicious"])
        if vtype == C.WHISPERING:
            return w["cats"]["whisper"]
        if vtype == C.VOICE_MISMATCH:
            return None if w.get("speaker_sim") is None else 1 - w["speaker_sim"]
        return w["speech_frac"]

    def _details(self, windows, idx, vtype):
        ws = [windows[i] for i in idx]
        if vtype == C.SUSPICIOUS_SOUND:
            best = {c: max(w["cats"][c] for w in ws) for c in self.cfg["yamnet"]["suspicious"]}
            return {"category": max(best, key=best.get), "scores": {k: round(v, 3) for k, v in best.items()}}
        if vtype == C.VOICE_MISMATCH:
            sims = [w["speaker_sim"] for w in ws if w.get("speaker_sim") is not None]
            return {"min_similarity": min(sims) if sims else None, "threshold": self.cfg["speaker"]["verify_thr"]}
        if vtype == C.WHISPERING:
            vf = [w["voiced_frac"] for w in ws if w.get("voiced_frac") is not None]
            return {"yamnet_whisper_max": max(w["cats"]["whisper"] for w in ws),
                    "voiced_frac_min": round(min(vf), 3) if vf else None}
        return {}


def analyze_audio(x, cfg, voiceprint=None):
    """Offline analysis of a whole waveform (16 kHz mono float32), or None -> unavailable."""
    if x is None:
        return {"status": "unavailable", "duration_s": 0.0, "windows": [], "segments": [], "chunks": [],
                "speech_s": 0.0, "events": [], "voiceprint": voiceprint is not None}
    aa = AudioAnalyzer(cfg, voiceprint)
    aa._live = False
    aa.feed(x)
    return aa.finalize()
