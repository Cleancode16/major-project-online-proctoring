"""
Live proctoring over WebRTC (flow.txt PART II-E, §E.2).

The browser sends its webcam + microphone tracks; this module receives them with aiortc, runs the
v2 VideoAnalyzer / AudioAnalyzer on them and pushes one JSON status per video sample back over the
WebRTC data channel the client opened ("status"). stop() builds the same report as the CLI.

Threading: all aiortc work happens on the server's asyncio loop. Model work runs in two single-thread
executors per session (video, audio) so MediaPipe/TF objects are always used from the same thread and
the asyncio loop never blocks.
"""

import asyncio
import json
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

import config as C
from aggregation.evidence import counter as evidence_counter
from aggregation.report import _jsonable, build_report, save_report
from audio import io as aio
from audio.monitor import AudioAnalyzer
from video.pipeline import SCENE_TYPES, VideoAnalyzer
from video.temporal import LivePersistence, samples_for

SR = 16000

# (key used by the React client / AlertPanel, violation type, label)
VIDEO_MONITORS = [
    ("identity", C.IDENTITY_MISMATCH, "Identity"),
    ("pose", C.HEAD_POSE_DEVIATION, "Head pose"),
    ("gaze", C.GAZE_AWAY, "Gaze"),
    ("person", C.MULTIPLE_PERSONS, "Persons"),
    ("object", C.PROHIBITED_OBJECT, "Objects"),
    ("face", C.FACE_NOT_VISIBLE, "Face visible"),
    ("spoof", C.FACE_SPOOF, "Liveness"),          # FAS1
]

# live alert key -> evidence group (flow.txt §C.6); final counts come from the report
ALERT_GROUP = {"object": "object", "person": "multiple_faces", "pose": "head_pose", "voice": "voice", "whisper": "voice",
               "spoof": "spoof"}

# FUS1 live alert key -> violation type (the confirmation rule is the report's buffer rule for that type)
LIVE_TYPE = {"identity": C.IDENTITY_MISMATCH, "pose": C.HEAD_POSE_DEVIATION, "gaze": C.GAZE_AWAY,
             "person": C.MULTIPLE_PERSONS, "object": C.PROHIBITED_OBJECT, "face": C.FACE_NOT_VISIBLE,
             "spoof": C.FACE_SPOOF, "voice": C.VOICE_MISMATCH, "sound": C.SUSPICIOUS_SOUND, "whisper": C.WHISPERING}

ALERT_TEXT = {
    "identity": "Face does not match the enrolled candidate",
    "pose": "Head turned away from the screen",
    "gaze": "Eyes looking away from the screen",
    "person": "More than one person in view",
    "object": "Prohibited object in view",
    "face": "Face not visible",
    "spoof": "Face looks like a photo or a screen (spoofing)",
    "voice": "Voice does not match the enrolled candidate",
    "sound": "Suspicious sound",
    "whisper": "Whispering detected",
}


def _sample_details(s):
    def f(v, fmt):
        return format(v, fmt) if v is not None else "-"
    pose = ("calibrating" if s.get("phase") == "calibration" else
            (f"Δyaw {s['rel_yaw']:+.0f}°, Δpitch {s['rel_pitch']:+.0f}°, Δroll {s['rel_roll']:+.0f}°"
             if "rel_yaw" in s else "not measured"))
    return {
        C.IDENTITY_MISMATCH: (f"match score {f(s.get('identity_score'), '.2f')}"
                              if s.get("identity_score") is not None else "no face"),
        C.HEAD_POSE_DEVIATION: pose,
        C.GAZE_AWAY: (s.get("gaze") or {}).get("label") or "not measured",
        C.MULTIPLE_PERSONS: f"{s.get('persons', 0)} person(s)",
        C.PROHIBITED_OBJECT: ", ".join(s.get("objects", {})) or "none",
        C.FACE_NOT_VISIBLE: "visible" if s.get("face_visible") else "not visible",
        C.FACE_SPOOF: (f"real-face score {s['spoof_score']:.2f}" if s.get("spoof_score") is not None
                       else "not measured"),
    }


def monitors_for_sample(s, live=None):
    """Per-monitor state for one video sample: na (not measured) / ok / warn (raw hit) / bad (confirmed)."""
    det = _sample_details(s)
    out = {}
    for key, vtype, label in VIDEO_MONITORS:
        raw = s["raw"].get(vtype)
        confirmed = (live or {}).get(vtype, raw)
        state = "na" if raw is None else ("bad" if confirmed else ("warn" if raw else "ok"))
        out[key] = {"label": label, "type": vtype, "state": state, "detail": det[vtype]}
    return out


class LiveSession:
    def __init__(self, session_id, candidate, cfg, face_embedding, voiceprint=None):
        self.id = session_id
        self.candidate = candidate
        self.cfg = cfg
        self.face_embedding = np.asarray(face_embedding)
        self.voiceprint = voiceprint
        self.fps = cfg["sampling"]["fps"]
        self.pc = RTCPeerConnection()
        self.channel = None
        self.vexec = ThreadPoolExecutor(1, thread_name_prefix=f"video-{session_id}")
        self.aexec = ThreadPoolExecutor(1, thread_name_prefix=f"audio-{session_id}")
        self.va = None                      # created inside the video thread (MediaPipe graphs)
        self.aa = AudioAnalyzer(cfg, voiceprint)
        self.tasks = []
        self.closed = False
        self.started = time.time()
        # video state (video thread / loop)
        self.t0 = None
        self.last_t = 0.0
        self.next_t = 0.0
        self.busy = False
        self.brightness = []
        self.light = deque(maxlen=200)
        k = samples_for(1.0, self.fps)
        self.vpers = {vt: LivePersistence(k) for _, vt, _ in VIDEO_MONITORS}
        self.vprev = {}
        # audio state (audio thread / loop)
        self.audio_buf, self.audio_pending, self.audio_samples = [], 0, 0
        self.apers = {key: LivePersistence(2) for key in ("voice", "sound", "whisper")}
        self.aprev = {}
        self.audio_status = {"state": "waiting"}
        # shared
        self.counts = {}
        # live session shares for the evidence limits (LIM2): flagged / analysed samples, flagged / audio ticks
        self.vn = self.vn_analysis = 0
        self.von = {C.HEAD_POSE_DEVIATION: 0, C.MULTIPLE_PERSONS: 0, C.PROHIBITED_OBJECT: 0, C.FACE_SPOOF: 0}
        self.aticks = self.avoice = 0
        self.warnings = set()
        self.pending_alerts = deque(maxlen=50)
        self.live_events = {}               # FUS1: alert key -> open live event {id, t0, confirmed}
        self.live_seq = {}
        self.lock = threading.Lock()
        self.timing = {"video_process": 0.0, "audio_process": 0.0}
        self.connection_state = "new"

    # ------------------------------------------------------------------ signalling
    async def accept(self, sdp, sdp_type):
        @self.pc.on("datachannel")
        def on_datachannel(channel):
            self.channel = channel

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                self.tasks.append(asyncio.ensure_future(self._video_loop(track)))
            elif track.kind == "audio":
                self.tasks.append(asyncio.ensure_future(self._audio_loop(track)))

        @self.pc.on("connectionstatechange")
        async def on_state():
            self.connection_state = self.pc.connectionState

        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)   # aiortc finishes ICE gathering here
        return self.pc.localDescription.sdp, self.pc.localDescription.type

    def _send(self, message):
        ch = self.channel
        if ch is not None and ch.readyState == "open":
            ch.send(json.dumps(message, default=_jsonable))

    # ------------------------------------------------------------------ live event lifecycle (FUS1)
    def _live_event(self, key, on, t, details, lead_s):
        """
        One live event per run of a debounced flag, with a stable event_id across its updates:
          provisional  the debounce fired (the signal lasted lead_s)
          confirmed    the run reached the type's confirmation duration (buffer.policies[type].confirm_s)
          ended        the signal stopped (final status confirmed / not confirmed)
        Only confirmed events are counted. The final report re-validates everything offline.
        """
        vtype = LIVE_TYPE[key]
        bc = self.cfg.get("buffer", {})
        confirm_s = bc["policies"].get(vtype, {}).get("confirm_s", 0.0) if bc.get("enabled") else 0.0
        ev, out = self.live_events.get(key), []

        def msg(stage, status, reason):
            return {"type": key, "event_id": ev["id"], "stage": stage, "status": status,
                    "message": ALERT_TEXT[key], "details": details, "t": round(ev["t0"], 2),
                    "t_update": round(t, 2), "duration_s": round(t - ev["t0"] + lead_s, 2), "reason": reason}

        if on and ev is None:
            n = self.live_seq[key] = self.live_seq.get(key, 0) + 1
            ev = self.live_events[key] = {"id": f"live-{key}-{n:03d}", "t0": t, "confirmed": False}
            out.append(msg("provisional", "provisional",
                           f"detected for {lead_s:.1f} s; confirmed after {confirm_s:.1f} s"))
        if on and not ev["confirmed"] and t - ev["t0"] + lead_s >= confirm_s:
            ev["confirmed"] = True
            with self.lock:
                self.counts[key] = self.counts.get(key, 0) + 1
            out.append(msg("confirmed", "confirmed",
                           f"persisted {t - ev['t0'] + lead_s:.1f} s (rule: {confirm_s:.1f} s)"))
        if not on and ev is not None:
            d = t - ev["t0"] + lead_s
            out.append(msg("ended", "confirmed" if ev["confirmed"] else "not_confirmed",
                           f"ended after {d:.1f} s" + ("" if ev["confirmed"] else
                                                      f", before the {confirm_s:.1f} s rule - not counted")))
            del self.live_events[key]
        if len(out) >= 2 and out[0]["stage"] == "provisional" and out[1]["stage"] == "confirmed":
            out.pop(0)                      # confirmed on the first tick: one alert, not two
        return out

    # ------------------------------------------------------------------ video
    async def _video_loop(self, track):
        loop = asyncio.get_running_loop()
        step = 1.0 / self.fps
        while not self.closed:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break
            now = time.monotonic()
            if self.t0 is None:
                self.t0 = now
            t = now - self.t0
            self.last_t = t
            if self.busy or t < self.next_t:
                continue                    # newest-frame policy: drop frames while busy
            self.next_t = t + step
            self.busy = True
            img = frame.to_ndarray(format="bgr24")
            fut = loop.run_in_executor(self.vexec, self._process_video, img, t)
            fut.add_done_callback(self._on_video_done)

    def _on_video_done(self, fut):
        self.busy = False
        try:
            status = fut.result()
        except Exception as exc:
            traceback.print_exc()
            self._send({"type": "error", "message": f"Video analysis error: {exc}"})
            return
        self._send(status)

    def _process_video(self, img, t):
        if self.va is None:
            self.va = VideoAnalyzer(self.face_embedding, self.cfg, self.fps, live=True)
        tp = time.perf_counter()
        s = self.va.process(img, t)
        self.timing["video_process"] += time.perf_counter() - tp

        if len(self.brightness) < 5:
            self.brightness.append(float(img.mean()))
            if len(self.brightness) == 5 and np.mean(self.brightness) < 5:
                with self.lock:
                    self.warnings.add("camera_black")

        lc = self.cfg.get("lighting", {})
        if lc and s.get("brightness") is not None:        # DET2: live low-light warning (rolling mean)
            self.light.append(s["brightness"])
            n = samples_for(lc["window_s"], self.va.rate)
            recent = list(self.light)[-n:]
            with self.lock:
                if len(recent) >= n and float(np.mean(recent)) < lc["low_mean"]:
                    self.warnings.add("low_light")
                else:
                    self.warnings.discard("low_light")

        live = {}
        alerts = []
        k = samples_for(1.0, self.va.rate)          # FPS10: 1 s debounce at the measured live rate
        k_scene = samples_for(1.0, self.va.scene_rate())   # PERF2: same 1 s over the real YOLO observations
        for vtype, p in self.vpers.items():
            p.resize(k_scene if vtype in SCENE_TYPES else k)
        for key, vtype, _ in VIDEO_MONITORS:
            if vtype in SCENE_TYPES and s.get("scene_held"):
                on = self.vprev.get(vtype, False)   # PERF2: reused YOLO result - keep the state, feed no evidence
            else:
                on = self.vpers[vtype](bool(s["raw"].get(vtype, False)))
            live[vtype] = on
            if on or self.vprev.get(vtype):
                alerts += self._live_event(key, on, t, _sample_details(s)[vtype], 1.0)   # 1 s debounce
            self.vprev[vtype] = on
        monitors = monitors_for_sample(s, live)

        with self.lock:
            self.vn += 1
            if s["phase"] != "calibration":
                self.vn_analysis += 1
            for vt in self.von:
                self.von[vt] += bool(live.get(vt))
            while self.pending_alerts:
                alerts.append(self.pending_alerts.popleft())
            counts = dict(self.counts)
            warnings = sorted(self.warnings)
            audio = dict(self.audio_status)
        base = self.va.baseline
        cal = self.cfg["calibration"]
        evidence = {}
        pct = self._live_percentages()
        for group, limit in self.cfg["evidence"]["limits"].items():
            n = sum(c for key, c in counts.items() if ALERT_GROUP.get(key) == group)
            evidence[group] = evidence_counter(group, n, pct[group], limit)
        return {
            "evidence": evidence,
            "session_violated": any(e["exceeded"] for e in evidence.values()),
            "type": "status", "t": round(t, 2), "phase": s["phase"],
            "calibrated": base is not None,
            "calibration": {"seconds": cal["duration_s"], "accepted": len(self.va.cal["yaw"]),
                            "pose_confidence": (self.va.pose_cal or {}).get("confidence")},   # HP3
            "monitors": monitors, "counts": counts, "alerts": alerts, "audio": audio,
            "warnings": warnings, "samples": self.va.processed,
            "sample_rate_hz": round(self.va.processed / max(t, 1e-6), 2) if t > 0 else None,
        }

    def _live_percentages(self):
        """Running estimate of each group's share of the session (the final values come from the report)."""
        with self.lock:
            f = lambda num, den: round(100.0 * num / den, 2) if den else 0.0  # noqa: E731
            return {"head_pose": f(self.von[C.HEAD_POSE_DEVIATION], self.vn_analysis),
                    "multiple_faces": f(self.von[C.MULTIPLE_PERSONS], self.vn),
                    "object": f(self.von[C.PROHIBITED_OBJECT], self.vn),
                    "spoof": f(self.von[C.FACE_SPOOF], self.vn),
                    "voice": f(self.avoice, self.aticks)}

    # ------------------------------------------------------------------ audio
    async def _audio_loop(self, track):
        import av
        loop = asyncio.get_running_loop()
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
        while not self.closed:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break
            for f in resampler.resample(frame):
                x = f.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
                self.audio_buf.append(x)
                self.audio_pending += len(x)
            if self.audio_pending >= SR // 2:     # hand over every 0.5 s
                chunk = np.concatenate(self.audio_buf)
                self.audio_buf, self.audio_pending = [], 0
                loop.run_in_executor(self.aexec, self._process_audio, chunk)

    def _process_audio(self, chunk):
        ta = time.perf_counter()
        try:
            self.aa.feed(chunk)
        except Exception:
            traceback.print_exc()
            return
        self.audio_samples += len(chunk)
        cfg = self.cfg
        if self.audio_samples >= 3 * SR and \
                aio.source_status(self.aa._tail, cfg["audio"]["silent_rms_dbfs"]) != "ok":
            with self.lock:
                self.warnings.add("mic_silent")

        st = self.aa.live_status()
        cats = st.get("categories", {})
        th = cfg["yamnet"]["thresholds"]
        sound = max(cfg["yamnet"]["suspicious"], key=lambda c: cats.get(c, 0.0)) if cats else None
        sound_on = bool(cats) and cats.get(sound, 0.0) >= th[sound]
        speech = st.get("speech_prob", 0.0) >= cfg["vad"]["start_threshold"]
        sim = st.get("speaker_sim")
        voice_bad = sim is not None and sim < cfg["speaker"]["verify_thr"]
        whisper = cats.get("whisper", 0.0) >= th["whisper"]

        flags = {"voice": voice_bad, "sound": sound_on, "whisper": whisper}
        new_alerts = []
        t_audio = self.audio_samples / SR
        for key, raw in flags.items():
            on = self.apers[key](raw)
            if on or self.aprev.get(key):
                details = {"voice": f"similarity {sim:.2f}" if sim is not None else "",
                           "sound": f"{sound} ({cats.get(sound, 0):.2f})" if sound else "",
                           "whisper": f"score {cats.get('whisper', 0):.2f}"}[key]
                new_alerts += self._live_event(key, on, t_audio, details, 0.0)
            self.aprev[key] = on

        def state(raw, confirmed):
            return "bad" if confirmed else ("warn" if raw else "ok")
        status = {
            "state": "ok", "t": round(self.audio_samples / SR, 2),
            "speech_prob": round(st.get("speech_prob", 0.0), 2), "speaker_sim": sim,
            "sound": sound if sound_on else None,
            "monitors": {
                "speech": {"label": "Speech", "state": "warn" if speech else "ok",
                           "detail": f"speech prob {st.get('speech_prob', 0.0):.2f}"},
                "voice": {"label": "Voice", "state": "na" if self.voiceprint is None else
                          state(voice_bad, self.aprev.get("voice")),
                          "detail": "no voiceprint" if self.voiceprint is None else
                          (f"similarity {sim:.2f}" if sim is not None else "no speech")},
                "sound": {"label": "Sounds", "state": state(sound_on or whisper,
                                                            self.aprev.get("sound") or self.aprev.get("whisper")),
                          "detail": (sound if sound_on else ("whisper" if whisper else "normal"))},
            },
        }
        self.timing["audio_process"] += time.perf_counter() - ta
        with self.lock:
            self.audio_status = status
            self.aticks += 1
            self.avoice += bool(self.aprev.get("voice") or self.aprev.get("whisper"))
            self.pending_alerts.extend(new_alerts)       # counted in _live_event once confirmed

    # ------------------------------------------------------------------ stop
    async def stop(self):
        """Close the connection, drain the analyzers and build/save the report. Returns (report, path)."""
        self.closed = True
        for task in self.tasks:
            task.cancel()
        await self.pc.close()
        loop = asyncio.get_running_loop()
        if self.audio_buf:
            chunk = np.concatenate(self.audio_buf)
            self.audio_buf, self.audio_pending = [], 0
            await loop.run_in_executor(self.aexec, self._process_audio, chunk)
        await loop.run_in_executor(self.aexec, lambda: None)      # wait for queued audio work
        try:
            return await loop.run_in_executor(self.vexec, self._finalize)   # after queued video work
        finally:
            self.vexec.shutdown(wait=False)
            self.aexec.shutdown(wait=False)

    def _finalize(self):
        if self.va is None or self.va.processed == 0:
            raise ValueError("No video was received from the browser; nothing to report.")
        elapsed = max(self.last_t, 1e-6)
        achieved = self.va.processed / elapsed
        if 0 < achieved < self.va.rate:     # persistence is defined in seconds (flow.txt §E.2)
            self.va.rate = achieved
        video = self.va.finalize()
        video["video"] = {"duration_s": round(elapsed, 1), "source": "webrtc"}
        audio = self.aa.finalize() if self.audio_samples > 0 else None
        meta = {"candidate": self.candidate, "mode": "live-webrtc", "session_id": self.id,
                "voiceprint": self.voiceprint is not None,
                "live_sample_rate_hz": round(achieved, 2),
                "camera_black": "camera_black" in self.warnings,
                "live_processing_seconds": {k: round(v, 1) for k, v in self.timing.items()}}
        report = build_report(video, audio, self.cfg, meta)
        if "camera_black" in self.warnings:
            report["warnings"].append("CAMERA_BLACK: the camera image was black (covered lens, privacy shutter "
                                      "or blocked camera access).")
        path = save_report(report, video, audio=audio)   # evidence package + counters + upload (flow.txt §C.5/§C.6)
        return report, path
