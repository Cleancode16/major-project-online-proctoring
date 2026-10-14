import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import {
  Play, Square, RotateCcw, Video, Upload, AlertTriangle, AlertCircle, Loader, Clock, Activity, FileVideo,
  Mic, Smartphone, Users, MoveHorizontal, Ban,
} from 'lucide-react';
import StatusIndicator from './StatusIndicator';
import AlertPanel from './AlertPanel';

// Test screen (flow.txt PART II-D, D.2).
//  Live: webcam + mic go to the server over WebRTC; the server answers with one JSON status per video
//        sample on the "status" data channel (server/webrtc_session.py).
//  Recorded: a video file is uploaded and analysed as a background job; progress is polled.

const STATE_STYLE = {
  ok: 'badge-success',
  warn: 'badge-warning',
  bad: 'badge-danger',
  na: 'badge-muted',
};
const STATE_WORD = { ok: 'OK', warn: 'check', bad: 'VIOLATION', na: 'n/a' };
const WARNING_TEXT = {
  camera_black: 'The camera image is black: the lens may be covered, a privacy shutter closed, or camera access blocked (Windows Settings > Privacy & security > Camera).',
  mic_silent: 'The microphone is silent: it may be muted or blocked (Windows Settings > Privacy & security > Microphone). Audio cannot be monitored.',
  low_light: 'The room is too dark: turn on a light in front of you. Phones, other persons and eye gaze are detected less reliably in a dark image.',
};

// Evidence groups (flow.txt §C.5/§C.6): live counters vs configured thresholds
const EVIDENCE_GROUPS = [
  ['voice', 'Voice', Mic],
  ['object', 'Objects', Smartphone],
  ['multiple_faces', 'Multiple faces', Users],
  ['head_pose', 'Head pose', MoveHorizontal],
];

const EvidenceCounters = ({ evidence }) => (
  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
    {EVIDENCE_GROUPS.map(([key, label, Icon]) => {
      const c = evidence?.[key] || { count: 0, threshold: '-', exceeded: false };
      return (
        <div key={key} className={`rounded-lg p-3 ring-1 ${c.exceeded
          ? 'bg-red-50 ring-red-200 text-red-900 dark:bg-red-950/50 dark:ring-red-900 dark:text-red-100'
          : 'surface-muted ring-slate-200 dark:ring-slate-800'}`}>
          <p className="text-xs font-medium opacity-80 inline-flex items-center gap-1"><Icon className="w-4 h-4" /> {label}</p>
          {c.percentage != null ? (
            <>
              <p className="text-xl font-bold">{c.percentage.toFixed(1)}%<span className="text-sm font-normal opacity-70"> / {c.limit_text}</span></p>
              <p className="text-xs opacity-80">{c.count} event(s){c.exceeded ? ' · limit exceeded' : ''}</p>
            </>
          ) : (
            <>
              <p className="text-xl font-bold">{c.count}<span className="text-sm font-normal opacity-70"> / {c.threshold}</span></p>
              <p className="text-xs opacity-80">{c.exceeded ? 'limit exceeded' : 'events / limit'}</p>
            </>
          )}
        </div>
      );
    })}
  </div>
);

const apiError = (err, fallback) => err.response?.data?.error || err.message || fallback;

const waitForIceGathering = (pc, timeoutMs = 3000) =>
  new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') {
      resolve();
      return;
    }
    const done = () => {
      pc.removeEventListener('icegatheringstatechange', check);
      resolve();
    };
    const check = () => pc.iceGatheringState === 'complete' && done();
    pc.addEventListener('icegatheringstatechange', check);
    setTimeout(done, timeoutMs);
  });

const MonitorBadges = ({ monitors, audio }) => {
  const items = [
    ...Object.entries(monitors || {}),
    ...Object.entries(audio?.monitors || {}),
  ];
  if (!items.length) return null;
  return (
    <div className="flex flex-wrap gap-2">
      {items.map(([key, m]) => (
        <div key={key} className={`badge text-xs shadow-sm ${STATE_STYLE[m.state] || STATE_STYLE.na}`} title={m.detail}>
          <span className="font-semibold">{m.label}</span> {STATE_WORD[m.state] || m.state}
          <span className="opacity-75 font-normal">· {m.detail}</span>
        </div>
      ))}
    </div>
  );
};

const ProctoringSession = ({ candidate, onFinished, onReset }) => {
  const [mode, setMode] = useState('live'); // 'live' | 'recorded'
  const [phase, setPhase] = useState('idle'); // idle | connecting | running | stopping
  const [connectionState, setConnectionState] = useState('closed');
  const [status, setStatus] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [error, setError] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  // recorded mode
  const [file, setFile] = useState(null);
  const [useAudio, setUseAudio] = useState(true);
  const [job, setJob] = useState(null);

  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const streamRef = useRef(null);
  const sessionIdRef = useRef(null);
  const timerRef = useRef(null);
  const pollRef = useRef(null);

  const releaseMedia = () => {
    clearInterval(timerRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    pcRef.current?.close();
    pcRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  useEffect(() => () => {
    // leaving the screen: end any running session on the server too
    const sid = sessionIdRef.current;
    releaseMedia();
    clearInterval(pollRef.current);
    if (sid) axios.post(`/api/session/${sid}/stop`).catch(() => {});
  }, []);

  const handleStatus = (msg) => {
    if (msg.type === 'error') {
      setError(msg.message);
      return;
    }
    setStatus(msg);
    if (msg.alerts?.length) setAlerts((prev) => [...prev, ...msg.alerts].slice(-200));
  };

  const startLive = async () => {
    setError(null);
    setAlerts([]);
    setStatus(null);
    setPhase('connecting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 15 } },
        audio: true,
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;

      const pc = new RTCPeerConnection(); // browser and server are on the same machine: host candidates suffice
      pcRef.current = pc;
      stream.getTracks().forEach((track) => pc.addTrack(track, stream));
      const channel = pc.createDataChannel('status');
      channel.onmessage = (ev) => handleStatus(JSON.parse(ev.data));
      pc.onconnectionstatechange = () => {
        setConnectionState(pc.connectionState);
        if (pc.connectionState === 'failed') setError('The WebRTC connection to the server failed.');
      };
      setConnectionState('connecting');

      await pc.setLocalDescription(await pc.createOffer());
      await waitForIceGathering(pc);
      const { data } = await axios.post('/api/webrtc/offer', {
        candidate,
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      });
      sessionIdRef.current = data.session_id;
      await pc.setRemoteDescription({ sdp: data.sdp, type: data.type });

      const started = Date.now();
      timerRef.current = setInterval(() => setElapsed((Date.now() - started) / 1000), 500);
      setPhase('running');
    } catch (err) {
      releaseMedia();
      setPhase('idle');
      setConnectionState('closed');
      if (err.name === 'NotAllowedError' || err.name === 'NotFoundError') {
        setError(`Camera/microphone not available (${err.message}). Allow access in the browser and in Windows privacy settings.`);
      } else {
        setError(apiError(err, 'Could not start the session'));
      }
    }
  };

  const stopLive = async () => {
    const sid = sessionIdRef.current;
    setPhase('stopping');
    releaseMedia();
    setConnectionState('closed');
    try {
      const { data } = await axios.post(`/api/session/${sid}/stop`);
      sessionIdRef.current = null;
      onFinished(data.report, data.report_id);
    } catch (err) {
      sessionIdRef.current = null;
      setError(apiError(err, 'Could not build the report'));
      setPhase('idle');
    }
  };

  const startRecorded = async () => {
    if (!file) return;
    setError(null);
    setJob({ status: 'uploading', progress: 0 });
    try {
      const form = new FormData();
      form.append('candidate', candidate);
      form.append('use_audio', useAudio ? 'true' : 'false');
      form.append('video', file);
      const { data } = await axios.post('/api/analyze-video', form, {
        onUploadProgress: (e) => setJob({ status: 'uploading', progress: e.total ? e.loaded / e.total : 0 }),
      });
      pollRef.current = setInterval(async () => {
        try {
          const { data: j } = await axios.get(`/api/analyze-video/${data.job_id}`);
          setJob(j);
          if (j.status === 'done') {
            clearInterval(pollRef.current);
            onFinished(j.report, j.report_id);
          } else if (j.status === 'error') {
            clearInterval(pollRef.current);
            setError(j.error);
          }
        } catch (err) {
          clearInterval(pollRef.current);
          setError(apiError(err, 'Lost contact with the analysis job'));
        }
      }, 1000);
    } catch (err) {
      setJob(null);
      setError(apiError(err, 'Upload failed'));
    }
  };

  const handleReset = () => {
    const sid = sessionIdRef.current;
    releaseMedia();
    clearInterval(pollRef.current);
    if (sid) axios.post(`/api/session/${sid}/stop`).catch(() => {});
    sessionIdRef.current = null;
    onReset();
  };

  const running = phase === 'running';
  const calibrationText = status && !status.calibrated
    ? `Calibrating: look at the screen (${status.calibration?.accepted ?? 0} samples)`
    : null;
  const totalCount = Object.values(status?.counts || {}).reduce((a, b) => a + b, 0);

  const fmtTime = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  const modeLocked = phase !== 'idle' || (job && job.status !== 'error');

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
        <div>
          <h1 className="page-title">Proctored test</h1>
          <p className="text-muted mt-1">
            Candidate <span className="font-semibold text-slate-900 dark:text-slate-100">{candidate}</span> · live video and audio monitoring
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {mode === 'live' && phase !== 'idle' && (
            <StatusIndicator connectionState={connectionState} calibrated={!!status?.calibrated} calibrationText={calibrationText} />
          )}
          {mode === 'live' && (phase === 'idle' ? (
            <button onClick={startLive} className="btn btn-primary">
              <Play className="w-4 h-4" /> Start session
            </button>
          ) : (
            <button onClick={stopLive} disabled={phase !== 'running'} className="btn btn-danger">
              {phase === 'stopping' || phase === 'connecting' ? <Loader className="w-4 h-4 animate-spin" /> : <Square className="w-4 h-4" />}
              {phase === 'stopping' ? 'Building report...' : phase === 'connecting' ? 'Connecting...' : 'Stop session'}
            </button>
          ))}
          <button onClick={handleReset} className="btn btn-secondary">
            <RotateCcw className="w-4 h-4" /> Reset
          </button>
        </div>
      </div>

      <div className="inline-flex gap-1 p-1 rounded-lg surface-muted" role="tablist" aria-label="Test mode">
        {[['live', 'Live (webcam + microphone)', Video], ['recorded', 'Recorded video', Upload]].map(([key, label, Icon]) => (
          <button
            key={key}
            role="tab"
            aria-selected={mode === key}
            onClick={() => setMode(key)}
            disabled={modeLocked}
            className={`tab ${mode === key ? 'tab-active' : 'tab-idle'}`}
          >
            <Icon className="w-4 h-4" /> {label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          {error && (
            <div className="alert alert-error"><AlertCircle className="w-5 h-5 flex-shrink-0" /> <span>{error}</span></div>
          )}
          {(status?.warnings || []).map((w) => (
            <div key={w} className="alert alert-warn"><AlertTriangle className="w-5 h-5 flex-shrink-0" /> <span>{WARNING_TEXT[w] || w}</span></div>
          ))}
          {running && status?.session_violated && (
            <div className="alert alert-error" role="alert">
              <Ban className="w-5 h-5 flex-shrink-0" />
              <span><span className="font-semibold">Session marked as violated.</span> A violation limit was exceeded; its evidence will be
                uploaded with the session report.</span>
            </div>
          )}

          {mode === 'live' ? (
            <div className="card space-y-4">
              <div className="relative bg-slate-900 rounded-lg overflow-hidden aspect-video ring-1 ring-slate-200 dark:ring-slate-800">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" style={{ transform: 'scaleX(-1)' }} />
                {phase === 'idle' && (
                  <div className="absolute inset-0 flex items-center justify-center bg-slate-950/70">
                    <div className="text-center text-white px-6">
                      <Video className="w-14 h-14 mx-auto mb-3 opacity-70" />
                      <p className="text-lg font-medium">Session not active</p>
                      <p className="text-sm opacity-75 mt-1">Click "Start session". Look at the screen for the first few seconds while the system calibrates.</p>
                    </div>
                  </div>
                )}
                {running && (
                  <span className="absolute bottom-3 left-3 badge bg-red-600 text-white text-xs">
                    <span className="w-2 h-2 rounded-full bg-white animate-pulse" /> LIVE {fmtTime(elapsed)}
                  </span>
                )}
                {running && status && (
                  <div className="absolute top-3 left-3 right-3">
                    <MonitorBadges monitors={status.monitors} audio={status.audio} />
                  </div>
                )}
              </div>
              {running && (
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="stat stat-blue">
                    <p className="stat-label inline-flex items-center gap-1"><Clock className="w-4 h-4" /> Elapsed</p>
                    <p className="stat-value">{fmtTime(elapsed)}</p>
                  </div>
                  <div className="stat stat-purple">
                    <p className="stat-label inline-flex items-center gap-1"><Activity className="w-4 h-4" /> Samples</p>
                    <p className="stat-value">{status?.samples ?? 0}</p>
                    <p className="text-xs opacity-80">{status?.sample_rate_hz ? `${status.sample_rate_hz}/s` : ''}</p>
                  </div>
                  <div className="stat stat-red">
                    <p className="stat-label inline-flex items-center gap-1"><AlertTriangle className="w-4 h-4" /> Violations</p>
                    <p className="stat-value">{totalCount}</p>
                  </div>
                </div>
              )}
              {running && (
                <div>
                  <p className="text-sm font-medium text-muted mb-2">Evidence counters (events / limit)</p>
                  <EvidenceCounters evidence={status?.evidence} />
                </div>
              )}
            </div>
          ) : (
            <div className="card space-y-5">
              <div>
                <h2 className="card-title">Analyse a recorded exam video</h2>
                <p className="text-sm text-muted mt-1">The video (and optionally its audio track) is analysed on this computer with the same checks as a live session.</p>
              </div>
              <label className={`flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-8 text-center transition-colors cursor-pointer
                border-slate-300 hover:border-primary-500 dark:border-slate-700 dark:hover:border-primary-400 ${modeLocked ? 'pointer-events-none opacity-60' : ''}`}>
                <FileVideo className="w-10 h-10 text-slate-400" />
                <span className="font-medium">{file ? file.name : 'Choose a video file'}</span>
                <span className="text-xs text-subtle">{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : 'MP4, WebM, AVI, MOV...'}</span>
                <input type="file" accept="video/*" className="hidden" onChange={(e) => setFile(e.target.files[0] || null)} disabled={!!job && job.status !== 'error'} />
              </label>
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input type="checkbox" className="checkbox" checked={useAudio} onChange={(e) => setUseAudio(e.target.checked)} />
                Also analyse the video's audio track
              </label>
              <button onClick={startRecorded} disabled={!file || (job && job.status !== 'error')} className="btn btn-primary">
                <Upload className="w-4 h-4" /> Analyse
              </button>
              {job && job.status !== 'error' && (
                <div>
                  <div className="flex justify-between text-sm text-muted mb-1">
                    <span>{job.status === 'uploading' ? 'Uploading...' : `Analysing ${job.t ?? 0} / ${job.duration ?? '?'} s`}</span>
                    <span className="font-mono">{Math.round(100 * (job.progress || 0))}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                    <div className="h-2 bg-primary-600 rounded-full transition-all" style={{ width: `${100 * (job.progress || 0)}%` }} />
                  </div>
                  {job.monitors && <div className="mt-4"><MonitorBadges monitors={job.monitors} /></div>}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="lg:col-span-1">
          <AlertPanel alerts={alerts} />
        </div>
      </div>
    </div>
  );
};

export default ProctoringSession;
