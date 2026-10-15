import { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import {
  Camera, CheckCircle, AlertCircle, Upload, Mic, Square, RefreshCw, ShieldCheck, AlertTriangle, Loader, Check,
  History, Image as ImageIcon, Sun, User as UserIcon, RotateCcw, ArrowRight, Camera as CameraIcon,
} from 'lucide-react';
import { SkeletonText } from './ui/Skeleton';
import EmptyState from './ui/EmptyState';

// Enrollment (flow.txt PART II-D, D.1): consent -> reference photo -> voice reading -> continue.
// All requests go to the local Python server through the Vite /api proxy.

const CONSENT_TEXT =
  'For proctoring, this system stores a reference photo of your face and a voiceprint computed from you ' +
  'reading a short passage aloud. During the test your webcam and microphone are analysed on the proctoring ' +
  'server. When a violation limit is exceeded, the evidence for that violation (snapshots, short audio clips ' +
  'and their details) is sent to the evidence server configured by your institution, together with the session ' +
  'ID. The raw enrollment recording is not kept, only the voiceprint.';

const apiError = (err, fallback) => err.response?.data?.error || err.message || fallback;

// Float32 chunks -> mono 16-bit PCM WAV blob (fallback recorder; the server resamples to 16 kHz)
const encodeWav = (parts, sampleRate) => {
  const n = parts.reduce((a, p) => a + p.length, 0);
  const view = new DataView(new ArrayBuffer(44 + n * 2));
  const str = (o, s) => [...s].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF'); view.setUint32(4, 36 + n * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  str(36, 'data'); view.setUint32(40, n * 2, true);
  let o = 44;
  parts.forEach((p) => p.forEach((v) => {
    const s = Math.max(-1, Math.min(1, v));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }));
  return new Blob([view], { type: 'audio/wav' });
};

const VoiceRecorder = ({ candidate, onResult, disabled }) => {
  const [passage, setPassage] = useState(null);
  const [recording, setRecording] = useState(false);
  const [requesting, setRequesting] = useState(false);
  const [remaining, setRemaining] = useState(0);
  const [level, setLevel] = useState(-90);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const recRef = useRef(null);
  const cleanupRef = useRef(() => {});
  const unmountedRef = useRef(false);

  const loadPassage = async (exclude) => {
    try {
      const { data } = await axios.get('/api/enrollment/passage', { params: exclude != null ? { exclude } : {} });
      setPassage(data);
    } catch (err) {
      setError(apiError(err, 'Could not load the passage'));
    }
  };

  useEffect(() => {
    unmountedRef.current = false;
    loadPassage();
    return () => {
      unmountedRef.current = true;   // a recording stopped by leaving the page is discarded, not uploaded
      cleanupRef.current();
    };
  }, []);

  const upload = async (blob, filename) => {
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('candidate', candidate);
      form.append('passage', String(passage.index));
      form.append('audio', blob, filename);
      const { data } = await axios.post('/api/enrollment/voice', form);
      setResult(data);
      onResult(data);
    } catch (err) {
      setError(apiError(err, 'Voice check failed'));
    } finally {
      setBusy(false);
    }
  };

  const micErrorText = (err) => {
    switch (err?.name) {
      case 'NotAllowedError':
      case 'SecurityError':
        return 'Microphone permission was denied or dismissed. Click the lock/camera icon in the address bar, allow ' +
          'the microphone for this site, and check Windows Settings > Privacy & security > Microphone ' +
          '("Let desktop apps access your microphone").';
      case 'NotFoundError':
      case 'OverconstrainedError':
        return 'No microphone was found. Connect a microphone (or select one in Windows sound settings) and try again.';
      case 'NotReadableError':
      case 'AbortError':
        return 'The microphone is in use by another application or could not be opened. Close other apps using it and try again.';
      default:
        return `Could not start recording: ${err?.message || err}`;
    }
  };

  // Records the stream. Primary: MediaRecorder (webm/opus). Fallback: raw Web Audio samples -> 16-bit WAV.
  const startRecorder = (stream, ctx, source, onDone) => {
    if (typeof window.MediaRecorder !== 'undefined') {
      try {
        const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus']
          .find((m) => MediaRecorder.isTypeSupported(m));
        const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
        const chunks = [];
        rec.ondataavailable = (e) => e.data && e.data.size && chunks.push(e.data);
        rec.onstop = () => {
          const type = rec.mimeType || mime || 'audio/webm';
          onDone(new Blob(chunks, { type }), type.includes('ogg') ? 'reading.ogg' : 'reading.webm');
        };
        rec.start(250);
        return { stop: () => rec.state === 'recording' && rec.stop() };
      } catch (err) {
        console.warn('MediaRecorder failed, using the WAV fallback:', err);
      }
    }
    // Fallback: capture PCM with a ScriptProcessor (connected through a muted gain so nothing is played back)
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const mute = ctx.createGain();
    mute.gain.value = 0;
    const parts = [];
    proc.onaudioprocess = (e) => parts.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    source.connect(proc);
    proc.connect(mute);
    mute.connect(ctx.destination);
    let stopped = false;
    return {
      stop: () => {
        if (stopped) return;
        stopped = true;
        proc.disconnect();
        mute.disconnect();
        onDone(encodeWav(parts, ctx.sampleRate), 'reading.wav');
      },
    };
  };

  const start = async () => {
    setError(null);
    setResult(null);
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setError('The browser blocks the microphone on this address. Open the app at http://localhost:5173 ' +
        '(not an IP address) in Chrome or Edge.');
      return;
    }
    setRequesting(true);
    let stream = null;
    let ctx = null;
    let meter = null;
    let countdown = null;
    try {
      // same (browser default) audio processing as the live WebRTC session, so voiceprints compare fairly
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const track = stream.getAudioTracks()[0];
      if (!track || track.readyState !== 'live') throw new Error('The microphone track is not live.');

      // live level meter (dBFS of the input)
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      if (ctx.state === 'suspended') await ctx.resume();
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      const buf = new Float32Array(analyser.fftSize);
      meter = setInterval(() => {
        analyser.getFloatTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i += 1) sum += buf[i] * buf[i];
        setLevel(20 * Math.log10(Math.sqrt(sum / buf.length) + 1e-9));
      }, 100);

      const release = () => {
        clearInterval(meter);
        clearInterval(countdown);
        stream.getTracks().forEach((t) => t.stop());
        ctx.close().catch(() => {});
      };
      const recorder = startRecorder(stream, ctx, source, (blob, filename) => {
        release();
        cleanupRef.current = () => {};
        if (unmountedRef.current) return;
        setRecording(false);
        if (!blob || blob.size === 0) {
          setError('The recording is empty. Check that the right microphone is selected and not muted, then try again.');
          return;
        }
        upload(blob, filename);
      });

      const seconds = passage?.record_s || 25;
      const startedAt = Date.now();
      countdown = setInterval(() => {
        const left = seconds - (Date.now() - startedAt) / 1000;
        setRemaining(Math.max(0, left));
        if (left <= 0) recorder.stop();
      }, 200);

      recRef.current = recorder;
      cleanupRef.current = () => { recorder.stop(); release(); };
      setRemaining(seconds);
      setRecording(true);
    } catch (err) {
      console.error('Voice recording failed to start:', err);
      clearInterval(meter);
      clearInterval(countdown);
      stream?.getTracks().forEach((t) => t.stop());
      ctx?.close().catch(() => {});
      setError(micErrorText(err));
    } finally {
      setRequesting(false);
    }
  };

  const stop = () => {
    recRef.current?.stop();
  };

  const m = result?.metrics || {};
  const levelPct = Math.min(100, Math.max(0, ((level + 60) / 60) * 100));

  return (
    <div className="space-y-4">
      <div>
        <h3 className="card-title">Voice enrollment</h3>
        <p className="text-[13px] text-muted mt-1">Press Record and read this passage aloud at a normal pace.</p>
      </div>
      <blockquote className="rounded-md border border-line border-l-2 border-l-accent px-4 py-3 text-sm leading-relaxed surface-muted">
        {passage ? passage.text : (
          <>
            <span className="sr-only">Loading the passage…</span>
            <SkeletonText lines={4} />
          </>
        )}
      </blockquote>
      <div className="flex flex-wrap gap-2">
        {!recording ? (
          <button onClick={start} disabled={disabled || busy || requesting || !passage} className="btn btn-primary">
            {requesting ? <Loader className="w-4 h-4 animate-spin" /> : <Mic className="w-4 h-4" />}
            {requesting ? 'Waiting for microphone permission…' : `Record (${passage?.record_s || 25} s)`}
          </button>
        ) : (
          <button onClick={stop} className="btn btn-danger">
            <Square className="w-4 h-4" /> Stop early ({remaining.toFixed(0)} s left)
          </button>
        )}
        <button onClick={() => loadPassage(passage?.index)} disabled={recording || requesting || busy} className="btn btn-secondary">
          <RefreshCw className="w-4 h-4" /> Another passage
        </button>
      </div>

      {recording && (
        <div>
          <div className="flex justify-between text-xs text-muted mb-1">
            <span className="inline-flex items-center gap-1.5"><span className="dot bg-critical animate-pulse" /> Recording · input level</span>
            <span className="font-mono">{level.toFixed(0)} dBFS</span>
          </div>
          <div className="track" role="meter" aria-valuemin={-60} aria-valuemax={0} aria-valuenow={Math.round(level)} aria-label="Microphone level">
            <div className="track-fill duration-100" style={{ width: `${levelPct}%` }} />
          </div>
        </div>
      )}
      {busy && (
        <p className="text-sm text-muted inline-flex items-center gap-2"><Loader className="w-4 h-4 animate-spin" /> Checking the recording…</p>
      )}
      {error && (
        <div className="alert alert-error"><AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" /> <span>{error}</span></div>
      )}
      {result && (
        <div className="space-y-2">
          {result.ok && (
            <div className="alert alert-ok"><CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" /> <span>Voice recording accepted.</span></div>
          )}
          {result.problems.map((p) => (
            <div key={p} className="alert alert-error"><AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" /> <span>{p}</span></div>
          ))}
          {result.warnings.map((w) => (
            <div key={w} className="alert alert-warn"><AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" /> <span>{w}</span></div>
          ))}
          {m.voiced_s !== undefined && (
            <div className="grid grid-cols-3 sm:grid-cols-5 gap-2 text-xs">
              {[
                ['Speech', `${m.voiced_s.toFixed(1)} s`],
                ['Level', `${m.level_dbfs.toFixed(0)} dBFS`],
                ['SNR', `${m.snr_db.toFixed(0)} dB`],
                ['Clipping', `${(100 * m.clip_frac).toFixed(2)} %`],
                ['Consistency', (m.consistency ?? 0).toFixed(2)],
              ].map(([k, v]) => (
                <div key={k} className="rounded-md border border-line px-2.5 py-2">
                  <div className="text-subtle">{k}</div>
                  <div className="font-mono font-semibold tabular-nums mt-0.5">{v}</div>
                </div>
              ))}
            </div>
          )}
          {result.asr && result.asr.status !== 'skipped' && (
            <p className="text-xs text-muted">Passage check: {result.asr.status} (word match {(100 * result.asr.match).toFixed(0)}%)</p>
          )}
          <p className="text-xs text-subtle">Attempt {result.attempt} of {result.max_attempts}</p>
        </div>
      )}
    </div>
  );
};

const SetupScreen = ({ onEnrolled }) => {
  const [candidate, setCandidate] = useState('');
  const [consent, setConsent] = useState(false);
  const [captureMode, setCaptureMode] = useState('webcam'); // 'webcam' or 'upload'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [faceOk, setFaceOk] = useState(false);
  const [voice, setVoice] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [savedInfo, setSavedInfo] = useState(null);
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);

  const candidateId = candidate.trim();
  const voiceOk = !!voice?.ok;
  const attemptsUsed = voice && !voice.ok && voice.attempt >= voice.max_attempts;

  const handleCapture = () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) setCapturedImage(imageSrc);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setCapturedImage(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleSubmitFace = async () => {
    if (!capturedImage) {
      setError('Please capture or upload an image first');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const { data } = await axios.post('/api/enrollment/face', { candidate: candidateId, consent, image: capturedImage });
      if (data.success) {
        setFaceOk(true);
        setVoice(null);
      }
    } catch (err) {
      setError(apiError(err, 'Failed to enroll the reference image'));
    } finally {
      setLoading(false);
    }
  };

  const handleRetake = () => {
    setCapturedImage(null);
    setFaceOk(false);
    setError(null);
  };

  const checkSaved = async () => {
    if (!candidateId) return;
    try {
      const { data } = await axios.get(`/api/enrollment/${encodeURIComponent(candidateId)}`);
      setSavedInfo(data);
    } catch {
      setSavedInfo(null);
    }
  };

  const ready = consent && !!candidateId;
  // numbered step marker: neutral until the step is complete
  const StepDot = ({ n, done }) => (
    <span className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-2xs font-semibold flex-shrink-0 ${
      done ? 'bg-success text-white' : 'border border-line text-fg-muted'
    }`}>
      {done ? <Check className="w-3 h-3" aria-label="done" /> : n}
    </span>
  );

  return (
    <div className="max-w-5xl mx-auto space-y-4">
      <div className="mb-2">
        <h1 className="page-title">Enrollment</h1>
        <p className="text-muted mt-1">Register your face and voice so the system can recognise you during the test.</p>
      </div>

      {/* 1. candidate + consent */}
      <section className="card space-y-4">
        <div className="flex items-center gap-2.5">
          <StepDot n={1} done={ready} />
          <h2 className="card-title">Candidate and consent</h2>
        </div>
        <div className="flex flex-col md:flex-row gap-4 md:items-end">
          <div className="flex-1">
            <label htmlFor="candidate-id" className="label">Candidate ID</label>
            <input
              id="candidate-id"
              value={candidate}
              onChange={(e) => { setCandidate(e.target.value); setSavedInfo(null); }}
              onBlur={checkSaved}
              placeholder="e.g. 21BCE1234"
              autoComplete="off"
              spellCheck={false}
              aria-describedby="candidate-id-hint"
              className="input max-w-sm"
            />
            <p id="candidate-id-hint" className="field-hint">Your roll or registration number, exactly as on your admit card.</p>
          </div>
          {savedInfo?.face && (
            <button onClick={() => onEnrolled(candidateId)} className="btn btn-secondary whitespace-normal h-auto min-h-8 py-1.5 text-left">
              <History className="w-4 h-4" /> Use saved enrollment (voice: {savedInfo.voice ? 'yes' : 'no'})
            </button>
          )}
        </div>
        <div className="alert alert-info">
          <ShieldCheck className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <p>{CONSENT_TEXT}</p>
            <label className="flex items-center gap-2 mt-3 font-medium cursor-pointer">
              <input type="checkbox" className="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
              I agree to face and voice enrollment for this exam.
            </label>
          </div>
        </div>
      </section>

      <div className={`grid grid-cols-1 xl:grid-cols-2 gap-4 transition-opacity ${ready ? '' : 'opacity-50 pointer-events-none select-none'}`} aria-disabled={!ready}>
        {/* 2. reference photo */}
        <section className="card space-y-4">
          <div className="flex items-center gap-2.5">
            <StepDot n={2} done={faceOk} />
            <h2 className="card-title">Reference photo</h2>
          </div>
          <div className="segmented flex w-full" role="tablist" aria-label="Photo source">
            <button
              role="tab"
              aria-selected={captureMode === 'webcam'}
              onClick={() => { setCaptureMode('webcam'); handleRetake(); }}
              className={`tab flex-1 justify-center ${captureMode === 'webcam' ? 'tab-active' : 'tab-idle'}`}
            >
              <Camera className="w-4 h-4" /> Webcam
            </button>
            <button
              role="tab"
              aria-selected={captureMode === 'upload'}
              onClick={() => { setCaptureMode('upload'); handleRetake(); }}
              className={`tab flex-1 justify-center ${captureMode === 'upload' ? 'tab-active' : 'tab-idle'}`}
            >
              <Upload className="w-4 h-4" /> Upload
            </button>
          </div>

          <div className="media-frame">
            {!capturedImage ? (
              captureMode === 'webcam' ? (
                <Webcam ref={webcamRef} screenshotFormat="image/jpeg" className="w-full h-full object-cover" mirrored />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center gap-3 text-white/70 px-6 text-center">
                  <ImageIcon className="w-6 h-6 text-white/60" aria-hidden="true" />
                  <p className="text-xs">A clear, front-facing photo (JPG or PNG)</p>
                  <button onClick={() => fileInputRef.current.click()} className="btn btn-primary">
                    <Upload className="w-4 h-4" /> Choose image
                  </button>
                  <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
                </div>
              )
            ) : (
              <img src={capturedImage} alt="Captured reference" className="w-full h-full object-cover" />
            )}
            {faceOk && (
              <span className="absolute top-3 left-3 media-chip"><CheckCircle className="w-3.5 h-3.5 text-success" /> Enrolled</span>
            )}
          </div>

          <ul className="text-[13px] text-muted space-y-1">
            <li className="flex gap-2"><Sun className="w-4 h-4 mt-0.5 flex-shrink-0" /> Face clearly visible and well lit, looking straight at the camera</li>
            <li className="flex gap-2"><UserIcon className="w-4 h-4 mt-0.5 flex-shrink-0" /> Only you in the picture</li>
          </ul>

          {error && (
            <div className="alert alert-error"><AlertCircle className="w-5 h-5 flex-shrink-0" /> <span>{error}</span></div>
          )}

          <div className="flex flex-wrap gap-2">
            {!capturedImage ? (
              <button onClick={handleCapture} disabled={captureMode !== 'webcam'} className="btn btn-primary flex-1">
                <Camera className="w-4 h-4" /> Capture photo
              </button>
            ) : (
              <>
                <button onClick={handleRetake} className="btn btn-secondary flex-1" disabled={loading}>
                  <RotateCcw className="w-4 h-4" /> Retake
                </button>
                <button onClick={handleSubmitFace} className="btn btn-primary flex-1" disabled={loading || faceOk}>
                  {loading ? <Loader className="w-4 h-4 animate-spin" /> : <CheckCircle className="w-4 h-4" />}
                  {loading ? 'Checking…' : faceOk ? 'Enrolled' : 'Confirm photo'}
                </button>
              </>
            )}
          </div>
        </section>

        {/* 3. voice */}
        <section className="card">
          {faceOk ? (
            <VoiceRecorder candidate={candidateId} onResult={setVoice} disabled={voiceOk || attemptsUsed} />
          ) : (
            <EmptyState
              className="h-full"
              icon={Mic}
              title="Voice enrollment"
              description="Confirm your reference photo first. Then you will read a short passage aloud to create your voiceprint."
              action={<span className="badge badge-muted"><CameraIcon className="w-3.5 h-3.5" /> Waiting for step 2</span>}
            />
          )}
        </section>
      </div>

      <div className="flex flex-col-reverse sm:flex-row gap-2 justify-end pt-2 [&>.btn]:w-full sm:[&>.btn]:w-auto">
        {faceOk && !voiceOk && attemptsUsed && (
          <button onClick={() => onEnrolled(candidateId)} className="btn btn-secondary">
            Continue without voice (voice checks disabled)
          </button>
        )}
        <button onClick={() => onEnrolled(candidateId)} disabled={!faceOk || !voiceOk} className="btn btn-primary">
          Continue to the test <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default SetupScreen;
