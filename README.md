# Online Proctoring System (v2)

An AI exam proctor with two parts:

- A **React frontend** in `client/proctoring`.
- A **Python backend** in `server/`.

During a live test the browser sends the candidate's webcam and microphone to the backend over **WebRTC**. The backend checks:

- **Video:** identity, head pose, gaze, extra people, phones/books/laptops, and whether the face is out of view.
- **Audio:** speech, a different voice, several speakers, whispering, and phone ringing / music / TV.

Every result is streamed back live over a WebRTC data channel. At the end the backend combines everything into one **violation score (0–100)** and a verdict: **VALID**, **SUSPICIOUS** (needs review) or **INVALID**.

Everything runs on this computer. The backend listens on localhost only, and face images, voiceprints and reports stay in `server/data/`.

- How every step works, every threshold and weight, and the API: **`flow.txt` PART II** (sections A–E)
- What was changed and why, with test evidence: **`plan.md`**
- All thresholds and weights: **`server/config.py`**

## Layout

```
client/proctoring/          React (Vite + Tailwind)
  src/App.jsx               steps: enrollment -> test -> results
  src/components/
    SetupScreen.jsx         consent, reference photo, voice reading with level meter
    ProctoringSession.jsx   live test over WebRTC (status via data channel) + recorded-video analysis
    Statistics.jsx          results: verdict, score, % chart, timeline, evidence, audio events, JSON download
    AlertPanel.jsx          live alerts
    StatusIndicator.jsx     WebRTC / calibration state
server/
  server.py                 Flask REST API + WebRTC signalling (port 5000, localhost)
  webrtc_session.py         live session: WebRTC tracks -> analyzers -> data-channel status -> report
  config.py                 all thresholds / weights ("v2" profile; "legacy" = v1 behaviour)
  video/ audio/ aggregation/  detection, audio monitoring, scoring + reports
  proctor.py, app.py        offline analysis (library + CLI)
  enrollment_store.py       local face/voice references (server/data/enrollment)
  tests/                    test harness, labels, results
  legacy/                   v1 app.py, v1 flow.txt, v1 Flask/Socket.IO server
start-all.ps1             starts backend + frontend (two windows) and opens the app
start-backend.ps1, start-frontend.ps1
```

## Setup (Windows, Python 3.10, Node 18+)

**Backend.** The Python environment lives at the repository root in `.venv`:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade "pip<25"
.venv\Scripts\python.exe -m pip install -r server\requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps Resemblyzer==0.1.4 silero-vad==6.2.3 mediapipe==0.10.21
cd server; ..\.venv\Scripts\python.exe -m video.l2cs --download; cd ..   # L2CS-Net gaze weights (GZ3)
```

`start-backend.ps1` does the same automatically if a package is missing. That includes `aiortc` (WebRTC), which is new in this version.

**Frontend:**

```powershell
cd client\proctoring
npm install
```

Run `npm install` again after this update: `socket.io-client` was removed from `package.json`.

**Optional passage check (Vosk):** download `vosk-model-small-en-us-0.15` into `server\models\`, then set `"asr_enabled": True` in `server/config.py` → `enroll`.

**First-run downloads (public, cached):**
- YAMNet: into `server/models/tfhub`.
- `yolov8s.pt`: by Ultralytics, into `server/`.

## Run

One command starts both. It opens a window each for the backend and the frontend, waits for the backend, then opens the app in your browser:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-all.ps1
```

Or start them separately:

```powershell
.\start-backend.ps1     # http://localhost:5000  (REST + WebRTC signalling)
.\start-frontend.ps1    # http://localhost:5173  (open this in Chrome/Edge)
```

1. **Enrollment:**
   - Enter the candidate ID and accept the consent notice.
   - Take or upload a reference photo.
   - Read the passage shown on screen aloud for 25 s. A live level meter is shown while recording.
   - The backend checks speech length, level, noise, clipping and voice consistency, and explains how to fix a bad recording. You get up to 3 attempts.
2. **Test.** Choose one:
   - **Live:** the browser connects to the backend with WebRTC. Look at the screen for the first 3 s, which is calibration. Badges and alerts update with every analysed sample. Click **Stop Session** to build the report.
   - **Recorded video:** upload an exam recording. It's analysed as a background job, with a progress bar.
3. **Results:**
   - the verdict and score
   - a percentage bar chart per violation type
   - one timeline with both video and audio events
   - evidence snapshots and an audio event table
   - the per-violation numbers, and a JSON report download

**If the camera image is black or the microphone is silent:** allow camera/microphone access for the browser and in *Windows Settings → Privacy & security*. The app shows a warning, and the report marks the result for review.

## PDF reports

Every session writes `report.pdf` next to `report.json` in `server/data/reports/<id>/`. The results page has a **Download PDF** button that calls `GET /api/reports/<id>/pdf`.

To convert an existing JSON report, such as one you downloaded:

```powershell
cd server
..\.venv\Scripts\python.exe -m aggregation.pdf_report "C:\path\to\proctoring_report_<id>.json"
```

The PDF is written next to the JSON. Evidence snapshots are taken from `server/data/reports/<id>/` automatically; pass `--evidence-dir` to point elsewhere.

## Command line (offline analysis)

```powershell
cd server
..\.venv\Scripts\python.exe app.py --video exam.mp4 --reference me.jpg --candidate alice
..\.venv\Scripts\python.exe app.py --candidate alice --video exam.mp4   # uses alice's saved enrollment
..\.venv\Scripts\python.exe app.py --profile legacy                     # original v1 behaviour
```

## Tests

```powershell
cd server
..\.venv\Scripts\python.exe -m tests.run_all --quick
```

Results are written to `server/tests/results/SUMMARY.md`. The test data and its limits (real, public and synthetic sources) are described in `plan.md`.

## REST API (summary; full list in flow.txt §E.1)

```
GET  /api/health                         GET  /api/enrollment/passage
GET  /api/enrollment/<candidate>         POST /api/enrollment/face      POST /api/enrollment/voice
POST /api/webrtc/offer                   POST /api/session/<id>/stop
POST /api/analyze-video                  GET  /api/analyze-video/<job>
GET  /api/reports                        GET  /api/reports/<id>         GET /api/reports/<id>/evidence/<file>
```
