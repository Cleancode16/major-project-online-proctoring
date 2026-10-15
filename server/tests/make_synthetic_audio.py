"""
Build SYNTHETIC audio test scenes (16 kHz mono) + labels in tests/data/audio.

Speech comes from the two Windows SAPI voices on this machine (offline TTS):
  "Microsoft David Desktop" = the enrolled CANDIDATE, "Microsoft Zira Desktop" = ANOTHER SPEAKER.
Non-speech sounds are generated: ring tone (dual tone bursts), music (chords + melody + beat),
keyboard typing (filtered clicks), whisper (David's speech with its excitation replaced by noise).
All scenes get low-level pink room noise. Label segments are in seconds.

These are synthetic and much cleaner than a real room; results on them are an upper bound.

Run:  .venv\\Scripts\\python.exe -m tests.make_synthetic_audio
"""

import json
import subprocess
import sys

import numpy as np

from audio import io as aio
from audio.passages import SYNTHETIC_PASSAGES as PASSAGES  # recordings were made from these
from tests import paths

SR = 16000
OUT = paths.DATA / "audio"
TTS = OUT / "tts"
DAVID, ZIRA = "Microsoft David Desktop", "Microsoft Zira Desktop"
rng = np.random.default_rng(7)

SENTENCES = [
    "Can you tell me the answer to question number four?",
    "I think the formula for the area of a circle is pi r squared.",
    "Write down option C, then move on to the next page quickly.",
    "The capital of Australia is Canberra, not Sydney.",
    "Hold on, let me check my notes before you submit it.",
    "We have about ten minutes left, so hurry up please.",
]


def tts(text, voice, name):
    path = TTS / f"{name}.wav"
    if not path.exists():
        safe = text.replace("'", "''")
        ps = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.SelectVoice('{voice}'); $s.SetOutputToWaveFile('{path}'); "
              f"$s.Speak('{safe}'); $s.Dispose()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    return aio.load_file(path)


def pink(n, level_db=-50):
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    f /= np.sqrt(np.maximum(1, np.arange(len(f))))
    x = np.fft.irfft(f, n)
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-12) * 10 ** (level_db / 20)).astype(np.float32)


def ring(seconds):
    t = np.arange(int(seconds * SR)) / SR
    tone = 0.5 * np.sin(2 * np.pi * 1300 * t) + 0.5 * np.sin(2 * np.pi * 1700 * t)
    gate = ((t % 0.1) < 0.05) & ((t % 2.0) < 1.0)          # warble 20 Hz, 1 s on / 1 s off
    return (0.3 * tone * gate).astype(np.float32)


def music(seconds):
    t = np.arange(int(seconds * SR)) / SR
    chords = [(261.6, 329.6, 392.0), (220.0, 261.6, 329.6), (174.6, 220.0, 261.6), (196.0, 246.9, 293.7)]
    melody = [523.3, 587.3, 659.3, 784.0, 659.3, 587.3, 523.3, 493.9]
    x = np.zeros_like(t)
    for i, ch in enumerate(chords * int(seconds // 8 + 1)):
        seg = (t >= i * 2) & (t < i * 2 + 2)
        for f in ch:
            x[seg] += 0.12 * (np.sin(2 * np.pi * f * t[seg]) + 0.3 * np.sin(4 * np.pi * f * t[seg]))
    for i in range(int(seconds * 2)):
        seg = (t >= i * 0.5) & (t < i * 0.5 + 0.45)
        f = melody[i % len(melody)]
        env = np.exp(-6 * (t[seg] - i * 0.5))
        x[seg] += 0.2 * env * np.sin(2 * np.pi * f * t[seg])
    beat = ((t % 0.5) < 0.03) * rng.standard_normal(len(t)) * 0.3
    return (0.6 * (x + beat)).astype(np.float32)


def typing(seconds):
    x = np.zeros(int(seconds * SR), np.float32)
    tt = 0.0
    while tt < seconds - 0.1:
        tt += rng.uniform(0.08, 0.3)
        i = int(tt * SR)
        click = rng.standard_normal(200) * np.exp(-np.arange(200) / 25)
        x[i:i + 200] += 0.25 * click[: len(x[i:i + 200])]
    return x


def whisperize(x):
    """Keep the spectral envelope of speech, replace the voiced excitation by noise."""
    import librosa
    n_fft, hop = 512, 128
    s = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    mag = np.abs(s)
    env = np.apply_along_axis(lambda c: np.convolve(c, np.ones(9) / 9, mode="same"), 0, mag)
    noise = librosa.stft(rng.standard_normal(len(x)).astype(np.float32), n_fft=n_fft, hop_length=hop)
    y = librosa.istft(env * noise / (np.abs(noise) + 1e-9), hop_length=hop, length=len(x))
    return (y / (np.abs(y).max() + 1e-9) * 0.15).astype(np.float32)


def place(dst, src, at_s, gain=1.0):
    i = int(at_s * SR)
    n = min(len(src), len(dst) - i)
    dst[i:i + n] += gain * src[:n]
    return (at_s, at_s + n / SR)


def norm(x, peak=0.5):
    return (x / (np.abs(x).max() + 1e-9) * peak).astype(np.float32)


def main():
    TTS.mkdir(parents=True, exist_ok=True)
    david = [norm(tts(s, DAVID, f"david_s{i}")) for i, s in enumerate(SENTENCES)]
    zira = [norm(tts(s, ZIRA, f"zira_s{i}")) for i, s in enumerate(SENTENCES)]
    for i, p in enumerate(PASSAGES):
        aio.save_wav(OUT / f"enroll_david_p{i}.wav", norm(tts(p, DAVID, f"david_p{i}")) + pink(1, -60)[0])
        aio.save_wav(OUT / f"enroll_zira_p{i}.wav", norm(tts(p, ZIRA, f"zira_p{i}")))

    scenes = {}

    def scene(name, dur, build):
        x = pink(int(dur * SR))
        labels = build(x)
        aio.save_wav(OUT / f"{name}.wav", np.clip(x, -1, 1))
        scenes[name] = {"duration_s": dur, "labels": labels}

    scene("silence", 20, lambda x: {})
    scene("candidate_speech", 25, lambda x: {
        "SPEECH": [place(x, david[0], 3), place(x, david[1], 12)], "speaker": "candidate"})
    scene("other_speaker", 25, lambda x: {
        "SPEECH": [place(x, zira[0], 3), place(x, zira[1], 12)], "VOICE_MISMATCH": "all_speech"})

    def dialogue(x):
        segs, who, at = [], [], 2.0
        for i in range(4):
            a = place(x, david[2 + i], at); segs.append(a); who.append("candidate"); at = a[1] + 0.4
            b = place(x, zira[2 + i], at); segs.append(b); who.append("other"); at = b[1] + 0.4
        return {"SPEECH": segs, "speakers": who, "MULTIPLE_SPEAKERS": [(segs[0][0], segs[-1][1])]}
    scene("dialogue", 40, dialogue)

    scene("music", 20, lambda x: {"SUSPICIOUS_SOUND": [place(x, music(16), 2)], "kind": "music"})
    scene("phone_ring", 20, lambda x: {"SUSPICIOUS_SOUND": [place(x, ring(10), 5)], "kind": "ring"})
    scene("typing", 20, lambda x: {"TYPING_INFO": [place(x, typing(15), 2)]})
    def whisper(x):
        seg = place(x, whisperize(np.concatenate(david[:2])), 3)
        return {"WHISPERING": [seg], "SPEECH": [seg]}  # whispered speech is still speech
    scene("whisper", 20, whisper)

    (OUT / "scenes.json").write_text(json.dumps(scenes, indent=1))
    for k, v in scenes.items():
        print(k, v["duration_s"], "s", {a: b for a, b in v["labels"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
