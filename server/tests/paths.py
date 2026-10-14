"""Locations of test inputs. The sample clip/reference are the candidate's own recordings (real data)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
RESULTS = TESTS / "results"
DATA = TESTS / "data"

SAMPLE_VIDEO = Path(r"C:\Users\bhara\OneDrive\Pictures\Camera Roll\WIN_20260207_13_29_05_Pro.mp4")
SAMPLE_REF = Path(r"C:\Users\bhara\OneDrive\Pictures\Camera Roll\WIN_20260206_21_26_58_Pro.jpg")
BASELINE_V1 = RESULTS / "baseline_app_v1.txt"
