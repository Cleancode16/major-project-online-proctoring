"""Enrollment reading text (flow.txt §B.3).

The candidate reads a single honesty declaration aloud (~60 words, ~20 s), which also serves as the
voiceprint recording. It is long enough to give the required voiced speech (enroll.min_voiced_s)
inside the 25 s recording window.

SYNTHETIC_PASSAGES are the earlier phonetically varied passages. They are kept only because the
synthetic test recordings (tests/data/audio/enroll_*_p<i>.wav) were generated from them.
"""

import random

DECLARATIONS = (
    "I acknowledge that I am responsible for maintaining a fair and transparent examination environment. I will not communicate with others, switch to unauthorized applications, use prohibited devices, or attempt to bypass the proctoring system. I understand that my camera and microphone may be monitored in accordance with the stated examination policy. Any suspected misconduct may be investigated and may result in appropriate disciplinary action."
)
PASSAGES = [DECLARATIONS]

def pick(rng=None):
    """Return (index, passage)."""
    rng = rng or random.Random()
    i = rng.randrange(len(PASSAGES))
    return i, PASSAGES[i]
