"""Enrollment reading passages (original text, ~45-55 words, ~15-20 s read aloud; flow.txt §B.3).

Written to cover most English phonemes (plosives, fricatives, nasals, vowels, the "th", "zh",
"ng" and "ch" sounds) so the voiceprint is not built from a narrow slice of the speaker's voice.
One passage is picked at random per enrollment so a single prerecorded clip cannot be replayed.
"""

import random

PASSAGES = [
    "On a quiet Thursday morning, Joseph checked the weather and packed a yellow umbrella. "
    "The bus was late again, so he walked past the bakery, where fresh bread and sweet buns "
    "were cooling by the window. He reached the station just as the big clock struck nine.",

    "My younger sister enjoys measuring things. Last week she counted every step from the garden "
    "gate to the old oak tree, then wrote the number on a blue card. She thinks careful habits "
    "and patient questions usually lead to the most surprising answers.",

    "The village fair opens at noon each June. Children judge the vegetable contest, a brass band "
    "plays cheerful music, and visitors queue for hot chocolate. By evening the lights glow over "
    "the square, and everyone shares stories about the largest pumpkin they have ever seen.",

    "Please read this sentence clearly and at a normal pace. A thick fog covered the harbour, "
    "but the fishing boats still moved slowly toward the shore. Their engines hummed, gulls "
    "circled above the masts, and a warm light shone from the lighthouse on the rocky point.",
]


def pick(rng=None):
    """Return (index, passage)."""
    rng = rng or random.Random()
    i = rng.randrange(len(PASSAGES))
    return i, PASSAGES[i]
