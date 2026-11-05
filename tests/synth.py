"""Synthetic speech-like audio with KNOWN label timing, so every test knows the right answer.

The "speech" is a harmonic tone with a moving pitch and a formant-like filter, chopped into word-sized bursts
with short pauses inside each utterance, on top of a steady noise floor. It is not real speech (the real-data
script covers that); it has what the detector looks at: loud stretches with short dips, between quiet stretches.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SR = 16000


def make_segments(duration: float, seed: int = 0, density: float = 0.55):
    """Random utterances [(start, end)] in seconds covering about `density` of the time."""
    r = np.random.RandomState(seed)
    segs, t = [], r.uniform(0.3, 1.5)
    while t < duration - 1.0:
        d = r.uniform(1.5, 6.0)
        end = min(t + d, duration - 0.2)
        segs.append((round(t, 3), round(end, 3)))
        t = end + r.uniform(0.4, 3.0) * (1.0 / density - 0.5)
    return segs


def synth_speech(segs, duration: float, sr: int = SR, snr_db: float = 25.0, seed: int = 0) -> np.ndarray:
    """Mono float32 audio: speech-like bursts inside each (start, end), noise floor everywhere."""
    r = np.random.RandomState(seed + 1000)
    n = int(duration * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for (a, b) in segs:
        # words: 0.15-0.5 s of voiced sound, 0.05-0.12 s pauses
        u = a
        while u < b:
            wlen = min(r.uniform(0.15, 0.5), b - u)
            i0, i1 = int(u * sr), int((u + wlen) * sr)
            tt = np.arange(i1 - i0) / sr
            f0 = r.uniform(100, 220)
            ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.08 * np.sin(2 * np.pi * 3 * tt + r.rand() * 6))) / sr
            w = sum(np.sin(k * ph + r.rand() * 6) / k for k in range(1, 15))
            env = np.hanning(len(tt)) ** 0.5 * (0.6 + 0.4 * np.sin(2 * np.pi * r.uniform(3, 6) * tt) ** 2)
            x[i0:i1] += w * env
            u += wlen + r.uniform(0.05, 0.12)
    x = signal.lfilter(*signal.butter(2, [250 / (sr / 2), 3500 / (sr / 2)], "band"), x)
    sp = np.sqrt(np.mean(x[np.abs(x) > 0.05 * np.abs(x).max()] ** 2)) if np.abs(x).max() > 0 else 1.0
    noise = r.randn(n)
    noise = signal.lfilter(*signal.butter(1, 0.3), noise)
    noise *= sp * 10 ** (-snr_db / 20) / max(np.sqrt(np.mean(noise ** 2)), 1e-12)
    y = x + noise
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


def shifted(segs, offset=0.0, scale=1.0):
    """Label times as a labeller or a faulty tool would write them: true time * scale - offset.

    With scale=1 the labels are `offset` seconds EARLIER than the audio, so driftcheck should report +offset.
    """
    return [(a * scale - offset, b * scale - offset) for a, b in segs]


def jittered(segs, sd=0.05, seed=0):
    """Sloppy but honest labelling: each boundary is off by a random amount (seconds)."""
    r = np.random.RandomState(seed + 7)
    out = []
    for a, b in segs:
        a2, b2 = a + r.randn() * sd, b + r.randn() * sd
        out.append((a2, max(b2, a2 + 0.2)))
    return out


def text_for(a: float, b: float, cps: float = 14.0) -> str:
    """Placeholder text with about `cps` characters per second (spaces included), so rate checks have something to read."""
    n = max(int(round((b - a) * cps)), 1)
    return (("la " * n)[:n]).strip() or "la"


def _stamp(t: float, sep: str = ",") -> str:
    t = max(t, 0.0)
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def write_srt(path, segs, cps: float = 14.0, texts=None):
    lines = []
    for i, (a, b) in enumerate(segs, 1):
        t = texts[i - 1] if texts else text_for(a, b, cps)
        lines.append(f"{i}\n{_stamp(a)} --> {_stamp(b)}\n{t}\n")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
