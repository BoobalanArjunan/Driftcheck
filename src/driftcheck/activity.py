"""Where is there sound? A plain energy-based activity detector (no model, no training).

Audio is cut into 10 ms frames, the energy of each frame is measured in the speech band (100 Hz to 4 kHz), and a
frame counts as "active" when it is clearly louder than the quiet parts of the same file. Short gaps are filled
and tiny blips are removed. This is deliberately simple. It works when speech stands out from the background;
it does not work on music or on constant loud noise, and it says so (``Activity.reliable``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

FRAME_S = 0.01            # 10 ms frames
CONTRAST_DB = 12.0        # needs at least this much difference between loud and quiet parts to trust the result
THRESHOLD_FRACTION = 0.3  # a frame is active above floor + 30% of the way up to the loud level
FILL_GAP_S = 0.25         # silence shorter than this inside speech is filled in
MIN_RUN_S = 0.10          # active runs shorter than this are dropped


class DriftcheckError(ValueError):
    """A problem the user can fix (bad input or option). The CLI shows it without a traceback."""


@dataclass
class Envelope:
    db: np.ndarray        # energy per frame in dB (relative, not calibrated)
    hop_s: float          # real length of one frame in seconds (sr and frame size rarely divide exactly)
    duration_s: float


@dataclass
class Activity:
    active: np.ndarray    # bool per frame
    hop_s: float
    duration_s: float
    reliable: bool
    floor_db: float
    top_db: float
    reason: str = ""

    @property
    def n(self) -> int:
        return len(self.active)


def _sos(sr: int):
    hi = min(4000.0, 0.45 * sr)
    lo = min(100.0, 0.5 * hi)
    return signal.butter(4, [lo, hi], "band", fs=sr, output="sos")


def envelope_from_array(x: np.ndarray, sr: int) -> Envelope:
    """x: mono float signal."""
    hop = max(int(round(sr * FRAME_S)), 1)
    nf = len(x) // hop
    if nf < 1:
        return Envelope(np.zeros(0), hop / sr, len(x) / sr)
    y = signal.sosfilt(_sos(sr), np.asarray(x, dtype=np.float64)[: nf * hop])
    e = (y.reshape(nf, hop) ** 2).mean(axis=1)
    return Envelope(10.0 * np.log10(np.maximum(e, 1e-14)), hop / sr, len(x) / sr)


def envelope_from_file(path: Path, block_s: float = 60.0) -> Envelope:
    """Read a file block by block (any length, constant memory) and measure its envelope."""
    try:
        f = sf.SoundFile(str(path))
    except Exception as e:
        raise DriftcheckError(f"cannot read {Path(path).name}: {e}") from None
    with f:
        sr = f.samplerate
        total = f.frames
        hop = max(int(round(sr * FRAME_S)), 1)
        block = hop * max(int(block_s * sr / hop), 1)
        sos = _sos(sr)
        zi = np.zeros((sos.shape[0], 2))
        parts, carry = [], np.zeros(0)
        while True:
            a = f.read(block, dtype="float32", always_2d=True)
            if len(a) == 0:
                break
            x = a.mean(axis=1).astype(np.float64)
            y, zi = signal.sosfilt(sos, x, zi=zi)
            y = np.concatenate([carry, y])
            nf = len(y) // hop
            if nf:
                parts.append((y[: nf * hop].reshape(nf, hop) ** 2).mean(axis=1))
            carry = y[nf * hop:]
    e = np.concatenate(parts) if parts else np.zeros(0)
    return Envelope(10.0 * np.log10(np.maximum(e, 1e-14)), hop / sr, total / sr)


def _runs(b: np.ndarray):
    """(start, end) index pairs of the True runs in a bool array (end exclusive)."""
    if len(b) == 0:
        return []
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def clean_runs(active: np.ndarray, hop_s: float, fill_gap_s: float = FILL_GAP_S, min_run_s: float = MIN_RUN_S) -> np.ndarray:
    """Fill short gaps between active runs, then drop short active runs."""
    out = active.copy()
    gap = int(round(fill_gap_s / hop_s))
    for s, e in _runs(~active):
        if s > 0 and e < len(active) and e - s <= gap:       # a gap with sound on both sides
            out[s:e] = True
    mn = int(round(min_run_s / hop_s))
    for s, e in _runs(out):
        if e - s < mn:
            out[s:e] = False
    return out


def detect_activity(env: Envelope) -> Activity:
    n = len(env.db)
    if n < 20:
        return Activity(np.zeros(n, bool), env.hop_s, env.duration_s, False, 0.0, 0.0, "audio is too short")
    floor = float(np.percentile(env.db, 10))
    top = float(np.percentile(env.db, 95))
    if top - floor < CONTRAST_DB:
        return Activity(np.zeros(n, bool), env.hop_s, env.duration_s, False, floor, top,
                        "no clear difference between loud and quiet parts (silent file, constant noise, or sound all the way through)")
    thr = floor + THRESHOLD_FRACTION * (top - floor)
    act = clean_runs(env.db > thr, env.hop_s)
    return Activity(act, env.hop_s, env.duration_s, True, floor, top)
