"""How far are the labels from the audio, and does the gap change over time?

Two on/off signals are compared, both at 10 ms steps:
  * the label signal: ON where some label says there is speech,
  * the audio signal: ON where the audio has sound (see activity.py).
We slide one against the other and see where they agree best. Agreement is counted +1 when both are ON or both
are OFF, -1 when they differ, and 0 where one of them has no information (shifted off the end).

Sign convention (used everywhere): **offset = the number of seconds to ADD to the label times to match the
audio.** A positive offset means the labels are early; the sound happens later than the labels say.

For long files this is done in windows (default 60 s each), giving an offset over time. A straight line through
those points is a steady drift (for example a subtitle file made for a different frame rate); a sudden step is a
jump (for example a cut or an inserted ad).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal, stats

from .activity import Activity
from .labels import Segment

MIN_SCORE = 0.3          # agreement needed at the best position to trust it
NULL_MARGIN = 0.1        # ...and it must beat the best score that unrelated made-up labels reach by this much
N_NULL = 60              # number of made-up unrelated label layouts tried for a whole file
N_NULL_WINDOW = 30       # ...and for each window of a long file
AMBIGUOUS_RATIO = 0.85   # another peak this close to the best one (far away from it) makes the answer ambiguous
EXCLUDE_S = 0.5          # "far away" for the rule above
MIN_WINDOWS = 4          # fewer good windows than this: no drift is reported


def label_indicator(segments: list[Segment], n: int, hop_s: float) -> np.ndarray:
    """Bool per audio frame: True where a label covers the frame's centre."""
    out = np.zeros(n, bool)
    for s in segments:
        if s.end <= s.start:
            continue
        i0 = int(np.ceil(s.start / hop_s - 0.5))
        i1 = int(np.ceil(s.end / hop_s - 0.5))
        i0, i1 = max(i0, 0), min(i1, n)
        if i1 > i0:
            out[i0:i1] = True
    return out


def _pm(b: np.ndarray) -> np.ndarray:
    return np.where(b, 1.0, -1.0)


def lag_scores(xw: np.ndarray, y: np.ndarray, w0: int, max_lag: int, center: int = 0) -> np.ndarray:
    """Agreement between the label window `xw` (+-1, starting at audio frame w0) and the audio signal y (+-1)
    shifted by each lag in -max_lag..max_lag.

    Lag L means: label frame t is compared with audio frame t+L. Audio outside the file counts as 0 (no
    information). Result has length 2*max_lag+1, divided by the window size. With `center`, the lags examined are
    center-max_lag..center+max_lag instead.
    """
    n = len(y)
    w1 = w0 + len(xw)
    lo, hi = w0 + center - max_lag, w1 + center + max_lag
    seg = np.zeros(hi - lo)
    a, b = max(lo, 0), min(hi, n)
    seg[a - lo:b - lo] = y[a:b]
    return signal.fftconvolve(seg, xw[::-1], mode="valid") / max(len(xw), 1)


def _runs(b: np.ndarray):
    """(start, end) pairs of the True runs in a bool array (end exclusive)."""
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def null_layouts(lab: np.ndarray, k: int, rng) -> list[np.ndarray]:
    """`k` made-up label layouts with the same ingredients as the real one but no link to the audio.

    The lengths of the labelled stretches and the lengths of the gaps between them are each shuffled, and the
    result is moved to a random place. These show how well UNRELATED labels of this kind can agree with this audio
    by coincidence, which is what a real measurement has to beat.
    """
    n = len(lab)
    runs = _runs(lab)
    if not runs:
        return []
    lens = np.array([e - s for s, e in runs])
    gaps = np.array([runs[0][0]] + [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)])
    out = []
    for _ in range(k):
        L, G = rng.permutation(lens), rng.permutation(gaps)
        z = np.zeros(n, bool)
        t = 0
        for g, ln in zip(G, L):
            t += int(g)
            z[t:min(t + int(ln), n)] = True
            t += int(ln)
            if t >= n:
                break
        out.append(np.roll(z, int(rng.randint(0, n))))
    return out


@dataclass
class Peak:
    offset_s: float | None
    score: float
    confident: bool
    reason: str = ""


def pick_peak(scores: np.ndarray, max_lag: int, hop_s: float, x_transitions: int, y_transitions: int,
              null_max: float | None = None, center: int = 0) -> Peak:
    """Choose the best lag and decide whether to trust it."""
    k = int(np.argmax(scores))
    best = float(scores[k])
    delta = 0.0
    if 0 < k < len(scores) - 1:
        a, b, c = scores[k - 1], scores[k], scores[k + 1]
        den = a - 2 * b + c
        if den < -1e-12:
            delta = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    offset = (k - max_lag + delta + center) * hop_s
    if x_transitions < 1:
        return Peak(None, best, False, "the labels cover all of this stretch or none of it, so there is nothing to line up")
    if y_transitions < 1:
        return Peak(None, best, False, "the audio has no change between sound and silence here")
    if best < MIN_SCORE or (null_max is not None and best < null_max + NULL_MARGIN):
        return Peak(offset, best, False, "labels and audio do not agree better than unrelated labels would (wrong labels for this audio?)")
    ex = max(int(round(EXCLUDE_S / hop_s)), 1)
    pk, _ = signal.find_peaks(scores)
    for j in pk:
        if abs(int(j) - k) > ex and scores[j] >= AMBIGUOUS_RATIO * best:
            return Peak(offset, best, False, "more than one position fits equally well (repetitive pattern)")
    return Peak(offset, best, True)


def _transitions(b: np.ndarray) -> int:
    return int(np.count_nonzero(np.diff(b.astype(np.int8))))


def _peak_for(x: np.ndarray, y: np.ndarray, lab: np.ndarray, act_bits: np.ndarray, w0: int, w1: int,
              max_lag: int, hop_s: float, nulls: list[np.ndarray], center: int = 0) -> Peak:
    M = int(max(min(max_lag, (w1 - w0 - 8) // 4), 1))
    xw = x[w0:w1]
    sc = lag_scores(xw, y, w0, M, center)
    nm = None
    if nulls:
        nm = max(float(lag_scores(_pm(z[w0:w1]), y, w0, M, center).max()) for z in nulls)
    return pick_peak(sc, M, hop_s, _transitions(lab[w0:w1]), _transitions(act_bits[w0:w1]), nm, center)


def estimate_offset(act: Activity, segments: list[Segment], max_offset_s: float = 5.0, seed: int = 0) -> Peak:
    """One offset for the whole file."""
    if not act.reliable:
        return Peak(None, 0.0, False, act.reason)
    n = act.n
    lab = label_indicator(segments, n, act.hop_s)
    M = max(int(round(max_offset_s / act.hop_s)), 1)
    return _peak_for(_pm(lab), _pm(act.active), lab, act.active, 0, n, M, act.hop_s,
                     null_layouts(lab, N_NULL, np.random.RandomState(seed)))


@dataclass
class WindowOffset:
    start_s: float
    end_s: float
    offset_s: float | None
    score: float
    confident: bool
    reason: str = ""


@dataclass
class Drift:
    windows: list[WindowOffset]
    n_good: int = 0
    slope_s_per_min: float | None = None
    offset_start_s: float | None = None
    offset_end_s: float | None = None
    offset_median_s: float | None = None
    scatter_s: float | None = None
    jumps: list[dict] = field(default_factory=list)


def estimate_drift(act: Activity, segments: list[Segment], window_s: float = 60.0, max_offset_s: float = 5.0,
                   jump_s: float = 0.3, seed: int = 0) -> Drift:
    """Offset in consecutive windows of `window_s`, then a straight line through them and a search for jumps."""
    if not act.reliable:
        return Drift(windows=[])
    n = act.n
    hop = act.hop_s
    lab = label_indicator(segments, n, hop)
    x, y = _pm(lab), _pm(act.active)
    M = max(int(round(max_offset_s / hop)), 1)
    w = max(int(round(window_s / hop)), 10)
    nulls = null_layouts(lab, N_NULL_WINDOW, np.random.RandomState(seed))
    wins: list[WindowOffset] = []
    seen: list[tuple[float, float]] = []          # (window centre time, offset) of the good windows so far
    for w0 in range(0, n - w // 2, w):
        w1 = min(w0 + w, n)
        if w1 - w0 < w // 2:
            break
        # Each window searches around where the previous good windows say the offset should be now, so a steady
        # drift (which can add up to many seconds) is followed instead of being lost beyond the search range.
        tc = 0.5 * (w0 + w1) * hop
        centre = 0.0
        if seen:
            centre = seen[-1][1]
            if len(seen) >= 2:
                (t1, o1), (t2, o2) = seen[-2], seen[-1]
                centre = o2 + (o2 - o1) / (t2 - t1) * (tc - t2)
        pk = _peak_for(x, y, lab, act.active, w0, w1, M, hop, nulls, int(round(centre / hop)))
        if pk.confident:
            seen.append((tc, pk.offset_s))
        wins.append(WindowOffset(round(w0 * hop, 3), round(w1 * hop, 3), None if pk.offset_s is None else round(pk.offset_s, 4),
                                 round(pk.score, 3), pk.confident, pk.reason))
    d = Drift(windows=wins)
    good = [(0.5 * (v.start_s + v.end_s), v.offset_s) for v in wins if v.confident]
    d.n_good = len(good)
    if good:
        d.offset_median_s = float(np.median([g[1] for g in good]))
    if len(good) < MIN_WINDOWS:
        return d
    t = np.array([g[0] for g in good])
    o = np.array([g[1] for g in good])
    # jumps first: a step between neighbouring good windows that the steady drift cannot explain
    dt = np.diff(t)
    adj = np.diff(o) / dt                                   # drift between neighbours, seconds per second
    typical = float(np.median(adj))
    steps = np.diff(o) - typical * dt                       # what is left over at each step
    noise = max(float(1.4826 * np.median(np.abs(steps - np.median(steps)))), 0.02)
    jump_at = np.zeros(len(o), float)                       # size of the jump just before each good window
    for i, st in enumerate(steps):
        if abs(st) >= jump_s and abs(st) >= 4 * noise:
            d.jumps.append({"at_s": float(0.5 * (t[i] + t[i + 1])), "step_s": float(st)})
            jump_at[i + 1] = st
    o_adj = o - np.cumsum(jump_at)                          # offsets with the jumps taken out
    slope, icpt, _, _ = stats.theilslopes(o_adj, t)
    res = o_adj - (icpt + slope * t)
    d.scatter_s = float(1.4826 * np.median(np.abs(res - np.median(res))))
    d.slope_s_per_min = float(slope * 60.0)
    d.offset_start_s = float(icpt)
    t_end = min(act.duration_s, max(sg.end for sg in segments))     # the end of the labels, in label time
    d.offset_end_s = float(icpt + slope * t_end + jump_at.sum())
    return d
