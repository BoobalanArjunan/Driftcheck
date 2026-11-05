"""The per-file checks: put the labels next to the audio and list what does not fit."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .activity import Activity, DriftcheckError, detect_activity, envelope_from_file, _runs
from .align import MIN_WINDOWS, Drift, estimate_drift, estimate_offset, label_indicator
from .labels import Segment

# Judgement defaults, not standards. Change them with --warn / --fail.
WARN_S = 0.15                # an offset this big is noticeable in subtitles and starts to hurt alignment-based training
FAIL_S = 0.5
UNLABELLED_WARN, UNLABELLED_FAIL = 0.10, 0.40     # share of the speech that no label covers
SILENT_WARN, SILENT_FAIL = 0.10, 0.40             # share of the labelled time that is silent
PAD_S = 0.15                 # labels are given this much slack on each side before saying "speech outside labels"
MIN_UNLABELLED_S = 0.5       # speech outside the labels shorter than this is ignored
MIN_SILENT_LABEL_S = 1.0     # shorter labels are not judged for silence
SILENT_ACTIVE_SHARE = 0.2    # a label with less than this share of sound counts as silent
RATE_RATIO = 3.0             # characters per second more than this many times the usual (or less than 1/RATE_RATIO^1.33)
RATE_ABS_MAX = 35.0          # characters (without spaces) per second nobody can speak

# Pairs of frame rates that are commonly mixed up: label times written for one, audio/video made at the other.
FRAME_RATE_RATIOS = {
    "25 vs 23.976 fps": 25 / 23.976, "23.976 vs 25 fps": 23.976 / 25,
    "25 vs 24 fps": 25 / 24, "24 vs 25 fps": 24 / 25,
    "30 vs 29.97 fps": 30 / 29.97, "29.97 vs 30 fps": 29.97 / 30,
    "24 vs 23.976 fps": 24 / 23.976, "23.976 vs 24 fps": 23.976 / 24,
}


@dataclass
class Finding:
    level: str            # INFO, WARN or FAIL
    code: str
    message: str
    at_s: float | None = None


@dataclass
class FileResult:
    name: str
    duration_s: float | None
    n_labels: int
    findings: list[Finding] = field(default_factory=list)
    measurable: bool = False
    offset_s: float | None = None             # best single number (the middle of the file when it drifts)
    offset_start_s: float | None = None
    offset_end_s: float | None = None
    drift_s_per_min: float | None = None
    score: float | None = None
    windows: list = field(default_factory=list)
    jumps: list = field(default_factory=list)
    unlabelled_speech_share: float | None = None
    silent_label_share: float | None = None
    suggested_scale: float | None = None

    def verdict(self) -> str:
        lv = {f.level for f in self.findings}
        return "FAIL" if "FAIL" in lv else "WARN" if "WARN" in lv else "PASS"


def _level(x: float, warn: float, fail: float) -> str | None:
    return "FAIL" if x >= fail else "WARN" if x >= warn else None


def _ms(x: float) -> str:
    return f"{x * 1000:+.0f} ms" if abs(x) < 1 else f"{x:+.2f} s"


def _chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def chars_per_second(segs: list[Segment], min_s: float = 1.0) -> list[float]:
    return [_chars(s.text) / s.duration for s in segs if s.text and s.duration >= min_s]


def _structure_checks(segs: list[Segment], duration: float | None, F: list[Finding]):
    bad = [s for s in segs if s.end <= s.start]
    if bad:
        F.append(Finding("FAIL", "bad-length", f"{len(bad)} label(s) end before they start or have zero length "
                         f"(first at {bad[0].start:.2f} s)", bad[0].start))
    if any(b.start < a.start - 1e-9 for a, b in zip(segs, segs[1:])):
        F.append(Finding("WARN", "out-of-order", "labels are not in time order"))
    ordered = sorted((s for s in segs if s.end > s.start), key=lambda s: s.start)
    ov = [(a, b) for a, b in zip(ordered, ordered[1:]) if a.end - b.start > 0.5]
    if ov:
        F.append(Finding("WARN", "overlap", f"{len(ov)} label(s) overlap the one before by more than 0.5 s "
                         f"(first at {ov[0][1].start:.2f} s)", ov[0][1].start))
    if duration is not None:
        after = [s for s in segs if s.start >= duration]
        if after:
            F.append(Finding("FAIL", "after-end", f"{len(after)} label(s) start after the audio has ended "
                             f"({duration:.1f} s); first at {after[0].start:.1f} s", after[0].start))
        else:
            past = [s for s in segs if s.end > duration + 0.5]
            if past:
                F.append(Finding("WARN", "past-end", f"{len(past)} label(s) run past the end of the audio ({duration:.1f} s)",
                                 past[0].start))
    texts = [bool(s.text.strip()) for s in segs]
    if any(texts) and not all(texts):
        F.append(Finding("WARN", "empty-text", f"{texts.count(False)} label(s) have no text"))


def _rate_checks(segs: list[Segment], ref_rate: float | None, F: list[Finding]):
    rates = [(_chars(s.text) / s.duration, s) for s in segs if s.text and s.duration >= 1.0]
    if not rates:
        return
    ref = ref_rate if ref_rate else (float(np.median([r for r, _ in rates])) if len(rates) >= 8 else None)
    fast = [(r, s) for r, s in rates if r > RATE_ABS_MAX or (ref and r > RATE_RATIO * ref)]
    slow = [(r, s) for r, s in rates if ref and r < ref / RATE_RATIO ** 1.33]
    if fast:
        r, s = max(fast, key=lambda t: t[0])
        F.append(Finding("WARN", "too-fast", f"{len(fast)} label(s) hold more text than anyone speaks in that time "
                         f"(fastest: {r:.0f} characters per second at {s.start:.1f} s): the text may belong to a longer stretch", s.start))
    if slow:
        r, s = min(slow, key=lambda t: t[0])
        F.append(Finding("WARN", "too-slow", f"{len(slow)} label(s) hold very little text for their length "
                         f"(slowest: {r:.1f} characters per second at {s.start:.1f} s): the label may cover more than the text", s.start))


def _timing(act: Activity, segs: list[Segment], res: FileResult, F: list[Finding], *, warn_s, fail_s, window_s, max_offset_s):
    long_file = act.duration_s >= 2 * window_s
    drift: Drift | None = estimate_drift(act, segs, window_s, max_offset_s) if long_file else None
    if drift is not None and drift.n_good >= MIN_WINDOWS:
        res.windows = [vars(w) for w in drift.windows]
        res.jumps = drift.jumps
        res.score = float(np.median([w.score for w in drift.windows if w.confident]))
        total = drift.offset_end_s - drift.offset_start_s - sum(j["step_s"] for j in drift.jumps)
        steady = abs(total) >= warn_s and abs(total) >= 3 * drift.scatter_s
        res.offset_start_s, res.offset_end_s = drift.offset_start_s, drift.offset_end_s
        res.measurable = True
        if steady:
            res.drift_s_per_min = drift.slope_s_per_min
            res.offset_s = 0.5 * (drift.offset_start_s + drift.offset_end_s)
            ratio = 1.0 + drift.slope_s_per_min / 60.0
            res.suggested_scale = ratio
            worst = max(abs(drift.offset_start_s), abs(drift.offset_end_s))
            lvl = _level(worst, warn_s, fail_s) or "WARN"
            hint = next((k for k, v in FRAME_RATE_RATIOS.items() if abs(ratio - v) < 0.0004), None)
            F.append(Finding(lvl, "drift",
                             f"the labels drift against the audio: offset is {_ms(drift.offset_start_s)} at the start and "
                             f"{_ms(drift.offset_end_s)} at the end ({drift.slope_s_per_min * 1000:+.0f} ms per minute). "
                             f"Multiplying the label times by about {ratio:.5f} would remove the drift"
                             + (f"; that is the ratio of {hint}, a common frame-rate mix-up" if hint else "")))
        else:
            res.offset_s = drift.offset_median_s
            res.drift_s_per_min = drift.slope_s_per_min
            lvl = _level(abs(res.offset_s), warn_s, fail_s)
            F.append(Finding(lvl or "INFO", "offset",
                             f"labels are {_ms(res.offset_s)} from the audio (add this to the label times to line up)"
                             if lvl else f"labels line up with the audio within {1000 * max(abs(res.offset_s), 0.01):.0f} ms "
                             f"and do not drift (steady over {drift.n_good} windows)"))
        for j in drift.jumps:
            F.append(Finding("FAIL" if abs(j["step_s"]) >= fail_s else "WARN", "jump",
                             f"the offset suddenly changes by {_ms(j['step_s'])} near {j['at_s']:.0f} s "
                             f"(a cut, an inserted section, or labels from two different versions)", j["at_s"]))
        return
    # not enough good windows (short file, or windows that could not be measured): one offset for the whole file
    pk = estimate_offset(act, segs, max_offset_s)
    res.score = pk.score
    if drift is not None:
        res.windows = [vars(w) for w in drift.windows]
    if pk.confident:
        res.measurable = True
        res.offset_s = pk.offset_s
        lvl = _level(abs(pk.offset_s), warn_s, fail_s)
        msg = (f"labels are {_ms(pk.offset_s)} from the audio (add this to the label times to line up)" if lvl else
               f"labels line up with the audio within {1000 * max(abs(pk.offset_s), 0.01):.0f} ms")
        if int(act.duration_s // window_s) < MIN_WINDOWS:
            note = f" (drift is not measured on this file: it needs at least {MIN_WINDOWS} windows of {window_s:g} s)"
        else:
            note = " (drift could not be measured: too few good windows)"
        F.append(Finding(lvl or "INFO", "offset", msg + note))
    else:
        F.append(Finding("INFO", "offset-unmeasurable", f"offset could not be measured: {pk.reason}"))


def _coverage(act: Activity, segs: list[Segment], res: FileResult, F: list[Finding], offset_note: bool):
    hop = act.hop_s
    lab = label_indicator(segs, act.n, hop)
    pad = max(int(round(PAD_S / hop)), 1)
    dil = ndimage.binary_dilation(lab, iterations=pad) if lab.any() else lab
    total_active = float(act.active.sum()) * hop
    # speech outside the labels
    runs = [(s, e) for s, e in _runs(act.active & ~dil) if (e - s) * hop >= MIN_UNLABELLED_S]
    un = sum(e - s for s, e in runs) * hop
    if total_active > 0:
        res.unlabelled_speech_share = un / total_active
        lvl = _level(res.unlabelled_speech_share, UNLABELLED_WARN, UNLABELLED_FAIL)
        if lvl:
            s0, e0 = max(runs, key=lambda r: r[1] - r[0])
            F.append(Finding(lvl, "unlabelled-speech",
                             f"{100 * res.unlabelled_speech_share:.0f}% of the sound ({un:.0f} s in {len(runs)} stretches) is outside every label; "
                             f"longest: {(e0 - s0) * hop:.1f} s at {s0 * hop:.1f} s"
                             + ("; this may be a result of the offset above" if offset_note else ""), s0 * hop))
    # labels over silence
    sil, tot = 0.0, 0.0
    first = None
    for sg in segs:
        d = sg.end - sg.start
        if d < MIN_SILENT_LABEL_S:
            continue
        i0, i1 = max(int(round(sg.start / hop)), 0), min(int(round(sg.end / hop)), act.n)
        if i1 - i0 < 10:
            continue
        tot += d
        if act.active[i0:i1].mean() < SILENT_ACTIVE_SHARE:
            sil += d
            first = first if first is not None else sg.start
    if tot > 0:
        res.silent_label_share = sil / tot
        lvl = _level(res.silent_label_share, SILENT_WARN, SILENT_FAIL)
        if lvl:
            F.append(Finding(lvl, "silent-labels",
                             f"{100 * res.silent_label_share:.0f}% of the labelled time ({sil:.0f} s) has almost no sound in the audio; "
                             f"first at {first:.1f} s"
                             + ("; this may be a result of the offset above" if offset_note else ""), first))


def check_file(name: str, audio_path: Path | None, segs: list[Segment], *, warn_s: float = WARN_S, fail_s: float = FAIL_S,
               window_s: float = 60.0, max_offset_s: float = 5.0, ref_rate: float | None = None) -> FileResult:
    """Check one recording's labels against its audio. audio_path=None runs only the checks that need no audio."""
    F: list[Finding] = []
    act = None
    duration = None
    if audio_path is not None:
        env = envelope_from_file(audio_path)
        duration = env.duration_s
        act = detect_activity(env)
    res = FileResult(name, duration, len(segs), F)
    _structure_checks(segs, duration, F)
    _rate_checks(segs, ref_rate, F)
    if act is None:
        return res
    if not act.reliable:
        F.append(Finding("INFO", "audio-unreliable", f"timing was not checked: {act.reason}"))
        return res
    valid = [s for s in segs if s.end > s.start]
    if not valid:
        return res
    _timing(act, valid, res, F, warn_s=warn_s, fail_s=fail_s, window_s=window_s, max_offset_s=max_offset_s)
    off_flag = any(f.code in ("offset", "drift", "jump") and f.level in ("WARN", "FAIL") for f in F)
    _coverage(act, valid, res, F, off_flag)
    return res
