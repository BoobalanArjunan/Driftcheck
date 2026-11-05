"""Reading timed labels: subtitles (SRT, VTT), Audacity label files, CSV, and Lhotse supervision manifests.

Every reader returns a list of ``Segment(start, end, text)`` with times in seconds, in the order they appear in
the file (we do not sort them: a file out of order is something we report).
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .activity import DriftcheckError

LABEL_EXT = {".srt": "srt", ".vtt": "vtt", ".tsv": "audacity", ".txt": "audacity", ".csv": "csv",
             ".jsonl": "lhotse", ".json": "lhotse", ".gz": "lhotse"}
FORMATS = ("srt", "vtt", "audacity", "csv", "lhotse")


@dataclass
class Segment:
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
_ARROW = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}")


def _t(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def _read_text(path: Path) -> str:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise DriftcheckError(f"cannot read {p.name}: {e}") from None
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8-sig", errors="replace")


def parse_subtitles(text: str, name: str = "labels") -> list[Segment]:
    """SRT and WebVTT share one shape: a line 'start --> end', then the text lines, then a blank line."""
    segs: list[Segment] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _ARROW.match(lines[i])
        if not m:
            i += 1
            continue
        g = m.groups()
        start, end = _t(*g[:4]), _t(*g[4:])
        i += 1
        body = []
        while i < len(lines) and lines[i].strip() != "":
            body.append(lines[i].strip())
            i += 1
        segs.append(Segment(start, end, " ".join(body)))
    if not segs:
        raise DriftcheckError(f"{name}: no 'start --> end' lines found (is this an SRT or VTT file?)")
    return segs


def parse_audacity(text: str, name: str = "labels") -> list[Segment]:
    """Audacity label track: start<TAB>end<TAB>text, times in seconds. Lines starting with a backslash are skipped."""
    segs = []
    for n, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        if not line.strip() or line.startswith("\\"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise DriftcheckError(f"{name} line {n}: expected 'start<TAB>end<TAB>text'")
        try:
            segs.append(Segment(float(parts[0]), float(parts[1]), parts[2].strip() if len(parts) > 2 else ""))
        except ValueError:
            raise DriftcheckError(f"{name} line {n}: start and end must be numbers (seconds)") from None
    if not segs:
        raise DriftcheckError(f"{name}: no labels found")
    return segs


_START = ("start", "start_s", "begin", "onset", "start_time", "xmin")
_END = ("end", "end_s", "stop", "offset", "end_time", "xmax")
_DUR = ("duration", "dur", "duration_s")
_TEXT = ("text", "transcript", "label", "word", "sentence")
_ID = ("recording_id", "file", "filename", "audio", "id")


def _col(header: list[str], names) -> int | None:
    low = [h.strip().lower() for h in header]
    for nm in names:
        if nm in low:
            return low.index(nm)
    return None


def parse_csv(text: str, name: str = "labels") -> dict[str, list[Segment]]:
    """CSV with a header row. Needs start plus end (or duration). Text is optional; so is a file id column."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise DriftcheckError(f"{name}: empty file")
    head = rows[0]
    s, e, d, t, r = (_col(head, c) for c in (_START, _END, _DUR, _TEXT, _ID))
    if s is None or (e is None and d is None):
        raise DriftcheckError(f"{name}: need a header row with start and end (or duration) columns; found: {', '.join(head)}")
    out: dict[str, list[Segment]] = {}
    for n, row in enumerate(rows[1:], 2):
        if not any(c.strip() for c in row):
            continue
        try:
            a = float(row[s])
            b = float(row[e]) if e is not None else a + float(row[d])
        except (ValueError, IndexError):
            raise DriftcheckError(f"{name} line {n}: start and end/duration must be numbers (seconds)") from None
        txt = row[t].strip() if t is not None and t < len(row) else ""
        out.setdefault(row[r].strip() if r is not None and r < len(row) else "", []).append(Segment(a, b, txt))
    if not out:
        raise DriftcheckError(f"{name}: no label rows found")
    return out


def parse_lhotse(text: str, name: str = "labels") -> dict[str, list[Segment]]:
    """Lhotse supervisions (one JSON object per line: recording_id, start, duration, text)."""
    out: dict[str, list[Segment]] = {}
    for n, line in enumerate(text.split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            a = float(o["start"])
            b = a + float(o["duration"])
            rid = str(o["recording_id"])
        except (ValueError, KeyError, TypeError):
            raise DriftcheckError(f"{name} line {n}: expected a Lhotse supervision with recording_id, start and duration") from None
        out.setdefault(rid, []).append(Segment(a, b, str(o.get("text") or "")))
    if not out:
        raise DriftcheckError(f"{name}: no supervisions found")
    return out


def guess_format(path: Path) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".gz":
        ext = Path(Path(path).stem).suffix.lower()
        return "lhotse" if ext in (".jsonl", ".json") else "audacity"
    if ext not in LABEL_EXT:
        raise DriftcheckError(f"cannot tell the label format of {Path(path).name}; use --format ({', '.join(FORMATS)})")
    return LABEL_EXT[ext]


def read_labels(path: Path, fmt: str | None = None) -> dict[str, list[Segment]]:
    """Read a label file. Returns {recording id: segments}; formats that describe one recording use the key ''."""
    path = Path(path)
    fmt = fmt or guess_format(path)
    if fmt not in FORMATS:
        raise DriftcheckError(f"unknown label format {fmt!r}; choose one of: {', '.join(FORMATS)}")
    text = _read_text(path)
    if fmt in ("srt", "vtt"):
        return {"": parse_subtitles(text, path.name)}
    if fmt == "audacity":
        return {"": parse_audacity(text, path.name)}
    if fmt == "csv":
        return parse_csv(text, path.name)
    return parse_lhotse(text, path.name)
