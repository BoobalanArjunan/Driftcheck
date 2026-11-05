"""Finding the audio and its labels, pairing them up, and running the checks over a whole dataset."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .activity import DriftcheckError
from .checks import FAIL_S, WARN_S, FileResult, Finding, check_file, chars_per_second
from .labels import LABEL_EXT, Segment, read_labels

AUDIO_EXT = {".wav", ".flac", ".ogg", ".opus", ".mp3", ".aif", ".aiff"}
SIBLING_EXT = (".srt", ".vtt", ".tsv", ".csv", ".jsonl")      # looked for next to the audio when --labels is not given


@dataclass
class Item:
    name: str
    audio: Path
    segments: list[Segment] | None
    error: str | None = None


@dataclass
class Report:
    files: list[FileResult]
    notes: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    def verdict(self) -> str:
        v = {f.verdict() for f in self.files}
        return "FAIL" if "FAIL" in v else "WARN" if "WARN" in v else "PASS"


def find_audio(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise DriftcheckError(f"{path} is not a file or a folder")
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise DriftcheckError(f"no audio files found in {path}")
    return files


def _keys(p: Path, root: Path | None) -> list[str]:
    keys = [p.stem, p.name]
    if root is not None:
        try:
            rel = p.relative_to(root)
            keys += [rel.as_posix(), rel.with_suffix("").as_posix()]
        except ValueError:
            pass
    return keys


def collect(audio: Path, labels: Path | None = None, fmt: str | None = None) -> tuple[list[Item], list[str]]:
    """Pair each audio file with its labels. Returns the items and notes about anything left over."""
    audio = Path(audio)
    files = find_audio(audio)
    root = audio if audio.is_dir() else None
    notes: list[str] = []
    items: list[Item] = []
    table: dict[str, list[Segment]] | None = None
    by_stem: dict[str, Path] = {}
    if labels is not None:
        labels = Path(labels)
        if labels.is_file():
            table = read_labels(labels, fmt)
        elif labels.is_dir():
            for p in sorted(labels.rglob("*")):
                if p.is_file() and p.suffix.lower() in LABEL_EXT and p.suffix.lower() != ".gz":
                    by_stem.setdefault(p.stem, p)
        else:
            raise DriftcheckError(f"{labels} is not a file or a folder")
    used: set[str] = set()
    for a in files:
        name = a.relative_to(root).as_posix() if root else a.name
        try:
            segs = None
            if table is not None:
                if "" in table and len(table) == 1:
                    if len(files) != 1:
                        raise DriftcheckError("this label file describes one recording; give a folder of label files "
                                              "or a manifest with recording ids")
                    segs, used = table[""], used | {""}
                else:
                    hit = next((k for k in _keys(a, root) if k in table), None)
                    if hit is not None:
                        segs, used = table[hit], used | {hit}
            else:
                cand = None
                if by_stem:
                    cand = by_stem.get(a.stem)
                else:
                    cand = next((a.with_suffix(e) for e in SIBLING_EXT if a.with_suffix(e).is_file()), None)
                if cand is not None:
                    got = read_labels(cand, fmt)
                    segs = got.get("") or (next(iter(got.values())) if len(got) == 1 else got.get(a.stem))
            items.append(Item(name, a, segs, None if segs is not None else "no labels found for this audio file"))
        except DriftcheckError as e:
            items.append(Item(name, a, None, str(e)))
    if table is not None:
        left = sorted(set(table) - used - {""})
        if left:
            notes.append(f"{len(left)} recording id(s) in the label file have no audio file (first: {left[0]})")
    return items, notes


def run(audio: Path, labels: Path | None = None, fmt: str | None = None, *, warn_s: float = WARN_S, fail_s: float = FAIL_S,
        window_s: float = 60.0, max_offset_s: float = 5.0, max_files: int | None = None, progress=None) -> Report:
    if warn_s <= 0 or fail_s < warn_s:
        raise DriftcheckError("--warn must be above 0 and --fail must not be below --warn")
    if window_s < 10:
        raise DriftcheckError("--window must be at least 10 seconds")
    items, notes = collect(audio, labels, fmt)
    if max_files:
        if len(items) > max_files:
            notes.append(f"only the first {max_files} of {len(items)} files were checked (--max-files)")
        items = items[:max_files]
    rates = [r for it in items if it.segments for r in chars_per_second(it.segments)]
    ref_rate = float(np.median(rates)) if len(rates) >= 20 else None
    out: list[FileResult] = []
    for i, it in enumerate(items, 1):
        if progress:
            progress(i, len(items), it.name)
        if it.segments is None:
            out.append(FileResult(it.name, None, 0, [Finding("WARN" if it.error.startswith("no labels") else "FAIL",
                                                             "no-labels", it.error)]))
            continue
        try:
            out.append(check_file(it.name, it.audio, it.segments, warn_s=warn_s, fail_s=fail_s, window_s=window_s,
                                  max_offset_s=max_offset_s, ref_rate=ref_rate))
        except DriftcheckError as e:
            out.append(FileResult(it.name, None, len(it.segments), [Finding("FAIL", "unreadable", str(e))]))
    missing = sum(1 for f in out if f.findings and f.findings[0].code == "no-labels")
    if missing and missing < len(out):
        notes.append(f"{missing} audio file(s) had no labels")
    return Report(out, notes, {"warn_s": warn_s, "fail_s": fail_s, "window_s": window_s, "max_offset_s": max_offset_s})


def stderr_progress(i: int, n: int, name: str):
    if n > 1:
        print(f"[{i}/{n}] {name}", file=sys.stderr)
