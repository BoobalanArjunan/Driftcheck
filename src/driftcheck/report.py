from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .scan import Report

NOTE = ("How to read this: 'offset' is the number of seconds to ADD to the label times to match the audio. It is found "
        "by lining up where the labels say there is speech with where the audio has sound, using a simple energy "
        "detector (no speech model). It works when speech stands out from the background and needs a minute or more "
        "of audio to be certain; on short clips it says 'could not be measured' instead of guessing. A PASS means "
        "none of these checks found a problem, not that every label is right.")


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def report_to_dict(rep: Report) -> dict:
    return {
        "verdict": rep.verdict(), "settings": rep.settings, "notes": rep.notes, "note": NOTE,
        "files": [{
            "name": f.name, "verdict": f.verdict(), "duration_s": _r(f.duration_s, 2), "n_labels": f.n_labels,
            "timing_measured": f.measurable, "offset_s": _r(f.offset_s), "offset_start_s": _r(f.offset_start_s),
            "offset_end_s": _r(f.offset_end_s), "drift_s_per_min": _r(f.drift_s_per_min, 4), "agreement_score": _r(f.score, 3),
            "suggested_scale": _r(f.suggested_scale, 6), "jumps": f.jumps,
            "unlabelled_speech_share": _r(f.unlabelled_speech_share), "silent_label_share": _r(f.silent_label_share),
            "findings": [asdict(x) for x in f.findings], "windows": f.windows,
        } for f in rep.files]}


def report_to_text(rep: Report, max_files: int = 15) -> str:
    n = len(rep.files)
    cnt = {k: sum(1 for f in rep.files if f.verdict() == k) for k in ("FAIL", "WARN", "PASS")}
    L = [f"driftcheck verdict: {rep.verdict()}"]
    if n > 1:
        L.append(f"{n} files: {cnt['PASS']} pass, {cnt['WARN']} warn, {cnt['FAIL']} fail")
        meas = [f for f in rep.files if f.measurable]
        L.append(f"timing could be measured on {len(meas)} of {n} files (the rest are too short, too quiet or have no labels)")
    L.append("")
    shown = sorted(rep.files, key=lambda f: ("FAIL", "WARN", "PASS").index(f.verdict()))
    if n > 1:
        shown = [f for f in shown if f.verdict() != "PASS"][:max_files] or shown[:3]
    for f in shown:
        dur = f"{f.duration_s:.1f} s, " if f.duration_s is not None else ""
        L.append(f"{f.name}: {f.verdict()} ({dur}{f.n_labels} labels)")
        for x in f.findings:
            if n > 1 and x.level == "INFO":
                continue
            L.append(f"  {x.level:4s} {x.message}")
        L.append("")
    if n > 1 and len([f for f in rep.files if f.verdict() != "PASS"]) > max_files:
        L.append(f"... and {len([f for f in rep.files if f.verdict() != 'PASS']) - max_files} more files with findings (see files.csv / findings.csv)")
        L.append("")
    for x in rep.notes:
        L.append(f"note: {x}")
    L += ["", NOTE]
    return "\n".join(L)


def _write(path: Path, text: str):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_reports(rep: Report, out: Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "driftcheck_report.json", json.dumps(report_to_dict(rep), indent=2) + "\n")
    _write(out / "driftcheck_report.txt", report_to_text(rep) + "\n")
    with open(out / "files.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["file", "verdict", "duration_s", "n_labels", "timing_measured", "offset_s", "offset_start_s", "offset_end_s",
                    "drift_ms_per_min", "agreement_score", "unlabelled_speech_share", "silent_label_share", "findings"])
        for f in rep.files:
            w.writerow([f.name, f.verdict(), _r(f.duration_s, 2), f.n_labels, f.measurable, _r(f.offset_s), _r(f.offset_start_s),
                        _r(f.offset_end_s), None if f.drift_s_per_min is None else round(f.drift_s_per_min * 1000, 1),
                        _r(f.score, 3), _r(f.unlabelled_speech_share), _r(f.silent_label_share),
                        "; ".join(x.code for x in f.findings if x.level != "INFO")])
    with open(out / "findings.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["file", "level", "code", "at_s", "message"])
        for f in rep.files:
            for x in f.findings:
                w.writerow([f.name, x.level, x.code, x.at_s, x.message])
    with open(out / "windows.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["file", "start_s", "end_s", "offset_s", "agreement_score", "confident", "reason"])
        for f in rep.files:
            for x in f.windows:
                w.writerow([f.name, x["start_s"], x["end_s"], x["offset_s"], x["score"], x["confident"], x["reason"]])
