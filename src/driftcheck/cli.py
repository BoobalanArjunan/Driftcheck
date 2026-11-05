from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._version import __version__
from .activity import DriftcheckError
from .labels import FORMATS
from .report import report_to_dict, report_to_text, write_reports
from .scan import run, stderr_progress


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="driftcheck",
                                description="Check that timed labels (subtitles, transcripts with times) still line up with their audio.")
    p.add_argument("--version", action="version", version=f"driftcheck {__version__}")
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("check", help="check audio file(s) against their labels",
                       description="AUDIO is an audio file or a folder of them. Labels are found next to each audio file "
                                   "(same name with .srt, .vtt, .tsv, .csv or .jsonl) or given with --labels (a file, a folder, "
                                   "or a Lhotse supervisions manifest).")
    c.add_argument("audio", type=Path)
    c.add_argument("--labels", type=Path, help="label file or folder (default: next to each audio file)")
    c.add_argument("--format", choices=FORMATS, help="label format (default: guessed from the file extension)")
    c.add_argument("--out", type=Path, help="write the report files into this folder")
    c.add_argument("--json", action="store_true", help="print JSON instead of text")
    c.add_argument("--warn", type=float, default=0.15, metavar="SECONDS", help="offset that gives a WARN (default 0.15)")
    c.add_argument("--fail", type=float, default=0.5, metavar="SECONDS", help="offset that gives a FAIL (default 0.5)")
    c.add_argument("--window", type=float, default=60.0, metavar="SECONDS", help="window length for drift (default 60)")
    c.add_argument("--max-offset", type=float, default=5.0, metavar="SECONDS", help="largest offset to look for (default 5)")
    c.add_argument("--max-files", type=int, help="check only the first N files")
    return p


def main(argv=None) -> int:
    p = _parser()
    a = p.parse_args(argv)
    if a.cmd != "check":
        p.print_help()
        return 2
    try:
        rep = run(a.audio, a.labels, a.format, warn_s=a.warn, fail_s=a.fail, window_s=a.window, max_offset_s=a.max_offset,
                  max_files=a.max_files, progress=stderr_progress)
        if a.out:
            write_reports(rep, a.out)
    except DriftcheckError as e:
        print(f"driftcheck: {e}", file=sys.stderr)
        return 2
    if a.json:
        import json
        print(json.dumps(report_to_dict(rep), indent=2))
    else:
        print(report_to_text(rep))
        if a.out:
            print(f"\nreport files written to {a.out}")
    return 1 if rep.verdict() == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
