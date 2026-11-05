"""Real-speech check for driftcheck, using LibriSpeech (or any folder of speech with transcripts).

LibriSpeech has one short utterance per file, so we build LONG recordings from it: many utterances from the same
reader joined with pauses of random length. The true time of every utterance is known exactly, so the true labels
are known exactly. Then we damage the labels in known ways and see whether driftcheck notices and measures it.

    python scripts/real_data_check.py --librispeech ~/LibriSpeech/dev-clean --talks 6 --out real_check

Nothing in your LibriSpeech folder is changed. Everything is written under --out.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from driftcheck import Segment, check_file
except ImportError:
    sys.exit("driftcheck is not installed: pip install --pre driftcheck   (or pip install -e . from a checkout)")

SR = 16000


def find_utterances(root: Path):
    """{(speaker, chapter): [(audio path, text)]} from LibriSpeech's *.trans.txt files."""
    out = {}
    for tf in sorted(root.rglob("*.trans.txt")):
        key = tf.parent.parent.name + "-" + tf.parent.name
        for line in tf.read_text().splitlines():
            uid, _, text = line.partition(" ")
            p = tf.parent / f"{uid}.flac"
            if p.is_file():
                out.setdefault(key, []).append((p, text))
    return out


def build_talk(utts, n_utts: int, rng, out_wav: Path):
    """Join utterances with random pauses. Returns [(start, end, text)] in seconds (the exact truth)."""
    chosen = utts[:n_utts]
    parts, truth, t = [], [], 0.0
    lead = rng.uniform(0.5, 2.0)
    parts.append(0.0005 * rng.randn(int(lead * SR)))
    t += lead
    for p, text in chosen:
        x, sr = sf.read(str(p), dtype="float64")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            raise SystemExit(f"{p.name} is {sr} Hz; this script expects 16 kHz (LibriSpeech)")
        parts.append(x)
        truth.append((t, t + len(x) / SR, text.lower()))
        t += len(x) / SR
        gap = rng.uniform(0.5, 3.0)
        parts.append(0.0005 * rng.randn(int(gap * SR)))      # a quiet room, about -66 dBFS
        t += gap
    sf.write(str(out_wav), np.concatenate(parts).astype(np.float32), SR, subtype="PCM_16")
    return truth


def segs(rows):
    return [Segment(a, b, tx) for a, b, tx in rows]


def run_one(wav: Path, rows, **kw):
    return check_file(wav.name, wav, segs(rows), **kw)


def find(res, code):
    return next((f for f in res.findings if f.code == code), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--librispeech", type=Path, required=True, help="a LibriSpeech folder, for example dev-clean")
    ap.add_argument("--talks", type=int, default=6, help="how many long recordings to build (default 6)")
    ap.add_argument("--utts", type=int, default=40, help="utterances per recording (default 40, about 5 minutes)")
    ap.add_argument("--out", type=Path, default=Path("real_check"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    books = find_utterances(a.librispeech)
    books = {k: v for k, v in books.items() if len(v) >= a.utts}
    if len(books) < a.talks:
        sys.exit(f"found only {len(books)} chapters with at least {a.utts} utterances in {a.librispeech}; "
                 f"lower --utts or --talks")
    rng = np.random.RandomState(a.seed)
    keys = sorted(books)
    rng.shuffle(keys)
    keys = keys[: a.talks]
    a.out.mkdir(parents=True, exist_ok=True)

    talks = []
    for k in keys:
        wav = a.out / f"talk_{k}.wav"
        truth = build_talk(books[k], a.utts, rng, wav)
        talks.append((k, wav, truth))
        print(f"built {wav.name}: {truth[-1][1]:.0f} s, {len(truth)} labels", flush=True)

    t0 = time.time()
    shortest = min(t[2][-1][1] for t in talks)
    L = ["# driftcheck: real-speech check (LibriSpeech)", "",
         f"{len(talks)} recordings built from LibriSpeech readers, about {a.utts} utterances each. "
         "True label times are exact; the defects below are planted by us.", ""]
    if shortest < 240:
        L += [f"> **Note:** the shortest recording is only {shortest:.0f} s. Drift and jumps need at least 4 windows of 60 s, "
              "so sections 3 and 4 cannot work on recordings this short. Use more utterances (--utts 40 or more).", ""]

    # --- 1. undamaged labels
    base = {}
    verdicts, offs, unl, sil = [], [], [], []
    for k, wav, truth in talks:
        r = run_one(wav, truth)
        base[k] = r.offset_s or 0.0
        verdicts.append(r.verdict())
        offs.append(r.offset_s)
        unl.append(r.unlabelled_speech_share)
        sil.append(r.silent_label_share)
        print(f"  clean {k}: {r.verdict()}  offset {r.offset_s}", flush=True)
    meas = [o for o in offs if o is not None]
    L += ["## 1. Undamaged labels (should PASS)", "",
          f"- verdicts: {', '.join(f'{v}={verdicts.count(v)}' for v in ('PASS', 'WARN', 'FAIL'))}",
          f"- timing measured on {len(meas)} of {len(talks)} recordings",
          f"- measured offset (seconds; a systematic value here is the natural looseness of utterance-level labels): "
          f"median {st.median(meas) if meas else 'n/a'}, range {min(meas) if meas else 'n/a'} to {max(meas) if meas else 'n/a'}",
          f"- unlabelled-speech share: median {st.median([u for u in unl if u is not None]):.3f}" if any(u is not None for u in unl) else "- unlabelled-speech share: n/a",
          f"- silent-label share: median {st.median([u for u in sil if u is not None]):.3f}" if any(u is not None for u in sil) else "- silent-label share: n/a",
          "", "If any recording here is WARN or FAIL, read its findings below: that is a false alarm on correct labels.", ""]
    for (k, wav, truth), v in zip(talks, verdicts):
        if v != "PASS":
            r = run_one(wav, truth)
            L.append(f"- `{wav.name}` {v}: " + " | ".join(f"{f.level} {f.message}" for f in r.findings if f.level != "INFO"))
    L.append("")

    # --- 2. planted offsets
    L += ["## 2. Planted offsets", "",
          "Error = measured minus (planted + the clean baseline above). Offsets are what driftcheck should report "
          "(seconds to add to the labels).", "",
          "| planted offset (s) | found | median abs error (ms) | worst abs error (ms) |", "|---|---|---|---|"]
    for planted in (0.2, -0.2, 0.5, -1.0, 2.0):
        errs, found = [], 0
        for k, wav, truth in talks:
            moved = [(max(s - planted, 0.0), e - planted, tx) for s, e, tx in truth]      # labels EARLY by `planted`
            r = run_one(wav, moved)
            if r.offset_s is not None and r.measurable:
                found += 1
                errs.append(abs(r.offset_s - (planted + base[k])) * 1000)
        L.append(f"| {planted:+.1f} | {found}/{len(talks)} | {st.median(errs):.0f} | {max(errs):.0f} |" if errs
                 else f"| {planted:+.1f} | 0/{len(talks)} | n/a | n/a |")
        print(f"  offset {planted:+.1f}: found {found}/{len(talks)}", flush=True)
    L.append("")

    # --- 3. planted drift
    L += ["## 3. Planted steady drift", "",
          "Labels stretched by a factor. 'Slope error' compares the measured drift (seconds per minute of label time) "
          "with the planted one.", "",
          "| label time factor | drift reported | median slope error (s/min) | frame-rate name given |", "|---|---|---|---|"]
    for scale, label in ((1.001, "1.001"), (0.9990, "0.999"), (1.004, "1.004"), (23.976 / 25, "23.976/25")):
        rep, errs, named = 0, [], 0
        for k, wav, truth in talks:
            moved = [(s * scale, e * scale, tx) for s, e, tx in truth]
            r = run_one(wav, moved)
            d = find(r, "drift")
            if d is not None:
                rep += 1
                errs.append(abs(r.drift_s_per_min - (1 / scale - 1) * 60))
                named += "fps" in d.message
        L.append(f"| {label} | {rep}/{len(talks)} | {st.median(errs):.3f} | {named}/{rep} |" if errs else f"| {label} | 0/{len(talks)} | n/a | 0/0 |")
        print(f"  drift {label}: reported {rep}/{len(talks)}", flush=True)
    L += ["", "(A factor of 1.001 is 60 ms per minute; on a 5-minute recording that is 0.3 s in total, so it is expected to be "
          "reported as WARN or not at all, depending on the warn limit.)", ""]

    # --- 4. planted jump
    L += ["## 4. Planted jump", "", "Labels in the second half moved 1.0 s early.", "",
          "| recording | jump found | place (s) | size (s) |", "|---|---|---|---|"]
    hit = 0
    for k, wav, truth in talks:
        mid = truth[len(truth) // 2][0]
        moved = [(s, e, tx) if s < mid else (s - 1.0, e - 1.0, tx) for s, e, tx in truth]
        r = run_one(wav, moved)
        j = r.jumps[0] if r.jumps else None
        hit += j is not None
        L.append(f"| {k} | {'yes' if j else 'no'} | {j['at_s']:.0f} (planted near {mid:.0f}) | {j['step_s']:+.2f} |" if j else f"| {k} | no | | |")
    L.append("")
    print(f"  jump found {hit}/{len(talks)}", flush=True)

    # --- 5. missing labels and wrong labels
    L += ["## 5. Missing labels, and labels from a different recording", ""]
    un_hit = 0
    for k, wav, truth in talks:
        r = run_one(wav, truth[::2])
        un_hit += find(r, "unlabelled-speech") is not None
    L.append(f"- every other label removed: 'speech outside every label' reported on {un_hit}/{len(talks)} recordings (expected: all)")
    conf, fails = 0, 0
    for i, (k, wav, truth) in enumerate(talks):
        other = talks[(i + 1) % len(talks)][2]
        d = max(e for _, e, _ in truth)
        other = [(s, e, tx) for s, e, tx in other if e <= d]
        r = run_one(wav, other)
        conf += (r.measurable)
        fails += r.verdict() == "FAIL"
    L.append(f"- labels of another recording: a confident offset was claimed on {conf}/{len(talks)} recordings (expected: 0); "
             f"verdict FAIL on {fails}/{len(talks)}")
    L.append("")

    # --- 6. short clips
    L += ["## 6. Single utterances (LibriSpeech as it comes)", ""]
    sample = [u for k in keys for u in books[k][:10]][:60]
    measurable, verdicts = 0, {"PASS": 0, "WARN": 0, "FAIL": 0}
    for p, text in sample:
        x, sr = sf.read(str(p), dtype="float64")
        dur = len(x) / sr
        r = check_file(p.name, p, [Segment(0.0, dur, text.lower())])
        measurable += r.measurable
        verdicts[r.verdict()] += 1
    L.append(f"- {len(sample)} utterances, each labelled as one segment covering the whole file: timing measured on {measurable}; "
             f"verdicts {verdicts}. (Whole-file labels leave little to line up; the useful checks here are the speaking-rate "
             f"and speech-outside-label ones.)")
    L += ["", f"_Run time: {time.time() - t0:.0f} s._", ""]
    (a.out / "results.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten to {a.out / 'results.md'}")


if __name__ == "__main__":
    main()
