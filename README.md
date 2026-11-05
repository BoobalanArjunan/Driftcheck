# driftcheck

Check that timed labels (subtitles, transcripts with times) still line up with their audio.

Speech and ASR datasets pair audio with text that has start and end times. When those times go wrong (shifted, slowly stretching, or cut in the middle), a model trains on the wrong sounds for the words, and nothing raises an error. driftcheck puts the labels next to the audio and tells you where they do not fit, in plain numbers.

> **Status: alpha (0.1.0a1).** The measurements were tested on synthetic speech where the right answer is known, and the code was checked by deliberately breaking it (see "How it was checked" below). It has **not yet been run on real speech**; the steps and the results table are in [docs/REAL_DATA_CHECK.md](docs/REAL_DATA_CHECK.md). It is not a speech recogniser and not a forced aligner.

## Install

```
pip install --pre driftcheck
```

(`--pre` is needed while it is an alpha release.)

## Use

```
driftcheck check talks/                      # labels next to each audio file: talk1.wav + talk1.srt
driftcheck check talk.wav --labels talk.vtt  # one file
driftcheck check audio/ --labels supervisions.jsonl.gz   # a Lhotse supervisions manifest, matched by recording id
driftcheck check audio/ --labels transcripts.csv --out report/
```

Label formats: **SRT, WebVTT, Audacity label files** (`.tsv`/`.txt`), **CSV** (columns `start` and `end` or `duration`, optional `text` and a file-id column) and **Lhotse supervisions** (`.jsonl`, `.jsonl.gz`). Audio: wav, flac, ogg, opus, mp3 (whatever `soundfile` reads). The exit code is 1 when any file is a FAIL, so it works in a CI job.

## What it checks

| Check | What it tells you |
|---|---|
| **Offset** | The labels are early or late by a number of seconds (and which number fixes it). |
| **Drift** | The offset grows over the file, for example subtitles made for 25 fps media on 23.976 fps audio. It gives the stretch factor, and names the frame-rate mix-up when the factor matches one. |
| **Jump** | The offset suddenly changes at one place (a cut, an inserted ad, two versions glued together). |
| **Speech nobody labelled** | Share of the sound that is outside every label. |
| **Labels over silence** | Share of the labelled time that has almost no sound. |
| **Text too long or too short for its time** | Characters per second far from the dataset's usual (or impossible), which usually means the text belongs to a different stretch. |
| **Structure** | Labels that end before they start, are out of order, overlap a lot, run past the end of the audio, or have no text. |

Output (the layout is real, **the numbers are made up** to show the format):

```
driftcheck verdict: FAIL
1 files: 0 pass, 0 warn, 1 fail

episode12.wav: FAIL (3601.4 s, 612 labels)
  FAIL the labels drift against the audio: offset is -0.15 s at the start and +9.80 s at the end (+163 ms per minute).
       Multiplying the label times by about 1.00272 would remove the drift
  WARN 14% of the sound (310 s in 41 stretches) is outside every label; longest: 12.1 s at 1840.5 s; this may be a result of the offset above
```

`--out report/` writes `driftcheck_report.json`, `files.csv` (one row per file), `findings.csv` and `windows.csv` (the offset over time, per window).

## How it works (no speech model)

1. The audio is cut into 10 ms pieces and each is marked "sound" or "quiet" by its loudness in the speech band, compared with the quiet and loud parts of the same file.
2. The labels are turned into the same kind of on/off line: on where some label says there is speech.
3. One line is slid against the other. The position where they agree best is the offset. In windows of 60 seconds this gives the offset over time: a straight line is drift, a step is a jump.
4. To avoid answers that are just coincidence, every measurement is compared with what **made-up unrelated labels of the same kind** (same label lengths and gaps, shuffled) score on this same audio. An offset is only reported when the real labels beat that clearly.

## What was measured (synthetic speech, so far)

Speech-like audio (voiced bursts with pauses over a noise floor) with known label times, damaged in known ways.

| Situation | Result |
|---|---|
| Planted offset, 2 minutes, noise 10 to 30 dB below the speech, label edges off by up to 0.15 s | recovered to within 30 ms (80 ms with the sloppiest edges), with the right sign |
| Steady drift over 5 minutes (0.1% to 0.4%) | slope recovered within 0.015 s per minute |
| A 25 vs 23.976 fps mix-up (4% stretch, 10 s off after 4 minutes) | followed correctly past the 5 s search range, and named |
| A 1 s jump in the middle of 10 minutes | found, placed within a minute, sized within 0.1 s, and not mistaken for drift |
| Labels that belong to a different recording (30 trials, 1 minute) | a confident offset claimed in at most 3 of 30 (about 2% in a larger trial); otherwise "no better than unrelated labels" |
| Short clips | **8 s: never certain; 20 s: about a third certain; 60 s and up: all certain.** Short clips say "could not be measured" instead of guessing |
| Sloppy labels, no real problem (10 minutes, edges off by 0.15 s) | no jump, no drift reported |

## Limits (please read)

- **Not yet validated on real speech.** Synthetic speech is cleaner than real speech. Real results will be added here after the run described in [docs/REAL_DATA_CHECK.md](docs/REAL_DATA_CHECK.md).
- **It sees loud and quiet, not words.** It cannot tell that the text says something different from the audio, and it cannot tell one speaker from another. It finds timing problems and gross mismatches only.
- **Needs speech that stands out from the background.** On music, constant loud noise or a file that is sound from start to end it says "timing was not checked" and gives the reason. Background music under speech will confuse it.
- **Needs enough audio.** An offset needs about a minute of speech with pauses to be certain; drift needs at least 4 windows (4 minutes at the default 60 s). Datasets of single short utterances get the speaking-rate, structure and speech-outside-label checks, and an offset only when the evidence is strong.
- **The first offset must be within 5 seconds** (`--max-offset`); after that the search follows the drift.
- **Offset is measured to about 10 ms steps**, and real label edges are rarely that exact. Treat anything under about 50 ms as noise.
- **The limits are judgement calls**: WARN at 0.15 s, FAIL at 0.5 s (`--warn`, `--fail`); speech outside labels or labels over silence WARN at 10% and FAIL at 40%; the speaking-rate check compares with the dataset's own median and is rough for languages where one character is a whole syllable or word.
- **No fixing yet.** It tells you the number to add or the factor to multiply by; it does not rewrite your label files.
- **The speech-outside-labels numbers are measured on the labels as given**, so a big offset also shows up there (the report says so).

## How it was checked

- 110 tests with known answers (planted offsets, drift, jumps, unrelated labels, short clips, repeating patterns, every label format, pairing, CLI exit codes).
- A mutation run: 69 deliberate bugs were put into the code one at a time (wrong sign, a threshold removed, a null test switched off, ...) and the tests were run each time. The first run caught 53; the 16 it missed showed weak tests, which were strengthened, and all 69 are now caught.

## Related work

- Forced aligners such as the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) work out word timings from the text with an acoustic model and a pronunciation dictionary. driftcheck is lighter and answers a different question: given labels you already have, do they still fit the audio? It needs no model, dictionary or language.
- [Lhotse](https://github.com/lhotse-speech/lhotse) and the NeMo toolkit have their own dataset-preparation and validation tools; driftcheck reads Lhotse supervisions so it can sit next to them.
- [stemcheck](https://github.com/BoobalanArjunan/stemcheck), [mixforge](https://github.com/BoobalanArjunan/mixforge) and [splitsafe](https://github.com/BoobalanArjunan/splitsafe): sibling tools for music stems, test mixes and train/test leakage.

## Development

```
pip install -e ".[dev]"
python -m pytest -q
```

## License

MIT. Copyright (c) 2026 Boobalan Arjunan.
