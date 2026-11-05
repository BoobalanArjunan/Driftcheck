# Real-data check (LibriSpeech)

This tests driftcheck on real speech. It answers: does driftcheck stay quiet on correct labels, and does it find and measure the damage we plant (shifted labels, stretched labels, a jump, missing labels, labels from another recording)?

## What you need

1. **LibriSpeech**: the `dev-clean` set from openslr.org/12 (about 340 MB, free). Unpack it. You should see folders like `LibriSpeech/dev-clean/84/121123/` with `.flac` files and a `*.trans.txt` file. driftcheck does not ship any speech.
2. `pip install --pre driftcheck` (or work from a project checkout to get the `scripts/` folder).

## Part A: planted defects

LibriSpeech is one short sentence per file. The script joins about 40 sentences from one reader, with pauses, into a recording of about 5 minutes. It knows the exact time of every sentence, so the true labels are exact. Then it damages the labels in known ways.

```
python scripts/real_data_check.py --librispeech ~/LibriSpeech/dev-clean --talks 6 --out real_check
```

It takes a minute or two. Open `real_check/results.md`. What the sections mean:

1. **Undamaged labels.** Should all PASS. A WARN or FAIL here is a false alarm on correct labels: send me the findings it lists. (LibriSpeech sentence clips have a little quiet time at both ends, so a measured offset of a few tens of milliseconds is the natural looseness of the labels, not an error.)
2. **Planted offsets.** Labels moved by 0.2, -0.2, 0.5, -1.0 and 2.0 s. "found" should be all recordings; the error should be small (tens of milliseconds).
3. **Planted drift.** Labels stretched by 1.001, 0.999, 1.004 and 23.976/25. The two small ones are small on purpose: a 0.1% stretch is only 0.3 s over 5 minutes, so reporting nothing is acceptable. The last two should be reported; the last one should be named as the 25 vs 23.976 fps mix-up.
4. **Planted jump.** Second half moved by 1 s. Should be found near the middle with a size close to 1 s.
5. **Missing labels and wrong labels.** Removing every other label should be reported on all recordings. Labels from a different recording should NOT give a confident offset.
6. **Single utterances.** LibriSpeech as it comes. Expect timing to be "not measurable" on most; that is the honest answer for a single short clip.

## Part B: your own data (optional, the most useful part)

If you have any real dataset that has timed labels (subtitles for a talk or a podcast, a meeting transcript with times), run:

```
driftcheck check your_audio_folder/ --out your_report/
```

Open `your_report/driftcheck_report.txt`. Pick one file that is WARN or FAIL and check it by hand: open the audio and the labels in Audacity (File > Import > Labels) and look at whether the first and last labels sit on the speech they say they do. That is the real test of whether driftcheck points at real problems. Tell me: right, wrong, or could not tell.

Your original files are never modified. Everything is written under `real_check/` and the folders you name.
