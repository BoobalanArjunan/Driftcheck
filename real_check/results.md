# driftcheck: real-speech check (LibriSpeech)

6 recordings built from LibriSpeech readers, about 40 utterances each. True label times are exact; the defects below are planted by us.

## 1. Undamaged labels (should PASS)

- verdicts: PASS=4, WARN=2, FAIL=0
- timing measured on 6 of 6 recordings
- measured offset (seconds; a systematic value here is the natural looseness of utterance-level labels): median 0.001, range -0.175 to 0.05
- unlabelled-speech share: median 0.000
- silent-label share: median 0.000

If any recording here is WARN or FAIL, read its findings below: that is a false alarm on correct labels.

- `talk_3170-137482.wav` WARN: WARN the offset suddenly changes by -460 ms near 60 s (a cut, an inserted section, or labels from two different versions) | WARN the offset suddenly changes by +430 ms near 120 s (a cut, an inserted section, or labels from two different versions)
- `talk_777-126732.wav` WARN: WARN labels are -175 ms from the audio (add this to the label times to line up)

## 2. Planted offsets

Error = measured minus (planted + the clean baseline above). Offsets are what driftcheck should report (seconds to add to the labels).

| planted offset (s) | found | median abs error (ms) | worst abs error (ms) |
|---|---|---|---|
| +0.2 | 6/6 | 10 | 170 |
| -0.2 | 6/6 | 55 | 180 |
| +0.5 | 6/6 | 21 | 207 |
| -1.0 | 6/6 | 39 | 1026 |
| +2.0 | 6/6 | 12 | 88 |

## 3. Planted steady drift

Labels stretched by a factor. 'Slope error' compares the measured drift (seconds per minute of label time) with the planted one.

| label time factor | drift reported | median slope error (s/min) | frame-rate name given |
|---|---|---|---|
| 1.001 | 5/6 | 0.011 | 3/5 |
| 0.999 | 3/6 | 0.019 | 2/3 |
| 1.004 | 6/6 | 0.014 | 0/6 |
| 23.976/25 | 1/6 | 0.178 | 0/1 |

(A factor of 1.001 is 60 ms per minute; on a 5-minute recording that is 0.3 s in total, so it is expected to be reported as WARN or not at all, depending on the warn limit.)

## 4. Planted jump

Labels in the second half moved 1.0 s early.

| recording | jump found | place (s) | size (s) |
|---|---|---|---|
| 1919-142785 | yes | 240 (planted near 233) | +1.21 |
| 2428-83705 | no | | |
| 3081-166546 | yes | 120 (planted near 132) | +1.00 |
| 3170-137482 | no | | |
| 777-126732 | no | | |
| 2086-149220 | no | | |

## 5. Missing labels, and labels from a different recording

- every other label removed: 'speech outside every label' reported on 6/6 recordings (expected: all)
- labels of another recording: a confident offset was claimed on 0/6 recordings (expected: 0); verdict FAIL on 1/6

## 6. Single utterances (LibriSpeech as it comes)

- 60 utterances, each labelled as one segment covering the whole file: timing measured on 0; verdicts {'PASS': 60, 'WARN': 0, 'FAIL': 0}. (Whole-file labels leave little to line up; the useful checks here are the speaking-rate and speech-outside-label ones.)

_Run time: 17 s._
