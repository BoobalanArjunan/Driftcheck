import re

import numpy as np
import pytest
import soundfile as sf

from driftcheck import checks as C
from driftcheck.checks import FileResult, Finding, check_file
from driftcheck.labels import Segment, read_labels
from conftest import speech
from synth import SR, jittered, shifted


@pytest.fixture
def run(case):
    """run(**case_args, **check_args) -> FileResult for a synthetic talk and its SRT labels."""
    def go(warn_s=C.WARN_S, fail_s=C.FAIL_S, window_s=60.0, ref_rate=None, **kw):
        path = case(**kw)
        segs = read_labels(path.with_suffix(".srt"))[""]
        return check_file(path.name, path, segs, warn_s=warn_s, fail_s=fail_s, window_s=window_s, ref_rate=ref_rate)
    return go


def codes(res, level=None):
    return {f.code for f in res.findings if level is None or f.level == level}


def msg(res, code):
    return next(f.message for f in res.findings if f.code == code)


# ------------------------------------------------------------------ the good case

def test_correct_labels_pass_on_a_short_and_a_long_file(run):
    for dur, seed in ((120.0, 1), (300.0, 3)):
        segs, _ = speech(dur, seed)
        res = run(duration=dur, seed=seed, labels=jittered(segs, 0.05, seed=3))
        assert res.verdict() == "PASS", [f.message for f in res.findings]
        assert res.measurable and abs(res.offset_s) < 0.1
        assert res.unlabelled_speech_share < 0.1 and res.silent_label_share < 0.1
        assert "offset" in codes(res) and not codes(res, "WARN") | codes(res, "FAIL")


# ------------------------------------------------------------------ offsets

def test_late_labels_warn_and_the_message_gives_the_fix(run):
    segs, _ = speech(120.0, 1)
    res = run(labels=shifted(segs, -0.3))          # labels 0.3 s LATE: add -0.3 s
    assert res.verdict() == "WARN"
    assert re.search(r"-(2[89]\d|3[01]\d) ms", msg(res, "offset")), msg(res, "offset")
    assert "add this to the label times" in msg(res, "offset")


def test_a_large_offset_fails(run):
    segs, _ = speech(120.0, 1)
    res = run(labels=shifted(segs, -1.0))
    assert res.verdict() == "FAIL"
    assert [f.level for f in res.findings if f.code == "offset"] == ["FAIL"]
    assert re.search(r"-(99\d ms|1\.00 s)", msg(res, "offset")), msg(res, "offset")
    assert "at least 4 windows of 60 s" in msg(res, "offset")
    assert res.offset_s == pytest.approx(-1.0, abs=0.05)


def test_thresholds_are_adjustable(run):
    segs, _ = speech(120.0, 1)
    lab = jittered(shifted(segs, -0.1), 0.03, seed=2)
    assert run(labels=lab).verdict() == "PASS"
    assert run(labels=lab, warn_s=0.05).verdict() == "WARN"
    assert run(labels=lab, warn_s=0.05, fail_s=0.08).verdict() == "FAIL"


def test_offset_is_also_found_through_the_long_file_path(run):
    long_segs, _ = speech(300.0, 3)
    res = run(duration=300.0, seed=3, labels=shifted(long_segs, -0.3))
    assert res.offset_s == pytest.approx(-0.3, abs=0.05)
    assert res.windows and res.measurable


def test_unmeasurable_offset_is_info_not_a_verdict(run):
    # an 8 s clip: not enough evidence, so no offset claim and no WARN/FAIL for timing
    segs, _ = speech(8.0, 60)
    res = run(duration=8.0, seed=60, labels=shifted(segs, -0.4))
    if not res.measurable:
        assert "offset-unmeasurable" in codes(res, "INFO")
        assert not codes(res, "WARN") & {"offset"} and not codes(res, "FAIL") & {"offset"}


# ------------------------------------------------------------------ drift and jumps

def test_frame_rate_mixup_is_named(run):
    segs, _ = speech(240.0, 11)
    res = run(duration=240.0, seed=11, labels=shifted(segs, 0.0, 23.976 / 25))
    assert res.verdict() == "FAIL"
    m = msg(res, "drift")
    assert "25 vs 23.976 fps" in m and "ratio" in m
    assert res.suggested_scale == pytest.approx(25 / 23.976, abs=0.0006)
    assert res.drift_s_per_min == pytest.approx(2.5646, abs=0.1)


def test_other_drift_is_reported_without_a_frame_rate_claim(run):
    segs, _ = speech(300.0, 3)
    res = run(duration=300.0, seed=3, labels=jittered(shifted(segs, 0.0, 1.004), 0.04, seed=1))
    m = msg(res, "drift")
    assert "fps" not in m and "per minute" in m
    assert res.suggested_scale == pytest.approx(1 / 1.004, abs=0.0005)
    assert res.verdict() in ("WARN", "FAIL")


def test_tiny_drift_below_the_warning_level_is_not_reported_as_drift(run):
    segs, _ = speech(300.0, 3)
    res = run(duration=300.0, seed=3, labels=jittered(shifted(segs, 0.0, 1.0002), 0.04, seed=1))
    assert "drift" not in codes(res) and res.verdict() == "PASS"


def test_a_jump_is_reported_with_its_size_and_place(run):
    segs, _ = speech(600.0, 3)
    moved = [(a, b) if a < 300 else (a - 1.0, b - 1.0) for a, b in segs]
    res = run(duration=600.0, seed=3, labels=jittered(moved, 0.04, seed=6))
    assert res.verdict() == "FAIL"
    j = next(f for f in res.findings if f.code == "jump")
    assert j.level == "FAIL" and re.search(r"\+1\.0\d s", j.message), j.message
    assert j.at_s == pytest.approx(300.0, abs=60.0)
    assert res.jumps and "drift" not in codes(res)


# ------------------------------------------------------------------ coverage

def test_speech_nobody_labelled_is_reported_with_the_right_level(run):
    segs, _ = speech(120.0, 1)
    some = run(labels=[s for i, s in enumerate(segs) if i % 5])        # one utterance in five left out
    assert 0.1 < some.unlabelled_speech_share < 0.4
    assert [f.level for f in some.findings if f.code == "unlabelled-speech"] == ["WARN"]
    half = run(labels=segs[::2])                                       # every other utterance left out
    assert half.unlabelled_speech_share > 0.4
    assert [f.level for f in half.findings if f.code == "unlabelled-speech"] == ["FAIL"]
    assert "outside every label" in msg(half, "unlabelled-speech")


def test_labels_that_stop_a_little_early_are_forgiven(run):
    """Speech 0.6 s past the end of every label is mostly covered by the 0.15 s slack, and what is left over (under
    0.5 s per stretch) is too short to count. Nothing should be reported."""
    segs, _ = speech(120.0, 1)
    res = run(labels=[(a, max(b - 0.6, a + 0.3)) for a, b in segs])
    assert "unlabelled-speech" not in codes(res)
    assert res.unlabelled_speech_share < 0.02
    res2 = run(labels=[(a, max(b - 0.7, a + 0.3)) for a, b in segs])       # 0.7 s: now it counts
    assert res2.unlabelled_speech_share > 0.05


def test_labels_over_silence_are_reported(run):
    segs, _ = speech(120.0, 1)
    gaps = [(b + 0.2, c - 0.2) for (_, b), (c, _) in zip(segs, segs[1:]) if c - b > 2.0]
    assert len(gaps) >= 3
    res = run(labels=sorted(segs + gaps))
    assert "silent-labels" in codes(res, "WARN") | codes(res, "FAIL")
    assert 0.1 < res.silent_label_share < 0.9
    assert "almost no sound" in msg(res, "silent-labels")


def test_coverage_messages_blame_the_offset_when_there_is_one(run):
    segs, _ = speech(120.0, 1)
    res = run(labels=shifted(segs, -1.0))
    assert "offset" in codes(res, "FAIL")
    for c in ("unlabelled-speech", "silent-labels"):
        if c in codes(res):
            assert "result of the offset" in msg(res, c)


def test_short_labels_and_short_speech_are_not_judged(tmp_path):
    segs, x = speech(60.0, 2)
    p = tmp_path / "a.wav"
    sf.write(str(p), x, SR)
    tiny = [Segment(a, a + 0.5, "x") for a, _ in segs]        # all labels shorter than the 1 s silence rule
    res = check_file("a", p, tiny)
    assert res.silent_label_share is None


# ------------------------------------------------------------------ structure and rate (no audio needed)

def struct(segs, duration=None):
    F = []
    C._structure_checks(segs, duration, F)
    return F


def test_bad_length_out_of_order_overlap_and_empty_text():
    F = struct([Segment(5, 5, "a"), Segment(6, 4, "b")])
    assert [(f.level, f.code) for f in F if f.code == "bad-length"] == [("FAIL", "bad-length")]
    assert "2 label(s)" in F[0].message
    assert any(f.code == "out-of-order" for f in struct([Segment(5, 6, "a"), Segment(1, 2, "b")]))
    assert not any(f.code == "out-of-order" for f in struct([Segment(1, 2, "a"), Segment(1, 3, "b")]))
    ov = struct([Segment(0, 5, "a"), Segment(3, 8, "b")])
    assert any(f.code == "overlap" and f.level == "WARN" for f in ov)
    assert not any(f.code == "overlap" for f in struct([Segment(0, 5, "a"), Segment(4.7, 8, "b")]))     # 0.3 s is normal
    et = struct([Segment(0, 1, "a"), Segment(2, 3, "  ")])
    assert any(f.code == "empty-text" and "1 label" in f.message for f in et)
    assert not any(f.code == "empty-text" for f in struct([Segment(0, 1, ""), Segment(2, 3, "")]))     # no text at all is fine


def test_labels_beyond_the_audio():
    after = struct([Segment(1, 2, "a"), Segment(11, 12, "b")], duration=10.0)
    assert [(f.level, f.code) for f in after] == [("FAIL", "after-end")]
    past = struct([Segment(1, 2, "a"), Segment(9, 10.8, "b")], duration=10.0)
    assert [(f.level, f.code) for f in past] == [("WARN", "past-end")]
    assert struct([Segment(1, 2, "a"), Segment(9, 10.3, "b")], duration=10.0) == []        # 0.3 s over is tolerated


def segs_with_rate(n, cps, extra=()):
    s = [Segment(i * 5.0, i * 5.0 + 4.0, "x" * int(cps * 4)) for i in range(n)]
    return s + list(extra)


def test_text_too_long_for_its_time_is_flagged():
    F = []
    C._rate_checks(segs_with_rate(20, 14, [Segment(200, 201.2, "y" * 80)]), None, F)
    assert [f.code for f in F] == ["too-fast"] and "67 characters per second" in F[0].message


def test_text_too_short_for_its_time_is_flagged():
    F = []
    C._rate_checks(segs_with_rate(20, 14, [Segment(200, 206, "yyy")]), None, F)
    assert [f.code for f in F] == ["too-slow"]


def test_rate_checks_need_a_reference_and_ignore_normal_variation():
    F = []
    C._rate_checks(segs_with_rate(5, 14, [Segment(200, 202, "y" * 60)]), None, F)       # 30 cps: under the hard limit, no reference
    assert F == []
    C._rate_checks(segs_with_rate(5, 14, [Segment(200, 202, "y" * 60)]), 14.0, F)       # with a dataset reference: 2x is fine
    assert F == []
    C._rate_checks(segs_with_rate(5, 14, [Segment(200, 202, "y" * 100)]), 14.0, F)      # 50 cps: over the hard limit
    assert [f.code for f in F] == ["too-fast"]
    H = []
    C._rate_checks([Segment(0, 2, "y" * 40)] * 3, 4.0, H)    # 20 cps against a reference of 4 cps: 5x
    assert [f.code for f in H] == ["too-fast"]


def test_text_faster_than_anyone_speaks_is_flagged_even_when_the_dataset_is_fast_too():
    F = []
    C._rate_checks(segs_with_rate(5, 14, [Segment(200, 202, "y" * 80)]), 14.0, F)      # 40 cps: under 3x the reference, over the hard limit
    assert [f.code for f in F] == ["too-fast"]


def test_chars_per_second_ignores_short_and_empty_labels():
    r = C.chars_per_second([Segment(0, 2, "ab cd"), Segment(5, 5.5, "xxxxxxxx"), Segment(8, 10, "")])
    assert r == [2.0]


def test_check_without_audio_runs_only_the_structure_and_rate_checks():
    res = check_file("x", None, [Segment(0, 1, "a"), Segment(5, 4, "b")])
    assert res.verdict() == "FAIL" and codes(res) == {"bad-length"} and res.duration_s is None


# ------------------------------------------------------------------ audio that cannot be judged

def test_audio_without_speech_contrast_is_info_and_still_gets_structure_checks(tmp_path):
    p = tmp_path / "noise.wav"
    sf.write(str(p), (0.1 * np.random.RandomState(0).randn(SR * 30)).astype(np.float32), SR)
    res = check_file("noise", p, [Segment(1, 3, "a"), Segment(9, 5, "b")])
    assert "audio-unreliable" in codes(res, "INFO") and "bad-length" in codes(res, "FAIL")
    assert not res.measurable and res.offset_s is None


def test_all_labels_invalid_leaves_only_structure_findings(tmp_path):
    segs, x = speech(60.0, 2)
    p = tmp_path / "a.wav"
    sf.write(str(p), x, SR)
    res = check_file("a", p, [Segment(5, 5, "a")])
    assert codes(res) == {"bad-length"} and not res.measurable


def test_verdict_is_the_worst_finding():
    r = FileResult("x", 1.0, 1, [Finding("INFO", "a", ""), Finding("WARN", "b", "")])
    assert r.verdict() == "WARN"
    r.findings.append(Finding("FAIL", "c", ""))
    assert r.verdict() == "FAIL"
    assert FileResult("x", 1.0, 1, [Finding("INFO", "a", "")]).verdict() == "PASS"
