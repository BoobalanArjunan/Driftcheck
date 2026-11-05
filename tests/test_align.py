import numpy as np
import pytest

from driftcheck import align as A
from driftcheck.activity import detect_activity, envelope_from_array
from driftcheck.labels import Segment
from conftest import speech
from synth import SR, jittered, make_segments, shifted


def S(pairs):
    return [Segment(a, b) for a, b in pairs]


def act_for(duration, seed, snr=20.0):
    segs, x = speech(duration, seed, snr)
    return segs, detect_activity(envelope_from_array(x, SR))


# ------------------------------------------------------------------ building blocks

def test_label_indicator_marks_frames_whose_centre_is_inside():
    lab = A.label_indicator(S([(0.10, 0.30)]), 100, 0.01)
    assert np.flatnonzero(lab).tolist() == list(range(10, 30))
    assert not A.label_indicator(S([(0.5, 0.5), (0.7, 0.6)]), 100, 0.01).any()      # empty and reversed labels do nothing
    assert A.label_indicator(S([(0.9, 5.0)]), 100, 0.01).sum() == 10                # clipped at the end of the audio
    assert A.label_indicator(S([(-1.0, 0.05)]), 100, 0.01).sum() == 5               # and at the start


def test_label_indicator_uses_frame_centres_not_frame_starts():
    lab = A.label_indicator(S([(0.104, 0.296)]), 100, 0.01)
    # frame i has its centre at (i + 0.5) * 0.01: centres inside 0.104..0.296 are frames 10 (0.105) to 29 (0.295)
    assert np.flatnonzero(lab).tolist() == list(range(10, 30))


def test_lag_scores_known_answer_and_sign():
    n = 200
    y = -np.ones(n)
    y[100:140] = 1.0                     # sound at frames 100..139
    x = -np.ones(n)
    x[80:120] = 1.0                      # labels say 80..119: EARLY by 20 frames
    sc = A.lag_scores(x, y, 0, 50)
    assert int(np.argmax(sc)) - 50 == 20      # positive lag = labels early = add time to labels
    assert sc.max() == pytest.approx(0.9)      # 180 of the 200 label frames have audio to compare with at lag 20


def test_lag_scores_window_and_centre_agree_with_full():
    rng = np.random.RandomState(0)
    y = np.where(rng.rand(500) > 0.5, 1.0, -1.0)
    x = np.roll(y, -7)
    full = A.lag_scores(x, y, 0, 40)
    win = A.lag_scores(x[100:300], y, 100, 40)
    assert int(np.argmax(full)) - 40 == 7 and int(np.argmax(win)) - 40 == 7
    shifted_search = A.lag_scores(x[100:300], y, 100, 10, center=7)
    assert int(np.argmax(shifted_search)) - 10 + 7 == 7


def test_unknown_audio_outside_the_file_counts_as_zero_not_as_silence():
    y = np.ones(50)
    x = np.ones(50)
    sc = A.lag_scores(x, y, 0, 10)
    assert sc.max() == pytest.approx(1.0)
    assert sc[0] == pytest.approx(40 / 50)        # lag -10: only 40 of 50 frames can be compared


def test_pick_peak_refuses_when_there_is_nothing_to_line_up():
    sc = np.zeros(21)
    sc[10] = 0.9
    assert A.pick_peak(sc, 10, 0.01, 0, 5).reason.startswith("the labels cover all")
    assert A.pick_peak(sc, 10, 0.01, 5, 0).reason.startswith("the audio has no change")
    weak = np.zeros(21)
    weak[10] = 0.1
    p = A.pick_peak(weak, 10, 0.01, 5, 5)
    assert not p.confident and "unrelated" in p.reason


def test_pick_peak_needs_to_beat_the_unrelated_label_level():
    sc = np.zeros(21)
    sc[10] = 0.8
    assert A.pick_peak(sc, 10, 0.01, 5, 5, null_max=0.5).confident
    assert not A.pick_peak(sc, 10, 0.01, 5, 5, null_max=0.75).confident     # within the 0.1 margin
    assert A.pick_peak(sc, 10, 0.01, 5, 5, null_max=None).confident


def test_pick_peak_flags_repetitive_patterns_as_ambiguous():
    sc = np.zeros(1001)
    sc[500] = 0.9
    sc[800] = 0.88                        # a second, almost equal peak 3 s away
    p = A.pick_peak(sc, 500, 0.01, 5, 5)
    assert not p.confident and "more than one" in p.reason
    sc[800] = 0.5
    assert A.pick_peak(sc, 500, 0.01, 5, 5).confident


def test_pick_peak_interpolates_between_frames():
    sc = np.zeros(21)
    sc[10], sc[11] = 0.9, 0.9             # peak exactly half a frame to the right
    assert A.pick_peak(sc, 10, 0.01, 5, 5).offset_s == pytest.approx(0.005)


def test_null_layouts_move_a_single_label_to_random_places():
    lab = np.zeros(2000, bool)
    lab[100:300] = True
    nulls = A.null_layouts(lab, 30, np.random.RandomState(1))
    starts = {int(np.flatnonzero(z)[0]) for z in nulls if z[0] == 0}
    assert len(starts) > 10 and all(int(z.sum()) == 200 for z in nulls)


def test_null_layouts_keep_the_ingredients_but_move_them():
    lab = np.zeros(2000, bool)
    lab[100:300] = lab[500:560] = lab[900:1300] = True
    nulls = A.null_layouts(lab, 20, np.random.RandomState(0))
    assert len(nulls) == 20
    runs = lambda z: sorted(e - s for s, e in A._runs(z))
    for z in nulls:
        assert z.sum() == lab.sum()
        # rolling can split one stretch in two at the wrap point, so compare totals, not stretch lengths
    assert any((z != lab).any() for z in nulls)
    assert A.null_layouts(np.zeros(100, bool), 5, np.random.RandomState(0)) == []


# ------------------------------------------------------------------ whole-file offset

@pytest.mark.parametrize("off", [0.0, 0.1, -0.35, 1.2, -2.0])
def test_offset_is_recovered_with_the_right_sign(off):
    segs, act = act_for(120.0, 1)
    p = A.estimate_offset(act, S(jittered(shifted(segs, off), 0.04, seed=2)))
    assert p.confident, p.reason
    assert p.offset_s == pytest.approx(off, abs=0.03)


def test_offset_survives_sloppy_boundaries_and_noise():
    errs = []
    for k in range(6):
        segs, act = act_for(120.0, 20 + k, snr=10.0)
        p = A.estimate_offset(act, S(jittered(shifted(segs, 0.25), 0.15, seed=k)))
        assert p.confident, (k, p.reason)
        errs.append(abs(p.offset_s - 0.25))
    assert max(errs) < 0.08, errs


def test_unrelated_labels_are_rarely_called_an_offset():
    """Labels for a different recording must not produce a confident offset (about 2% in our trials)."""
    trials, confident = 30, 0
    for k in range(trials):
        _, act = act_for(60.0, 40 + k)
        other = make_segments(60.0, seed=900 + k)
        confident += A.estimate_offset(act, S(other)).confident
    assert confident <= 3, confident


def test_short_clips_say_they_cannot_tell_instead_of_guessing():
    ok = 0
    for k in range(10):
        segs, act = act_for(8.0, 60 + k)
        p = A.estimate_offset(act, S(shifted(segs, 0.2)))
        ok += p.confident
        if p.confident:
            assert p.offset_s == pytest.approx(0.2, abs=0.1)     # when it does answer, it is right
    assert ok <= 5       # honest: an 8 s clip rarely carries enough evidence


def test_unreliable_audio_gives_no_offset():
    x = 0.1 * np.random.RandomState(0).randn(SR * 60)
    act = detect_activity(envelope_from_array(x, SR))
    p = A.estimate_offset(act, S([(1.0, 3.0), (5.0, 9.0)]))
    assert p.offset_s is None and not p.confident
    assert "no clear difference between loud and quiet" in p.reason


def test_repeating_pattern_is_ambiguous():
    from synth import synth_speech
    segs = [(2 + 4 * i, 4 + 4 * i) for i in range(14)]
    act = detect_activity(envelope_from_array(synth_speech(segs, 60.0, seed=9), SR))
    p = A.estimate_offset(act, S(segs))
    assert not p.confident            # refused: either "more than one position" or "no better than unrelated labels"


def test_estimate_is_repeatable():
    segs, act = act_for(60.0, 5)
    a = A.estimate_offset(act, S(shifted(segs, 0.2)))
    b = A.estimate_offset(act, S(shifted(segs, 0.2)))
    assert (a.offset_s, a.score, a.confident) == (b.offset_s, b.score, b.confident)


# ------------------------------------------------------------------ drift and jumps

def test_no_drift_when_labels_are_right():
    segs, act = act_for(300.0, 3)
    d = A.estimate_drift(act, S(jittered(segs, 0.04, seed=1)))
    assert d.n_good >= 4
    assert abs(d.slope_s_per_min) < 0.01 and abs(d.offset_end_s - d.offset_start_s) < 0.06
    assert d.jumps == []


@pytest.mark.parametrize("scale,off", [(1.001, 0.0), (0.999, 0.2), (1.004, -0.3)])
def test_steady_drift_is_measured(scale, off):
    D = 300.0
    segs, act = act_for(D, 3)
    d = A.estimate_drift(act, S(jittered(shifted(segs, off, scale), 0.04, seed=5)))
    # labels = scale * audio_time - off, so offset(label time t) = (t * (1 - scale) + off) / scale
    t_end = scale * max(b for _, b in segs) - off
    true_slope = (1 - scale) / scale * 60          # seconds per minute of label time
    assert d.slope_s_per_min == pytest.approx(true_slope, abs=0.015)
    assert d.offset_start_s == pytest.approx(off / scale, abs=0.06)
    assert d.offset_end_s == pytest.approx((t_end * (1 - scale) + off) / scale, abs=0.08)
    assert d.jumps == []


def test_large_drift_is_followed_beyond_the_search_range():
    """A 25 vs 23.976 fps mix-up gives 4% drift: 10 s out after four minutes, far beyond the 5 s search range."""
    D = 240.0
    segs, act = act_for(D, 11)
    d = A.estimate_drift(act, S(shifted(segs, 0.0, 23.976 / 25)))
    scale = 23.976 / 25
    assert d.n_good >= 4
    t_end = scale * max(b for _, b in segs)
    assert d.offset_end_s == pytest.approx(t_end * (1 - scale) / scale, abs=0.3)
    assert d.slope_s_per_min == pytest.approx(60 * (1 - scale) / scale, abs=0.05)


def test_a_sudden_jump_is_found_and_not_mistaken_for_drift():
    segs, act = act_for(600.0, 3)
    moved = [(a, b) if a < 300 else (a - 1.0, b - 1.0) for a, b in segs]     # labels 1 s early after 300 s
    d = A.estimate_drift(act, S(jittered(moved, 0.04, seed=6)))
    assert len(d.jumps) == 1
    assert d.jumps[0]["at_s"] == pytest.approx(300.0, abs=60.0)
    assert d.jumps[0]["step_s"] == pytest.approx(1.0, abs=0.1)
    assert abs(d.slope_s_per_min) < 0.01          # the jump is taken out before the drift line is fitted


def test_drift_needs_enough_good_windows():
    segs, act = act_for(150.0, 4)           # only 2 full windows
    d = A.estimate_drift(act, S(segs))
    assert d.n_good < A.MIN_WINDOWS and d.slope_s_per_min is None


def test_drift_on_unreliable_audio_is_empty():
    x = 0.1 * np.random.RandomState(0).randn(SR * 200)
    act = detect_activity(envelope_from_array(x, SR))
    d = A.estimate_drift(act, S([(1.0, 3.0)]))
    assert d.windows == [] and d.n_good == 0


def test_sloppy_labels_do_not_create_jumps_or_drift():
    segs, act = act_for(600.0, 3)
    d = A.estimate_drift(act, S(jittered(segs, 0.15, seed=11)))
    assert d.n_good >= 4 and d.jumps == []
    assert abs(d.slope_s_per_min) < 0.02


def test_a_small_step_is_not_a_jump():
    segs, act = act_for(600.0, 3)
    moved = [(a, b) if a < 300 else (a - 0.12, b - 0.12) for a, b in segs]       # 0.12 s: real, but under the 0.3 s jump size
    d = A.estimate_drift(act, S(jittered(moved, 0.04, seed=6)))
    assert d.jumps == []


def test_following_drift_needs_the_extrapolation_when_each_window_moves_more_than_the_search_range():
    """0.6 s of drift per window but a 0.4 s search range: only a prediction from the last windows can keep up."""
    segs, act = act_for(300.0, 3)
    d = A.estimate_drift(act, S(shifted(segs, 0.0, 0.99)), max_offset_s=0.4)
    assert d.n_good >= 4
    assert d.slope_s_per_min == pytest.approx((1 / 0.99 - 1) * 60, abs=0.05)


def test_labels_of_another_recording_give_no_good_windows_on_a_long_file():
    for seed in (3, 4, 5):
        _, act = act_for(300.0, seed)
        d = A.estimate_drift(act, S(make_segments(300.0, seed=800 + seed)))
        assert d.n_good == 0 and d.slope_s_per_min is None and d.jumps == []


def test_drift_at_the_end_means_the_end_of_the_labels_not_of_the_audio():
    segs, act = act_for(600.0, 3)
    early = [(a, b) for a, b in segs if b < 300]            # labels stop halfway through the audio
    scale = 1.004
    d = A.estimate_drift(act, S(jittered(shifted(early, 0.0, scale), 0.04, seed=5)))
    assert d.n_good >= 4
    t_end = scale * max(b for _, b in early)
    assert d.offset_end_s == pytest.approx(t_end * (1 - scale) / scale, abs=0.1)
