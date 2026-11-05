import numpy as np
import pytest
import soundfile as sf

from driftcheck.activity import (DriftcheckError, clean_runs, detect_activity, envelope_from_array,
                                 envelope_from_file)
from conftest import speech
from synth import SR


def test_frame_length_is_the_real_one_for_awkward_sample_rates():
    # 22050 Hz: a 10 ms frame is 220 samples = 9.977 ms. Using 10 ms would drift 140 ms per minute.
    env = envelope_from_array(np.zeros(22050 * 60), 22050)
    assert env.hop_s == pytest.approx(220 / 22050)
    assert len(env.db) * env.hop_s == pytest.approx(60.0, abs=0.01)


def test_detects_speech_stretches_where_they_really_are():
    segs, x = speech(120.0, 1)
    act = detect_activity(envelope_from_array(x, SR))
    assert act.reliable
    t = (np.arange(act.n) + 0.5) * act.hop_s
    truth = np.zeros(act.n, bool)
    for a, b in segs:
        truth[(t >= a) & (t < b)] = True
    agree = (truth == act.active).mean()
    assert agree > 0.93, agree


def test_file_reading_matches_array_reading_across_block_boundaries(tmp_path):
    _, x = speech(30.0, 2)
    p = tmp_path / "a.wav"
    sf.write(str(p), x, SR)
    a = envelope_from_array(x, SR)
    b = envelope_from_file(p, block_s=4.0)        # several blocks, not a multiple of the file length
    assert len(a.db) == len(b.db)
    assert np.abs(a.db - b.db).max() < 0.2        # float32 file versus float64 array
    assert b.duration_s == pytest.approx(30.0)


def test_stereo_file_is_averaged(tmp_path):
    _, x = speech(20.0, 3)
    p = tmp_path / "s.wav"
    sf.write(str(p), np.stack([x, np.zeros_like(x)], axis=1), SR)        # sound in the left channel only
    assert np.abs(envelope_from_file(p).db - envelope_from_array(0.5 * x, SR).db).max() < 0.2


def test_unreadable_file_is_a_clear_error(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"not audio")
    with pytest.raises(DriftcheckError, match="cannot read bad.wav"):
        envelope_from_file(p)


@pytest.mark.parametrize("kind", ["silence", "noise", "tone"])
def test_says_unreliable_when_there_is_no_speech_contrast(kind):
    n = SR * 30
    r = np.random.RandomState(0)
    x = {"silence": np.zeros(n), "noise": 0.1 * r.randn(n),
         "tone": 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)}[kind]
    act = detect_activity(envelope_from_array(x, SR))
    assert not act.reliable and act.reason
    assert not act.active.any()


def test_very_short_audio_is_unreliable_not_a_crash():
    act = detect_activity(envelope_from_array(np.zeros(100), SR))
    assert not act.reliable and "short" in act.reason


def test_clean_runs_fills_short_gaps_and_drops_blips():
    hop = 0.01
    a = np.zeros(300, bool)
    a[10:50] = True
    a[60:100] = True          # gap of 10 frames = 0.1 s between: filled
    a[150:153] = True         # blip of 0.03 s: dropped
    a[200:240] = True
    a[270:290] = True         # gap of 30 frames = 0.3 s: kept
    out = clean_runs(a, hop)
    assert out[10:100].all()
    assert not out[150:153].any()
    assert not out[240:270].any() and out[200:240].all() and out[270:290].all()


def test_clean_runs_does_not_fill_edges():
    a = np.zeros(100, bool)
    a[20:60] = True
    out = clean_runs(a, 0.01)
    assert not out[:20].any() and not out[60:].any()


def test_file_envelope_has_the_real_frame_length_too(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(str(p), np.zeros(22050 * 5, dtype=np.float32), 22050)
    env = envelope_from_file(p)
    assert env.hop_s == pytest.approx(220 / 22050)
    assert len(env.db) * env.hop_s == pytest.approx(5.0, abs=0.01)


def test_file_reading_copes_with_a_reader_that_returns_odd_sized_blocks(tmp_path, monkeypatch):
    """Some decoders return fewer samples than asked. Leftover samples must carry into the next block."""
    import driftcheck.activity as act_mod
    _, x = speech(20.0, 4)
    p = tmp_path / "a.wav"
    sf.write(str(p), x, SR)
    real = sf.SoundFile

    class Choppy:
        def __init__(self, path):
            self._f = real(path)
            self.samplerate, self.frames = self._f.samplerate, self._f.frames

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self._f.close()

        def read(self, n, **kw):
            return self._f.read(min(n, 1234), **kw)          # never more than 1234 samples at a time

    monkeypatch.setattr(act_mod.sf, "SoundFile", Choppy)
    b = envelope_from_file(p)
    a = envelope_from_array(x, SR)
    assert len(a.db) == len(b.db) and np.abs(a.db - b.db).max() < 0.2
