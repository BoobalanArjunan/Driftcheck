import functools
import sys
from pathlib import Path

import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from synth import SR, make_segments, synth_speech, write_srt  # noqa: E402


@functools.lru_cache(maxsize=None)
def speech(duration: float, seed: int, snr_db: float = 20.0):
    """(true segments, audio) for a synthetic talk. Cached: building audio is the slow part of the tests."""
    segs = make_segments(duration, seed=seed)
    return segs, synth_speech(segs, duration, snr_db=snr_db, seed=seed)


@pytest.fixture
def case(tmp_path):
    """case(name, duration, seed, labels=None, snr_db=20, cps=14) -> audio path; writes name.wav and name.srt.

    `labels` is a list of (start, end) label times; default is the true speech times."""
    def make(name="talk", duration=120.0, seed=1, labels=None, snr_db=20.0, cps=14.0, texts=None, folder=None):
        d = Path(folder) if folder else tmp_path
        d.mkdir(parents=True, exist_ok=True)
        segs, x = speech(duration, seed, snr_db)
        sf.write(str(d / f"{name}.wav"), x, SR)
        write_srt(d / f"{name}.srt", segs if labels is None else labels, cps=cps, texts=texts)
        return d / f"{name}.wav"
    return make
