import csv
import gzip
import json

import pytest
import soundfile as sf

from driftcheck import cli, scan
from driftcheck.activity import DriftcheckError
from conftest import speech
from synth import SR, shifted


# ------------------------------------------------------------------ pairing audio with labels

def test_siblings_are_paired_by_name_including_nested_folders(case, tmp_path):
    case("a", duration=30.0, seed=1)
    case("b", duration=30.0, seed=2, folder=tmp_path / "sub" / "deep")
    items, notes = scan.collect(tmp_path)
    assert [(i.name, i.segments is not None) for i in items] == [("a.wav", True), ("sub/deep/b.wav", True)]
    assert notes == []


def test_a_single_audio_file_and_a_labels_file(case, tmp_path):
    p = case("a", duration=30.0, seed=1)
    items, _ = scan.collect(p, tmp_path / "a.srt")
    assert len(items) == 1 and items[0].segments and items[0].name == "a.wav"


def test_labels_folder_is_matched_by_file_name(case, tmp_path):
    audio = tmp_path / "audio"
    case("a", duration=30.0, seed=1, folder=audio)
    (tmp_path / "labels").mkdir()
    (audio / "a.srt").rename(tmp_path / "labels" / "a.srt")
    items, _ = scan.collect(audio, tmp_path / "labels")
    assert items[0].segments is not None
    items2, _ = scan.collect(audio)                       # not next to the audio any more
    assert items2[0].segments is None and "no labels found" in items2[0].error


def test_csv_manifest_with_recording_ids_and_a_leftover_id(case, tmp_path):
    for n, sd in (("a", 1), ("b", 2)):
        case(n, duration=30.0, seed=sd)
        (tmp_path / f"{n}.srt").unlink()
    rows = [("a", 1.0, 3.0), ("b", 2.0, 4.0), ("ghost", 0.0, 1.0)]
    with open(tmp_path / "m.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "start", "end", "text"])
        w.writerows([(r, a, b, "x") for r, a, b in rows])
    items, notes = scan.collect(tmp_path, tmp_path / "m.csv")
    assert [i.segments is not None for i in items] == [True, True]
    assert any("ghost" in n for n in notes)


def test_lhotse_manifest_is_matched_by_recording_id(case, tmp_path):
    case("rec1", duration=30.0, seed=1)
    (tmp_path / "rec1.srt").unlink()
    body = "\n".join(json.dumps({"recording_id": "rec1", "start": s, "duration": d, "text": "t"})
                     for s, d in ((1.0, 2.0), (5.0, 3.0))) + "\n"
    (tmp_path / "sup.jsonl.gz").write_bytes(gzip.compress(body.encode()))
    items, _ = scan.collect(tmp_path, tmp_path / "sup.jsonl.gz")
    assert [(s.start, s.end) for s in items[0].segments] == [(1.0, 3.0), (5.0, 8.0)]


def test_a_one_recording_label_file_cannot_serve_several_audio_files(case, tmp_path):
    case("a", duration=30.0, seed=1)
    case("b", duration=30.0, seed=2)
    items, _ = scan.collect(tmp_path, tmp_path / "a.srt")
    assert all(i.segments is None and "describes one recording" in i.error for i in items)


def test_pairing_errors(tmp_path):
    with pytest.raises(DriftcheckError, match="not a file or a folder"):
        scan.collect(tmp_path / "missing")
    with pytest.raises(DriftcheckError, match="no audio files"):
        scan.collect(tmp_path)
    sf.write(str(tmp_path / "a.wav"), speech(30.0, 1)[1], SR)
    with pytest.raises(DriftcheckError, match="not a file or a folder"):
        scan.collect(tmp_path, tmp_path / "nolabels")


def test_unreadable_label_file_becomes_a_fail_not_a_crash(case, tmp_path):
    case("a", duration=30.0, seed=1)
    case("b", duration=30.0, seed=2)
    (tmp_path / "b.srt").write_text("this is not a subtitle file\n")
    rep = scan.run(tmp_path)
    v = {f.name: f for f in rep.files}
    assert v["b.wav"].verdict() == "FAIL" and "no 'start --> end'" in v["b.wav"].findings[0].message
    assert v["a.wav"].verdict() in ("PASS", "WARN")
    assert rep.verdict() == "FAIL"


def test_audio_without_labels_warns_and_is_counted(case, tmp_path):
    case("a", duration=30.0, seed=1)
    sf.write(str(tmp_path / "lonely.wav"), speech(30.0, 2)[1], SR)
    rep = scan.run(tmp_path)
    lone = next(f for f in rep.files if f.name == "lonely.wav")
    assert lone.verdict() == "WARN" and lone.findings[0].code == "no-labels"
    assert any("1 audio file(s) had no labels" in n for n in rep.notes)


def test_unreadable_audio_is_a_fail(case, tmp_path):
    case("a", duration=30.0, seed=1)
    (tmp_path / "a.wav").write_bytes(b"junk")
    rep = scan.run(tmp_path)
    assert rep.files[0].verdict() == "FAIL" and rep.files[0].findings[0].code == "unreadable"


def test_max_files_limits_and_says_so(case, tmp_path):
    for i, n in enumerate("abc"):
        case(n, duration=30.0, seed=i + 1)
    rep = scan.run(tmp_path, max_files=2)
    assert len(rep.files) == 2 and any("first 2 of 3" in n for n in rep.notes)


def test_run_validates_its_options(case, tmp_path):
    p = case("a", duration=30.0, seed=1)
    with pytest.raises(DriftcheckError, match="--warn"):
        scan.run(p, warn_s=0.5, fail_s=0.1)
    with pytest.raises(DriftcheckError, match="--warn"):
        scan.run(p, warn_s=0)
    with pytest.raises(DriftcheckError, match="--window"):
        scan.run(p, window_s=5)


def test_dataset_wide_speaking_rate_is_used_as_the_reference(case, tmp_path):
    # 24 files at ~6 characters per second; one file at 24 (4x faster, but under the hard limit of 35): only
    # the dataset's own usual rate can show it is wrong, because every label in that file is equally fast
    from synth import text_for
    for i in range(24):
        case(f"f{i:02d}", duration=20.0, seed=1 + i % 3, cps=6.0)
    zz = speech(20.0, 1)[0]
    case("zz", duration=20.0, seed=1, texts=[text_for(a, b, 24.0) for a, b in zz])
    rep = scan.run(tmp_path)
    zz = next(f for f in rep.files if f.name == "zz.wav")
    assert any(x.code == "too-fast" for x in zz.findings)
    assert not any(x.code in ("too-fast", "too-slow") for f in rep.files if f.name != "zz.wav" for x in f.findings)


# ------------------------------------------------------------------ command line

def test_cli_pass_returns_0_and_prints_the_verdict(case, capsys):
    p = case("a", duration=30.0, seed=1)
    assert cli.main(["check", str(p)]) == 0
    assert "driftcheck verdict: PASS" in capsys.readouterr().out


def test_cli_fail_returns_1(case, capsys):
    segs, _ = speech(120.0, 1)
    p = case("a", duration=120.0, seed=1, labels=shifted(segs, -1.0))
    assert cli.main(["check", str(p)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "add this to the label times" in out


def test_cli_thresholds_change_the_verdict(case):
    segs, _ = speech(120.0, 1)
    p = case("a", duration=120.0, seed=1, labels=shifted(segs, -1.0))
    assert cli.main(["check", str(p), "--warn", "1.5", "--fail", "3"]) == 0


def test_cli_usage_problems_return_2(case, capsys, tmp_path):
    assert cli.main([]) == 2
    assert cli.main(["check", str(tmp_path / "nothing")]) == 2
    assert "driftcheck:" in capsys.readouterr().err
    p = case("a", duration=30.0, seed=1)
    assert cli.main(["check", str(p), "--warn", "2", "--fail", "1"]) == 2


def test_cli_json_is_valid_and_complete(case, capsys):
    segs, _ = speech(120.0, 1)
    p = case("a", duration=120.0, seed=1, labels=shifted(segs, -1.0))
    cli.main(["check", str(p), "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["verdict"] == "FAIL" and d["settings"]["warn_s"] == 0.15
    f = d["files"][0]
    assert f["offset_s"] == pytest.approx(-1.0, abs=0.05) and f["findings"] and f["verdict"] == "FAIL"
    assert "ADD to the label times" in d["note"]


def test_cli_writes_the_report_files(case, tmp_path):
    segs, _ = speech(300.0, 3)
    p = case("a", duration=300.0, seed=3, labels=shifted(segs, 0.0, 1.004))
    out = tmp_path / "rep"
    cli.main(["check", str(p), "--out", str(out)])
    for n in ("driftcheck_report.json", "driftcheck_report.txt", "files.csv", "findings.csv", "windows.csv"):
        assert (out / n).is_file(), n
    rows = list(csv.DictReader(open(out / "files.csv")))
    assert rows[0]["file"] == "a.wav" and float(rows[0]["drift_ms_per_min"]) < -100
    w = list(csv.DictReader(open(out / "windows.csv")))
    assert len(w) >= 4 and {"start_s", "offset_s", "agreement_score", "confident"} <= set(w[0])
    assert any(r["code"] == "drift" for r in csv.DictReader(open(out / "findings.csv")))


def test_cli_many_files_summary_lists_the_problem_files_only(case, tmp_path, capsys):
    segs, _ = speech(120.0, 1)
    case("good", duration=120.0, seed=1)
    case("bad", duration=120.0, seed=1, labels=shifted(segs, -1.0))
    assert cli.main(["check", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "2 files: 1 pass, 0 warn, 1 fail" in out
    assert "bad.wav: FAIL" in out and "good.wav" not in out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and "driftcheck 0.1.0" in capsys.readouterr().out


def test_cli_window_option_changes_the_windows(case, tmp_path):
    segs, _ = speech(300.0, 3)
    p = case("a", duration=300.0, seed=3, labels=shifted(segs, 0.0, 1.004))
    out = tmp_path / "rep"
    cli.main(["check", str(p), "--out", str(out), "--window", "30"])
    w = list(csv.DictReader(open(out / "windows.csv")))
    assert len(w) >= 9 and float(w[0]["end_s"]) - float(w[0]["start_s"]) == pytest.approx(30.0, abs=0.1)
