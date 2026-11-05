import gzip
import json

import pytest

from driftcheck.activity import DriftcheckError
from driftcheck.labels import (guess_format, parse_audacity, parse_csv, parse_lhotse, parse_subtitles,
                               read_labels)

SRT = """1
00:00:01,500 --> 00:00:03,250
Hello there,
how are you?

2
01:02:03,004 --> 01:02:05,000
Second one
"""


def test_srt_times_text_and_multiline():
    s = parse_subtitles(SRT)
    assert len(s) == 2
    assert (s[0].start, s[0].end, s[0].text) == (1.5, 3.25, "Hello there, how are you?")
    assert s[1].start == pytest.approx(3723.004)
    assert s[1].end == pytest.approx(3725.0)


def test_srt_crlf_and_bom(tmp_path):
    p = tmp_path / "a.srt"
    p.write_bytes(b"\xef\xbb\xbf" + SRT.replace("\n", "\r\n").encode())
    s = read_labels(p)[""]
    assert len(s) == 2 and s[0].text.startswith("Hello")


def test_srt_short_milliseconds_are_not_misread():
    # "00:00:01,5" means 1.5 s, not 1.005 s
    s = parse_subtitles("1\n00:00:01,5 --> 00:00:02,25\nx\n")
    assert (s[0].start, s[0].end) == (1.5, 2.25)


def test_vtt_with_header_short_times_and_cue_settings():
    vtt = "WEBVTT\n\nNOTE comment\n\n00:01.000 --> 00:02.500 align:start position:0%\nfirst\n\ncue2\n01:00:00.000 --> 01:00:01.000\nsecond\n"
    s = parse_subtitles(vtt)
    assert [(x.start, x.end, x.text) for x in s] == [(1.0, 2.5, "first"), (3600.0, 3601.0, "second")]


def test_subtitles_without_cues_is_an_error():
    with pytest.raises(DriftcheckError, match="no 'start --> end'"):
        parse_subtitles("just some text\n")


def test_audacity_skips_frequency_lines_and_reads_text():
    s = parse_audacity("1.0\t2.5\thello\n\\\t100.0\t200.0\n3.0\t4.0\n")
    assert [(x.start, x.end, x.text) for x in s] == [(1.0, 2.5, "hello"), (3.0, 4.0, "")]


def test_audacity_bad_line_names_the_line():
    with pytest.raises(DriftcheckError, match="line 2"):
        parse_audacity("1\t2\tok\nabc\tdef\tbad\n")
    with pytest.raises(DriftcheckError, match="line 1"):
        parse_audacity("only-one-column\n")


def test_csv_start_end_and_start_duration_and_id():
    a = parse_csv("start,end,text\n1,2.5,hi\n3,4,yo\n")
    assert [(x.start, x.end, x.text) for x in a[""]] == [(1.0, 2.5, "hi"), (3.0, 4.0, "yo")]
    b = parse_csv("Start,Duration,Transcript,File\n1,1.5,hi,a\n3,1,yo,b\n5,1,again,a\n")
    assert [(x.start, x.end) for x in b["a"]] == [(1.0, 2.5), (5.0, 6.0)]
    assert [x.text for x in b["b"]] == ["yo"]


def test_csv_errors():
    with pytest.raises(DriftcheckError, match="header row"):
        parse_csv("a,b\n1,2\n")
    with pytest.raises(DriftcheckError, match="line 2"):
        parse_csv("start,end\nx,y\n")
    with pytest.raises(DriftcheckError, match="empty"):
        parse_csv("")


def test_lhotse_groups_by_recording_and_reads_gzip(tmp_path):
    rows = [{"recording_id": "r1", "start": 1.0, "duration": 2.0, "text": "a"},
            {"recording_id": "r2", "start": 0.5, "duration": 1.0, "text": "b"},
            {"recording_id": "r1", "start": 4.0, "duration": 1.0, "text": "c"}]
    body = "\n".join(json.dumps(r) for r in rows) + "\n"
    out = parse_lhotse(body)
    assert [(x.start, x.end) for x in out["r1"]] == [(1.0, 3.0), (4.0, 5.0)]
    p = tmp_path / "sup.jsonl.gz"
    p.write_bytes(gzip.compress(body.encode()))
    assert set(read_labels(p)) == {"r1", "r2"}


def test_lhotse_missing_field_is_an_error():
    with pytest.raises(DriftcheckError, match="line 1"):
        parse_lhotse('{"start": 1.0}\n')


def test_guess_format_and_unknown(tmp_path):
    assert guess_format(tmp_path / "x.SRT") == "srt"
    assert guess_format(tmp_path / "x.vtt") == "vtt"
    assert guess_format(tmp_path / "x.tsv") == "audacity"
    assert guess_format(tmp_path / "x.csv") == "csv"
    assert guess_format(tmp_path / "x.jsonl.gz") == "lhotse"
    with pytest.raises(DriftcheckError, match="--format"):
        guess_format(tmp_path / "x.docx")


def test_read_labels_unknown_format_and_missing_file(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text(SRT)
    with pytest.raises(DriftcheckError, match="unknown label format"):
        read_labels(p, "xml")
    with pytest.raises(DriftcheckError, match="cannot read"):
        read_labels(tmp_path / "nope.srt")


def test_csv_with_a_byte_order_mark_still_has_its_header(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes(b"\xef\xbb\xbfstart,end,text\n1,2,hi\n")
    assert read_labels(p)[""][0].text == "hi"
