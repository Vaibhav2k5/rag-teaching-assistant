from dataclasses import dataclass

import pytest

from rag_ta.chunker import chunk_segments, fmt_ts


@dataclass
class Seg:
    start: float
    end: float
    text: str


def make_segments(n: int, words: int = 10) -> list[Seg]:
    return [Seg(i * 5.0, i * 5.0 + 5.0, " ".join([f"w{i}"] * words)) for i in range(n)]


def test_empty_input_gives_no_chunks():
    assert chunk_segments([]) == []


def test_short_transcript_is_one_chunk():
    segs = make_segments(3)
    chunks = chunk_segments(segs, chunk_chars=800, overlap_chars=100)
    assert len(chunks) == 1
    assert chunks[0].start == 0.0 and chunks[0].end == 15.0


def test_chunks_respect_size_and_keep_timestamps():
    segs = make_segments(50)  # each segment is ~29 chars
    chunks = chunk_segments(segs, chunk_chars=200, overlap_chars=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200
        assert c.start < c.end
    assert chunks[0].start == 0.0
    assert chunks[-1].end == segs[-1].end  # nothing at the end is lost


def test_consecutive_chunks_overlap():
    segs = make_segments(50)
    chunks = chunk_segments(segs, chunk_chars=200, overlap_chars=60)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # b starts inside a: that's the overlap


def test_every_segment_lands_in_some_chunk():
    segs = make_segments(40)
    chunks = chunk_segments(segs, chunk_chars=150, overlap_chars=40)
    joined = " ".join(c.text for c in chunks)
    for s in segs:
        assert s.text in joined


def test_blank_segments_are_ignored():
    segs = [Seg(0, 1, "hello"), Seg(1, 2, "   "), Seg(2, 3, "world")]
    chunks = chunk_segments(segs)
    assert chunks[0].text == "hello world"


def test_indexes_are_sequential():
    chunks = chunk_segments(make_segments(60), chunk_chars=150, overlap_chars=30)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        chunk_segments(make_segments(2), chunk_chars=100, overlap_chars=100)
    with pytest.raises(ValueError):
        chunk_segments(make_segments(2), chunk_chars=0)


@pytest.mark.parametrize("secs,expected", [(0, "0:00"), (12.7, "0:12"), (754, "12:34"), (3725, "1:02:05")])
def test_fmt_ts(secs, expected):
    assert fmt_ts(secs) == expected
