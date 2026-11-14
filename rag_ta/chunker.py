"""Step 3: turn timestamped Whisper segments into retrieval chunks.

Why not split the raw text every N characters? Because we'd lose the
timestamps. Instead we glue whole segments together until a chunk is about
chunk_chars long, and remember the start of the first segment and the end of
the last one. The next chunk starts with the last ~overlap_chars of the
previous one, so an idea that crosses a chunk boundary is still findable.

This file has no heavy dependencies on purpose, so it's easy to unit test.
"""
from dataclasses import dataclass
from typing import Iterable, Protocol


class SegmentLike(Protocol):
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    text: str


def chunk_segments(
    segments: Iterable[SegmentLike],
    chunk_chars: int = 800,
    overlap_chars: int = 150,
) -> list[Chunk]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if not 0 <= overlap_chars < chunk_chars:
        raise ValueError("overlap_chars must be between 0 and chunk_chars")

    segs = [s for s in segments if s.text and s.text.strip()]
    chunks: list[Chunk] = []
    current: list[SegmentLike] = []
    current_len = 0

    def flush() -> None:
        text = " ".join(s.text.strip() for s in current)
        chunks.append(Chunk(index=len(chunks), start=current[0].start, end=current[-1].end, text=text))

    for seg in segs:
        seg_len = len(seg.text.strip()) + 1
        if current and current_len + seg_len > chunk_chars:
            flush()
            # carry over trailing segments as overlap
            carried: list[SegmentLike] = []
            carried_len = 0
            for prev in reversed(current):
                plen = len(prev.text.strip()) + 1
                if carried_len + plen > overlap_chars:
                    break
                carried.insert(0, prev)
                carried_len += plen
            current, current_len = carried, carried_len
        current.append(seg)
        current_len += seg_len

    # last chunk, unless it is nothing but overlap we already emitted
    if current:
        already_emitted = bool(chunks) and current[-1].end <= chunks[-1].end
        if not already_emitted:
            flush()

    return chunks


def fmt_ts(seconds: float) -> str:
    """12.5 -> '0:12', 3725 -> '1:02:05'."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
