"""Step 2: speech to text with Whisper (faster-whisper implementation).

faster-whisper runs the same Whisper models through CTranslate2, which is
about 4x faster on CPU and uses less memory. We keep every segment's start
and end time, because that's what lets the assistant cite timestamps later.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from .config import settings


@dataclass
class Segment:
    start: float  # seconds
    end: float
    text: str


@dataclass
class Transcript:
    language: str            # detected language code, e.g. "en", "hi"
    language_probability: float
    duration: float          # seconds
    segments: list[Segment]

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(
            language=d["language"],
            language_probability=d["language_probability"],
            duration=d["duration"],
            segments=[Segment(**s) for s in d["segments"]],
        )


def load_wav(path: Path):
    """Read the 16 kHz mono 16-bit WAV that audio.extract_audio produces.

    We hand Whisper the samples directly instead of a file path. That way
    Whisper never has to decode audio itself (FFmpeg already did it), and we
    don't depend on its PyAV decoder, whose API changes between versions.
    """
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path.name} must be 16 kHz mono 16-bit; run it through extract_audio first")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


@lru_cache(maxsize=2)
def _load_model(model_size: str):
    # imported here so the rest of the package can be used without Whisper installed
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


def transcribe(
    audio_path: Path,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Transcript:
    """Transcribe audio. language=None means auto-detect (works for ~99 languages).

    on_progress receives a fraction 0..1, used by the UI progress bar.
    """
    model = _load_model(model_size or settings.whisper_model)
    segments_iter, info = model.transcribe(
        load_wav(audio_path),
        language=language,
        vad_filter=True,     # skip long silences, faster and fewer made-up words
        beam_size=5,
    )

    segments: list[Segment] = []
    for seg in segments_iter:  # this is a generator: transcription happens as we iterate
        text = seg.text.strip()
        if text:
            segments.append(Segment(start=round(seg.start, 2), end=round(seg.end, 2), text=text))
        if on_progress and info.duration:
            on_progress(min(seg.end / info.duration, 1.0))

    if on_progress:
        on_progress(1.0)

    return Transcript(
        language=info.language,
        language_probability=round(info.language_probability, 3),
        duration=round(info.duration, 2),
        segments=segments,
    )
