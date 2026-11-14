"""The whole ingestion pipeline, in order:

    media file -> copy into data/media -> FFmpeg -> 16 kHz WAV
               -> Whisper -> timestamped transcript (saved as JSON)
               -> chunker -> chunks with start/end times
               -> embeddings -> FAISS (+ lectures.json registry)
"""
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Callable, Optional

from .audio import extract_audio
from .chunker import chunk_segments
from .config import settings
from .store import LectureStore
from .transcribe import Transcript, transcribe

StatusFn = Callable[[str], None]
ProgressFn = Callable[[float], None]


def make_lecture_id(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "lecture"
    return f"{slug}-{uuid.uuid4().hex[:6]}"


def ingest_file(
    src: Path,
    title: str,
    store: LectureStore,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    on_status: StatusFn = lambda msg: None,
    on_progress: ProgressFn = lambda frac: None,
) -> dict:
    settings.ensure_dirs()
    title = title.strip() or src.stem
    lecture_id = make_lecture_id(title)

    # keep a copy of the original so the UI can play it back at a timestamp
    media_name = f"{lecture_id}{src.suffix.lower()}"
    media_path = settings.media_dir / media_name
    shutil.copyfile(src, media_path)

    on_status("Extracting audio with FFmpeg...")
    wav = extract_audio(media_path, settings.audio_dir / f"{lecture_id}.wav")

    on_status(f"Transcribing with Whisper ({model_size or settings.whisper_model})...")
    transcript: Transcript = transcribe(wav, model_size=model_size, language=language,
                                        on_progress=on_progress)
    if not transcript.segments:
        raise RuntimeError("Whisper found no speech in this file.")
    (settings.transcript_dir / f"{lecture_id}.json").write_text(
        json.dumps({"title": title, **transcript.to_dict()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    wav.unlink(missing_ok=True)  # the WAV was only needed for Whisper

    on_status("Chunking and embedding...")
    chunks = chunk_segments(transcript.segments, settings.chunk_chars, settings.overlap_chars)
    lecture = {
        "id": lecture_id,
        "title": title,
        "media_file": media_name,
        "language": transcript.language,
        "duration": transcript.duration,
        "whisper_model": model_size or settings.whisper_model,
    }
    store.add_lecture(lecture, chunks)

    on_status(f"Done: {len(transcript.segments)} segments, {len(chunks)} chunks, "
              f"language '{transcript.language}'.")
    return store.get(lecture_id)


def load_transcript(lecture_id: str) -> Optional[dict]:
    path = settings.transcript_dir / f"{lecture_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
