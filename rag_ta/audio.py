"""Step 1: pull the audio track out of a lecture video with FFmpeg.

Whisper works best on 16 kHz mono audio, so we convert to exactly that.
We use the FFmpeg binary that ships with the imageio-ffmpeg package, so there
is nothing to install system-wide.
"""
import subprocess
from pathlib import Path

import imageio_ffmpeg


def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(media_path: Path, out_wav: Path) -> Path:
    """Convert any video/audio file into a 16 kHz mono WAV for Whisper."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(),
        "-y",                 # overwrite if it already exists
        "-i", str(media_path),
        "-vn",                # drop the video stream
        "-ac", "1",           # mono
        "-ar", "16000",       # 16 kHz, what Whisper expects
        "-f", "wav",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # last few lines of ffmpeg's stderr are the useful part
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"FFmpeg could not read {media_path.name}:\n{tail}")
    return out_wav
