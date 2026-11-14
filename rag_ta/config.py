"""All tunable settings in one place. Each one can be overridden from .env."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# project root = the folder that contains rag_ta/
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")  # also pick up the workspace-level .env (HF token)


def _env(name: str, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return type(default)(value)


@dataclass(frozen=True)
class Settings:
    # where everything is stored
    data_dir: Path = field(default_factory=lambda: Path(_env("RAG_DATA_DIR", str(ROOT / "data"))))

    # speech to text (faster-whisper). tiny/base/small/medium. "small" is the
    # best accuracy/speed trade-off on a laptop CPU, "base" is faster.
    whisper_model: str = _env("WHISPER_MODEL", "base")
    whisper_device: str = _env("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = _env("WHISPER_COMPUTE_TYPE", "int8")

    # chunking: we group whisper segments until a chunk reaches ~chunk_chars,
    # and carry the last overlap_chars worth of segments into the next chunk
    chunk_chars: int = _env("CHUNK_CHARS", 800)
    overlap_chars: int = _env("OVERLAP_CHARS", 150)

    # embeddings: multilingual model, so a question in one language can find
    # lecture text in another
    embedding_model: str = _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

    # retrieval
    top_k: int = _env("TOP_K", 5)
    fetch_k: int = _env("FETCH_K", 30)  # candidates fetched before lecture filtering

    # answer generation. LLM_PROVIDER: auto / gemini / openai / anthropic / huggingface.
    # "auto" picks the first provider that has an API key set, in that order.
    llm_provider: str = _env("LLM_PROVIDER", "auto")
    llm_model: str = _env("LLM_MODEL", "")  # empty = the provider's default below
    llm_temperature: float = _env("LLM_TEMPERATURE", 0.2)
    llm_max_new_tokens: int = _env("LLM_MAX_NEW_TOKENS", 700)
    hf_token: str = _env("HUGGINGFACEHUB_API_TOKEN", "")
    google_api_key: str = _env("GOOGLE_API_KEY", "")
    openai_api_key: str = _env("OPENAI_API_KEY", "")
    anthropic_api_key: str = _env("ANTHROPIC_API_KEY", "")

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def transcript_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "lectures.json"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.media_dir, self.audio_dir, self.transcript_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()

# model downloads (Whisper, embeddings) read HF_TOKEN; reuse the existing token so
# downloads are authenticated and don't print the "unauthenticated" warning
if settings.hf_token and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = settings.hf_token
# transformers prints a long load report for the embedding model; it's harmless noise
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
