"""Step 4: embeddings + FAISS vector store + a small registry of lectures.

Embeddings: intfloat/multilingual-e5-small (384 dimensions, ~100 languages).
E5 models are trained with prefixes: "passage: " for stored text and
"query: " for questions. Using them noticeably improves retrieval, so we wrap
the model in our own small Embeddings class instead of a generic one.

Vectors are L2-normalised, so the inner product FAISS computes is exactly the
cosine similarity (1.0 = same meaning).
"""
import json
import threading
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .chunker import Chunk
from .config import settings


class E5Embeddings(Embeddings):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            [f"passage: {t}" for t in texts],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(f"query: {text}", normalize_embeddings=True)
        return vec.tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> E5Embeddings:
    return E5Embeddings(settings.embedding_model)


class LectureStore:
    """Owns the FAISS index on disk and lectures.json (what's been ingested)."""

    _lock = threading.Lock()  # Streamlit can run callbacks concurrently

    def __init__(self):
        settings.ensure_dirs()
        self._index = None
        self._registry = self._load_registry()

    # ---------- registry ----------

    def _load_registry(self) -> dict:
        if settings.registry_path.exists():
            return json.loads(settings.registry_path.read_text(encoding="utf-8"))
        return {}

    def _save_registry(self) -> None:
        settings.registry_path.write_text(
            json.dumps(self._registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def lectures(self) -> list[dict]:
        return sorted(self._registry.values(), key=lambda l: l["created_at"], reverse=True)

    def get(self, lecture_id: str) -> Optional[dict]:
        return self._registry.get(lecture_id)

    # ---------- index ----------

    @property
    def index(self):
        if self._index is None and (settings.index_dir / "index.faiss").exists():
            from langchain_community.vectorstores import FAISS

            # allow_dangerous_deserialization: the docstore is a pickle we wrote ourselves
            self._index = FAISS.load_local(
                str(settings.index_dir),
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        return self._index

    def add_lecture(self, lecture: dict, chunks: list[Chunk]) -> None:
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy

        docs = [
            Document(
                page_content=c.text,
                metadata={
                    "lecture_id": lecture["id"],
                    "title": lecture["title"],
                    "language": lecture["language"],
                    "chunk_index": c.index,
                    "start": c.start,
                    "end": c.end,
                },
            )
            for c in chunks
        ]
        ids = [f"{lecture['id']}:{c.index}" for c in chunks]

        with self._lock:
            if self.index is None:
                self._index = FAISS.from_documents(
                    docs,
                    get_embeddings(),
                    ids=ids,
                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                )
            else:
                self._index.add_documents(docs, ids=ids)
            self._index.save_local(str(settings.index_dir))

            lecture = {**lecture, "num_chunks": len(chunks),
                       "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            self._registry[lecture["id"]] = lecture
            self._save_registry()

    def delete_lecture(self, lecture_id: str) -> None:
        with self._lock:
            lecture = self._registry.pop(lecture_id, None)
            if lecture is None:
                return
            if self.index is not None and lecture.get("num_chunks"):
                ids = [f"{lecture_id}:{i}" for i in range(lecture["num_chunks"])]
                self._index.delete(ids)
                self._index.save_local(str(settings.index_dir))
            self._save_registry()
            # remove the lecture's files too
            for path in (settings.transcript_dir / f"{lecture_id}.json",
                         settings.audio_dir / f"{lecture_id}.wav"):
                path.unlink(missing_ok=True)
            media = lecture.get("media_file")
            if media:
                (settings.media_dir / media).unlink(missing_ok=True)

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        lecture_ids: Optional[list[str]] = None,
    ) -> list[tuple[Document, float]]:
        """Top-k chunks for a question, optionally only from some lectures.

        Returns (document, cosine similarity) pairs, best first.
        """
        if self.index is None:
            return []
        k = k or settings.top_k
        allowed = set(lecture_ids) if lecture_ids else None
        return self.index.similarity_search_with_score(
            query,
            k=k,
            fetch_k=settings.fetch_k,
            filter=(lambda md: md["lecture_id"] in allowed) if allowed else None,
        )
