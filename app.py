"""Streamlit web app.

    streamlit run app.py

Sidebar: upload a lecture, watch it being processed, manage lectures.
Main area: chat with the assistant. Every answer shows its sources, and each
source has a button that plays the lecture from that exact moment.
"""
import tempfile
from pathlib import Path

import streamlit as st

from rag_ta.chunker import fmt_ts
from rag_ta.config import settings
from rag_ta.store import LectureStore

st.set_page_config(page_title="AI Teaching Assistant", page_icon="🎓", layout="wide")


@st.cache_resource
def get_store() -> LectureStore:
    return LectureStore()


store = get_store()
st.session_state.setdefault("messages", [])   # list of {"role", "content", "sources"}
st.session_state.setdefault("player", None)   # (lecture_id, start_seconds)

LANGUAGES = {
    "Auto-detect": None, "English": "en", "Hindi": "hi", "Bengali": "bn", "Tamil": "ta",
    "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml",
    "Punjabi": "pa", "Urdu": "ur", "Spanish": "es", "French": "fr", "German": "de",
    "Japanese": "ja", "Chinese": "zh", "Arabic": "ar", "Russian": "ru", "Portuguese": "pt",
    "Italian": "it", "Korean": "ko", "Nepali": "ne",
}

# ---------------- sidebar: add and manage lectures ----------------
with st.sidebar:
    st.header("📚 Lectures")

    with st.expander("➕ Add a lecture", expanded=not store.lectures()):
        upload = st.file_uploader(
            "Video or audio file",
            type=["mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a", "ogg", "flac"],
        )
        title = st.text_input("Title", placeholder="e.g. DBMS Lecture 3: Normalisation")
        col1, col2 = st.columns(2)
        model_size = col1.selectbox("Whisper model", ["tiny", "base", "small", "medium"],
                                    index=["tiny", "base", "small", "medium"].index(settings.whisper_model),
                                    help="Bigger = more accurate but slower on CPU.")
        lang_name = col2.selectbox("Spoken language", list(LANGUAGES))

        if st.button("Process lecture", type="primary", disabled=upload is None,
                     use_container_width=True):
            from rag_ta.ingest import ingest_file

            suffix = Path(upload.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.getbuffer())
                tmp_path = Path(tmp.name)
            status = st.status("Processing lecture...", expanded=True)
            bar = status.progress(0.0, text="Transcribing")
            try:
                lecture = ingest_file(
                    tmp_path, title or Path(upload.name).stem, store,
                    model_size=model_size, language=LANGUAGES[lang_name],
                    on_status=status.write,
                    on_progress=lambda f: bar.progress(f, text=f"Transcribing {f:.0%}"),
                )
                status.update(label=f"Added: {lecture['title']}", state="complete", expanded=False)
                st.rerun()
            except Exception as exc:  # show the error in the UI instead of a stack trace
                status.update(label="Failed", state="error")
                st.error(str(exc))
            finally:
                tmp_path.unlink(missing_ok=True)

    lectures = store.lectures()
    if lectures:
        st.caption("Ask about all lectures, or pick some:")
        selected = st.multiselect(
            "Search in", options=[l["id"] for l in lectures],
            format_func=lambda i: store.get(i)["title"], placeholder="All lectures",
        )
        for l in lectures:
            with st.expander(f"🎬 {l['title']}"):
                st.caption(f"Language: {l['language']} · Length: {fmt_ts(l['duration'])} · "
                           f"{l['num_chunks']} chunks · Whisper {l['whisper_model']}")
                c1, c2 = st.columns(2)
                if c1.button("Transcript", key=f"t_{l['id']}"):
                    st.session_state.player = (l["id"], 0)
                if c2.button("Delete", key=f"d_{l['id']}"):
                    store.delete_lecture(l["id"])
                    st.rerun()
    else:
        selected = []

    from rag_ta.rag import describe_llm

    llm_label = describe_llm()
    st.divider()
    if llm_label == "no LLM configured":
        st.warning("No LLM API key found. Add one to `.env` (see `.env.example`).")
    else:
        st.caption(f"🤖 Answer model: {llm_label}")

    if st.session_state.messages and st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- main area ----------------
st.title("🎓 AI Teaching Assistant")
st.caption("Ask questions about your lecture videos, in any language. "
           "Answers come only from the lectures, with timestamps.")

chat_col, player_col = st.columns([3, 2]) if st.session_state.player else (st.container(), None)

# video player + transcript for the clicked source
if player_col is not None:
    lecture_id, start = st.session_state.player
    lecture = store.get(lecture_id)
    with player_col:
        if lecture is None:
            st.session_state.player = None
        else:
            st.subheader(lecture["title"])
            media = settings.media_dir / lecture["media_file"]
            if media.suffix in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
                st.audio(str(media), start_time=int(start))
            else:
                st.video(str(media), start_time=int(start))
            from rag_ta.ingest import load_transcript

            transcript = load_transcript(lecture_id)
            if transcript:
                with st.container(height=320):
                    for seg in transcript["segments"]:
                        marker = "▶ " if seg["start"] <= start < seg["end"] + 1 else ""
                        st.markdown(f"`{fmt_ts(seg['start'])}` {marker}{seg['text']}")
            if st.button("Close player"):
                st.session_state.player = None
                st.rerun()


def render_sources(sources, key_prefix: str) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{s['title']}** · `{fmt_ts(s['start'])}–{fmt_ts(s['end'])}` "
                        f"· similarity {s['score']:.2f}\n\n> {s['text'][:300]}…")
            if c2.button("▶ Play", key=f"{key_prefix}_{i}"):
                st.session_state.player = (s["lecture_id"], s["start"])
                st.rerun()


with chat_col:
    if not lectures:
        st.info("👈 Start by adding a lecture video or audio file in the sidebar.")

    for n, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_sources(msg.get("sources"), key_prefix=f"m{n}")

question = st.chat_input("Ask a question about the lectures...", disabled=not lectures)
if question:
    from rag_ta.rag import answer_question

    history = [
        (m["content"], a["content"])
        for m, a in zip(st.session_state.messages[::2], st.session_state.messages[1::2])
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with chat_col:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the lectures..."):
                try:
                    result = answer_question(store, question, lecture_ids=selected or None,
                                             history=history)
                    content = result.text
                    sources = [s.__dict__ for s in result.sources]
                except Exception as exc:
                    content, sources = f"⚠️ Something went wrong: {exc}", []
            st.markdown(content)
    st.session_state.messages.append({"role": "assistant", "content": content, "sources": sources})
    st.rerun()
