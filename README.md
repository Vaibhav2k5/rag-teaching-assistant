# RAG-Based AI Teaching Assistant

Ask questions about lecture videos, in any language, and get answers that come
only from the lectures, with timestamps you can click to jump to that moment.

```
lecture video ──FFmpeg──> 16 kHz audio ──Whisper──> timestamped transcript
      │                                                   │
      │                                       chunk (~800 chars, overlap, keep times)
      │                                                   │
      │                                  multilingual embeddings (E5) ──> FAISS index
      │                                                                     │
 student question ──embed──> top-k similar chunks ──> LLM (Qwen 2.5) ──> answer + [Lecture @ 12:34]
```

## Quick start

```bash
# from the workspace folder, using the existing venv
cd rag_teaching_assistant
..\venv\Scripts\activate          # Windows
pip install -r requirements.txt

# set ONE LLM API key in .env (copy .env.example). Easiest free option:
# a Gemini key from https://aistudio.google.com/apikey  ->  GOOGLE_API_KEY=...

streamlit run app.py              # web app at http://localhost:8501
```

Or from the command line:

```bash
python cli.py ingest "lecture1.mp4" --title "DBMS Lecture 1"
python cli.py list
python cli.py ask "What is a primary key?"
python cli.py ask "प्राथमिक कुंजी क्या है?"          # Hindi question, English lecture
python cli.py transcript <lecture-id>
python cli.py delete <lecture-id>
```

The first run downloads the Whisper model (~150 MB for `base`) and the
embedding model (~470 MB). After that everything except the LLM runs offline.

## Project structure

| File | What it does |
|---|---|
| `rag_ta/config.py` | Every setting in one place, overridable from `.env` |
| `rag_ta/audio.py` | FFmpeg: video → 16 kHz mono WAV |
| `rag_ta/transcribe.py` | Whisper (faster-whisper): audio → segments with start/end times, auto language detection |
| `rag_ta/chunker.py` | Groups segments into overlapping chunks without losing timestamps |
| `rag_ta/store.py` | E5 multilingual embeddings, FAISS index on disk, registry of lectures |
| `rag_ta/ingest.py` | Runs the whole ingestion pipeline in order |
| `rag_ta/rag.py` | Retrieval, prompt construction, pluggable LLM (Gemini / OpenAI / Anthropic / Hugging Face), sources, readable errors |
| `app.py` | Streamlit UI: upload, progress, chat, sources, play-from-timestamp |
| `cli.py` | Command-line version of everything |
| `evaluate.py` | Retrieval evaluation: hit@k and MRR on your own test questions |
| `tests/` | Unit tests for chunking (`pytest`) |

Stored data (all under `data/`): original media, transcripts as JSON, the FAISS
index, and `lectures.json`.

## How it works, step by step

1. **Audio extraction (FFmpeg).** Lecture videos can be any format. FFmpeg drops the
   video and converts the audio to 16 kHz mono, which is what Whisper is trained on.
   The FFmpeg binary comes from the `imageio-ffmpeg` package, so nothing needs to be
   installed system-wide.

2. **Transcription (Whisper).** `faster-whisper` runs OpenAI's Whisper models through
   CTranslate2: same accuracy, about 4x faster on CPU, with int8 quantisation to save
   memory. It auto-detects the spoken language and returns segments with start and end
   times. A voice-activity filter skips silences, which is faster and reduces
   Whisper's tendency to invent words during silence.

3. **Chunking.** Segments are joined until a chunk reaches about 800 characters
   (roughly 150–200 tokens), keeping the start time of the first segment and the end
   time of the last. The next chunk repeats about 150 characters from the end of the
   previous one, so an idea that spans a boundary still appears whole in one chunk.
   Chunking by segments instead of raw characters is what keeps timestamps accurate.

4. **Embeddings (multilingual E5).** Each chunk becomes a 384-number vector with
   `intfloat/multilingual-e5-small`, which covers about 100 languages in one shared
   vector space. That is what makes cross-language search work: a question in Hindi
   lands near the matching English lecture text. E5 is trained with `passage:` and
   `query:` prefixes, so we add them. Vectors are normalised, so FAISS's inner
   product equals cosine similarity.

5. **Vector store (FAISS).** Chunks are stored with metadata: lecture id, title,
   start and end time, and language. The index is saved to disk after each lecture.
   Search can be limited to selected lectures.

6. **Answering (RAG).** The question is embedded, the top 5 chunks are retrieved, and
   they go into the prompt with their timestamps. The system prompt tells the model to
   answer only from those excerpts, to say so when the answer isn't there, to cite
   `[Lecture @ mm:ss]`, and to reply in the student's language. Temperature is 0.2
   for focused, factual answers. The last 3 turns of chat are included, so follow-ups
   like "explain that more simply" work.

## How the multilingual support works

Three pieces, each multilingual:

| Stage | Model | Languages |
|---|---|---|
| Speech to text | Whisper | ~99 spoken languages, auto-detected |
| Search | multilingual-e5-small | ~100 languages in one shared vector space |
| Answer | Your configured LLM (Gemini 2.5 Flash, GPT, Claude or Qwen 2.5) | All are multilingual; the prompt tells the model to reply in the question's language |

So a lecture can be in one language and the question in another. Nothing is
translated up front: the shared embedding space does the matching, and the LLM
writes the answer in the student's language. The UI lists 22 common languages for
the "spoken language" option, including the main Indian languages; auto-detect
covers the rest.

**Honest limits:** transcription quality varies by language (Whisper is strongest on
English and widely spoken languages, weaker on low-resource ones), and cross-language
retrieval is a little less precise than same-language retrieval.

## Design decisions (and why)

| Decision | Alternative | Why this one |
|---|---|---|
| faster-whisper | openai-whisper | Same models, ~4x faster on CPU, lower memory |
| Chunk by segments | Fixed character splits | Keeps exact timestamps for citations |
| 800 chars + 150 overlap | Bigger/smaller | Big enough for one idea, small enough to stay focused; overlap avoids cut-off ideas |
| multilingual-e5-small | all-MiniLM-L6-v2 | MiniLM is English-only and cuts text at 128 tokens; E5 handles ~100 languages and 512 tokens |
| FAISS | Chroma | No server or extra dependencies, very fast, saves to a file. Chroma would add richer metadata queries; at this scale FAISS is simpler |
| Pluggable LLM (Gemini, OpenAI, Anthropic, Hugging Face) | One fixed provider | Retrieval doesn't care which model writes the answer. LangChain gives all four the same interface, so switching is one line in `.env`. No GPU needed |
| Temperature 0.2 | Default ~0.7 | Factual answers, fewer made-up details |
| Strict "only from excerpts" prompt | Free answering | Grounded answers; says "not covered" instead of hallucinating |

## Evaluation

`evaluate.py` measures retrieval, the step that decides whether the LLM even sees
the answer. Write a few questions with the second at which the answer is spoken:

```json
{"question": "What is 2NF?", "lecture_id": "dbms-3-ab12cd", "time": 754}
{"question": "2NF क्या है?", "lecture_id": "dbms-3-ab12cd", "time": 754}
```

```bash
python evaluate.py eval_questions.jsonl --k 5
```

It reports **hit@5** (how often the right moment is in the top 5) and **MRR** (how
high it's ranked on average). Asking the same question in two languages shows how
well cross-language search holds up.

## Known limitations and next steps

- **Speed:** transcription on CPU takes roughly 0.3–1x the lecture length with `base`.
  A GPU, or the `tiny` model, makes it much faster.
- **Retrieval quality:** pure vector search can miss exact keywords (formula names,
  codes). Next step: hybrid search (BM25 plus vectors) and a cross-encoder re-ranker.
- **Scale:** FAISS here is a flat index in one file. For thousands of lectures, move to
  an IVF/HNSW index or a vector database, and transcribe in a background job queue.
- **Slides:** only speech is indexed. Adding OCR of slide frames would capture
  on-screen content.
- **Answer evaluation:** only retrieval is measured. Answer faithfulness could be
  scored with RAGAS or an LLM judge.
