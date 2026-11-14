"""Command-line interface. Handy for quick tests and for demos without the UI.

    python cli.py ingest lecture.mp4 --title "Intro to DBMS"
    python cli.py list
    python cli.py ask "What is normalisation?"
    python cli.py ask "¿Qué es una clave primaria?" --lecture intro-to-dbms-1a2b3c
    python cli.py transcript intro-to-dbms-1a2b3c
    python cli.py delete intro-to-dbms-1a2b3c
"""
import argparse
import sys
from pathlib import Path

from rag_ta.chunker import fmt_ts
from rag_ta.store import LectureStore

# Windows terminals default to cp1252; make sure non-English text prints
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def cmd_ingest(args, store: LectureStore) -> None:
    from rag_ta.ingest import ingest_file

    src = Path(args.file)
    if not src.exists():
        sys.exit(f"File not found: {src}")

    def progress(frac: float) -> None:
        print(f"\r  transcribing... {frac:5.0%}", end="", flush=True)

    lecture = ingest_file(
        src, args.title or src.stem, store,
        model_size=args.model, language=args.language,
        on_status=lambda m: print(f"\n{m}" if m.startswith("Done") else m),
        on_progress=progress,
    )
    print(f"Added lecture '{lecture['title']}' (id: {lecture['id']})")


def cmd_list(args, store: LectureStore) -> None:
    lectures = store.lectures()
    if not lectures:
        print("No lectures yet. Add one with: python cli.py ingest <file>")
        return
    for l in lectures:
        print(f"{l['id']:<40} {l['title'][:30]:<30} {l['language']:>4} "
              f"{fmt_ts(l['duration']):>8} {l['num_chunks']:>4} chunks")


def cmd_ask(args, store: LectureStore) -> None:
    from rag_ta.rag import LLMError, answer_question, retrieve

    try:
        result = answer_question(store, args.question, lecture_ids=args.lecture or None)
    except LLMError as exc:
        # still show what retrieval found, so the problem is clearly the LLM step
        print(f"\n[LLM unavailable] {exc}\n")
        print("Retrieved excerpts (no generated answer):")
        for s in retrieve(store, args.question, lecture_ids=args.lecture or None):
            print(f"  {s.label:<40} similarity {s.score:.3f}\n    {s.text[:160]}...")
        sys.exit(1)
    print("\n" + result.text + "\n")
    if result.sources:
        print("Sources:")
        for s in result.sources:
            print(f"  {s.label:<40} similarity {s.score:.3f}")


def cmd_transcript(args, store: LectureStore) -> None:
    from rag_ta.ingest import load_transcript

    t = load_transcript(args.lecture_id)
    if not t:
        sys.exit(f"No transcript for {args.lecture_id}")
    for seg in t["segments"]:
        print(f"[{fmt_ts(seg['start'])}] {seg['text']}")


def cmd_delete(args, store: LectureStore) -> None:
    if not store.get(args.lecture_id):
        sys.exit(f"No lecture with id {args.lecture_id}")
    store.delete_lecture(args.lecture_id)
    print(f"Deleted {args.lecture_id}")


def main() -> None:
    p = argparse.ArgumentParser(description="RAG Teaching Assistant")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="add a lecture video or audio file")
    pi.add_argument("file")
    pi.add_argument("--title")
    pi.add_argument("--model", help="whisper size: tiny/base/small/medium")
    pi.add_argument("--language", help="force a language code, e.g. en, hi (default: auto)")
    pi.set_defaults(fn=cmd_ingest)

    sub.add_parser("list", help="list lectures").set_defaults(fn=cmd_list)

    pa = sub.add_parser("ask", help="ask a question")
    pa.add_argument("question")
    pa.add_argument("--lecture", action="append", help="limit to a lecture id (repeatable)")
    pa.set_defaults(fn=cmd_ask)

    pt = sub.add_parser("transcript", help="print a lecture transcript")
    pt.add_argument("lecture_id")
    pt.set_defaults(fn=cmd_transcript)

    pd = sub.add_parser("delete", help="remove a lecture")
    pd.add_argument("lecture_id")
    pd.set_defaults(fn=cmd_delete)

    args = p.parse_args()
    args.fn(args, LectureStore())


if __name__ == "__main__":
    main()
