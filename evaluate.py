"""Retrieval evaluation: does the right part of the lecture come back in the top-k?

Write a small JSONL file of test questions, one per line:

    {"question": "What is 2NF?", "lecture_id": "dbms-3-ab12cd", "time": 754}
    {"question": "2NF क्या है?", "lecture_id": "dbms-3-ab12cd", "time": 754}

"time" is the moment (in seconds) in the lecture where the answer is spoken.
A question counts as a hit@k if any of the top-k retrieved chunks covers that
moment. We also report MRR (mean reciprocal rank): 1.0 means the right chunk
was always ranked first.

    python evaluate.py eval_questions.jsonl --k 5

This measures retrieval only, which is the part that decides whether the LLM
even sees the answer. Answer quality would need a human or an LLM judge on top.
"""
import argparse
import json
import sys
from pathlib import Path

from rag_ta.rag import retrieve
from rag_ta.store import LectureStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("questions", type=Path)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    store = LectureStore()
    rows = [json.loads(l) for l in args.questions.read_text(encoding="utf-8").splitlines() if l.strip()]
    hits, rr_total = 0, 0.0

    for row in rows:
        sources = retrieve(store, row["question"], k=args.k)
        rank = next(
            (i for i, s in enumerate(sources, 1)
             if s.lecture_id == row["lecture_id"] and s.start <= row["time"] <= s.end),
            None,
        )
        hits += rank is not None
        rr_total += 1 / rank if rank else 0
        print(f"{'HIT ' if rank else 'MISS'} rank={rank or '-':<2} {row['question']}")

    n = len(rows) or 1
    print(f"\nhit@{args.k}: {hits / n:.0%}   MRR: {rr_total / n:.2f}   ({len(rows)} questions)")


if __name__ == "__main__":
    main()
