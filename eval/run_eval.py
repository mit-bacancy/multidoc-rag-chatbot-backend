"""Run sample questions through the API's retrieval + generation and write a markdown report.

    python -m eval.run_eval --doc gpo [--strategy basic hierarchical]

Question file (eval/questions/<doc_id>.json):
    [{"id": "q1", "question": "...", "expected_answer": "...",   # optional
      "expected_pages": [12, 13],                                 # optional, for hit@k
      "answer_type": "text|table|chart|footnote"}]               # optional, Task 3 breakdown
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import DOCUMENTS, STRATEGIES, doc_strategies
from app.generation import answer
from app.llm import invoke_text

EVAL_DIR = Path(__file__).resolve().parent

JUDGE_PROMPT = (
    "You grade a RAG system's answer against an expected answer. "
    "Score 1 if it contains the same key facts (numbers, units, names) as the expected answer, "
    "0.5 if partially correct, 0 if wrong or missing. "
    'Reply with JSON only: {"score": <0|0.5|1>, "reason": "<short>"}'
)


def judge(question: str, expected: str, got: str) -> tuple[float | None, str]:
    raw = invoke_text(
        [
            SystemMessage(JUDGE_PROMPT),
            HumanMessage(f"Question: {question}\n\nExpected: {expected}\n\nAnswer: {got}"),
        ]
    )
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
        return float(data["score"]), data.get("reason", "")
    except (ValueError, KeyError, TypeError):
        return None, raw[:200]


def run(doc_id: str, strategy: str, questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        res = answer(doc_id, strategy, q["question"], [])
        pages = [s.page for s in res.sources if s.page is not None]
        expected_pages = q.get("expected_pages") or []
        hit = any(p in pages for p in expected_pages) if expected_pages else None
        score, reason = judge(q["question"], q["expected_answer"], res.answer) if q.get("expected_answer") else (None, "")
        rows.append(
            {
                **q,
                "strategy": strategy,
                "answer": res.answer,
                "retrieved": [f"p.{s.page} {s.section_path or ''} ({s.content_type})" for s in res.sources],
                "hit": hit,
                "score": score,
                "reason": reason,
            }
        )
        print(f"[{strategy}] {q['id']}: hit={hit} score={score}")
    return rows


def summary(rows: list[dict]) -> dict:
    hits = [r["hit"] for r in rows if r["hit"] is not None]
    scores = [r["score"] for r in rows if r["score"] is not None]
    return {
        "n": len(rows),
        "hit@k": f"{sum(hits)}/{len(hits)}" if hits else "-",
        "score": f"{sum(scores) / len(scores):.2f}" if scores else "-",
    }


def report(doc_id: str, results: dict[str, list[dict]]) -> str:
    lines = [f"# Eval — {DOCUMENTS[doc_id]['title']} ({date.today()})", "", "## Summary", "",
             "| strategy | questions | hit@k | mean score |", "|---|---|---|---|"]
    for strategy, rows in results.items():
        s = summary(rows)
        lines.append(f"| {strategy} | {s['n']} | {s['hit@k']} | {s['score']} |")

    # Task 3: breakdown by where the answer lives
    for strategy, rows in results.items():
        by_type = defaultdict(list)
        for r in rows:
            if r.get("answer_type"):
                by_type[r["answer_type"]].append(r)
        if by_type:
            lines += ["", f"### By answer type ({strategy})", "", "| type | questions | hit@k | mean score |", "|---|---|---|---|"]
            for t, rs in sorted(by_type.items()):
                s = summary(rs)
                lines.append(f"| {t} | {s['n']} | {s['hit@k']} | {s['score']} |")

    lines += ["", "## Questions"]
    first = next(iter(results.values()))
    for i, q in enumerate(first):
        lines += ["", f"### {q['id']}: {q['question']}"]
        if q.get("expected_answer"):
            lines.append(f"**Expected:** {q['expected_answer']}  ")
        if q.get("expected_pages"):
            lines.append(f"**Expected pages:** {q['expected_pages']}")
        for strategy, rows in results.items():
            r = rows[i]
            lines += [
                "",
                f"**{strategy}** — hit: {r['hit']}, score: {r['score']} {('— ' + r['reason']) if r['reason'] else ''}",
                "",
                "> " + r["answer"].replace("\n", "\n> "),
                "",
                "Retrieved: " + "; ".join(r["retrieved"]),
            ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, choices=list(DOCUMENTS))
    ap.add_argument("--strategy", nargs="+", choices=STRATEGIES, help="default: doc strategy (+ baseline)")
    args = ap.parse_args()

    qfile = EVAL_DIR / "questions" / f"{args.doc}.json"
    questions = json.loads(qfile.read_text())
    if not questions:
        raise SystemExit(f"No questions in {qfile}")

    strategies = args.strategy or doc_strategies(args.doc)
    results = {s: run(args.doc, s, questions) for s in strategies}

    out = EVAL_DIR / "results" / f"{args.doc}_{date.today()}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report(args.doc, results))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
