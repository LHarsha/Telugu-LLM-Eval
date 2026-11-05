"""Scoring: answer extraction, matching, aggregation and confidence intervals.

Scoring never calls an API. It reads the raw outputs saved by the runner (and,
optionally, cached LLM-judge verdicts) and recomputes everything from scratch,
so anyone can reproduce the leaderboard from the repository alone.

Rules
-----
* MCQ: the predicted letter is taken from the last "Answer: X" line. If there
  is none, a reply that is just a letter, or that exactly matches one option's
  text after normalisation, is accepted. Correct = letter equals gold_option.
* Short answer (strict): the extracted answer, normalised (NFC, zero-width
  characters, punctuation, Telugu digits, case, whitespace), must equal one of
  the normalised gold_answers.
* Short answer (judge, optional, labelled): items that fail the strict match
  can be re-checked by an LLM judge. Judge-assisted accuracy is always reported
  separately from strict accuracy; strict is the headline number.
* Missing outputs (e.g. a model has not finished because of free-tier quotas)
  count as missing, not wrong; a model is marked incomplete until every item
  has an output.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

from .normalize import clean_text, normalize_for_match
from .schema import LETTERS

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 12345

# "Answer: B", "**Answer:** (B)", "answer - b.", "Final answer: B) తెలంగాణ"
_ANSWER_LINE = re.compile(r"(?:final\s+)?answer\s*\**\s*[:：\-]\s*\**\s*(.*)", re.IGNORECASE)
_LEADING_LETTER = re.compile(r"^[\s\*\(\[\"']*([A-Fa-f])(?=$|[\s\)\]\.\:,\*\"'])")


@dataclass
class Extraction:
    answer: str | None       # letter for MCQ, text for short answers; None = could not parse
    method: str              # answer_line | bare_letter | option_text | whole_reply | none


def _answer_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        m = _ANSWER_LINE.search(line.strip())
        if m:
            out.append(m.group(1).strip())
    return out


def extract_mcq(text: str, options: list[str]) -> Extraction:
    text = clean_text(text or "")
    letters = LETTERS[: len(options)]
    for cand in reversed(_answer_lines(text)):
        m = _LEADING_LETTER.match(cand)
        if m and m.group(1).upper() in letters:
            return Extraction(m.group(1).upper(), "answer_line")
        hit = _match_option_text(cand, options)
        if hit:
            return Extraction(hit, "option_text")
    stripped = text.strip().strip("*").strip()
    m = re.fullmatch(r"[\(\[]?([A-Fa-f])[\)\]\.]?", stripped)
    if m and m.group(1).upper() in letters:
        return Extraction(m.group(1).upper(), "bare_letter")
    hit = _match_option_text(stripped, options)
    if hit:
        return Extraction(hit, "option_text")
    return Extraction(None, "none")


def _match_option_text(cand: str, options: list[str]) -> str | None:
    n = normalize_for_match(cand)
    if not n:
        return None
    hits = [LETTERS[i] for i, o in enumerate(options) if normalize_for_match(o) == n]
    return hits[0] if len(hits) == 1 else None


def extract_short(text: str) -> Extraction:
    text = clean_text(text or "")
    lines = _answer_lines(text)
    if lines:
        ans = lines[-1].strip().strip("*").strip()
        return Extraction(ans or None, "answer_line" if ans else "none")
    # No "Answer:" line: accept a one-line reply as the answer.
    nonempty = [ln for ln in text.splitlines() if ln.strip()]
    if len(nonempty) == 1 and len(nonempty[0]) <= 80:
        return Extraction(nonempty[0].strip(), "whole_reply")
    return Extraction(None, "none")


def short_match(pred: str | None, golds: list[str]) -> bool:
    if not pred:
        return False
    p = normalize_for_match(pred)
    return bool(p) and any(p == normalize_for_match(g) for g in golds)


@dataclass
class ItemScore:
    item_id: str
    category: str
    script: str
    difficulty: str
    region: str
    answer_type: str
    status: str              # scored | missing
    pred: str | None = None
    method: str = "none"
    correct: bool = False            # strict
    judge_correct: bool | None = None  # only for short answers judged
    response_excerpt: str = ""

    @property
    def correct_with_judge(self) -> bool:
        return self.correct or bool(self.judge_correct)


def score_item(item: dict, response_text: str | None) -> ItemScore:
    base = dict(item_id=item["id"], category=item["category"], script=item["script"],
                difficulty=item["difficulty"], region=item.get("region", "none"),
                answer_type=item["answer_type"])
    if response_text is None:
        return ItemScore(**base, status="missing")
    if item["answer_type"] == "mcq":
        ex = extract_mcq(response_text, item["options"])
        correct = ex.answer is not None and ex.answer == item["gold_option"]
    else:
        ex = extract_short(response_text)
        correct = short_match(ex.answer, item["gold_answers"])
    return ItemScore(**base, status="scored", pred=ex.answer, method=ex.method, correct=correct,
                     response_excerpt=" ".join(response_text.split())[:200])


# --- aggregation ------------------------------------------------------------

def bootstrap_ci(values: list[int], samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean of 0/1 values (resampling items)."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    lo_i = int((1 - level) / 2 * samples)
    hi_i = min(samples - 1, int((1 + level) / 2 * samples) - 1)
    return (means[lo_i], means[hi_i])


def summarize(values: list[int], with_ci: bool = True) -> dict:
    n = len(values)
    d = {"n": n, "correct": sum(values), "accuracy": (sum(values) / n) if n else None}
    if with_ci and n:
        lo, hi = bootstrap_ci(values)
        d["ci95"] = [lo, hi]
    return d


GROUPS = ("category", "script", "difficulty", "answer_type", "region")


def aggregate(scores: list[ItemScore]) -> dict:
    scored = [s for s in scores if s.status == "scored"]
    missing = [s.item_id for s in scores if s.status == "missing"]
    strict = [int(s.correct) for s in scored]
    out = {
        "n_items": len(scores),
        "n_scored": len(scored),
        "n_missing": len(missing),
        "complete": not missing,
        "missing_items": missing[:50],
        "overall": summarize(strict),
        "parse_failures": sum(1 for s in scored if s.method == "none"),
        "groups": {},
    }
    for g in GROUPS:
        buckets: dict[str, list[int]] = defaultdict(list)
        for s in scored:
            buckets[getattr(s, g)].append(int(s.correct))
        out["groups"][g] = {k: summarize(v, with_ci=(g == "category")) for k, v in sorted(buckets.items())}
    cats = [v["accuracy"] for v in out["groups"]["category"].values() if v["accuracy"] is not None]
    out["macro_category_accuracy"] = sum(cats) / len(cats) if cats else None

    shorts = [s for s in scored if s.answer_type == "short"]
    judged = [s for s in shorts if s.judge_correct is not None]
    if judged:
        out["judge"] = {
            "overall": summarize([int(s.correct_with_judge) for s in scored]),
            "short_answer_items": len(shorts),
            "judged_items": len(judged),
            "judge_flipped_to_correct": sum(1 for s in judged if s.judge_correct and not s.correct),
            "short_strict": summarize([int(s.correct) for s in shorts], with_ci=False),
            "short_with_judge": summarize([int(s.correct_with_judge) for s in shorts], with_ci=False),
        }
    return out


def chance_accuracy(items: list[dict]) -> float | None:
    """Expected accuracy of uniform random guessing (short answers count as 0)."""
    if not items:
        return None
    return sum((1 / len(it["options"])) if it["answer_type"] == "mcq" else 0.0 for it in items) / len(items)
