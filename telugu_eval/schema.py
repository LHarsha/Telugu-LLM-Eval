"""Dataset schema: field definitions and per-record checks.

One item = one JSON object per line in ``data/items.jsonl``. Reading
comprehension passages live in ``data/passages.jsonl`` and are referenced by
``passage_id`` so several questions can share one passage.

Item fields
-----------
id            str   "<prefix>-<NNN>", prefix by category: cul / cm / rc.
                    Example items use "ex-<prefix>-<N>".
category      enum  cultural | code_mixed | reading_comprehension
subtopic      str   free text, e.g. "festivals", "food", "chat", "news".
script        enum  telugu | romanized | mixed
region        enum  telangana | andhra | both | none
question      str   the question text (code-mixed items include the message
                    being asked about inside the question).
answer_type   enum  mcq | short
options       list  mcq only: 2-6 option strings (4 recommended), no "A)" prefix.
gold_option   str   mcq only: letter of the correct option ("A".."F").
gold_answers  list  short only: every acceptable answer (aliases, spellings,
                    scripts). Matched after normalisation.
passage_id    str   reading_comprehension only.
difficulty    enum  easy | medium | hard
source        str   reference backing the gold answer (URL, book, "common
                    knowledge"); strongly recommended for cultural facts.
author_note   str   free-form note (why this item, ambiguity, etc.).
author        str   who wrote the item.
status        enum  draft | reviewed | retired. Only "reviewed" is scored.
created       str   ISO date YYYY-MM-DD.
canary        str   the canary string; added automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from . import CANARY
from .normalize import detect_script, normalize_for_match, script_profile

CATEGORIES = ("cultural", "code_mixed", "reading_comprehension")
CATEGORY_PREFIX = {"cultural": "cul", "code_mixed": "cm", "reading_comprehension": "rc"}
PREFIX_CATEGORY = {v: k for k, v in CATEGORY_PREFIX.items()}
SCRIPTS = ("telugu", "romanized", "mixed")
REGIONS = ("telangana", "andhra", "both", "none")
ANSWER_TYPES = ("mcq", "short")
DIFFICULTIES = ("easy", "medium", "hard")
STATUSES = ("draft", "reviewed", "retired")
LETTERS = "ABCDEF"

REQUIRED_ITEM_FIELDS = (
    "id", "category", "subtopic", "script", "region", "question",
    "answer_type", "difficulty", "status", "author", "created", "canary",
)
OPTIONAL_ITEM_FIELDS = (
    "options", "gold_option", "gold_answers", "passage_id", "source", "author_note",
)
# Canonical key order when writing records (readable lines, clean diffs).
ALL_ITEM_FIELDS = (
    "id", "category", "subtopic", "script", "region", "passage_id", "question",
    "answer_type", "options", "gold_option", "gold_answers", "difficulty",
    "source", "author_note", "author", "status", "created", "canary",
)
assert set(ALL_ITEM_FIELDS) == set(REQUIRED_ITEM_FIELDS + OPTIONAL_ITEM_FIELDS)

REQUIRED_PASSAGE_FIELDS = ("passage_id", "title", "text", "script", "author", "created", "canary")
OPTIONAL_PASSAGE_FIELDS = ("source", "author_note")
ALL_PASSAGE_FIELDS = ("passage_id", "title", "script", "text", "source", "author_note", "author", "created", "canary")
assert set(ALL_PASSAGE_FIELDS) == set(REQUIRED_PASSAGE_FIELDS + OPTIONAL_PASSAGE_FIELDS)

ITEM_ID_RE = re.compile(r"^(cul|cm|rc)-(\d{3,})$")
EXAMPLE_ITEM_ID_RE = re.compile(r"^ex-(cul|cm|rc)-(\d+)$")
PASSAGE_ID_RE = re.compile(r"^p-(\d{3,})$")
EXAMPLE_PASSAGE_ID_RE = re.compile(r"^ex-p-(\d+)$")
# Options should not carry their own label; the prompt template adds letters.
OPTION_LABEL_RE = re.compile(r"^\s*\(?[A-Fa-f]\s*[\).:\-]\s+")


@dataclass
class Issue:
    level: str          # "error" | "warning"
    record_id: str
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper():7}] {self.record_id} :: {self.field}: {self.message}"


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)

    def error(self, rid: str, fld: str, msg: str) -> None:
        self.issues.append(Issue("error", rid, fld, msg))

    def warn(self, rid: str, fld: str, msg: str) -> None:
        self.issues.append(Issue("warning", rid, fld, msg))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    def extend(self, other: "Report") -> None:
        self.issues.extend(other.issues)


def _nonempty_str(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _check_enum(rep: Report, rid: str, rec: dict, name: str, allowed) -> bool:
    v = rec.get(name)
    if v not in allowed:
        rep.error(rid, name, f"{v!r} is not one of {list(allowed)}")
        return False
    return True


def _check_date(rep: Report, rid: str, v) -> None:
    try:
        date.fromisoformat(v)
    except (TypeError, ValueError):
        rep.error(rid, "created", f"{v!r} is not an ISO date (YYYY-MM-DD)")


def check_item(item: dict, *, example: bool = False, passage_ids: set[str] | None = None) -> Report:
    """Validate a single item in isolation (plus passage reference if given)."""
    rep = Report()
    if not isinstance(item, dict):
        rep.error("?", "-", "record is not a JSON object")
        return rep
    rid = str(item.get("id", "<no id>"))

    for f in REQUIRED_ITEM_FIELDS:
        v = item.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            rep.error(rid, f, "missing required field")
    for f in item:
        if f not in ALL_ITEM_FIELDS:
            rep.warn(rid, f, "unknown field (typo?)")
    for f in ("question", "subtopic", "source", "author_note", "author"):
        if f in item and item[f] is not None and not isinstance(item[f], str):
            rep.error(rid, f, "must be a string")

    # --- id ---------------------------------------------------------------
    id_re = EXAMPLE_ITEM_ID_RE if example else ITEM_ID_RE
    m = id_re.match(rid) if isinstance(item.get("id"), str) else None
    if not m:
        want = "ex-<cul|cm|rc>-<N>" if example else "<cul|cm|rc>-<NNN>"
        rep.error(rid, "id", f"bad id format, expected {want}")
    cat_ok = _check_enum(rep, rid, item, "category", CATEGORIES)
    if m and cat_ok and PREFIX_CATEGORY[m.group(1)] != item["category"]:
        rep.error(rid, "id", f"prefix '{m.group(1)}' does not match category '{item['category']}'")

    script_ok = _check_enum(rep, rid, item, "script", SCRIPTS)
    _check_enum(rep, rid, item, "region", REGIONS)
    at_ok = _check_enum(rep, rid, item, "answer_type", ANSWER_TYPES)
    _check_enum(rep, rid, item, "difficulty", DIFFICULTIES)
    _check_enum(rep, rid, item, "status", STATUSES)
    if "created" in item:
        _check_date(rep, rid, item["created"])
    if item.get("canary") is not None and item.get("canary") != CANARY:
        rep.error(rid, "canary", "canary string does not match the project canary")

    if example and item.get("status") == "reviewed":
        rep.error(rid, "status", "example items must not be 'reviewed' (they are never scored)")

    # --- answer -----------------------------------------------------------
    if at_ok:
        if item["answer_type"] == "mcq":
            _check_mcq(rep, rid, item)
        else:
            _check_short(rep, rid, item)

    # --- category-specific -----------------------------------------------
    if cat_ok:
        cat = item["category"]
        pid = item.get("passage_id")
        if cat == "reading_comprehension":
            if not _nonempty_str(pid):
                rep.error(rid, "passage_id", "required for reading_comprehension")
            elif passage_ids is not None and pid not in passage_ids:
                rep.error(rid, "passage_id", f"passage {pid!r} not found in passages file")
        elif pid not in (None, ""):
            rep.error(rid, "passage_id", "only reading_comprehension items take a passage")
        if cat == "cultural":
            if item.get("region") == "none":
                rep.warn(rid, "region", "cultural item with region 'none'; use telangana/andhra/both")
            if not _nonempty_str(item.get("source")):
                rep.warn(rid, "source", "no source for a cultural fact; add a reference if you can")

    # --- script consistency ----------------------------------------------
    if script_ok and _nonempty_str(item.get("question")):
        _check_script(rep, rid, item)

    # --- answer leakage ---------------------------------------------------
    q = normalize_for_match(item.get("question") or "")
    if item.get("answer_type") == "short" and isinstance(item.get("gold_answers"), list) and q:
        for g in item["gold_answers"]:
            ng = normalize_for_match(g) if isinstance(g, str) else ""
            if len(ng) >= 3 and f" {ng} " in f" {q} ":
                rep.warn(rid, "gold_answers", f"gold answer {g!r} appears verbatim in the question")
    return rep


def _check_mcq(rep: Report, rid: str, item: dict) -> None:
    opts = item.get("options")
    if not isinstance(opts, list) or not all(isinstance(o, str) for o in opts):
        rep.error(rid, "options", "mcq needs a list of option strings")
        return
    if not 2 <= len(opts) <= len(LETTERS):
        rep.error(rid, "options", f"mcq needs 2-{len(LETTERS)} options, got {len(opts)}")
        return
    if len(opts) != 4:
        rep.warn(rid, "options", f"{len(opts)} options; 4 is the v1 convention (random baseline 25%)")
    norm = []
    for i, o in enumerate(opts):
        if not o.strip():
            rep.error(rid, "options", f"option {LETTERS[i]} is empty")
        if OPTION_LABEL_RE.match(o):
            rep.warn(rid, "options", f"option {LETTERS[i]} looks pre-labelled ({o[:6]!r}); letters are added by the prompt")
        norm.append(normalize_for_match(o))
    seen: dict[str, int] = {}
    for i, n in enumerate(norm):
        if n and n in seen:
            rep.error(rid, "options", f"options {LETTERS[seen[n]]} and {LETTERS[i]} are duplicates after normalisation")
        seen.setdefault(n, i)
    g = item.get("gold_option")
    valid = tuple(LETTERS[: len(opts)])  # tuple, not str: "" in "ABCD" is True
    if not isinstance(g, str) or g not in valid:
        rep.error(rid, "gold_option", f"{g!r} must be one of {list(valid)}")
    if item.get("gold_answers"):
        rep.error(rid, "gold_answers", "mcq items use gold_option, not gold_answers")
    low = [n for n in norm if n in {"all of the above", "none of the above", "పైవన్నీ", "పైవేవీ కావు"}]
    if low:
        rep.warn(rid, "options", "'all/none of the above' options weaken MCQ items")


def _check_short(rep: Report, rid: str, item: dict) -> None:
    golds = item.get("gold_answers")
    if not isinstance(golds, list) or not golds or not all(_nonempty_str(g) for g in golds):
        rep.error(rid, "gold_answers", "short-answer items need a non-empty list of acceptable answers")
    else:
        norm = [normalize_for_match(g) for g in golds]
        if len(set(norm)) != len(norm):
            rep.warn(rid, "gold_answers", "some aliases are identical after normalisation (harmless, but redundant)")
        if any(len(g.split()) > 6 for g in norm):
            rep.warn(rid, "gold_answers", "long gold answer: exact match will rarely hit; consider MCQ or rely on the LLM judge")
    if item.get("options"):
        rep.error(rid, "options", "short-answer items must not have options")
    if item.get("gold_option"):
        rep.error(rid, "gold_option", "short-answer items use gold_answers, not gold_option")


def _check_script(rep: Report, rid: str, item: dict) -> None:
    declared = item["script"]
    text = item["question"]
    prof = script_profile(text)
    guessed = detect_script(text)
    if declared == "romanized" and prof["telugu"] > 0:
        rep.warn(rid, "script", "declared romanized but question contains Telugu script")
    elif declared == "telugu" and guessed == "mixed":
        rep.warn(rid, "script", "declared telugu but question has many Latin letters; is it 'mixed'?")
    elif declared == "mixed" and (prof["telugu"] == 0 or prof["latin"] == 0):
        rep.warn(rid, "script", "declared mixed but question uses only one script")
    if item.get("category") == "code_mixed" and declared == "telugu" and prof["latin"] == 0:
        rep.warn(rid, "script", "code_mixed item in pure Telugu script; English words written in Telugu script are fine, just double-check")


def check_passage(p: dict, *, example: bool = False) -> Report:
    rep = Report()
    if not isinstance(p, dict):
        rep.error("?", "-", "record is not a JSON object")
        return rep
    pid = str(p.get("passage_id", "<no id>"))
    for f in REQUIRED_PASSAGE_FIELDS:
        v = p.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            rep.error(pid, f, "missing required field")
    for f in p:
        if f not in ALL_PASSAGE_FIELDS:
            rep.warn(pid, f, "unknown field (typo?)")
    id_re = EXAMPLE_PASSAGE_ID_RE if example else PASSAGE_ID_RE
    if not (isinstance(p.get("passage_id"), str) and id_re.match(p["passage_id"])):
        rep.error(pid, "passage_id", "bad id format, expected " + ("ex-p-<N>" if example else "p-<NNN>"))
    _check_enum(rep, pid, p, "script", SCRIPTS)
    if "created" in p:
        _check_date(rep, pid, p["created"])
    if p.get("canary") is not None and p.get("canary") != CANARY:
        rep.error(pid, "canary", "canary string does not match the project canary")
    text = p.get("text")
    if isinstance(text, str):
        words = len(text.split())
        if words < 40:
            rep.warn(pid, "text", f"passage is short ({words} words)")
        elif words > 400:
            rep.warn(pid, "text", f"passage is long ({words} words); v1 aims for short passages")
    return rep
