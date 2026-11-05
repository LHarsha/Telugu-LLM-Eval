import json
from pathlib import Path

import pytest

from telugu_eval import CANARY, dataio
from telugu_eval.schema import check_item, check_passage
from telugu_eval.validate import validate_dataset

REPO_DATA = Path(__file__).resolve().parent.parent / "data"


def mcq(**over):
    it = {
        "id": "cul-001", "category": "cultural", "subtopic": "festivals", "script": "telugu",
        "region": "telangana", "question": "బతుకమ్మ పండుగను ఏ రాష్ట్రంలో జరుపుకుంటారు?",
        "answer_type": "mcq", "options": ["ఆంధ్రప్రదేశ్", "తెలంగాణ", "కర్ణాటక", "తమిళనాడు"],
        "gold_option": "B", "difficulty": "easy", "source": "x", "author": "t",
        "status": "reviewed", "created": "2026-11-05", "canary": CANARY,
    }
    it.update(over)
    return {k: v for k, v in it.items() if v is not None}


def short(**over):
    it = mcq(id="cul-002", answer_type="short", options=None, gold_option=None,
             question="ఆవకాయను ఏ కాయతో చేస్తారు?", gold_answers=["మామిడి", "mamidi"])
    it.update(over)
    return {k: v for k, v in it.items() if v is not None}


def fields(rep, level="error"):
    return {i.field for i in rep.issues if i.level == level}


def test_valid_items_pass():
    assert not check_item(mcq()).errors
    assert not check_item(short()).errors


@pytest.mark.parametrize("missing", ["question", "category", "difficulty", "canary", "subtopic"])
def test_missing_required_field(missing):
    it = mcq()
    del it[missing]
    assert missing in fields(check_item(it))


def test_blank_string_counts_as_missing():
    assert "question" in fields(check_item(mcq(question="   ")))


def test_bad_enum_and_id():
    assert "script" in fields(check_item(mcq(script="latin")))
    assert "id" in fields(check_item(mcq(id="cultural-1")))
    assert "id" in fields(check_item(mcq(id="cm-001")))  # prefix vs category


def test_mcq_option_checks():
    assert "gold_option" in fields(check_item(mcq(gold_option="E")))
    assert "gold_option" in fields(check_item(mcq(gold_option=None)))
    assert "options" in fields(check_item(mcq(options=["ఒకటి"])))
    assert "options" in fields(check_item(mcq(options=["తెలంగాణ", "తెలంగాణ.", "x", "y"])))  # dup after norm
    assert "options" in fields(check_item(mcq(options=["a", "", "c", "d"])))
    assert "options" in fields(check_item(mcq(options=["A) ఒకటి", "b", "c", "d"])), "warning")
    assert "options" in fields(check_item(mcq(options=["a", "b", "c"], gold_option="A")), "warning")


def test_short_answer_checks():
    assert "gold_answers" in fields(check_item(short(gold_answers=[])))
    assert "gold_answers" in fields(check_item(short(gold_answers=["  "])))
    assert "options" in fields(check_item(short(options=["a", "b"])))
    leak = short(question="మామిడి కాయతో ఏ పచ్చడి చేస్తారు?", gold_answers=["మామిడి"])
    assert "gold_answers" in fields(check_item(leak), "warning")


def test_reading_comprehension_needs_known_passage():
    rc = mcq(id="rc-001", category="reading_comprehension", region="none")
    assert "passage_id" in fields(check_item(rc))
    rc["passage_id"] = "p-001"
    assert "passage_id" in fields(check_item(rc, passage_ids={"p-002"}))
    assert not check_item(rc, passage_ids={"p-001"}).errors
    assert "passage_id" in fields(check_item(mcq(passage_id="p-001")))  # non-RC with passage


def test_script_consistency_warnings():
    assert "script" in fields(check_item(mcq(script="romanized")), "warning")
    cm = mcq(id="cm-001", category="code_mixed", script="telugu", region="none",
             question="నేను office కి వెళ్తున్నా, meeting late అవుతుంది. ఎందుకు?")
    assert "script" in fields(check_item(cm), "warning")


def test_examples_must_not_be_reviewed():
    ex = mcq(id="ex-cul-1")
    assert "status" in fields(check_item(ex, example=True))
    assert not check_item(mcq(id="ex-cul-1", status="draft"), example=True).errors
    assert "id" in fields(check_item(mcq(id="cul-001", status="draft"), example=True))


def test_wrong_canary_rejected():
    assert "canary" in fields(check_item(mcq(canary="something else")))


def test_passage_checks():
    p = {"passage_id": "p-001", "title": "t", "text": "పదం " * 50, "script": "telugu",
         "author": "t", "created": "2026-11-05", "canary": CANARY}
    assert not check_passage(p).errors
    assert "passage_id" in fields(check_passage({**p, "passage_id": "passage1"}))
    assert "text" in fields(check_passage({**p, "text": "చిన్న"}), "warning")


# --- dataset-level --------------------------------------------------------

def write(tmp: Path, items, passages=(), examples=()):
    dataio.write_jsonl(tmp / dataio.ITEMS_FILE, list(items))
    dataio.write_jsonl(tmp / dataio.PASSAGES_FILE, list(passages))
    dataio.write_jsonl(tmp / dataio.EXAMPLES_FILE, list(examples))


def test_duplicate_ids_and_questions(tmp_path):
    a = mcq()
    b = mcq(question=a["question"] + " ")  # same id, same question after norm
    write(tmp_path, [a, b])
    rep = validate_dataset(tmp_path).report
    msgs = " ".join(i.message for i in rep.errors)
    assert "duplicate item id" in msgs
    assert "duplicate of" in msgs


def test_near_duplicate_warning(tmp_path):
    a = mcq()
    b = mcq(id="cul-002", question="బతుకమ్మ పండుగను ఏ రాష్ట్రంలో జరుపుకుంటారు ?!")
    c = mcq(id="cul-003", question="బతుకమ్మ పండుగని ఏ రాష్ట్రంలో జరుపుకుంటారు?")
    write(tmp_path, [a, b, c])
    rep = validate_dataset(tmp_path).report
    assert any("duplicate of cul-001" in i.message for i in rep.errors)       # b: exact after norm
    assert any("near-duplicate" in i.message for i in rep.warnings)            # c: one letter changed


def test_retired_items_ignored_for_duplicates(tmp_path):
    write(tmp_path, [mcq(status="retired"), mcq(id="cul-002")])
    assert not validate_dataset(tmp_path).report.errors


def test_same_question_on_different_passages_is_fine(tmp_path):
    P = {"title": "t", "text": "పదం " * 50, "script": "telugu", "author": "t", "created": "2026-11-05", "canary": CANARY}
    ps = [{"passage_id": "p-001", **P}, {"passage_id": "p-002", **P}]
    rc1 = mcq(id="rc-001", category="reading_comprehension", region="none", passage_id="p-001", question="కథలో ముఖ్య పాత్ర ఎవరు?")
    rc2 = {**rc1, "id": "rc-002", "passage_id": "p-002"}
    write(tmp_path, [rc1, rc2], ps)
    assert not validate_dataset(tmp_path).report.errors


def test_gold_letter_imbalance_warning(tmp_path):
    items = [mcq(id=f"cul-{n:03d}", question=f"ప్రశ్న సంఖ్య {n} గురించి వేరే విషయం {'అ' * n}") for n in range(1, 11)]
    write(tmp_path, items)
    rep = validate_dataset(tmp_path).report
    assert any(i.field == "gold_option" and i.record_id == "dataset" for i in rep.warnings)


def test_invalid_json_line_reported(tmp_path):
    (tmp_path / dataio.ITEMS_FILE).write_text(json.dumps(mcq(), ensure_ascii=False) + "\n{not json\n", encoding="utf-8")
    out = validate_dataset(tmp_path)
    assert len(out.parse_errors) == 1 and out.parse_errors[0].line == 2
    assert not out.ok


def test_example_leaking_into_items_is_error(tmp_path):
    write(tmp_path, [mcq()], examples=[mcq(id="ex-cul-1", status="draft")])
    assert any("example item duplicated" in i.message for i in validate_dataset(tmp_path).report.errors)


def test_repo_dataset_is_valid():
    out = validate_dataset(REPO_DATA)
    assert out.ok, "\n".join(map(str, out.parse_errors + out.report.errors))
