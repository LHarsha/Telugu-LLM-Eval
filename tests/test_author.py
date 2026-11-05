from telugu_eval import CANARY, dataio
from telugu_eval.author.server import Store


def raw_item(**over):
    it = {"category": "cultural", "subtopic": "Festivals", "script": "telugu", "region": "telangana",
          "question": "  బతుకమ్మ పండుగను​ ఏ రాష్ట్రంలో జరుపుకుంటారు?  ", "answer_type": "mcq",
          "options": ["ఆంధ్రప్రదేశ్", "తెలంగాణ ", "కర్ణాటక", "తమిళనాడు"], "gold_option": "B",
          "gold_answers": ["stale value from switching answer type"],
          "difficulty": "easy", "source": "", "author_note": "", "author": "me"}
    it.update(over)
    return it


def test_save_assigns_id_cleans_and_fills_defaults(tmp_path):
    s = Store(tmp_path)
    item, rep = s.save_item(raw_item(), None)
    assert not rep.errors
    assert item["id"] == "cul-001"
    assert next(iter(item)) == "id"  # canonical key order on disk
    assert item["question"] == "బతుకమ్మ పండుగను ఏ రాష్ట్రంలో జరుపుకుంటారు?"
    assert item["options"][1] == "తెలంగాణ"
    assert item["subtopic"] == "festivals"
    assert item["canary"] == CANARY and item["status"] == "draft"
    assert "gold_answers" not in item and "source" not in item
    item2, _ = s.save_item(raw_item(question="ఆవకాయను ఏ కాయతో చేస్తారు?", answer_type="short",
                                    gold_answers=["మామిడి", ""], options=["x"]), None)
    assert item2["id"] == "cul-002" and item2["gold_answers"] == ["మామిడి"] and "options" not in item2
    assert len(dataio.read_jsonl(tmp_path / dataio.ITEMS_FILE).records) == 2


def test_save_refuses_errors_and_duplicates(tmp_path):
    s = Store(tmp_path)
    _, rep = s.save_item(raw_item(gold_option=""), None)
    assert rep.errors
    assert not (tmp_path / dataio.ITEMS_FILE).exists()
    s.save_item(raw_item(), None)
    _, rep = s.save_item(raw_item(), None)  # same question again
    assert any("duplicate of cul-001" in i.message for i in rep.errors)


def test_edit_in_place_and_id_is_immutable(tmp_path):
    s = Store(tmp_path)
    item, _ = s.save_item(raw_item(), None)
    edited, rep = s.save_item({**item, "status": "reviewed"}, "cul-001")
    assert not rep.errors and edited["status"] == "reviewed"
    recs = dataio.read_jsonl(tmp_path / dataio.ITEMS_FILE).records
    assert len(recs) == 1 and recs[0]["status"] == "reviewed"
    _, rep = s.save_item({**item, "id": "cul-009"}, "cul-001")
    assert rep.errors


def test_passage_save_and_rc_item(tmp_path):
    s = Store(tmp_path)
    p, rep = s.save_passage({"title": "కథ", "text": "ఒక ఊరిలో ఒక రైతు ఉండేవాడు. " * 10, "author": "me"}, None)
    assert not rep.errors and p["passage_id"] == "p-001" and p["script"] == "telugu"
    assert next(iter(p)) == "passage_id"
    item, rep = s.save_item(raw_item(category="reading_comprehension", region="none",
                                     passage_id="p-001", question="రైతు ఎక్కడ ఉండేవాడు?"), None)
    assert not rep.errors and item["id"] == "rc-001"
    _, rep = s.save_item(raw_item(category="reading_comprehension", region="none",
                                  passage_id="p-404", question="ఇంకో ప్రశ్న ఏమిటి?"), None)
    assert any(i.field == "passage_id" for i in rep.errors)
