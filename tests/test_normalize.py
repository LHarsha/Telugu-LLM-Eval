import unicodedata

from telugu_eval.normalize import clean_text, detect_script, normalize_for_match


def test_nfc_composes_telugu_ai_vowel_sign():
    # U+0C48 (ై) is canonically equivalent to U+0C46 U+0C56.
    composed = "కై"
    decomposed = "కై"
    assert composed != decomposed
    assert normalize_for_match(decomposed) == normalize_for_match(composed)
    assert clean_text(decomposed) == unicodedata.normalize("NFC", composed)


def test_joiners_removed_for_match_but_kept_in_clean_text():
    with_zwnj = "క‌ష"
    assert "‌" in clean_text(with_zwnj)
    assert normalize_for_match(with_zwnj) == normalize_for_match("కష")
    assert normalize_for_match("మా‍మిడి") == "మామిడి"


def test_invisible_chars_removed_everywhere():
    s = "﻿మామిడి​­"
    assert clean_text(s) == "మామిడి"


def test_whitespace_and_punctuation():
    assert normalize_for_match("  మామిడి   కాయ.  ") == "మామిడి కాయ"
    assert normalize_for_match("రెండు, ఎకరాలు!") == "రెండు ఎకరాలు"
    # danda and double danda
    assert normalize_for_match("అవును।") == "అవును"
    assert normalize_for_match("అవును॥") == "అవును"


def test_clean_text_keeps_paragraphs():
    assert clean_text("line one  \r\n\r\n\r\n\tline two") == "line one\n\nline two"


def test_telugu_digits_and_case():
    assert normalize_for_match("౨ ఎకరాలు") == "2 ఎకరాలు"
    assert normalize_for_match("Mamidi") == normalize_for_match("MAMIDI") == "mamidi"


def test_detect_script():
    assert detect_script("బతుకమ్మ పండుగ") == "telugu"
    assert detect_script("Nenu repu office ki raanu") == "romanized"
    assert detect_script("నేను office కి వెళ్తున్నా") == "mixed"
    # a short acronym inside Telugu text is not code-mixing
    assert detect_script("ఈ రోజు TV లో వార్తలు చూశాను, చాలా ఆసక్తికరంగా ఉన్నాయి") == "telugu"
