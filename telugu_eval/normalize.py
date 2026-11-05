"""Unicode normalization for Telugu (and Romanized Telugu) text.

There are two levels:

* ``clean_text``: light and lossless in meaning. Used when *storing* authored
  text: NFC, strip invisible junk (ZWSP, BOM, word joiner, soft hyphen),
  normalise whitespace. ZWJ/ZWNJ are kept because an author may use them
  deliberately to control conjunct rendering.

* ``normalize_for_match``: aggressive. Used when *comparing* strings (duplicate
  detection now, answer scoring later). On top of ``clean_text`` it removes
  ZWJ/ZWNJ, maps Telugu digits to ASCII, drops punctuation (including the
  danda), casefolds Latin script and collapses all whitespace.

Why NFC matters for Telugu: the same visible text can be encoded in more than
one way. For example the vowel sign AI (U+0C48) is canonically equivalent to
E + AI length mark (U+0C46 U+0C56). Different keyboards / IMEs emit different
forms, so without NFC two visually identical answers would not match.
"""

from __future__ import annotations

import re
import unicodedata

# Invisible characters that never carry meaning in Telugu text.
_INVISIBLE = {
    "​",  # ZERO WIDTH SPACE
    "⁠",  # WORD JOINER
    "﻿",  # BOM / ZERO WIDTH NO-BREAK SPACE
    "­",  # SOFT HYPHEN
    "᠎",  # MONGOLIAN VOWEL SEPARATOR (shows up from some copy-paste)
}
# Joiners: affect glyph shaping (half forms / conjunct breaking) but not the
# underlying word, so they are removed for matching only.
_JOINERS = {"‌", "‍"}  # ZWNJ, ZWJ

_INVISIBLE_RE = re.compile("[" + "".join(_INVISIBLE) + "]")
_JOINER_RE = re.compile("[" + "".join(_JOINERS) + "]")
_WS_RE = re.compile(r"\s+")

# Telugu digits ౦..౯ (U+0C66..U+0C6F) -> 0..9
_TELUGU_DIGITS = {0x0C66 + i: str(i) for i in range(10)}

TELUGU_RANGE = (0x0C00, 0x0C7F)


def clean_text(text: str) -> str:
    """Normalise authored text for storage (meaning-preserving)."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = _INVISIBLE_RE.sub("", text)
    # Normalise line endings, trim trailing spaces per line, collapse runs of
    # spaces/tabs, but keep newlines (passages have paragraphs).
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_punct(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def normalize_for_match(text: str) -> str:
    """Aggressive normalisation for comparing two strings."""
    text = clean_text(text)
    text = _JOINER_RE.sub("", text)
    # NFC again: removing joiners can leave sequences that compose.
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TELUGU_DIGITS)
    # Danda (U+0964) and double danda (U+0965) are category Po, so they are
    # caught by the punctuation filter. Replace punctuation with a space so
    # "a,b" does not become "ab".
    text = "".join(" " if _is_punct(ch) else ch for ch in text)
    text = text.casefold()
    return _WS_RE.sub(" ", text).strip()


def script_profile(text: str) -> dict[str, int]:
    """Count Telugu vs Latin letters in ``text`` (used for script checks)."""
    telugu = latin = 0
    for ch in text:
        cp = ord(ch)
        if TELUGU_RANGE[0] <= cp <= TELUGU_RANGE[1]:
            if unicodedata.category(ch).startswith("L"):
                telugu += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    return {"telugu": telugu, "latin": latin}


def detect_script(text: str) -> str:
    """Best-guess script label: 'telugu', 'romanized' or 'mixed'.

    'mixed' means Telugu script with embedded Latin-script English words,
    e.g. "నేను office కి వెళ్తున్నా". A handful of Latin letters (an acronym,
    a number unit) inside Telugu text does not make it mixed.
    """
    p = script_profile(text)
    t, l = p["telugu"], p["latin"]
    if t == 0 and l == 0:
        return "telugu"
    if t == 0:
        return "romanized"
    if l == 0:
        return "telugu"
    # Both present. Require a meaningful share of Latin letters for "mixed".
    if l >= 4 and l / (t + l) >= 0.08:
        return "mixed"
    return "telugu"
