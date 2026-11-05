"""The fixed prompt template.

Every model gets exactly the same prompt for an item: one user message, no
system prompt (some models served through the Gemini API, e.g. Gemma, do not
accept system instructions, so a single user turn keeps all models equal).

Instructions are in English; the item content (passage, question, options) is
Telugu / Romanized / mixed exactly as authored. Answers are requested in an
"Answer: ..." line so the scorer can extract them even from chatty models.

If you change anything here, bump PROMPT_VERSION. The template hash is stored
with every run, so results produced with different prompts are never mixed.
"""

from __future__ import annotations

import hashlib

from .schema import LETTERS

PROMPT_VERSION = "p1"

_HEADER = "You are taking a test of Telugu language understanding."

_PASSAGE = "Read the passage below and answer the question about it.\n\nPassage:\n{passage}\n"

_MCQ = (
    "{header}\n\n"
    "{passage_block}"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n\n"
    "Choose the single best option. Reply with one line in exactly this format:\n"
    "Answer: <letter>\n"
    "where <letter> is one of {letters}. Do not explain."
)

_SHORT = (
    "{header}\n\n"
    "{passage_block}"
    "Question:\n{question}\n\n"
    "Give a short answer (a word or a short phrase). Reply with one line in exactly this format:\n"
    "Answer: <your answer>\n"
    "Do not explain."
)


def template_hash() -> str:
    """Fingerprint of the template text (stored in run metadata)."""
    h = hashlib.sha256()
    for part in (PROMPT_VERSION, _HEADER, _PASSAGE, _MCQ, _SHORT):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def build_prompt(item: dict, passage_text: str | None = None) -> str:
    passage_block = ""
    if item.get("category") == "reading_comprehension":
        if not passage_text:
            raise ValueError(f"{item.get('id')}: reading comprehension item without passage text")
        passage_block = _PASSAGE.format(passage=passage_text.strip()) + "\n"
    if item["answer_type"] == "mcq":
        opts = item["options"]
        letters = LETTERS[: len(opts)]
        options = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
        letter_list = ", ".join(letters[:-1]) + f" or {letters[-1]}"
        return _MCQ.format(header=_HEADER, passage_block=passage_block, question=item["question"].strip(),
                           options=options, letters=letter_list)
    return _SHORT.format(header=_HEADER, passage_block=passage_block, question=item["question"].strip())
