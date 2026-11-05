# Contributing

## Adding benchmark items

Items must be **written directly in Telugu** (or natural Telugu–English code-mixing). Please do not translate items from English datasets. If you used an AI model to draft an item, say so in `author_note`; natively written items are preferred, and every item is verified by a native speaker before it is marked `reviewed`.

1. Fork the repo and start the authoring form:
   ```bash
   python -m telugu_eval author
   ```
   It opens at `http://127.0.0.1:8765`, saves to `data/items.jsonl`, and checks each item as you type.
2. Guidelines:
   - **One clear correct answer.** If two native speakers could reasonably disagree, rewrite the item or explain the reasoning in `author_note`.
   - **Cultural facts need a `source`** (a URL or reference) unless they are genuinely common knowledge.
   - **Short answers: list every acceptable form** in `gold_answers`: spelling variants, Telugu script, Romanized and English.
   - **Multiple choice: four plausible options**, correct answer placed at varied positions, no "all of the above".
   - **Code-mixed items should read like real messages**: how people actually type on WhatsApp or social media, in Telugu script, Romanized or mixed.
   - **Reading-comprehension passages should be original** (your own writing) or clearly licensed; note the source.
   - Use the difficulty rubric in [`data/README.md`](data/README.md).
   - Leave new items as `draft`.
3. Run the checks:
   ```bash
   python -m telugu_eval validate
   python -m pytest
   ```
4. Open a pull request describing what you added (capability, number of items, region). A maintainer reviews each item; reviewed items are switched to `reviewed` and included in scoring. Contributors are credited in the dataset (`author` field).

## Adding a model

See "Add a model" in the README. Only free tiers or local models, please, so anyone can reproduce the leaderboard at no cost. Commit the resulting `runs/<model>/` folder so the raw outputs are public.

## Code changes

Keep the tooling dependency-free (standard library only, plus pytest for tests) and add tests for anything that affects scoring.
