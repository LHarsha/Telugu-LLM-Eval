# Results

| Model | Type | Overall (95% CI) | Cultural | Code-mixed | Reading comp. | n |
|---|---|---|---|---|---|---|
| Gemma 4 31B (open weights) | open-weights | 93.3 (83.3–100.0) | 90.0 | 100.0 | 90.0 | 30 |
| Gemini 3.5 Flash-Lite | api | 86.7 (73.3–96.7) | 70.0 | 100.0 | 90.0 | 30 |
| Random guess | baseline | 16.7 (6.7–30.0) | 10.0 | 10.0 | 30.0 | 30 |
| Gemini 3.6 Flash *(incomplete)* | api | 100.0 (100.0–100.0) | 100.0 | 100.0 | – | 4 |

Accuracy in %, strict scoring (exact match after normalisation). Split: `reviewed`: Only items marked reviewed by the author are scored.
30 items; chance level 17.5%. Prompt `p1`, dataset v1 (`2221af0982ee1e42`), generated 2026-11-05.
