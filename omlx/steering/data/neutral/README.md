# Neutral quality corpus

Fixed corpus for the steering quality gate: per-token KL(unsteered ‖ steered)
and NLL change are measured on these paragraphs, two forwards per text.

- 200 paragraphs in four `para_*.txt` files, blank-line separated.
- `#` lines are comments and are skipped by the loader.
- **License:** the text is original, written for the oMLX project, and is
  distributed under the project license (Apache-2.0). No third-party text,
  no scraping, no attribution required beyond the license header.

The corpus deliberately avoids steering-related vocabulary (verbosity
instructions, persona prompts) so the measured KL reflects general
distribution shift, not topic overlap with the steering pairs.
