# Handover: where to adapt the Insights Lab for use at Roche

Line numbers refer to commit `19b01076` (branches `main` and `feat/comment-entries`); search for the function names if the
files have changed since. The prototype was built and tested without access to Roche systems: everything below
is what has to be connected or replaced on the Roche side.

## 1. The two external API calls

| Service | Used for | File and place | What to change |
|---|---|---|---|
| **DeepL** | German to English translation of the comments (`scripts/translation.py`) | `translate_missing()`, lines 136-185: translator created at lines 149-150, call `translator.translate_text(...)` at lines 167-170 (`source_lang="DE"`, `target_lang="EN-US"`, `tag_handling="xml"`, `ignore_tags=["keep"]`). Key: `DEEPL_API_KEY`, `scripts/config.py` line 23. Glossary: `scripts/glossary.py` (`ensure_glossary`) and `lexicon/glossary_de_en.csv`. | To use an approved translation service, pass another object as `translator` (the tests already inject a fake one, `tests/test_glossary.py`). It must accept a list of texts, keep the `<keep>...</keep>` placeholders untouched and return objects with a `.text` attribute. The DeepL glossary calls are specific to DeepL: without a glossary the translation still runs (see the `try/except` around line 154). |
| **Mistral API** (cloud) | Writing of the Smart Analysis (`scripts/llm_summary.py`) | `get_cloud_client()`, lines 354-356; `stream_summary_cloud()`, lines 364-376 with the call `chat.completions.create(...)` at lines 373-374. Settings: `MISTRAL_BASE_URL` (line 33), `MISTRAL_MODEL` (line 34, default `ministral-14b-latest`), `MAX_TOKENS` (line 35), temperature 0.2. Key: `MISTRAL_API_KEY`. | The client is the OpenAI-compatible `openai` library. If the approved service is OpenAI-compatible, set `MISTRAL_BASE_URL`, `MISTRAL_MODEL` and `MISTRAL_API_KEY` (environment or secrets) and nothing else changes. Otherwise adapt `get_cloud_client()` only. |
| **LM Studio** (local) | Same analysis on a local model, no data leaves the machine | `get_client()`, lines 41-43; `stream_summary()` from line 339. | A local or internal model server can be used by setting `LM_STUDIO_BASE_URL` and `LM_STUDIO_MODEL`. |

Which of the two models is used is decided in `scripts/app.py` (comment at lines 641-643, `render_ai_summary` (line 649),
call at line 800): the deployed copy (no `.env` file, `config.ON_CLOUD` in `scripts/config.py` line 11) uses the
cloud model with pseudonymised names; a local run uses LM Studio. On an internal server this switch should be
replaced by an explicit setting.

**Pseudonymisation before every external call**: `scripts/pseudonymise.py` replaces vendors, materials, MRP
controllers, purchasing groups, e-mails, persons, quantities and prices by placeholders (`VENDOR_1`,
`MATERIAL_2`...) and restores them in the answer (`PromptPseudonymiser`, line 178; template built in
`scripts/app.py`, `pseudonymiser_template`, line 644). Known limits: abbreviated or misspelt vendor names and person
names without an e-mail address are not detected. Please review the patterns against the Roche data policy.

## 2. Data input (the place to connect Snowflake)

At the moment the pipeline reads weekly CSV exports from `data/Snapshot_2026/`:

- `scripts/data_cleaning.py`, `SNAPSHOT_DIR` (line 52) and `merge_snapshots()` (line 167): the only function that
  reads the raw files (`glob` on `*.csv`, line 172). Replacing it with a query that returns the same columns, one row
  per order line and snapshot day, is the integration point; the steps after it (`keep_commented_rows`,
  `split_journal`, `deduplicate_comments`, language detection, abbreviations) work on the resulting DataFrame.
- `scripts/tracking.py` then builds `data/item_tracking.csv` and `data/current_snapshot.csv` (paths at lines 36-37).
- Output files published to the dashboard: see `FILES` in `scripts/data_bundle.py` (line 20).
- The expected columns are those of the Material Availability Dashboard export; the exact list is easiest to read in
  `merge_snapshots()` and `tracking.py` (module docstring).

## 3. Hosting and access (Streamlit Community Cloud specific)

- Password page: `scripts/app.py` (around line 50), `APP_PASSWORD` in `scripts/config.py`. To be replaced by Roche single sign-on.
- Encrypted data file `deploy/data.enc` (Fernet, key `DATA_KEY`): `scripts/data_bundle.py`. Needed only because the
  prototype data sits in a public repository; not needed on an internal server.
- Secrets are read from `.env` or from the Streamlit secrets (`scripts/config.py`, lines 15-21).
- The interface is Streamlit (`scripts/app.py`, `scripts/insights_pages.py`, `scripts/ui.py`). The analysis logic
  (`insights.py`, `stats_engine.py`, `tracking.py`, `llm_summary.py`) does not depend on it and can be wrapped in an
  API (for example FastAPI) for an R Shiny page.

## 4. Tests and limits

- `venv/bin/python -m pytest -q tests`: synthetic data, fake API clients, no network, no key, no Roche data.
- Not evaluated formally: the translation, the grouping of comments and the AI texts. The Watchlist model is a
  logistic regression trained on 126 items with a later impact, validated by random cross-validation (no temporal
  hold-out); its scores are a ranking, not a probability.
- Real comments, data files and API keys must never be pushed: the repository is public, the data reaches it only
  encrypted (`deploy/data.enc`) and the keys stay in `.env` or in the Streamlit secrets.
