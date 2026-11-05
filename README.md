# Roche AI PoC: Material Availability – Insights Lab (NLP on Planner Comments)

## 🎯 Project Overview
This project aims to **extract actionable insights** for Roche’s planner comments using **Natural Language Processing (NLP)**. The goal is to:
- **Quantify recurring issues** (e.g., "50% of delays for Supplier X are due to capacity issues").
- **Identify patterns** in comments (e.g., "urgent" + "Supplier Y" often leads to Stock Out).
- **Provide filters** in a Streamlit dashboard to analyze comments by **supplier, material, or root cause**.
- **Anticipate impacts**: find the early comments and contexts after which a Production / Market impact comes more often (weak signals).

## 📊 Current Status (29 October 2026)
- **Comments file** (`data/cleaned_comments.csv`, pipeline steps 1-7): the comments of the 174 daily snapshots of `Snapshot_2026`, one row per (material, vendor, comment) with how long it stayed visible (`First_Seen`, `Last_Seen`, `Days_Active`). A comment cell is a **journal**: its dated entries are split, the signatures (staff logins) removed and the entries kept in the cell, separated by ` ¦▸ `. Each step is one function and one notebook cell, so its result can be checked on its own.
- **Abbreviation library** (`lexicon/abbreviations.csv`): German and English abbreviations of the comments, reviewed by Matthieu and Alexis on 28.10; the 114 certain ones are written out (`Comment_Expanded`) before translation; names (suppliers, instruments, SAP codes) and ambiguous ones stay as written.
- **Translation**: German entries translated to English with DeepL (`Comment_EN`), after pseudonymisation; a jargon glossary (`lexicon/glossary_de_en.csv`, e.g. *Vorzug* → pull-in, *Orderbuch* → source list) and a cache (each entry paid for once).
- **Tracking table** (`data/item_tracking.csv`): one row per day and item from **all** snapshot rows (also without comment), for the trends, the impacts and the weak signals.
- **Streamlit dashboard "Insights Lab"** (`scripts/app.py`): one page, a menu over the full width, a password page when shared. Sections: **Overview** (what the dashboard shows today: all the order lines of the last snapshot, a material counted once; status tiles, clickable top root causes that set the Root Cause filter), **Most Frequent Comments** (similar wordings grouped, or the comments one by one with a column picker), **Root Cause Trends** (all snapshots, what changed between two weeks), **Impact by Root Cause**, **Watchlist** (open items at risk; the first entries, the context and the weights behind it are in the statistical details for analysts), **Smart Analysis** (two pages: the analysis and the key statistics it uses; LM Studio on this Mac, Mistral API pseudonymised once deployed; copy / Word / text export). Sidebar filters (MRP controller, purchasing group, vendor, material, stakeholder, root cause, impact, timeframe) apply everywhere; saved views; PNG export of every chart; explanations and statistics folded for the analysts.
- **Online access** (Streamlit Community Cloud): testers open the dashboard at a fixed link, behind a password; the data is pushed to GitHub only encrypted (`deploy/data.enc`).
- **Tests**: 65 pytest tests on synthetic data, no network.

## 🛠️ Tools & Libraries
Versions installed in `venv` (Python 3.12.4); minimum versions in `requirements.txt`.

| Tool / library | Version | Purpose |
|---|---|---|
| Python | 3.12.4 | Language of the whole project |
| pandas (+ numpy) | 2.2.3 (1.26.4) | Cleaning pipeline, statistics, tables |
| langdetect | 1.0.9 | Language of every journal entry (DE / EN) |
| deepl (DeepL API) | 1.32.0 | Translation DE → EN, glossary, usage |
| openpyxl | 3.1.5 | Excel sheet of the abbreviations for the planners |
| scikit-learn | 1.4.2 | Grouping of similar comments (TF-IDF), weak-signal model (logistic regression, cross-validation, MCC) |
| SciPy | 1.17.1 | Chi-square and Fisher tests |
| sentence-transformers (+ PyTorch, transformers) | 3.4.1 (2.14.1, 4.57.6) | Grouping by meaning: local model all-MiniLM-L6-v2 |
| openai (client) | 3.26.0 | Mistral API (cloud) and LM Studio (local), both OpenAI-compatible |
| Mistral API / LM Studio | ministral-14b-latest / Mistral 7B | Smart Analysis: deployed app / on this Mac (localhost) |
| Streamlit (+ Altair) | 1.66.1 (5.5.0) | Dashboard; clickable charts of the Overview |
| streamlit-sortables | 0.3.1 | Drag and drop of the table columns |
| matplotlib | 3.8.4 | Charts of the dashboard (PNG) |
| seaborn | 0.13.2 | Heatmap of a notebook exploration |
| spaCy | 3.8.16 | Notebook explorations (lemmas, keywords) |
| python-dotenv | 1.2.4 | Keys and password from `.env` |
| Jupyter | 1.1.1 | Notebooks: one cell per pipeline step |
| pytest | 9.1.1 | Tests (synthetic data, fake API clients) |
| Streamlit Community Cloud | free plan | Permanent hosting of the dashboard (data encrypted in the repository, `cryptography` Fernet) |
| Git / GitHub | — | Versioning (code and documentation only) |

*(See [docs/tools_justification.md](docs/tools_justification.md) for why each tool was chosen and its limits.)*

## 📁 Project Structure

```
Roche_AI_PoC/
├── data/                        # Confidential, never pushed (.gitignore)
│   ├── Snapshot_2026/           # Raw daily snapshots
│   ├── interim/                 # Translation cache, DeepL glossary id, embedding cache, saved views, review sheets
│   ├── cleaned_comments.csv     # Cleaned, expanded and translated comments
│   ├── item_tracking.csv        # One row per (day, material, vendor) from all snapshot rows
│   └── current_snapshot.csv     # All the order lines of the last snapshot day (Overview)
├── lexicon/
│   ├── abbreviations.csv        # Abbreviation library (DE / EN), reviewed 28.10
│   └── glossary_de_en.csv       # DeepL glossary of German jargon
├── scripts/                     # Python modules and notebooks (flat: the notebooks import the modules)
│   ├── data_cleaning.py         # Pipeline steps 1-6 (+ save with Comment_EN from the cache)
│   ├── entries.py               # Separator ' ¦▸ ' between the journal entries of a cell
│   ├── abbreviations.py         # Abbreviation library: expansion, checks, Excel sheet
│   ├── translation.py           # Step 7: pseudonymise + DeepL entry by entry, glossary, cache
│   ├── glossary.py              # DeepL glossary: checks, creation / reuse, forced terms
│   ├── pseudonymise.py          # Hides names, IDs, quantities, prices before DeepL / Mistral
│   ├── tracking.py              # Builds data/item_tracking.csv and data/current_snapshot.csv
│   ├── stats_engine.py          # Statistics of the dashboard and of the Smart Analysis
│   ├── insights.py              # Grouping, trends, root cause × impact, weak signals (analyses)
│   ├── insights_pages.py        # Sections Most Frequent Comments, Trends, Impact, Watchlist (Streamlit)
│   ├── llm_summary.py           # Smart Analysis: prompt, Mistral API, LM Studio
│   ├── app.py                   # Streamlit dashboard
│   ├── ui.py                    # Look, menu, switches, charts, tables, loading screen
│   ├── config.py                # Keys and password from .env (Streamlit secrets once deployed)
│   ├── data_bundle.py           # Encrypted copy of the data for the deployed app (deploy/data.enc)
│   ├── requirements.txt         # Dependencies of the deployed dashboard (Streamlit Cloud)
│   └── *.ipynb                  # data_cleaning, translation, llm_summary: step by step
├── tests/                       # pytest (65 tests)
├── docs/                        # project_steps.md, project_structure.md, tools_justification.md
├── assets/, .streamlit/         # Roche logo, favicon, theme
├── deploy/data.enc              # Data of the deployed app, encrypted (key DATA_KEY: .env + Streamlit secrets)
├── requirements.txt
└── .env / .env.example          # Keys and password (.env never pushed)
```
Every file is described in [docs/project_structure.md](docs/project_structure.md); the method, step by step, in [docs/project_steps.md](docs/project_steps.md).

## 🚀 How to Run
1. **Set up the environment**:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -m spacy download de_core_news_sm

   cp .env.example .env    # then fill the keys below
   ```
   | `.env` line | Used for |
   |---|---|
   | `DEEPL_API_KEY` | Translation (step 2: `translation.py`) |
   | `MISTRAL_API_KEY`, `MISTRAL_MODEL` | Smart Analysis of the deployed app ([console.mistral.ai](https://console.mistral.ai) → API Keys; default model `ministral-14b-latest`) |
   | `LM_STUDIO_BASE_URL`, `LM_STUDIO_MODEL` | Smart Analysis on this Mac (localhost): LM Studio server started, model loaded |
   | `APP_PASSWORD` | Password page (read again at every login: a change applies without restarting); leave empty for local use |

2. **Build and translate the comments** (after adding new snapshots to `data/Snapshot_2026/`):
   ```bash
   python scripts/data_cleaning.py
   python scripts/tracking.py                # tracking table of all snapshot rows (trends, impacts, watchlist) + last snapshot (Overview)
   python scripts/abbreviations.py --check --sheet    # validates the library, writes the Excel sheet for the planners (data/interim)
   python scripts/glossary.py --check        # validates the DeepL glossary
   python scripts/translation.py --dry-run   # characters to send to DeepL + cached translations made without the glossary
   # python scripts/translation.py --refresh-glossary   # (optional) also redo those, costs characters
   python scripts/translation.py
   ```
   The same steps can be run and checked one by one in `scripts/data_cleaning.ipynb` and `scripts/translation.ipynb` (see [docs/project_steps.md](docs/project_steps.md)). The rebuild already fills `Comment_EN` from the translation cache; `translation.py` is only needed for new German entries (the rebuild prints their number).

3. **Launch the Streamlit dashboard** (only on this Mac):
   ```bash
   venv/bin/streamlit run scripts/app.py
   ```
   Use the Streamlit of the project environment (`venv/bin/streamlit`, or `streamlit` after `source venv/bin/activate`): another Python (e.g. Anaconda) lacks some packages, such as `streamlit-sortables` for the column order. The dashboard opens at http://localhost:8501. If `APP_PASSWORD` is set in `.env`, the password page also appears locally.

4. **Put the dashboard online** (Streamlit Community Cloud: fixed link, this Mac can be off; Marco agreed to host the data there):
   - The data goes to GitHub **encrypted only**: `deploy/data.enc` (built by `python scripts/data_bundle.py`, which creates `DATA_KEY` in `.env` the first time). After a new run of the pipeline: rebuild it, commit and push `deploy/data.enc`; the app restarts by itself.
   - Once, on [share.streamlit.io](https://share.streamlit.io) (sign in with GitHub): *Create app* → repository `MatthieuSNR/Roche_AI_PoC`, branch `main`, main file `scripts/app.py`; *Advanced settings* → the secrets `APP_PASSWORD`, `DATA_KEY` (the complete value from `.env`) and, for the cloud AI summary, `MISTRAL_API_KEY`.
   - Send the testers the `….streamlit.app` link and the password. A new password in the secrets applies within a minute.
   - The Cloud installs `scripts/requirements.txt` (dashboard only, CPU PyTorch), not the root file. Without `APP_PASSWORD` the deployed app refuses to open.

5. **Run the tests**:
   ```bash
   venv/bin/python -m pytest -q tests
   ```
   65 tests, synthetic data only, no network and no key needed.












