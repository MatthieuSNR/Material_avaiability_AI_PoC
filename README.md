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
- **Streamlit dashboard "Insights Lab"** (`scripts/app.py`): one page, a menu over the full width, a password page when shared. Sections: **Overview** (material status tiles, top root causes with stakeholder options), **Filtered Comments** (column picker), **Most Frequent Comments** (similar wordings grouped), **Root Cause Trends** (all snapshots, what changed between two weeks), **Key Statistics**, **Impact by Root Cause**, **Weak Signals** (phrases, context, watch list), **Smart Analysis** (Mistral API, pseudonymised; copy / Word / text export). Sidebar filters (MRP controller, purchasing group, vendor, material, root cause, impact, timeframe) apply everywhere; saved views; PNG export of every chart; explanations and statistics folded for the analysts.
- **Remote access** (`share.sh`): testers open the dashboard in their own browser through an ngrok tunnel or a Cloudflare tunnel (for networks that block ngrok, e.g. Roche), behind a password; the data stays on this Mac.
- **Tests**: 64 pytest tests on synthetic data, no network.

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
| Mistral API / LM Studio | ministral-14b-latest / Mistral 7B | Smart Analysis (cloud) / local backup |
| Streamlit | 1.32.0 | Dashboard |
| streamlit-sortables | 0.3.1 | Drag and drop of the table columns |
| matplotlib | 3.8.4 | Charts of the dashboard (PNG) |
| seaborn | 0.13.2 | Heatmap of a notebook exploration |
| spaCy | 3.8.16 | Notebook explorations (lemmas, keywords) |
| python-dotenv | 1.2.4 | Keys and password from `.env` |
| Jupyter | 1.1.1 | Notebooks: one cell per pipeline step |
| pytest | 9.1.1 | Tests (synthetic data, fake API clients) |
| ngrok | 3.20.0 (free plan) | Tunnel for the remote usability tests |
| cloudflared | 2026.10.7 (quick tunnel, no account) | Second tunnel, for networks that block ngrok (e.g. Roche) |
| Git / GitHub | — | Versioning (code and documentation only) |

*(See [docs/tools_justification.md](docs/tools_justification.md) for why each tool was chosen and its limits.)*

## 📁 Project Structure

```
Roche_AI_PoC/
├── data/                        # Confidential, never pushed (.gitignore)
│   ├── Snapshot_2026/           # Raw daily snapshots
│   ├── interim/                 # Translation cache, DeepL glossary id, embedding cache, saved views, review sheets
│   ├── cleaned_comments.csv     # Cleaned, expanded and translated comments
│   └── item_tracking.csv        # One row per (day, material, vendor) from all snapshot rows
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
│   ├── tracking.py              # Builds data/item_tracking.csv
│   ├── stats_engine.py          # Statistics of the dashboard and of the Smart Analysis
│   ├── insights.py              # Grouping, trends, root cause × impact, weak signals (analyses)
│   ├── insights_pages.py        # Sections Most Frequent Comments, Trends, Impact, Weak Signals (Streamlit)
│   ├── llm_summary.py           # Smart Analysis: prompt, Mistral API, LM Studio
│   ├── app.py                   # Streamlit dashboard
│   ├── ui.py                    # Look, menu, switches, charts, tables, loading screen
│   ├── config.py                # Keys and password from .env
│   └── *.ipynb                  # data_cleaning, translation, llm_summary: step by step
├── tests/                       # pytest (64 tests)
├── docs/                        # project_steps.md, project_structure.md, tools_justification.md
├── assets/, .streamlit/         # Roche logo, favicon, theme
├── share.sh                     # Shares the dashboard through ngrok + Cloudflare (usability tests)
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
   | `MISTRAL_API_KEY`, `MISTRAL_MODEL` | Cloud AI summary ([console.mistral.ai](https://console.mistral.ai) → API Keys; default model `ministral-14b-latest`) |
   | `LM_STUDIO_BASE_URL`, `LM_STUDIO_MODEL` | Local AI summary (LM Studio server started) |
   | `APP_PASSWORD` | Password page (read again at every login: a change applies without restarting); leave empty for local use |

2. **Build and translate the comments** (after adding new snapshots to `data/Snapshot_2026/`):
   ```bash
   python scripts/data_cleaning.py
   python scripts/tracking.py                # tracking table of all snapshot rows (trends, impacts, weak signals)
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

4. **Share the dashboard for a usability test** (testers in their own browser):
   ```bash
   ./share.sh
   ```
   - Copy the `Forwarding https://….ngrok-free.app` line and send this link + the password (`grep APP_PASSWORD .env`) to the tester.
   - The tester clicks **Visit Site** on the ngrok page, then enters the password.
   - **Network that blocks ngrok (e.g. Roche)**: send instead the Cloudflare link printed before the ngrok screen (`https://….trycloudflare.com`, copied to the clipboard and in `outputs/cloudflare_link.txt`); no warning page, straight to the password. It changes at each start. Needs `brew install cloudflared` once.
   - Keep the terminal open and the Mac awake and online during the test; **Ctrl+C** stops the sharing.
   - Run it **once** only (free plan: one tunnel at a time); the link changes at each start, unless `NGROK_DOMAIN` is set in `.env` (the free static domain of the ngrok account, dashboard.ngrok.com → Domains): then the link stays the same and the ngrok page shows only once per browser and week.
   - A new `APP_PASSWORD` applies at the next login, without restarting.

5. **Run the tests**:
   ```bash
   venv/bin/python -m pytest -q tests
   ```
   64 tests, synthetic data only, no network and no key needed.












