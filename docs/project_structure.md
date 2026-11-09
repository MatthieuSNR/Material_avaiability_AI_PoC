## 📁 Project Structure

State on 29 October 2026. Every file listed here is used; files that no longer served (an early spaCy root-cause
matcher in French, its visualisation script, an empty NLP notebook) were removed on 26 October and remain in the
Git history.

```
Roche_AI_PoC/
├── data/                          # Confidential: never pushed to GitHub (.gitignore)
│   ├── Snapshot_2026/             # Raw daily snapshots: 26 CSV files (W14 to W38), 174 days
│   ├── interim/                   # Files generated or kept on this Mac
│   │   ├── translation_cache.csv          # Every DeepL translation already paid for (+ glossary version)
│   │   ├── cleaned_comments_before_rebuild.csv  # Older build (its translations are all in the cache)
│   │   ├── deepl_glossary.json            # Id and hash of the glossary created on DeepL
│   │   ├── embedding_cache.npz            # Vectors of the language model per comment (grouping by meaning)
│   │   ├── abbreviations_for_review.xlsx  # Sheet for the planners: abbreviation, meaning, language
│   │   ├── abbreviations_review_alexis_2026-10-29.csv  # The reviewed sheet (Matthieu OK / ?, Alexis meaning + context)
│   │   ├── abbreviation_evidence.csv      # Real excerpts that justify the proposed meanings (read by --report)
│   │   └── saved_views.json               # Saved views of the dashboard
│   ├── cleaned_comments.csv       # One row per (material, vendor, comment): cleaned, expanded, translated
│   ├── item_tracking.csv          # One row per (day, material, vendor) from ALL snapshot rows
│   └── current_snapshot.csv       # All the order lines of the last snapshot day (Overview)
│
├── lexicon/                       # Versioned reference tables, no real comment and no supplier name in them
│   ├── abbreviations.csv          # Abbreviation library (DE / EN): meaning, context, expand flag, review status
│   └── glossary_de_en.csv         # DeepL glossary of the German supply-chain jargon (use = yes / no / force)
│
├── scripts/                       # Python code of the PoC (flat: the notebooks import the modules directly)
│   │   # --- Pipeline (data/Snapshot_2026 -> data/cleaned_comments.csv)
│   ├── data_cleaning.py           # Steps 1-6: merge, commented rows, journal entries, one row per comment, language,
│   │                              #   abbreviations; save with Comment_EN from the translation cache
│   ├── entries.py                 # Separator " ¦▸ " between the journal entries of a cell (used by every module)
│   ├── abbreviations.py           # Abbreviation library: expansion entry by entry, --check (also a meaning expanded
│   │                              #   twice), --sheet (Excel for the planners), --report, --discover
│   ├── translation.py             # Step 7: language per entry, pseudonymise, DeepL with cache and glossary -> Comment_EN
│   ├── glossary.py                # DeepL glossary: --check, creation / reuse on DeepL, forced terms
│   ├── pseudonymise.py            # Hides names, IDs, quantities and prices before DeepL or the Mistral API
│   ├── tracking.py                # Builds data/item_tracking.csv (worst status, impact level) and current_snapshot.csv
│   │   # --- Analyses
│   ├── stats_engine.py            # Statistics of the dashboard and of the Smart Analysis (pandas only)
│   ├── insights.py                # Grouping of similar comments, weekly trends, status change, root cause x impact,
│   │                              #   stakeholders, weak signals (phrases, context, prediction model)
│   ├── llm_summary.py             # Smart Analysis: facts, prompt, Mistral API (deployed app) or LM Studio (on this Mac)
│   │   # --- Dashboard
│   ├── app.py                     # Streamlit dashboard: password page, sidebar filters and views, Overview, Filtered
│   │                              #   Comments, Smart Analysis and its key statistics; calls insights_pages for 4 sections
│   ├── insights_pages.py          # Sections Most Frequent Comments, Root Cause Trends, Impact by Root Cause, Weak
│   │                              #   Signals; context of the Smart Analysis (smart_context)
│   ├── ui.py                      # Look and components: ION-like top bar and menu, section titles with Options,
│   │                              #   switches shown in the browser, charts (kept per data, PNG link), column picker,
│   │                              #   coloured comparisons, loading screen, Smart Analysis card (copy / Word / text)
│   ├── config.py                  # Reads the keys from .env (Streamlit secrets once deployed); the app password is read again at every login
│   ├── data_bundle.py             # Encrypted copy of the data for the deployed app (deploy/data.enc, key DATA_KEY)
│   ├── requirements.txt           # Dependencies of the dashboard deployed on Streamlit Community Cloud
│   │   # --- Notebooks: run each step and check its result
│   ├── data_cleaning.ipynb        # Steps 1-6, one cell per step, + explorations A-C (spaCy tokens, keywords, heatmap)
│   ├── translation.ipynb          # Step 7 (7.1-7.5)
│   └── llm_summary.ipynb          # Statistics, prompt, pseudonymised prompt, cloud / local analysis (switches)
│
├── tests/                         # pytest, synthetic data only, no network (69 tests)
│   ├── test_comment_cleaning.py   # Journal entries, signatures, snapshot merge, abbreviations (library, expanded twice),
│   │                              #   entry-by-entry translation, Comment_EN from the cache
│   ├── test_glossary.py           # DeepL glossary: validation, hash, forced terms, refresh of stale translations
│   ├── test_pseudonymise.py       # Placeholders (names, IDs, quantities, prices) replaced and restored
│   ├── test_cloud_summary.py      # No real name in the Mistral request, aliases restored, no asterisk in the rendering
│   └── test_insights.py           # Grouping (opposite meanings kept apart), tracking, status change, root cause x impact,
│                                  #   stakeholders merged, weak signals, Lab filters, charts drawn once per data
│
├── docs/
│   ├── project_steps.md           # Method, step by step, and data update procedure
│   ├── project_structure.md       # This file
│   └── tools_justification.md     # Why every tool and library, and its limits
│
├── assets/                        # Roche logo and favicon
├── .streamlit/config.toml         # Theme close to the ION Material Availability dashboard
├── outputs/                       # Local outputs, not pushed (demo teaser)
├── deploy/data.enc                # Data of the deployed app, encrypted (Fernet); the only data file on GitHub
├── requirements.txt               # Python dependencies (minimum versions)
├── .env / .env.example            # API keys and app password (.env never pushed)
└── README.md                      # Overview, tools, how to run
```

### Key Notes
- **Data flow, comments**: `Snapshot_2026/` → `data_cleaning.py` (steps 1-6, with `entries.py`, `abbreviations.py` + `lexicon/abbreviations.csv`) → `cleaned_comments.csv` → `translation.py` (step 7, with `pseudonymise.py`, `glossary.py` + `lexicon/glossary_de_en.csv`, the cache) → `Comment_EN`.
- **Data flow, items**: `Snapshot_2026/` → `tracking.py` → `item_tracking.csv` → `insights.py` → `insights_pages.py`.
- **Dashboard**: `app.py` reads both files, applies the sidebar filters, draws its own sections with `stats_engine.py` and `ui.py`, and the four analysis sections with `insights_pages.py`; the Smart Analysis (`llm_summary.py`) receives the statistics and the findings of these sections (`smart_context`).
- **Sections of the page, in the menu order**: Overview (with Top Root Causes; last snapshot only), Most Frequent Comments, Root Cause Trends, Impact by Root Cause, Watchlist, Smart Analysis (always last; second page: the key statistics it uses).
- **One function per cleaning step**, one notebook cell per step: the result of every step can be checked on its own. The notebooks call the functions of the scripts, they do not duplicate code.
- **`stats_engine.py` and `insights.py` compute every figure**: the dashboard displays them and the LLM only receives them, so the LLM never computes anything.
- **`lexicon/` vs `data/interim/`**: `lexicon/` holds the reference tables, versioned and free of real comments and supplier names (the GitHub repository is public); `data/interim/` holds files generated from the confidential data, never pushed.
- **`scripts/` is flat on purpose**: a modular `src/` package was tried and set aside (history of the repository); a flat folder lets the notebooks import the modules without installation.
- **`data/`, `.env`, `venv/`, `outputs/`**: confidential or machine-specific, excluded by `.gitignore`. Only code and documentation are pushed. Notebooks are pushed **without outputs** (their outputs show real comments).
- **Run with the project environment**: `venv/bin/streamlit run scripts/app.py`; another Python (e.g. Anaconda) lacks some packages.
