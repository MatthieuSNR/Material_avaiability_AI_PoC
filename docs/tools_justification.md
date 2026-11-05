# 🛠️ Tools and Libraries Justification

This document justifies the **selection of tools, libraries, and frameworks** used in the **Roche AI PoC: Material Availability – Insights Lab** project (state on 29 October 2026; installed versions in `venv`, minimum versions in `requirements.txt`).

| Section | Tool | Used in |
|---|---|---|
| 1 | Python 3.12.4 | everything |
| 2 | VS Code | development |
| 3 | pandas 2.2.3 (+ numpy 1.26.4) | pipeline, statistics, dashboard |
| 4 | spaCy 3.8.16 | notebook explorations only |
| 5 | langdetect 1.0.9, DeepL API (`deepl` 1.32.0) | pipeline steps 5 and 7 |
| 6 | Mistral API, LM Studio, `openai` client 3.26.0 | Smart Analysis |
| 7 | matplotlib 3.8.4, seaborn 0.13.2 | charts (seaborn: one notebook heatmap) |
| 8 | Streamlit 1.66.1, Altair 5.5.0, streamlit-sortables 0.3.1 | dashboard |
| 9 | `venv` | environment |
| 10 | `.env` + python-dotenv 1.2.4 | keys and password |
| 11 | project structure | — |
| 12 | Streamlit Community Cloud + `cryptography` 50.0.2 | online access for the usability tests |
| 13 | pytest 9.1.1 | tests |
| 14 | openpyxl 3.1.5 | Excel sheet for the planners |
| 15 | scikit-learn 1.4.2, SciPy 1.17.1, sentence-transformers 3.4.1 (+ PyTorch 2.14.1, transformers 4.57.6) | Insights Lab analyses |
| 16 | Jupyter 1.1.1 (+ nbconvert) | step-by-step notebooks |
| 17 | Git / GitHub | versioning |
Each choice is grounded in **technical requirements, academic standards, and industry best practices** to ensure **reproducibility, scalability, and maintainability**.

---

---

## **📌 1. Programming Language: Python (v3.12.4)**

### **Why Python?**
   **Criteria**          | **Justification**                                                                                                                                                                                                 |
 |-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | **Industry Standard** | Python is the **most widely used language** for **NLP (Natural Language Processing)** and **Machine Learning (ML)** in both academia and industry (e.g., [Stack Overflow Survey 2023](https://survey.stackoverflow.co/2023/)). |
 | **Library Ecosystem** | Rich ecosystem of **open-source libraries** (e.g., `pandas`, `spaCy`, `scikit-learn`) tailored for data science and NLP tasks.                                                                               |
 | **Readability**       | Clean and **intuitive syntax**, making it easier to document and share code (critical for thesis reproducibility).                                                                                           |
 | **Compatibility**     | Version **3.12.4** ensures compatibility with all required libraries (e.g., `spaCy 3.8.16`, `pandas 2.2.3`).                                                                                     |
 | **Academic Adoption** | Python is **the most cited language** in NLP/ML research papers (e.g., [arXiv](https://arxiv.org/), [ACL Anthology](https://aclanthology.org/)).                                                                   |

### **Alternatives Considered**
 | **Alternative** | **Rejection Reason**                                                                                     |
 |-----------------|-------------------------------------------------------------------------------------------------------|
 | R               | Strong in statistics but **less suited for NLP/ML pipelines** and lacks native spaCy/scikit-learn support. |
 | Java            | Verbose syntax and **steeper learning curve** for data science tasks.                              |
 | JavaScript      | Limited **NLP/ML libraries** compared to Python.                                                     |

### **References**
- [Python Official Documentation](https://www.python.org/doc/)
- [Stack Overflow Developer Survey 2023](https://survey.stackoverflow.co/2023/)
- [arXiv: Python in NLP Research](https://arxiv.org/)

---

---

## **📌 2. Integrated Development Environment (IDE): Visual Studio Code (VS Code)**

### **Why VS Code?**
 | **Criteria**          | **Justification**                                                                                                                                                                                                 |
 |-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | **Python Support**    | Built-in **Python extension** (Pylance) for autocompletion, linting, and debugging.                                                                                                                             |
 | **Lightweight**       | Faster and **less resource-intensive** than alternatives like PyCharm.                                                                                                                                       |
 | **Git Integration**   | Native **Git support** for version control, critical for collaborative projects.                                                                                                                               |
 | **Extensions**        | Rich marketplace (e.g., **Jupyter Notebooks**, **Mermaid**, **GitLens**) to enhance productivity.                                                                                                               |
 | **Cross-Platform**    | Works seamlessly on **macOS, Windows, and Linux**.                                                                                                                                                              |
 | **Academic Use**      | Widely adopted in **universities and research** for data science projects.                                                                                                                                |

### **Alternatives Considered**
 | **Alternative** | **Rejection Reason**                                                                                     |
 |-----------------|-------------------------------------------------------------------------------------------------------|
 | PyCharm         | Heavy and **slower** for small projects like this PoC.                                              |
 | Jupyter Notebook | Limited **file management** and less suitable for script-based workflows.                          |
 | Sublime Text    | Lacks **native Python debugging** and Git integration.                                             |

### **References**
- [VS Code Official Documentation](https://code.visualstudio.com/docs)
- [Stack Overflow Survey 2023: Most Popular IDEs](https://survey.stackoverflow.co/2023/)

---

---

## **📌 3. Data Manipulation: pandas (v2.2.3)**

### **Why pandas?**
 | **Criteria**          | **Justification**                                                                                                                                                                                                 |
 |-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | **Industry Standard** | **De facto standard** for data manipulation in Python (used by 80% of data scientists, per [Kaggle Surveys](https://www.kaggle.com/surveys)).                                                           |
 | **Performance**       | Optimized for **large datasets** with C-based backend.                                                                                                                                                         |
 | **Functionality**     | Provides **DataFrame** structure for easy cleaning, filtering, and aggregation.                                                                                                                               |
 | **Integration**       | Works seamlessly with **spaCy, matplotlib and Streamlit**.                                                                                                                                                   |
 | **Academic Use**      | Cited in **thousands of research papers** (e.g., [Nature](https://www.nature.com/), [IEEE](https://ieeexplore.ieee.org/)).                                                                                     |

### **Key Features Used in the Project**
- **numpy** (installed with pandas) is used directly for vectorised similarities (grouping of comments), the embedding cache (`.npz`) and the charts.
- **`pd.read_csv()` + `glob`**: Load the 174 daily snapshots of `data/Snapshot_2026/` (≈2 million rows; 87,177 commented rows kept).
- **One function per cleaning step** (`scripts/data_cleaning.py`): each step takes a DataFrame and returns a DataFrame, so that its result can be checked in a notebook cell.
- **`groupby().agg()`**: One row per (material, vendor, comment) with `First_Seen`, `Last_Seen`, `N_Snapshots`, `Days_Active` (2,231 comments); a stable sort keeps the "latest value" of each column reproducible.
- **`to_datetime()`**: Convert snapshot dates to datetime objects (period filters, weekly trends).
- **`value_counts()` / `groupby()`**: All dashboard statistics (`scripts/stats_engine.py`), e.g. "X% of this vendor's comments have root cause Y".

### **Alternatives Considered**
 | **Alternative** | **Rejection Reason**                                                                                     |
 |-----------------|-------------------------------------------------------------------------------------------------------|
 | Polars          | Faster but **less mature**, and Streamlit, spaCy and the notebooks are built around pandas.             |
 | Dask            | Overkill for **medium datasets** (≈2 million snapshot rows, read in ≈15 s; ≈2,200 comments after cleaning). |
 | R (data.table)  | Not compatible with **Python-based NLP libraries** like spaCy.                                       |

### **References**
- [pandas Official Documentation](https://pandas.pydata.org/docs/)
- [Kaggle: State of Data Science 2023](https://www.kaggle.com/surveys/2023)

---

---

## **📌 4. Natural Language Processing (NLP): spaCy (v3.8.16)**

### **Why spaCy?**
 | **Criteria**          | **Justification**                                                                                                                                                                                                 |
 |-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | **Performance**       | **Faster** than NLTK for tokenization, lemmatization, and entity recognition.                                                                                                                                   |
 | **Multilingual**      | Supports **German (`de_core_news_sm`)** and **English (`en_core_web_sm`)**, the two languages of the planner comments.                                                                                               |
 | **Pre-trained Models**| Includes **pre-trained models** for NLP tasks (e.g., entity recognition, dependency parsing).                                                                                                               |
 | **Industry Adoption** | Widely used in **industrial NLP pipelines** (production-oriented design, maintained by Explosion AI).                                                                                                              |
 | **Academic Use**      | Cited in **NLP research** (e.g., [ACL Anthology](https://aclanthology.org/)).                                                                                                                                     |

### **Key Features Used in the Project**
Used in the explorations of `data_cleaning.ipynb` (A: tokens, B: keywords by root cause), not in the pipeline that builds the comments file:
- **Tokenization and lemmatization** entry by entry, with the model of the language detected in step 5 (`de_core_news_sm` or `en_core_web_sm`), on `Comment_Expanded` (e.g., "Lieferungen" → "Lieferung").
- **Stop word removal**: Filter out common words (e.g., "the", "und").
- **Keyword analysis by root cause**: the most frequent lemmas of each root cause.

### **Alternatives Considered**
 | **Alternative**       | **Rejection Reason**                                                                                     |
 |-----------------------|-------------------------------------------------------------------------------------------------------|
 | NLTK                  | **Slower** and less optimized for German.                                                             |
 | Hugging Face Transformers | Overkill for **simple NLP tasks** (tokens, keywords). Requires GPU for large models. |
 | Stanford NLP          | **Java-based**, not compatible with Python workflows.                                               |

### **References**
- [spaCy Official Documentation](https://spacy.io/)
- [spaCy Models: German and English](https://spacy.io/models)
- [ACL Anthology: spaCy in Research](https://aclanthology.org/)

---

---

## **📌 5. Language Detection and Translation: langdetect (v1.0.9) + DeepL API (deepl v1.32.0)**

### **Why langdetect + DeepL?**
 | **Criteria**          | **Justification** |
 |-----------------------|-------------------|
 | **Need**              | About two thirds of the planner comments are written in **German**; translating them to English gives one language for analysis, dashboard and LLM summary. |
 | **Translation quality** | DeepL handles **German supply-chain jargon and abbreviations** (e.g., "best. LT", "Stk") better than generic free tools. |
 | **Cost control**      | Only German comments are sent, and every translation is stored in `data/interim/translation_cache.csv`: a comment is **paid for only once**, even after a data rebuild (the rebuild itself fills `Comment_EN` from the cache). A `--dry-run` mode counts the characters before sending (free plan: 500,000 characters/month). |
 | **Detection accuracy** | `langdetect` alone labelled short German comments as Hungarian, Catalan, Norwegian… (≈12 %), and these were never translated. Detection is therefore **restricted to English/German**, with a small German lexicon as fallback. |

### **Key Features Used in the Project** (`scripts/translation.py`)
- **`detect_langs()`**: Language probabilities, reduced to `en` / `de`, computed **per journal entry** (a cell can mix German and English entries) and **before** the abbreviations are written out (pipeline step 5), because an abbreviation can have another meaning in each language. Known limit: short mixed entries (e.g., "via SafetyStock gedeckt") can be labelled English.
- **`Translator.translate_text()`**: Batch translation (50 entries per call), with retry on rate limit; German entries only, pseudonymised, after the abbreviations are written out (pipeline step 7).
- **Glossary** (`lexicon/glossary_de_en.csv`, `scripts/glossary.py`): German supply-chain jargon with a fixed English term (`Vorzug` → pull-in); created once on DeepL and reused while its content does not change. DeepL does not always apply it (16 of 140 *Vorzug* entries still came out as *priority* or *benefits*): terms marked `use = force` are written in English inside an ignored `<keep>` tag before sending, so their translation is guaranteed (`Vorzug`, `Vorzüge`, `Vorzugsanfrage`, and since 29.10 `Orderbuch` → source list).
- **Abbreviations before translation** (`scripts/abbreviations.py`, `lexicon/abbreviations.csv`, reviewed by Matthieu and Alexis on 28.10): the certain abbreviations are written out in the language of each entry, so DeepL receives full words (`LT` → *Liefertermin* → *delivery date*).
- **`Translator.get_usage()`**: Monthly character usage, printed after each run.

### **Alternatives Considered**
 | **Alternative** | **Rejection Reason** |
 |-----------------|----------------------|
 | Google Translate API | Comparable quality but **paid from the first character** and less precise on technical German. |
 | Local translation model (e.g., MarianMT, Hugging Face) | **No data leaves the machine**, but lower quality on jargon and slower on CPU; a possible option if data governance requires it. |
 | Local LLM (LM Studio) | Possible, but **slower** and less consistent than a dedicated translation engine. |

### **Data Governance**
**Classification** (Marco, project sponsor, 26.10): prices and quantities are sensitive; the other data is not confidential, so the pseudonymisation below is an additional safeguard. Quantities and prices written inside the comments are masked too (`QTY_n`, `PRICE_n`, the unit stays so that it is translated), since 26.10.

DeepL is a **cloud service**: only the comment text is sent (no vendor, material or quantity column). Before sending, the text is **pseudonymised** (`scripts/pseudonymise.py`): e-mail addresses, @-mentions, supplier names, material numbers, long numbers (PO, batch) and the company name are replaced by placeholders, which DeepL copies unchanged; the original values are restored locally after translation. The same module protects the prompt of the Smart Analysis (section 6). The comment cell is first cut into its **journal entries** and the signatures (staff logins, dates) are removed (`data_cleaning.py`), so that no login reaches the translation; abbreviations are written out before pseudonymisation (`scripts/abbreviations.py`, `lexicon/abbreviations.csv`), so that DeepL receives full words. Names of the sourcing managers (column `SOURCING_MANAGER`) are pseudonymised too.

**Residual limits** (thesis, 3.8):
- Supplier names written in an **abbreviated or misspelled** form, and supplier names shorter than 4 characters, are not detected.
- **Person names** written without "@" or e-mail address (e.g., "laut Hr. Müller") are not detected.
- The **rest of the comment** (delays, quantities, problem descriptions) is still sent to a cloud service; pseudonymisation reduces, but does not remove, the governance question. The local alternatives above remain possible.
- The placeholder pattern relies on DeepL respecting the ignored tags; a placeholder that is altered would not be restored (checked by `tests/test_pseudonymise.py` for the local part only).

### **References**
- [DeepL API Documentation](https://developers.deepl.com/docs)
- [langdetect (port of Google's language-detection)](https://pypi.org/project/langdetect/)

---

---

## **📌 6. Smart Analysis: Mistral API (deployed app) and a local LLM on this Mac (LM Studio)**

### **What the LLM does, and what it does not**
**pandas computes every number** (`scripts/stats_engine.py`); the LLM only **writes** the analysis from these precomputed statistics and the most frequent comments, and is instructed never to compute or invent a figure. This is why a small model is sufficient and why the figures of the analysis can be checked against the dashboard.

### **Why the Mistral API (since 16 Oct.)?**
 | **Criteria**          | **Justification** |
 |-----------------------|-------------------|
 | **Need**              | The local model (Mistral 7B on the laptop) took 40–60 s per analysis, made the Mac heat up, and cannot run on a deployed or shared app. A hosted model removes the load from the laptop. |
 | **Same family, same interface** | Mistral's API is **OpenAI-compatible**: the same `openai` client and the same prompt are used (only `MISTRAL_BASE_URL` and the key change), so cloud and local answers can be compared on the same selections. |
 | **Model**             | `ministral-14b-latest` by default (`MISTRAL_MODEL` in `.env`). On the free plan, the small and medium models were often rate-limited (HTTP 429) and the large one not included (HTTP 403) when tested on 16 Oct.; this choice is a practical one and may change with the plan. |
 | **Provider**          | European provider, free tier sufficient for a prototype. For an adoption by Roche, the choice would follow the contract Roche has already approved (e.g., a model hosted in an EU region by a provider under Roche's agreement). |
 | **Cost**              | Negligible at this volume (one request of about 1,000 input tokens per analysis); the free plan quotas are shared by all visitors of a shared demo. |

### **Key Features Used in the Project** (`scripts/llm_summary.py`, `scripts/pseudonymise.py`)
- **`chat.completions.create()`** with low temperature (0.2) for factual answers; the same prompt for cloud and local (`build_prompt`).
- **Compact prompt** (`build_facts`): only the key statistics, without indentation or duplicated figures, and every key states its unit (`_pct` = percentage, otherwise a count) after the model turned "+75 comments" into "+75%" in a test.
- **Structured, in-depth analysis**: 250–320 words in five fixed sections (Key findings, What drives the issues, Early warnings, Risks to escalate, Recommended actions). The prompt includes the most frequent informative comments of the three main root causes, the findings of the Insights Lab for the same selection (root cause → impact, weak signals, watch list, long-open comments) and the filters (so that the model does not "discover" them); the model must quote comments word for word, so the analysis explains *why* and not only *how much*.
- **Display and export** (`scripts/ui.py`): one bold phrase per bullet at most, no markdown asterisk left; **Copy** (formatted, for Word or an e-mail), **Download Word** and **Download text**, in the browser.
- **Streaming** (`stream_summary_cloud`): the analysis appears word by word.
- **Pseudonymised prompt** (`PromptPseudonymiser`): suppliers, MRP controllers, purchasing groups and material numbers are replaced by aliases (`VENDOR_1`, `MRP_2`, `PGROUP_3`, `MATERIAL_4`), and e-mails, @-mentions, long numbers and the company name in the comments too. The prompt tells the model to copy the aliases unchanged. The alias table stays in memory on the Mac; aliases in the answer are replaced by the real names before display (`stream_restored` holds back the last word so that half an alias is never shown). Pseudonymisation is always on for the cloud model.
- **Nothing else is sent**: no price and no volume; only shares, counts, trends and a few short comments. A test (`tests/test_cloud_summary.py`) checks that no real supplier, MRP controller, purchasing group, material number or e-mail appears in the request, and that the answer comes back with the real names.

### **Data Governance and Limits**
- **Pseudonymisation is a safeguard, not an approval**: the use of an external LLM API on Roche data has still to be validated with Roche (open point). Business-confidential content (shares per supplier, planner comments) leaves the machine in aliased form.
- **Residual limits** are those of section 5: supplier names written in abbreviated or misspelled form and person names without "@" are not detected; MRP controllers and purchasing groups are short codes, replaced in the statistics but not searched inside free text.
- **Provider terms**: Mistral's retention and training terms for the plan in use have not been verified yet; they must be checked and cited before the thesis states that the data is not stored.
- **Quality**: LLM output is not deterministic and there is no formal evaluation yet (faithfulness of figures and quotes, comparison cloud vs local). The figures are protected by construction (pandas), the interpretation is not.
- **Availability**: the cloud analysis needs the network and a valid key; the free plan can answer 429 under load.

### **Local model: LM Studio (Mistral 7B Instruct v0.3)**
 | **Criteria**          | **Justification** |
 |-----------------------|-------------------|
 | **Confidentiality**   | The model runs **on the laptop** (`http://127.0.0.1:1234/v1`): no Roche data leaves the machine. Still the reference if data governance forbids any external API. |
 | **Cost**              | No API fee, no usage limit. |
 | **Limits**            | ≈5 tokens/s on an Apple M3 with 16 GB when memory is short, so 40–60 s per analysis and a hot laptop; Mistral 7B does not accept the `system` role (instructions are merged into the user message). |
 | **Status**            | Used when the dashboard runs on this Mac (localhost, e.g. demos): no API limit or overload (Mistral's 429 "out of capacity" on the free plan). The deployed app (Streamlit Cloud) uses the Mistral API. |

### **Alternatives Considered**
 | **Alternative** | **Reason** |
 |-----------------|-----------|
 | Other cloud APIs (OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, Google Vertex AI) | Comparable quality for this task, so the choice depends on data governance and contracts rather than benchmarks. Not retained for the prototype; the code only needs another base URL and key. Under an existing Roche agreement, one of these (EU region) would be the natural choice. |
 | Local only (LM Studio, Ollama) | Best confidentiality but slow, hot laptop, impossible to deploy or share. Ollama is equivalent to LM Studio; LM Studio was chosen for its graphical interface. |
 | Larger local models (≥ 13B) | Too slow on a laptop for an interactive dashboard. |

### **References**
- [Mistral AI Documentation](https://docs.mistral.ai/)
- [LM Studio Documentation](https://lmstudio.ai/docs)
- [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [GDPR (Regulation (EU) 2016/679), Art. 4(5): definition of pseudonymisation](https://eur-lex.europa.eu/eli/reg/2016/679/oj)


## **📌 7. Visualization: matplotlib (v3.8.4) + seaborn (v0.13.2)**

### **Why matplotlib + seaborn?**
 | **Tool**       | **Justification**                                                                                                                                                                                                 |
 |----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | **matplotlib** | **Standard library** for basic plots (histograms, bar charts). **Integrated with pandas** (e.g., `df.plot()`).                                                                                                   |
 | **seaborn**    | **Enhanced aesthetics** for statistical plots (e.g., heatmaps, boxplots). Built on top of matplotlib.                                                                                                               |

### **Key Features Used in the Project**
- **Bar Charts**: Top root causes with count and share of the selection; weak signals against the rate of all items (colour = how many times more often, solid / light = significant or not).
- **100% stacked bars**: impact communicated and material status per root cause (ION colours, legend above the bars), drawn with Altair: a click on a segment sets the sidebar filters.
- **Pie Charts**: Root cause share.
- **Line Charts**: Items per root cause (or stakeholder) and week.
- **Heatmaps** (seaborn, notebook only): Root causes per vendor.
- **Drawn once per data** (`ui.chart`): a chart is a function of its data; its PNG is kept for the same inputs, so a rerun of the dashboard does not draw the unchanged charts again (4 s → 0.6 s per rerun). The PNG is put directly in the page, with a **⬇ PNG** link under it (download in the browser).

### **Alternatives Considered**
 | **Alternative** | **Rejection Reason**                                                                                     |
 |-----------------|-------------------------------------------------------------------------------------------------------|
 | Plotly          | **Interactive** but heavier and requires JavaScript for full functionality; static PNGs are also easier to put into the thesis and the slides. |
 | ggplot          | **R-based**, not compatible with Python.                                                             |

### **References**
- [matplotlib Official Documentation](https://matplotlib.org/stable/contents.html)
- [seaborn Official Documentation](https://seaborn.pydata.org/)

---

---

## **📌 8. Dashboard: Streamlit (v1.66.1)**

### **Why Streamlit?**
 | **Criteria**          | **Justification**                                                                                                                                                                                                 |
 |-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | **Ease of Use**       | **No frontend development** required. Write Python scripts and get a **fully interactive web app**.                                                                                                             |
 | **Integration**       | Works with **pandas and matplotlib** out of the box.                                                                                                                                               |
 | **Deployment**        | Free hosting on [Streamlit Cloud](https://streamlit.io/cloud) is possible, but not used with the confidential data (see section 12).                                                                                                                                             |
 | **Industry Adoption** | Widely used by **data science teams** for internal prototypes (acquired by Snowflake in 2022).                                                                                                                      |
 | **Academic Use**      | Increasingly used in **research** for sharing interactive results (e.g., [arXiv](https://arxiv.org/)).                                                                                                         |

### **Key Features Used in the Project**
- **Widgets**: Sidebar filters (MRP controller, purchasing group, vendor, material, root cause, impact, timeframe); every filter is a **multiselect** (empty = All), so a planner can view their own scope plus the colleagues they back up (user feedback, demo of 16 Oct.). The filters **cascade**: each one only offers the values that exist within the filters above it. They fold (`st.expander`) so that they fit on one screen.
- **Saved views** (like SAP *personalised views*): filters, timeframe and table columns saved under a name and re-applied in one click (`data/interim/saved_views.json`, local).
- **▥ Columns panel** on the large tables: `st.popover` (opens in the browser, floats over the table) with a checkbox per column and **drag and drop** of the order with [streamlit-sortables](https://github.com/ohtaman/streamlit-sortables) (without the package the panel still works, without drag and drop).
- **One scrolling page** with six sections (Overview, Most Frequent Comments, Root Cause Trends, Impact by Root Cause, Watchlist, Smart Analysis with its key statistics) and a menu that stays at the top. The menu is HTML; a small script installed with `st.iframe` highlights the section on screen and brings a section's title just under the menu.
- **Options** of a section: `st.popover` on the right of its blue title (settings, *Show explanations* and, on Most Frequent Comments and the Watchlist, the **▥ Columns** of the table beside them). The CSS of *Show explanations* is in the section title, not in the panel: Streamlit takes a closed panel out of the page.
- **Clickable charts**: the Top Root Causes and the bars of Impact by Root Cause are drawn with Altair (`st.altair_chart(on_select=...)`, Streamlit 1.35+): a click sets the filters of the sidebar. The other charts stay matplotlib pictures. Moved from Streamlit 1.32 to the version of Streamlit Community Cloud on 05.11, so that the Mac and the deployed app run the same; the page styles (`scripts/ui.py`) target its HTML.
- **Work in the browser, without a rerun**: Streamlit reruns the whole script at every click. The actions that change nothing in the data are therefore done in the browser by CSS and a little JavaScript (`scripts/ui.py`): *Show explanations*, the switch between the two charts of *Impact by Root Cause*, the PNG export, the copy and download of the Smart Analysis.
- **Caching**: `st.cache_data` (comments file, analyses), `st.cache_resource` (tracking table shared without copy, language model), plus the charts kept per data (section 7).
- **DataFrames**: `st.dataframe` with column help (**?**) and, for comparisons, a pandas Styler (coloured cells).
- **Theme + custom CSS** (`.streamlit/config.toml`, `scripts/ui.py`): a look close to the existing Roche **ION Material Availability** dashboard (Roche logo, blue bar and section titles, square corners, material status tiles), so planners recognise the tool. The logo is in `assets/roche_logo.png`.

### **Alternatives Considered**
 | **Alternative** | **Rejection Reason**                                                                                     |
 |-----------------|-------------------------------------------------------------------------------------------------------|
 | Dash (Plotly)   | **More complex** to set up for simple dashboards.                                                     |
 | Flask/Django    | Requires **HTML/CSS/JavaScript knowledge** for frontend development.                                  |

### **References**
- [Streamlit Official Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

---

---

## **📌 9. Virtual Environment: `venv`**

### **Why a Virtual Environment?**
 | **Criteria**          | **Justification**                                                                                                                                                                                                 |
 |-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | **Isolation**         | Ensures **dependencies for this project** (e.g., `spaCy 3.8.16`) do not conflict with other Python projects.                                                                                                       |
 | **Reproducibility**   | **`requirements.txt`** allows others to recreate the exact same environment.                                                                                                                                   |
 | **Portability**       | Can be **shared across machines** (e.g., your laptop, Roche’s servers).                                                                                                                                             |
 | **Version Control**   | Avoids issues with **multiple Python versions** (e.g., Python 3.9 vs. 3.12).                                                                                                                                       |

### **How It Was Implemented**
```bash
python3.12 -m venv venv              # creation
source venv/bin/activate             # activation (or call venv/bin/python, venv/bin/streamlit directly)
pip install -r requirements.txt      # dependencies
deactivate                           # deactivation
```
**Pitfall met on 29.10**: the dashboard started with another Python (Anaconda) failed with `No module named 'streamlit_sortables'`; it must be started with `venv/bin/streamlit run scripts/app.py`.

### **References**
- [Virtual Environments and Packages](https://docs.python.org/3/tutorial/venv.html)


---
---

## **📌 10. Secrets Management: `.env` file + python-dotenv (v1.2.4)**

### **Why?**
 | **Criteria**          | **Justification** |
 |-----------------------|-------------------|
 | **Security**          | The secrets are stored in a local `.env` file, **excluded from Git** by `.gitignore`: `DEEPL_API_KEY` (translation), `MISTRAL_API_KEY` (Smart Analysis) and `APP_PASSWORD` (password page of the shared app). `scripts/config.py` and `scripts/llm_summary.py` only read them, so the code can be pushed to GitHub without any key. |
 | **Simplicity**        | One file to edit when a key changes; `.env.example` documents the expected variables (`LM_STUDIO_*` and `MISTRAL_MODEL` are settings, not secrets). The app password is read again from `.env` at every login (`config.app_password`), so a new password applies without restarting; a value given on the command line wins (local test without password). |

### **Lesson Learned**
An earlier version stored the key directly in `scripts/config.py`, which was pushed to GitHub. A key that has been pushed must be **revoked**: deleting the file does not remove it from the Git history.

### **References**
- [python-dotenv Documentation](https://pypi.org/project/python-dotenv/)
- [GitHub: Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)


## **📌 11. Project Structure: Why This Organization?**

### **Why This Structure?**
The project follows a **modular and scalable** structure inspired by **best practices in data science** (e.g., [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)).
Each folder and file has a **clear purpose**, making the project **easy to maintain, debug, and extend**.
   **Folder/File**       | **Purpose**                                                                                     | **Justification**                                                                                                                                                                                                 |
 |-----------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | `data/`               | **Input data**: daily snapshots, cleaned comments, translation cache (never pushed to GitHub)   | **Separation of concerns**: Raw data is isolated from scripts and outputs to avoid confusion.                                                                                                               |
 | `scripts/`            | **Python scripts and notebooks** of the PoC (pipeline, dashboard, Smart Analysis)                 | **Modularity**: Each script has a **single responsibility**: `data_cleaning.py` (pipeline steps 1-6), `entries.py` (separator of the journal entries), `abbreviations.py` (abbreviation library), `translation.py` + `glossary.py` (step 7, DeepL), `pseudonymise.py` (hides names before DeepL / Mistral), `stats_engine.py` (statistics), `tracking.py` (tracking table), `insights.py` + `insights_pages.py` (analysis sections), `llm_summary.py` (Smart Analysis), `ui.py` (layout and components), `config.py` (keys), `app.py` (dashboard). One notebook per stage (`data_cleaning`, `translation`, `llm_summary`) runs the functions of the scripts step by step, without duplicating code. |
| `lexicon/`            | **Reference tables**: abbreviation library, DeepL glossary                                       | **Traceability**: versioned in Git, free of real comments; the planners validate them, the pipeline reads them. Files generated from the data stay in `data/interim/`. |
 | `tests/`              | **pytest tests** (comment cleaning, glossary, pseudonymisation, cloud summary, Insights Lab: 64 tests) | **Trust**: the privacy guarantee (no real name in the request) is checked automatically, with fake API clients and no Roche data. |
 | `assets/`, `.streamlit/` | **Roche logo, favicon, theme**                                                               | **Recognition**: a look close to the existing ION dashboard, kept out of the code. |
 | `deploy/`             | **Encrypted data of the deployed app** (`data.enc`)                                              | **Confidentiality**: the only data file on the public GitHub, unreadable without `DATA_KEY` (section 12). |
 | `outputs/`            | **Local outputs** (e.g., demo teaser), not pushed                                               | **Separation**: material made for presentations stays apart from code and data.                                                                                                                                 |
 | `venv/`               | **Virtual environment** for Python dependencies                                                | **Isolation**: Ensures dependencies (e.g., `spaCy 3.8.16`) do not conflict with other projects or system-wide Python.                                                                                              |
 | `docs/`               | **Project documentation** (this file, `project_steps.md`, `project_structure.md`)                                | **Clarity**: Centralizes all documentation for easy reference by stakeholders (e.g., Roche team) and for your thesis.                                                                                              |
 | `requirements.txt`    | **List of Python dependencies**                                                               | **Reproducibility**: Allows others to recreate the exact same environment using `pip install -r requirements.txt`.                                                                                                         |
 | `README.md`           | **Project overview and instructions**                                                          | **Onboarding**: Helps new users (e.g., Roche stakeholders or thesis reviewers) understand how to run the project.                                                                                                  |

---
### **How the Structure Supports the PoC Goals**
1. **Clarity**:
   - Each folder/file has a **clear name and purpose** (e.g., `translation.py` for step 7); files that no longer serve are removed (26 Oct.: an early French spaCy matcher, its plot script and an empty notebook; they stay in the Git history).
   - **Easy to navigate** for both developers and stakeholders.

2. **Reproducibility**:
   - **`requirements.txt`** and **`venv/`** ensure the project can be **recreated on any machine**.
   - **Scripts can be re-run** at each new snapshot delivery (see `docs/project_steps.md`).

3. **Scalability**:
   - **Modular scripts** (e.g., `data_cleaning.py`, `translation.py`) can be **extended or replaced** without affecting the entire project; a cleaning step is a function that can be changed and re-checked on its own.
   - **Outputs are separated** from inputs, making it easy to **add new data or scripts**.

4. **Collaboration**:
   - **Git-friendly**: The structure is compatible with **version control** (e.g., Git).
   - **Documentation is centralized** in `docs/`, making it easy to share with Roche or thesis reviewers.

---
### **References**
- [Cookiecutter Data Science Project Template](https://drivendata.github.io/cookiecutter-data-science/)
- [Good Enough Practices in Scientific Computing (arXiv)](https://arxiv.org/abs/1609.00037)
- [Google’s Rules for Repository Structure](https://github.com/google/styleguide/blob/gh-pages/docguide/style.md#repository-structure)



---

---

## **📌 12. Online Access for Usability Testing: Streamlit Community Cloud (free plan) + cryptography (v50.0.2)**

### **Why Streamlit Community Cloud?**
 | **Criteria**          | **Justification** |
 |-----------------------|-------------------|
 | **Need**              | Usability tests with planners who are **not next to the laptop**, on the Roche network: they open the dashboard in their own browser, with the real data. |
 | **Reachability**      | A `….streamlit.app` link is ordinary web hosting, fixed, and works while this Mac is off. Tunnels from this Mac (ngrok, Cloudflare quick tunnel) were tried first: the Roche network blocks them. |
 | **Effort**            | Deployed from the GitHub repository (`main`, `scripts/app.py`); every push updates the app. No server to run. Suited to a **prototype**: if Roche adopts the dashboard, it will be rebuilt on an internal platform (e.g., R Shiny, Posit Connect). |
 | **Cost**              | Free plan. |
 | **Data agreement**    | Marco (data owner) agreed to host the pseudonymised data there. |

### **How It Works in the Project**
- **Data never in clear on GitHub** (the repository is public): `scripts/data_bundle.py` zips the files the dashboard reads (`cleaned_comments.csv`, `item_tracking.csv`, embedding cache, saved views) and encrypts them with **Fernet** (`cryptography`: AES-128 + HMAC) into `deploy/data.enc`, the only data file pushed. The key `DATA_KEY` is in `.env` and in the app's secrets; at its start the deployed app unpacks the files (`unpack_if_needed`). After a new run of the pipeline: rebuild, commit and push `deploy/data.enc`.
- **Secrets**: no `.env` on the Cloud; `config.py` copies the Streamlit secrets (`APP_PASSWORD`, `DATA_KEY`, `MISTRAL_API_KEY`) into the environment, so the modules are unchanged.
- **Password page** (`check_password` in `scripts/app.py`): required on the Cloud (without `APP_PASSWORD` the deployed app does not open); locally active when `APP_PASSWORD` is set in `.env`, read again at every login (changing it needs no restart).
- **Dependencies**: `scripts/requirements.txt` (next to the entry point, read by the Cloud instead of the root file): the dashboard only, PyTorch CPU build, minimum versions (the Cloud runs Python 3.14).

### **Limits**
- The data runs on **Streamlit's (Snowflake) servers in the US**, decrypted there to be shown; the protection for visitors is the password.
- A simple shared password, no individual accounts.
- An old `data.enc` stays in the Git history: readable only with the key, which must never be pushed or shared.
- The free app sleeps after a few days without visits (one click wakes it up).
- Each cloud AI summary generated by a visitor uses the same Mistral API key (free plan quotas are shared).

### **Alternatives Considered**
 | **Alternative** | **Rejection Reason** |
 |-----------------|----------------------|
 | ngrok / Cloudflare quick tunnel from this Mac | Tried first (data stays on the Mac), but **blocked by the Roche network**; the Mac must also stay on during the tests. |
 | Internal Roche hosting (Posit Connect, Streamlit in Snowflake...) | The right option for an adopted tool, but requires IT approval: too heavy for a prototype test. |
 | Upload of the data in the app at each start | No data on GitHub at all, but the files would have to be uploaded again after every restart of the app. |

### **References**
- [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- [cryptography: Fernet](https://cryptography.io/en/latest/fernet/)


---

---

## **📌 13. Tests: pytest (v9.1.1)**

### **Why pytest?**
 | **Criteria**          | **Justification** |
 |-----------------------|-------------------|
 | **Need**              | Two promises are negative ones: no login stays in the cleaned comments, and no real name leaves the machine. They are checked by tests rather than by a one-off look at the data. The cleaning rules (signature chains, typos, `n/a`) are also fixed by tests, so a change in one rule does not silently break another. |
 | **No dependency on data or network** | The tests use a small synthetic table and a fake API client (`tests/test_cloud_summary.py`), so they run without Roche data, without a Mistral key and offline. |
 | **Simplicity**        | Plain functions with `assert`; `python -m pytest -q tests`. |

### **What Is Tested** (67 tests)
- `tests/test_comment_cleaning.py` (23): journal entries and signatures (chains, typos, signature in a line, `n/a`), snapshot merge (partial week skipped), one row per comment without login, abbreviations by language and on the real library, a meaning that another row would expand again, entry-by-entry translation, `Comment_EN` taken from the cache when the cleaned file is saved (DeepL never called).
- `tests/test_glossary.py` (10): validation of the glossary, whole-word case-sensitive matching, glossary created once and reused, fallback without glossary, forced terms (also at the start of a sentence), stale translations redone only on request.
- `tests/test_pseudonymise.py` (7): identifying elements, quantities and prices are replaced and restored; the same value gets the same placeholder.
- `tests/test_cloud_summary.py` (8): the request of the Smart Analysis contains no real supplier, MRP controller, purchasing group, material number or e-mail; aliases of the streamed answer are replaced by the real names; the filters are known and trivial statistics dropped; an invented alias becomes a neutral label; no asterisk in the rendered analysis.
- `tests/test_insights.py` (19): grouping rules (case, containment, root cause, meaning, opposite meanings), journal entries, tracking table, status change, root cause × impact and status, stakeholders merged, weak signals (phrases, context, model), filters of the Lab, embedding cache, a chart drawn once for the same data.

### **Limits**
The tests do not call the real DeepL or Mistral API and do not evaluate the quality of the translation or of the analysis; the detection limits of section 5 (abbreviated names, person names) are not covered. The look of the dashboard (CSS, browser-side switches) is checked by hand in a browser, not by tests.

### **References**
- [pytest Documentation](https://docs.pytest.org/)

---

---

## **📌 14. Excel Sheet for the Planners: openpyxl (v3.1.5)**

### **Why?**
 | **Criteria**          | **Justification** |
 |-----------------------|-------------------|
 | **Need**              | The planners validate the abbreviation library. They work in Excel, not in a CSV with twelve technical columns: `data/interim/abbreviations_for_review.xlsx` has three columns (abbreviation, meaning, language of the entries where it appears). |
 | **Simplicity**        | `pandas.ExcelWriter(engine="openpyxl")` writes the sheet; openpyxl only adds a bold header, column widths and a frozen first row (`abbreviations.write_sheet`). |

### **Alternatives Considered**
 | **Alternative** | **Rejection Reason** |
 |-----------------|----------------------|
 | CSV             | Opens in Excel, but with encoding and separator problems for umlauts and commas; no layout. |
 | XlsxWriter      | Equivalent for writing; openpyxl can also read the sheet back once the planners have filled it. |

### **References**
- [openpyxl Documentation](https://openpyxl.readthedocs.io/)

---

---

## **📌 15. Insights Lab: scikit-learn (v1.4.2) + SciPy (v1.17.1) + sentence-transformers (v3.4.1, with PyTorch 2.14.1 and transformers 4.57.6)**

### **Why?**
 | **Criteria**          | **Justification** |
 |-----------------------|-------------------|
 | **Need**              | Group near-identical comments (US24), measure the link between root cause and impact, and test whether early comments announce an impact (weak signals), with figures rather than impressions. |
 | **Explainability**    | TF-IDF of character n-grams (similarity of wordings), logistic regression (weights per word) and statistical tests give a reason for every result: why two comments were grouped, which words raise the risk, how likely a difference is to be chance. |
 | **Fit to the data**   | Comments are short and code-like (px13, LT, svc). Usuga-Cadavid et al. found classic models as good as transformers on such texts (their Company C), with far less effort and no GPU. |
 | **Honest evaluation** | 5-fold stratified cross-validation and the Matthews correlation coefficient (MCC), which stays meaningful when severe impacts are rare (10% of the items), as in the reference paper. |

### **Key Features Used in the Project** (`scripts/insights.py`)
- **`TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))`** + cosine similarity: similar wordings, robust to typos and plural forms; merges also require the same main root cause, and a word negated in one wording and affirmed in the other blocks the merge (*critical* / *non-critical*). 1,504 wordings → 1,057 issues.
- **`CountVectorizer(ngram_range=(2, 3))`** + **`scipy.stats.fisher_exact`**: phrases of 2 to 3 words of the first comment followed by a severe impact more often than for all items (lift, one-sided test).
- **`LogisticRegression(class_weight="balanced")`**, **`StratifiedKFold`**, **`matthews_corrcoef`**, **`roc_auc_score`**: can the first comment (and its root cause) predict a Production / Market impact? Out-of-fold probabilities rank the open items of the watch list.
- **`scipy.stats.chi2_contingency`** + Cramér's V: independence of root cause and impact (on the items whose impact was communicated only: an empty impact is unknown, not "no impact").
- **`SentenceTransformer("all-MiniLM-L6-v2")`** (sentence-transformers + PyTorch, run on this Mac, ~90 MB model, 7 s for 1,504 texts): grouping by meaning, as an option next to the character similarity. Chosen over an embedding API (e.g. Mistral) because no comment leaves the machine; English model because the groups are made on the English translation. The vectors are kept on disk (`data/interim/embedding_cache.npz`): the model (~8 s to load) is only used for comments it has never seen, so switching to *Meaning* takes under a second.

### **Alternatives Considered**
 | **Alternative** | **Rejection Reason** |
 |-----------------|----------------------|
 | Embedding API (Mistral `mistral-embed`, OpenAI) | No installation, but the comments would leave the machine (pseudonymised); the local model is fast enough. |
| Multilingual model on the German text | Larger (~470 MB); not needed since the groups are made on the English translation. Neither model knows the jargon (*expedite* / *pull-in*: 0.20). |
 | Fine-tuned transformers (BERT family), as in the reference paper | Too few labelled items (126 severe impacts) to fine-tune reliably; no gain expected on code-like texts. |
 | Topic models (LDA) | Topics are hard to name on very short comments; used only as a baseline in the reference paper. |

### **References**
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [SciPy statistics](https://docs.scipy.org/doc/scipy/reference/stats.html)
- Usuga-Cadavid, J. P., Lamouri, S., Grabot, B., Fortin, A.: *Using deep learning to value free-form text data for predictive maintenance*, International Journal of Production Research.

---

---

## **📌 16. Notebooks: Jupyter (v1.1.1)**

### **Why?**
 | **Criteria**          | **Justification** |
 |-----------------------|-------------------|
 | **Checkability**      | One notebook per stage (`data_cleaning.ipynb`, `translation.ipynb`, `llm_summary.ipynb`), one cell per pipeline step: the rows before and after each step and a few examples are shown, so a supervisor or a planner can follow the cleaning. |
 | **No duplicated code** | The cells call the functions of `scripts/`; the scripts remain the reference. |
 | **Safe "Run All"** | Calls to paid or external services are behind switches (`RUN_CLOUD`, `RUN_LOCAL` in `llm_summary.ipynb`; the DeepL call of `translation.ipynb` only translates what is not cached). |
 | **Confidentiality**   | Notebooks are pushed **without outputs** (outputs show real comments); `nbconvert` (installed with Jupyter) is used to run them from the command line and to clear the outputs. |

### **References**
- [Jupyter Documentation](https://docs.jupyter.org/)

---

---

## **📌 17. Versioning: Git and GitHub**

### **Why?**
 | **Criteria**          | **Justification** |
 |-----------------------|-------------------|
 | **Traceability**      | Every change is a commit with a message that explains it; the history shows how the prototype evolved (useful for the thesis). |
 | **Backup**            | The code and the documentation are pushed to GitHub after each change (branch `feat/comment-entries`, merged into `main`). |
 | **Confidentiality**   | The repository is **public**: only code, documentation and the reference tables are pushed. `data/`, `.env`, `venv/` and `outputs/` are excluded by `.gitignore`; the abbreviation library keeps no supplier name; notebooks are pushed without outputs. A DeepL key once pushed must be revoked (section 10). |

### **References**
- [Git Documentation](https://git-scm.com/doc)
- [GitHub: About .gitignore](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files)
