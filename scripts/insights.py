"""
Analyses of the Insights Lab sections of the dashboard (drawn by scripts/insights_pages.py). pandas / scikit-learn / scipy
only, no Streamlit here, so that every function can be tested and run in a notebook.

1. group_comments       near-identical comments counted as one issue (US24, Johannes 23.10)
2. weekly_items         root cause trend on ALL snapshot rows (data/item_tracking.csv), by root cause or stakeholder
   status_change        why the number of stock-outs changed between two weeks (US26)
3. root_cause_impact    root cause x impact: contingency table, chi-square, Cramér's V, lift per root cause
4. weak_signals         do the first comments of an item announce a Production / Market impact to come?

Grouping (1) is explainable on purpose: every merge has a reason shown in the dashboard.
  a. same text after normalisation (case, punctuation, "stock-out" / "stock out" / "stockout"), whatever the root cause;
  b. one text contained in the other (at most MAX_EXTRA_WORDS more words) AND the same main root cause:
     "px13" and "px13 - No stockout detected";
  c. similar wording (cosine of character 3-5-grams >= threshold) AND the same main root cause (or none entered):
     "Request for pull-in" and "Request to pull in the LT".
  A variant joins the group of the most frequent similar variant (no chaining A~B~C), so a group stays readable.
Classic text features are used rather than a deep model: on short, code-like texts (px13, LT, svc) they are
as good as transformers (Usuga-Cadavid et al., predictive maintenance logs) and every decision can be explained.
"""

import os
import re

import numpy as np
import pandas as pd

from entries import split_cell

NO_ROOT_CAUSE = "(no root cause)"
ITEM = ["MATERIAL_NUMBER", "VENDOR_ACCOUNT_NUMBER"]
SEVERE_IMPACTS = ["Market", "Production"]
NOT_COMMUNICATED = "Not communicated"   # no impact entered: unknown, NOT "no impact"
IMPACT_COLUMNS = ["Market", "Production", "Logistics", "None (no impact)", NOT_COMMUNICATED]
MAX_EXTRA_WORDS = 3
# Words that turn the next word into its opposite: "critical" / "non-critical" are never grouped
NEGATIONS = {"no", "not", "non", "without", "never", "cannot", "unable", "isn", "aren", "don", "doesn", "won"}
# Stakeholder responsible for a root cause = prefix of its name (legend from Johannes Höttler, 21.10)
STAKEHOLDERS = {"S": "Supplier", "B": "Buying", "P": "Procurement", "I": "Import/Export", "M": "Material Management",
                "PE": "PE", "PL": "Planning", "PR": "Production", "C": "Complaint Center", "TSM": "TSM", "E": "External"}


def stakeholder(root_cause):
    """'PL_Shortterm demand' -> 'Planning'; a root cause without prefix ('QM Lot') -> 'Other'."""
    if pd.isna(root_cause):
        return NO_ROOT_CAUSE
    prefix = str(root_cause).split("_", 1)[0]
    return STAKEHOLDERS.get(prefix, "Other") if "_" in str(root_cause) else "Other"


def cause_name(root_cause):
    """The root cause without its stakeholder prefix: 'PL_Internal Errors' -> 'Internal Errors'."""
    if pd.isna(root_cause):
        return root_cause
    prefix, _, rest = str(root_cause).partition("_")
    return rest if rest and prefix in STAKEHOLDERS else str(root_cause)


def merge_stakeholders(root_causes):
    """Same root cause whatever the stakeholder: 'PL_Internal Errors', 'P_Internal Error', 'S_Internal Error' ->
    one name (the most frequent spelling; singular / plural merged). Returns the series renamed."""
    names = root_causes.map(cause_name)
    key = names.map(lambda n: re.sub(r"s\b", "", n.lower()).strip() if pd.notna(n) else n)
    label = names.groupby(key).agg(lambda s: s.value_counts().index[0])
    return key.map(label)


# ------------------------------------------------------------------ 1. grouping of similar comments
def normalise(text):
    text = str(text).lower()
    text = re.sub(r"stock[\s-]?outs?", "stockout", text)
    text = re.sub(r"pull[\s-]?ins?", "pullin", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def polarity(norm):
    """(negated words, plain words) of a normalised text: 'stock coverage non critical' ->
    ({'critical'}, {'stock', 'coverage'}). The word right after a negation is negated."""
    negated, plain, after_negation = set(), set(), False
    for word in norm.split():
        if word in NEGATIONS:
            after_negation = True
            continue
        (negated if after_negation else plain).add(word)
        after_negation = False
    return negated, plain


def opposite(a, b):
    """True when a word is negated in one text and affirmed in the other (opposite meanings)."""
    return bool(a[0] & b[1]) or bool(b[0] & a[1])


def comment_entries(df, which="all", text_column="Comment_EN"):
    """One row per journal entry. The journal is written newest first (checked on the data: 171 cells
    newest first, 8 oldest first), so the LAST entry of a cell describes the problem and the FIRST one
    the latest status (Johannes, US25). which = 'all' | 'problem' | 'latest'."""
    rows = df.dropna(subset=[text_column]).copy()
    rows["_entries"] = rows[text_column].map(lambda c: [e for e in split_cell(c) if e])
    if which == "problem":
        rows["_entries"] = rows["_entries"].map(lambda e: e[-1:])
    elif which == "latest":
        rows["_entries"] = rows["_entries"].map(lambda e: e[:1])
    rows = rows.explode("_entries").dropna(subset=["_entries"])
    return rows.rename(columns={"_entries": "Entry"})


DATE = re.compile(r"\b\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b|\b\d{4}-\d{2}-\d{2}\b")


def neutral_label(wordings):
    """Group label without the dates and numbers that vary between its wordings: the group
    "25 estimated time of arrival 9/7-9/9", "40 estimated time of arrival 7/28"... is labelled
    "[n] estimated time of arrival [date]". Unchanged when the numbers are the same in all wordings."""
    label = wordings[0]
    without = {" ".join(re.sub(r"\d+", " ", normalise(w)).split()) for w in wordings}
    digits = {tuple(re.findall(r"\d+", w)) for w in wordings}
    if len(wordings) < 2 or len(digits) < 2 or len(without) == len(wordings):
        return label
    label = DATE.sub("[date]", label)
    label = re.sub(r"\[date\]\s*-\s*\[date\]", "[date]", label)
    return re.sub(r"\b\d+(?:[.,]\d+)?\b", "[n]", label)


def _main_root_cause(series):
    known = series.dropna()
    return known.mode().iloc[0] if len(known) else NO_ROOT_CAUSE


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 90 MB, runs on this Mac, no data leaves it
METHODS = {"chars": 0.55, "meaning": 0.80}   # default similarity threshold of each method


EMBEDDING_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "interim",
                               "embedding_cache.npz")


def embed(texts, model_name=EMBEDDING_MODEL, load_model=None, cache_path=EMBEDDING_CACHE):
    """Sentence embeddings (normalised), computed locally and kept on disk (data/interim/embedding_cache.npz):
    a text is encoded once; the model is only loaded when a text is not in the cache yet (it takes ~8 s).
    load_model = function returning the model (the dashboard passes its cached one)."""
    texts = list(texts)
    cached = {}
    if os.path.exists(cache_path):
        with np.load(cache_path) as data:
            if str(data["model"]) == model_name:
                cached = dict(zip(data["texts"].tolist(), data["vectors"]))
    missing = sorted({t for t in texts if t not in cached})
    if missing:
        if load_model is None:
            from sentence_transformers import SentenceTransformer
            load_model = lambda: SentenceTransformer(model_name)
        vectors = load_model().encode(missing, normalize_embeddings=True, batch_size=64)
        cached.update(zip(missing, np.asarray(vectors, dtype=np.float32)))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, model=model_name, texts=np.array(list(cached)), vectors=np.stack(list(cached.values())))
    return np.stack([cached[t] for t in texts])


def group_comments(entries, threshold=None, method="chars", embeddings=None):
    """Groups of near-identical entries. entries: one row per entry with Entry, Root_Cause, MATERIAL_NUMBER,
    VENDOR_NAME, Days_Active. method = 'chars' (similarity of the character 3-5-grams: same words, typos) or
    'meaning' (similarity of sentence embeddings: paraphrases such as "No stockout detected" / "No stockout
    apparent"); embeddings = function texts -> vectors for 'meaning' (default: embed). Returns (groups, variants):
      groups   one row per group: Group, Comment (most frequent wording), Variants, Occurrences, Materials,
               Vendors, Days_visible, Main_root_cause, Root_causes ("A 29 · B 22")
      variants one row per normalised wording with its Group and the reason it joined the group."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    entries = entries.assign(_norm=entries["Entry"].map(normalise))
    entries = entries[entries["_norm"] != ""]
    variants = (entries.groupby("_norm")
                .agg(Wording=("Entry", lambda s: s.value_counts().index[0]), Occurrences=("Entry", "size"),
                     Main_root_cause=("Root_Cause", _main_root_cause),
                     Spellings=("Entry", "nunique"))
                .sort_values(["Occurrences", "Wording"], ascending=[False, True]).reset_index())
    if variants.empty:
        return pd.DataFrame(), variants

    threshold = METHODS[method] if threshold is None else threshold
    if method == "meaning":
        vectors = np.asarray((embeddings or embed)(variants["Wording"]), dtype=np.float32)
        similarity_matrix = lambda: vectors @ vectors.T
        kind = "similar meaning"
    else:
        vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(variants["_norm"])
        similarity_matrix = lambda: (vectors @ vectors.T).toarray()
        kind = "similar wording"
    words = [set(n.split()) for n in variants["_norm"]]
    n_words = np.array([len(w) for w in words])
    causes = variants["Main_root_cause"].to_numpy()
    polarities = [polarity(n) for n in variants["_norm"]]
    sim = similarity_matrix()          # all pairs at once (1,500 wordings: about 2 million similarities)
    anchors, group_of, reason = [], [], []
    for i in range(len(variants)):
        best, best_reason = None, ""
        if anchors:
            a = np.asarray(anchors)
            scores = sim[i, a]
            same = causes[a] == causes[i]
            # 1. contained in the other wording (at most MAX_EXTRA_WORDS more words), same root cause
            close = a[same & (np.abs(n_words[a] - n_words[i]) <= MAX_EXTRA_WORDS)]
            contained = [c for c in close if (words[c] <= words[i] or words[i] <= words[c])
                         and not opposite(polarities[i], polarities[c])]
            if contained:
                best = max(contained, key=lambda c: sim[i, c])
                best_reason = "contained in the other wording, same root cause"
            else:
                # 2. similar wording / meaning, same root cause (or none entered): the most similar anchor
                no_cause = (causes[a] == NO_ROOT_CAUSE) | (causes[i] == NO_ROOT_CAUSE)
                ok = (same | no_cause) & (scores >= threshold)
                # the most similar one that does not say the opposite
                j = next((j for j in np.argsort(-np.where(ok, scores, -1.0))[:int(ok.sum())]
                          if not opposite(polarities[i], polarities[a[j]])), None)
                if j is not None:
                    best = int(a[j])
                    best_reason = f"{kind} ({scores[j]:.2f}), " + ("same root cause" if same[j] else "no root cause entered")
        if best is None:
            anchors.append(i)
            group_of.append(i)
            reason.append("group label (most frequent wording)")
        else:
            group_of.append(group_of[best])
            reason.append(best_reason)
    variants["Group"] = group_of
    variants["Reason"] = [r + (f" + {s} spellings (case, punctuation)" if s > 1 else "")
                          for r, s in zip(reason, variants["Spellings"])]

    entries = entries.merge(variants[["_norm", "Group"]], on="_norm")
    labels = variants.groupby("Group")["Wording"].agg(lambda w: neutral_label(list(w)))
    groups = (entries.groupby("Group")
              .agg(Occurrences=("Entry", "size"), Materials=("MATERIAL_NUMBER", "nunique"),
                   Vendors=("VENDOR_NAME", "nunique"), Days_visible=("Days_Active", "sum"),
                   Main_root_cause=("Root_Cause", _main_root_cause),
                   Root_causes=("Root_Cause", lambda s: " · ".join(f"{k} {v}" for k, v in
                                                                   s.fillna(NO_ROOT_CAUSE).value_counts().items())),
                   Variants=("_norm", "nunique"), Spellings=("Entry", "nunique"))
              .reset_index())
    groups.insert(1, "Comment", groups["Group"].map(labels))
    groups = groups.sort_values(["Occurrences", "Days_visible"], ascending=False).reset_index(drop=True)
    return groups, variants.drop(columns="_norm")


# ------------------------------------------------------------------ 2. trends on all snapshots
def weekly_items(tracking, by="Root_Cause"):
    """Number of items (material, vendor) carrying each root cause (or stakeholder) during each week,
    from every snapshot row, commented or not. Index = Monday of the week, one column per value."""
    rc = tracking.dropna(subset=["Root_Cause"]).copy()
    if rc.empty:
        return pd.DataFrame()
    rc["Week"] = pd.to_datetime(rc["Snapshot_Date"]).dt.to_period("W").dt.start_time
    rc["Stakeholder"] = rc["Root_Cause"].map(stakeholder)
    rc["Item"] = rc["MATERIAL_NUMBER"].astype(str) + "|" + rc["VENDOR_ACCOUNT_NUMBER"].astype(str)
    return rc.groupby(["Week", by])["Item"].nunique().unstack(fill_value=0)


def week_end_days(tracking):
    """{Monday of the week: last snapshot day of that week}."""
    days = pd.Series(pd.to_datetime(tracking["Snapshot_Date"].unique())).sort_values()
    return days.groupby(days.dt.to_period("W").dt.start_time).max().to_dict()


def status_change(tracking, day_a, day_b, statuses=("Actual Stock Out",)):
    """Items in the given status(es) on day_a vs day_b: which items entered / left, and their root causes.
    Returns (summary by root cause, items that entered with their root cause and comment)."""
    t = tracking.assign(Snapshot_Date=pd.to_datetime(tracking["Snapshot_Date"]))
    in_status = t[t["MATERIAL_STATUS"].isin(statuses)]
    a = in_status[in_status["Snapshot_Date"] == day_a].set_index(ITEM)
    b = in_status[in_status["Snapshot_Date"] == day_b].set_index(ITEM)
    entered = b.loc[b.index.difference(a.index)]
    left = a.loc[a.index.difference(b.index)]
    cause = lambda frame: frame["Root_Cause"].fillna(NO_ROOT_CAUSE)
    summary = pd.DataFrame({"Entered": cause(entered).value_counts(), "Left": cause(left).value_counts(),
                            f"On {day_b:%d.%m}": cause(b).value_counts()}).fillna(0).astype(int)
    summary["Net change"] = summary["Entered"] - summary["Left"]
    summary = summary.sort_values(["Net change", "Entered"], ascending=False)
    summary.index.name = "Root cause"
    return summary.reset_index(), entered.reset_index(), len(a), len(b)


# ------------------------------------------------------------------ 3. root cause x impact
def item_root_cause_impact(tracking):
    """One row per (item, root cause): the most severe impact entered WHILE this root cause was entered for the item
    ('Not communicated' if none). An impact entered under another root cause belongs to that one: so a root cause
    gives the same figures with or without the Root Cause filter (which keeps the rows of that cause)."""
    order = {level: i for i, level in enumerate(IMPACT_COLUMNS)}
    rows = tracking.dropna(subset=["Root_Cause"])
    worst = (rows.dropna(subset=["Impact_Level"]).assign(_r=lambda d: d["Impact_Level"].map(order))
             .sort_values("_r").groupby(ITEM + ["Root_Cause"])["Impact_Level"].first())
    pairs = rows.drop_duplicates(ITEM + ["Root_Cause"])[ITEM + ["Root_Cause"]]
    pairs = pairs.join(worst, on=ITEM + ["Root_Cause"])
    pairs["Impact_Level"] = pairs["Impact_Level"].fillna(NOT_COMMUNICATED)
    return pairs


def root_cause_impact(tracking, min_items=5):
    """Contingency table root cause x impact (items), the chi-square test of independence on the items whose impact was communicated,
    Cramér's V, and per root cause: share of severe impacts (Production + Market) among the items whose
    impact was communicated (items without impact entered are left out: their impact is unknown),
    lift vs all root causes and Fisher's exact test (this root cause vs the others). With too few items for the
    chi-square test (e.g. one root cause selected), the test gives chi2 = None and the rest still works; without any
    communicated impact: ValueError."""
    from scipy.stats import chi2_contingency, fisher_exact

    pairs = item_root_cause_impact(tracking)
    table = pd.crosstab(pairs["Root_Cause"], pairs["Impact_Level"]).reindex(columns=IMPACT_COLUMNS, fill_value=0)
    assessed = table.drop(columns=NOT_COMMUNICATED)
    tested = assessed[assessed.sum(axis=1) >= min_items]
    tested = tested.loc[:, tested.sum() > 0]
    if not assessed.to_numpy().sum():
        raise ValueError("no communicated impact")
    n = tested.to_numpy().sum()
    chi2, p, dof, cramers_v = None, float("nan"), 0, None
    if min(tested.shape) >= 2:
        chi2, p, dof, _ = chi2_contingency(tested)
        chi2, cramers_v = round(float(chi2), 1), round(float(np.sqrt(chi2 / (n * (min(tested.shape) - 1)))), 2)

    severe = assessed[SEVERE_IMPACTS].sum(axis=1)
    total_assessed = assessed.sum(axis=1)
    base = severe.sum() / total_assessed.sum()
    rows = []
    for cause in table.index:
        s, a = int(severe[cause]), int(total_assessed[cause])
        other_s, other_a = int(severe.sum() - s), int(total_assessed.sum() - a)
        p_cause = fisher_exact([[s, a - s], [other_s, other_a - other_s]], alternative="two-sided")[1] if a else np.nan
        rows.append({"Root cause": cause, "Stakeholder": stakeholder(cause), "Items": int(table.loc[cause].sum()),
                     "Impact communicated": a, "Severe (Prod.+Market)": s,
                     "% severe": round(s / a * 100, 1) if a else np.nan,
                     "Lift": round(s / a / base, 2) if a and base else np.nan,
                     "p-value": round(p_cause, 4) if a else np.nan})
    per_cause = pd.DataFrame(rows).sort_values(["p-value", "Items"], ascending=[True, False]).reset_index(drop=True)
    test = {"chi2": chi2, "dof": int(dof), "p": float(p), "cramers_v": cramers_v,
            "n_items": int(n), "n_causes": len(tested), "severe_base_pct": round(base * 100, 1)}
    return table, per_cause, test


STATUS_ORDER = ["Actual Stock Out", "Potential Stock Out", "Below Safety", "Good Part"]


def root_cause_status(tracking):
    """Items per root cause and material status: the worst status the item had while the root cause was entered
    ('Unknown Status (Good Part)' counts as Good Part)."""
    rank = {status: i for i, status in enumerate(STATUS_ORDER)}
    rows = tracking.dropna(subset=["Root_Cause"]).assign(
        _status=lambda d: d["MATERIAL_STATUS"].replace({"Unknown Status (Good Part)": "Good Part"}))
    rows = rows[rows["_status"].isin(STATUS_ORDER)].assign(_r=lambda d: d["_status"].map(rank))
    worst = rows.sort_values("_r").drop_duplicates(ITEM + ["Root_Cause"])
    return pd.crosstab(worst["Root_Cause"], worst["_status"]).reindex(columns=STATUS_ORDER, fill_value=0)


# ------------------------------------------------------------------ 4. weak signals
# A weak signal = something in the FIRST comment on an item (its wording, the root cause, the vendor, the
# material status at that time) that is followed by a Production / Market impact communicated LATER.
# Single words say little ("pull"): phrases of 2 to 3 words are used, with their context.
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z0-9]*(?:-[a-zA-Z0-9]+)*\b"   # "pull-in" stays one word


def weak_signal_items(tracking, comments_en):
    """One row per item with a comment: its first comment (in English) and the context at that time (root cause,
    vendor, material status), and whether a severe impact (Production / Market) was communicated LATER
    (False = none communicated, which may also mean that the impact was never entered). Items already severe
    on the day of their first comment are left out: there is nothing to anticipate."""
    t = tracking.assign(Snapshot_Date=pd.to_datetime(tracking["Snapshot_Date"]))
    commented = t.dropna(subset=["Comment"]).sort_values("Snapshot_Date")
    first = commented.groupby(ITEM).first()[[c for c in ["Snapshot_Date", "Comment", "Root_Cause", "VENDOR_NAME",
                                                         "MATERIAL_STATUS", "MRP_CONTROLLER", "MATERIAL_DESC", "PLANT"]
                                             if c in commented]]
    first = first.rename(columns={"Snapshot_Date": "First_comment"})
    severe_day = t[t["Impact_Level"].isin(SEVERE_IMPACTS)].groupby(ITEM)["Snapshot_Date"].min().rename("First_severe")
    last = t.sort_values("Snapshot_Date").groupby(ITEM).last()
    items = first.join(severe_day).join(last["Snapshot_Date"].rename("Last_seen"))
    items = items.join(last["MATERIAL_STATUS"].rename("Status_now")).reset_index()
    items = items[~(items["First_severe"] <= items["First_comment"])]
    items["Severe_later"] = items["First_severe"].notna()
    items["Days_before_impact"] = (items["First_severe"] - items["First_comment"]).dt.days
    items["Text"] = (items["Comment"].map(comments_en).fillna(items["Comment"]).str.replace("¦▸", " ", regex=False)
                     .str.replace(r"(?i)\bpull[\s-]?in", "pull-in", regex=True))   # one spelling for the phrases
    items["Root_Cause"] = items["Root_Cause"].fillna(NO_ROOT_CAUSE)
    items["VENDOR_NAME"] = items["VENDOR_NAME"].fillna("Unknown vendor")
    items["MATERIAL_STATUS"] = items["MATERIAL_STATUS"].fillna("Unknown")
    return items.reset_index(drop=True)


def _fisher_greater(k, m, n_pos, n):
    from scipy.stats import fisher_exact
    return fisher_exact([[k, m - k], [n_pos - k, n - m - (n_pos - k)]], alternative="greater")[1]


def _top(values, n=2):
    counts = pd.Series(values).value_counts()
    return ", ".join(f"{k} ({v})" for k, v in counts.head(n).items())


def signal_phrases(items, min_items=4, max_p=0.1):
    """Phrases of 2 to 3 words of the first comments followed by a severe impact more often than average, with
    their context: main root cause, main vendors, an example. A phrase found in exactly the same items as a
    phrase already listed (e.g. "request pull-in" / "request pull-in lt") is shown once."""
    from sklearn.feature_extraction.text import CountVectorizer

    if len(items) < min_items:
        return pd.DataFrame()
    vec = CountVectorizer(binary=True, ngram_range=(2, 3), min_df=min_items, stop_words="english", token_pattern=TOKEN)
    try:
        x = vec.fit_transform(items["Text"].str.lower()).tocsc()
    except ValueError:          # no phrase frequent enough
        return pd.DataFrame()
    y = items["Severe_later"].to_numpy()
    n_pos, n = int(y.sum()), len(y)
    rows = []
    for j, phrase in enumerate(vec.get_feature_names_out()):
        idx = x[:, j].nonzero()[0]
        k, m = int(y[idx].sum()), len(idx)
        if not n_pos or k / m <= n_pos / n:
            continue
        p = _fisher_greater(k, m, n_pos, n)
        if p > max_p:
            continue
        sub = items.iloc[idx]
        example = sub.loc[sub["Severe_later"], "Text"] if k else sub["Text"]
        rows.append({"Phrase": phrase, "Items": m, "Impact later": k, "% impact later": round(k / m * 100, 1),
                     "× average": round(k / m / (n_pos / n), 1), "p-value": round(p, 4),
                     "Main root cause": _top(sub["Root_Cause"], 1), "Main vendors": _top(sub["VENDOR_NAME"]),
                     "Example": example.loc[example.str.len().idxmin()][:160], "_items": frozenset(idx)})
    if not rows:
        return pd.DataFrame()
    phrases = pd.DataFrame(rows).sort_values(["p-value", "Phrase"], key=lambda c: c if c.name == "p-value" else -c.str.len())
    phrases = phrases.drop_duplicates("_items").drop(columns="_items")
    return phrases.sort_values(["p-value", "× average"], ascending=[True, False]).reset_index(drop=True)


def signal_context(items, column, min_items=3):
    """Same question for a context column (vendor, root cause, material status at the first comment):
    share of the items followed by a severe impact, compared with the average."""
    y = items["Severe_later"]
    n_pos, n = int(y.sum()), len(y)
    rows = []
    for value, group in items.groupby(column):
        k, m = int(group["Severe_later"].sum()), len(group)
        if m < min_items:
            continue
        rows.append({column: value, "Items": m, "Impact later": k, "% impact later": round(k / m * 100, 1),
                     "× average": round(k / m / (n_pos / n), 1) if n_pos else np.nan,
                     "p-value": round(_fisher_greater(k, m, n_pos, n), 4) if n_pos else np.nan})
    return pd.DataFrame(rows).sort_values(["% impact later", "Items"], ascending=False).reset_index(drop=True) \
        if rows else pd.DataFrame()


CONTEXT = {"Root_Cause": "root cause", "VENDOR_NAME": "vendor", "MATERIAL_STATUS": "status"}


def signal_model(items, folds=5, seed=0):
    """Can the first comment and its context anticipate a severe impact? Logistic regression (balanced classes)
    on the phrases of the comment (2-3 words, TF-IDF) and the context (root cause, vendor, material status),
    5-fold stratified cross-validation (every item is scored by a model that never saw it).
    Returns a dict:
      scores   per model: share of the items followed by an impact among the 10% the model ranks highest
               (vs the average), share of all impacts found there, MCC and ROC AUC (statistical details)
      risk     out-of-fold risk of each item (comment + context model)
      reasons  per item, the 3 elements that raise its risk most (phrases, root cause, vendor, status)
      weights  weight of every element in the final model."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder

    y = items["Severe_later"].to_numpy().astype(int)
    if y.sum() < folds or (len(y) - y.sum()) < folds:
        return {}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    models = {"cause": ("Root cause only", ["Root_Cause"], False),
              "text": ("Comment phrases only", [], True),
              "all": ("Comment phrases + context (root cause, vendor, status)", list(CONTEXT), True)}

    def features(train, test, columns, text):
        parts_train, parts_test, names = [], [], []
        if text:
            # min_df=3: a phrase must appear in 3 first comments (keeps isolated names out)
            tf = TfidfVectorizer(ngram_range=(2, 3), min_df=3, stop_words="english", token_pattern=TOKEN, sublinear_tf=True)
            parts_train.append(tf.fit_transform(items["Text"].iloc[train]))
            parts_test.append(tf.transform(items["Text"].iloc[test]))
            names += [f'phrase "{t}"' for t in tf.get_feature_names_out()]
        for column in columns:
            oh = OneHotEncoder(handle_unknown="ignore", min_frequency=3)
            parts_train.append(oh.fit_transform(items[[column]].iloc[train]))
            parts_test.append(oh.transform(items[[column]].iloc[test]))
            names += [f"{CONTEXT[column]}: " + ("other (rare)" if "infrequent" in c else c[len(column) + 1:])
                      for c in oh.get_feature_names_out([column])]
        return hstack(parts_train).tocsr(), hstack(parts_test).tocsr(), names

    base = y.mean()
    top_n = max(1, int(round(len(y) * 0.1)))
    scores, risk = [], pd.Series(np.nan, index=items.index)
    for key, (label, columns, text) in models.items():
        prob = np.zeros(len(y))
        for train, test in cv.split(items, y):
            x_train, x_test, _ = features(train, test, columns, text)
            model = LogisticRegression(class_weight="balanced", max_iter=2000).fit(x_train, y[train])
            prob[test] = model.predict_proba(x_test)[:, 1]
        top = np.argsort(-prob)[:top_n]
        hit = y[top].mean()
        scores.append({"Model": label, "Impact rate in the top 10%": round(hit * 100, 1),
                       "× average": round(hit / base, 1), "Share of all impacts found": round(y[top].sum() / y.sum() * 100, 1),
                       "MCC": round(matthews_corrcoef(y, (prob >= 0.5).astype(int)), 3),
                       "ROC AUC": round(roc_auc_score(y, prob), 3)})
        if key == "all":
            risk[:] = prob
    rows = np.arange(len(y))
    x_all, _, names = features(rows, rows[:1], list(CONTEXT), True)
    model = LogisticRegression(class_weight="balanced", max_iter=2000).fit(x_all, y)
    coef = model.coef_[0]
    contrib = x_all.multiply(coef).tocsr()      # contribution of each element to each item's risk
    names = np.array(names)
    reasons = []
    for i in range(len(y)):
        row = contrib.getrow(i)
        order = np.argsort(-row.data)
        reasons.append(", ".join(names[row.indices[j]] for j in order[:3] if row.data[j] > 0))
    weights = pd.DataFrame({"Element": names, "Weight": coef.round(3)}).sort_values("Weight", ascending=False)
    scores = pd.DataFrame(scores)
    scores.attrs.update(n_items=len(y), n_severe=int(y.sum()), base_pct=round(base * 100, 1), top_n=top_n)
    return {"scores": scores, "risk": risk, "reasons": pd.Series(reasons, index=items.index),
            "weights": weights.reset_index(drop=True)}
