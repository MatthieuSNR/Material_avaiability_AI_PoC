"""
Insights Lab: the analyses built on the tracking table and the journal entries of the comments.
The sidebar filters (and the timeframe) of app.py apply everywhere. The analyses are in scripts/insights.py.

Sections of the dashboard (one page; the first two replaced the first versions once validated):
  frequent-comments  Most frequent comments, near-identical wordings grouped (US24)
  trends             Root cause trends on all snapshot rows + what changed between two weeks (US26)
  cause-impact       Which root causes lead to which impact and material status
  watchlist          Open items most at risk; under it, for analysts, what the ranking rests on: which first entries
                     and contexts (vendor, root cause, status) announce an impact to come

Written for planners, buyers and managers: plain-language insights and coloured charts first. The settings and
the "Show explanations" switch are behind a small "Options" button on the right of each section title; Impact by
Root Cause (two charts) shows one part at a time, chosen in a switch that works in the browser; the mechanics and
statistics are in folded panels "for analysts", with pages (tabs) when they are many; a "?" on every indicator. smart_context() gives the same findings to the Smart Analysis.
"""

import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import insights as ins
import ui
from tracking import CURRENT_PATH, STATUS_RANK, TRACKING_PATH

IMPACT_COLORS = {"Market": "#9B1C1C", "Production": "#E0443A", "Logistics": "#F5C518", "None (no impact)": "#2FA84F"}
SIGNIFICANT = 0.05

UNMERGED = "Unmerged comments"   # option of Most Frequent Comments: nothing grouped

# Timeframe, in the Options of each section. A comment or an item is in the period if it was visible during it.
LAST_SNAPSHOT = "Last snapshot"   # first choice of Top Root Causes
TIMEFRAMES = ["All snapshots", "Last 4 weeks", "Last 8 weeks", "Last 16 weeks", "Custom period"]
TIMEFRAME_KEYS = ["tf_top", "tf_frequent", "tf_trends", "tf_impact", "tf_watchlist", "tf_smart"]

# Column "times the rate of the whole selection" (the reference of every comparison: all the items of the
# selection, i.e. the sidebar filters and the timeframe)
LIFT = "× whole selection"

# Tooltips ("?") of the indicators: what it is, and how to read it
HELP = {
    "lift": "How many times more often than for all the items of the selection. 1.0 = the same; 2.0 = twice as "
            "often; 0.5 = half as often.",
    "p": "How likely the difference would appear by pure chance. Below 0.05 (5%): unlikely to be chance, the "
         "link is considered real. Above: it may be chance, read with care.",
    "significant": "Yes = the difference with the whole selection is unlikely to be chance (p-value below 0.05).",
    "severe": "Share of the items whose communicated impact is Production or Market.",
    "communicated": "Items for which planning entered an impact. Items without an impact entered are left out: "
                    "their impact is unknown (not communicated), which does not mean there is none.",
    "impact_later": "Share of the items that received a Production or Market impact AFTER this first comment.",
}


def _with_labels(t):
    """The columns of the sidebar filters, on the tracking table or the last snapshot."""
    t["Snapshot_Date"] = pd.to_datetime(t["Snapshot_Date"])
    t["MATERIAL_LABEL"] = t["MATERIAL_DESC"].fillna("No description") + " — " + t["MATERIAL_NUMBER"].astype("int64").astype(str)
    t["VENDOR_LABEL"] = (t["VENDOR_NAME"].fillna("Unknown vendor") + " — "
                         + t["VENDOR_ACCOUNT_NUMBER"].astype("Int64").astype(str).replace("<NA>", "no vendor"))
    t["MRP_CONTROLLER"] = t["MRP_CONTROLLER"].fillna("Unknown")
    t["PURCHASING_GROUP"] = t["PURCHASING_GROUP"].fillna("Unknown")
    t["Stakeholder"] = t["Root_Cause"].map(ins.stakeholder).where(t["Root_Cause"].notna())
    return t


@st.cache_resource   # one shared table, not copied at every use (read only: every step makes new frames)
def load_tracking():
    if not os.path.exists(TRACKING_PATH):
        return None
    return _with_labels(pd.read_csv(TRACKING_PATH))


@st.cache_resource
def load_current():
    """All the order lines of the last snapshot (Overview); None when scripts/tracking.py has not built them."""
    if not os.path.exists(CURRENT_PATH):
        return None
    return _with_labels(pd.read_csv(CURRENT_PATH))


def timeframe_option(panel, key, choices=TIMEFRAMES):
    """The timeframe of a section, chosen in its Options: returns (choice, period); period = (start, end), or None
    for all the snapshots (and for LAST_SNAPSHOT). One day: a custom period that starts and ends on that day."""
    tracking = load_tracking()
    first, last = tracking["Snapshot_Date"].min(), tracking["Snapshot_Date"].max()
    choice = panel.selectbox("Timeframe:", choices, key=key,
                             help="Which snapshots this section looks at. For a month, a quarter or one day: Custom period.")
    if choice == "Custom period":
        dates = panel.date_input("Date range:", value=(first.date(), last.date()), min_value=first.date(),
                                 max_value=last.date(), key=f"{key}_dates")
        if isinstance(dates, tuple) and len(dates) == 2:   # a tuple of two only when BOTH dates are selected
            return choice, (pd.to_datetime(dates[0]), pd.to_datetime(dates[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    elif choice.split()[1].isdigit():   # "Last 4 weeks"
        return choice, (last - pd.Timedelta(weeks=int(choice.split()[1])), last)
    return choice, None


def comments_in_period(comments, period):
    """The comments visible at least once during the period ([First_Seen, Last_Seen] overlaps it)."""
    if period is None:
        return comments
    return comments[(comments["Last_Seen"] >= period[0]) & (comments["First_Seen"] <= period[1])]


def status_of_materials(comments, tracking):
    """The status of the material of each comment on the last day it was visible: the worst of all its lines that
    day, as in the tiles of the Overview. A comment row of the file only knows the status of one line."""
    rank = tracking["MATERIAL_STATUS"].map(STATUS_RANK).fillna(9)
    worst = (tracking.assign(_rank=rank).sort_values("_rank", kind="stable")
             .groupby(["Snapshot_Date", "MATERIAL_NUMBER"])["MATERIAL_STATUS"].first())
    keys = pd.MultiIndex.from_arrays([comments["Last_Seen"], comments["MATERIAL_NUMBER"]])
    return pd.Series(worst.reindex(keys).to_numpy(), index=comments.index).fillna(comments["MATERIAL_STATUS"])


def label_key(label):
    """'Description — 1234567' -> '1234567' (the number after the last ' — ' of a filter label)."""
    return str(label).rsplit(" — ", 1)[-1]


def filter_tracking(tracking, selections, period=None):
    """The sidebar filters and the timeframe, on the tracking table (one row per day and item):
    * MRP controller, purchasing group, stakeholder, root cause: on the rows;
    * vendor and material: on their number (the label of the filter ends with it); MATERIAL_NUMBER: these numbers
      (the materials of the Material Status filter);
    * impact: on the items whose impact over the timeframe is one of the selected levels; 'Not communicated'
      = items with no impact entered at all ('None (no impact)' = planning entered that there is none)."""
    if period:
        start, end = period
        tracking = tracking[(tracking["Snapshot_Date"] >= start) & (tracking["Snapshot_Date"] <= end)]
    for column in ("MRP_CONTROLLER", "PURCHASING_GROUP", "Stakeholder", "Root_Cause"):
        if selections.get(column):
            tracking = tracking[tracking[column].isin(selections[column])]
    if selections.get("VENDOR_LABEL"):
        keys = {label_key(v) for v in selections["VENDOR_LABEL"]}
        tracking = tracking[tracking["VENDOR_LABEL"].map(label_key).isin(keys)]
    if selections.get("MATERIAL_LABEL"):
        keys = {label_key(v) for v in selections["MATERIAL_LABEL"]}
        tracking = tracking[tracking["MATERIAL_NUMBER"].astype("int64").astype(str).isin(keys)]
    if selections.get("MATERIAL_NUMBER"):
        tracking = tracking[tracking["MATERIAL_NUMBER"].isin(selections["MATERIAL_NUMBER"])]
    if selections.get("Impact_Level"):
        wanted = set(selections["Impact_Level"])
        levels = tracking.groupby(ins.ITEM)["Impact_Level"].agg(lambda s: set(s.dropna()))
        keep = levels[levels.map(lambda s: bool(s & wanted) or (not s and ins.NOT_COMMUNICATED in wanted))].index
        tracking = tracking.set_index(ins.ITEM).loc[lambda d: d.index.isin(keep)].reset_index()
    return tracking


# ------------------------------------------------------------------ explanations
_explain_section = ""


def options_with_columns(anchor_id, title):
    """options() for a section with a table: settings on the left of the panel (returned first), the '▥ Columns' of
    the table beside them."""
    panel = ui.section_with_options(anchor_id, title)
    global _explain_section
    _explain_section = anchor_id
    with panel:
        left, right = st.columns([1, 1.7], gap="large")
    with left:
        ui.explain_switch(anchor_id)
    return left, right


def options(anchor_id, title):
    """Blue header of a section with a small 'Options' button on its right; the button opens the settings of
    the section and its 'Show explanations' switch (off by default). Returns the panel so that the section adds
    its own settings to it (`with panel:`). One switch per section.
    The switch works in the browser (ui.explain_switch): showing the explanations does not rerun the dashboard."""
    global _explain_section
    _explain_section = anchor_id
    panel = ui.section_with_options(anchor_id, title)
    with panel:
        ui.explain_switch(anchor_id)
    return panel


def explain(text):
    """A 'how to read' box of the current section, hidden until its 'Show explanations' switch is on."""
    ui.explain_box(_explain_section, text)


def details(title="Statistical details (for analysts)"):
    return st.expander(title, expanded=False)


def fold(title):
    """A blue panel header that opens on click (arrow on the right): mechanics and details, folded by default."""
    panel = st.expander(title, expanded=False)
    panel.markdown('<span class="ion-fold"></span>', unsafe_allow_html=True)   # marker for the blue style (ui.py)
    return panel


def insights_box(lines):
    """The main findings of a page, in plain words."""
    if lines:
        st.success("\n\n".join(lines))


@st.cache_resource(show_spinner="Loading the language model (first time only)...")
def _embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(ins.EMBEDDING_MODEL)


@st.cache_data(show_spinner="Grouping similar comments...")
def _groups(entries, threshold, method):
    # vectors kept on disk: the model is only loaded for comments it has never seen
    embeddings = (lambda texts: ins.embed(texts, load_model=_embedding_model)) if method == "meaning" else None
    return ins.group_comments(entries, threshold, method, embeddings)


def _stacked_rows(shares, counts, colors):
    """The segments of the 100% bars, for ui.clickable_stacked: one row per bar and part, with their position."""
    rows = []
    for cause in shares.index:
        start = 0.0
        for part in shares.columns:
            share = float(shares.at[cause, part])
            rows.append({"Cause": cause, "Part": part, "Share": round(share, 1), "Items": int(counts.at[cause, part]),
                         "Start": start, "End": start + share, "Mid": start + share / 2,
                         "Label": f"{share:.0f}%" if share >= 7 else "",
                         "Ink": "black" if colors[part] == "#F5C518" else "white"})
            start += share
    return pd.DataFrame(rows)


def _pick_segment(chart_key, cause_filter, part_filter):
    """A click on a segment of a bar: the sidebar filters take its root cause (or stakeholder) and its part. It runs
    before the next rerun, when the filters may still be set."""
    points = st.session_state[chart_key]["selection"].get("pick") or []
    if points:
        st.session_state[cause_filter] = [points[0]["Cause"]]
        st.session_state[part_filter] = [points[0]["Part"]]


# ------------------------------------------------------------------ main page: most frequent comments
def frequent_comments_section(filtered_df, selection_name, unmerged_table):
    """unmerged_table() draws the comments one by one (option 'Unmerged comments')."""
    settings, columns_panel = options_with_columns("frequent-comments", "Most Frequent Comments — " + selection_name)
    _, period = timeframe_option(settings, "tf_frequent")
    filtered_df = comments_in_period(filtered_df, period)
    method_label = settings.selectbox("Group comments with a similar:",
                                ["Wording (characters)", "Meaning (language model)", UNMERGED],
                                key="lab_method",
                                help="Wording: same words, typos, plural forms. Meaning: a small language model "
                                     "(run on this machine, no data leaves it) also groups paraphrases such as "
                                     "'No stockout detected' / 'No stockout apparent'. Unmerged comments: every "
                                     "comment on its own line, nothing grouped.")
    if method_label == UNMERGED:
        return unmerged_table(columns_panel, filtered_df)
    top_n = int(settings.selectbox("Select a view:", ["Top 10", "Top 20", "Top 50"], key="lab_top").split()[1])
    method = "meaning" if method_label.startswith("Meaning") else "chars"
    threshold = settings.slider("How similar to be grouped:", 0.40, 0.95, ins.METHODS[method], 0.05, key=f"lab_threshold_{method}",
                          help="Higher = stricter (fewer, more certain groups). Lower = more comments grouped. "
                               "Comments with different root causes are never grouped, except when only the case "
                               "or the punctuation differs.")
    explain("Comments that say the same thing in slightly different words are counted as **one issue**, e.g. `PX13` "
            "and `px13 - No stockout detected`. A comment with several dated entries counts each entry.\n\n"
            "Each line is an issue; open 'Why were these wordings grouped?' under the table to see which wordings "
            "were grouped, and why.")
    entries = ins.comment_entries(filtered_df)[["Entry", "Root_Cause", "MATERIAL_NUMBER", "VENDOR_NAME", "Days_Active"]]
    if entries.empty:
        st.info("No comments match the current filters.")
        return
    groups, variants = _groups(entries.reset_index(drop=True), threshold, method)

    m1, m2, m3 = st.columns(3)
    m1.metric("Entries", f"{len(entries):,}", help="Journal entries of the comments of the selection (a comment "
                                                    "cell can hold several dated entries).")
    m2.metric("Distinct wordings", f"{entries['Entry'].nunique():,}", help="Different texts, before grouping.")
    m3.metric("Issues after grouping", f"{len(groups):,}", delta=f"{len(groups) - entries['Entry'].nunique():,}",
              delta_color="off", help="Distinct issues once the similar wordings are grouped. The smaller this "
                                      "number compared with the wordings, the more repetition in the comments.")

    table = groups.head(top_n).copy()
    table.insert(3, "% of entries", (table["Occurrences"] / len(entries) * 100).round(1))
    ui.picked_table("t_grouped", panel=columns_panel, frame=
        table.drop(columns=["Group", "Spellings"]).rename(columns={
            "Comment": "Comment (EN, most frequent wording)", "Variants": "Wordings grouped",
            "Days_visible": "Days visible (total)", "Main_root_cause": "Main root cause",
            "Root_causes": "Root causes (entries)"}),
        width="stretch", hide_index=True,
        column_config={
            "Comment (EN, most frequent wording)": st.column_config.TextColumn(width="large"),
            "Occurrences": st.column_config.NumberColumn(help="Number of entries of this issue."),
            "% of entries": st.column_config.NumberColumn(help="Share of all the entries of the selection."),
            "Wordings grouped": st.column_config.NumberColumn(help="How many different wordings were grouped."),
            "Days visible (total)": st.column_config.NumberColumn(help="Total days these comments stayed in the dashboard."),
            "Root causes (entries)": st.column_config.TextColumn(width="large", help="Root causes entered with these "
                                                                                     "comments, with their count.")})

    with details("Why were these wordings grouped? (for analysts)"):
        choice = st.selectbox("Issue:", table["Comment"].tolist(), key="lab_group")
        group_id = table.loc[table["Comment"] == choice, "Group"].iloc[0]
        st.dataframe(variants[variants["Group"] == group_id][["Wording", "Occurrences", "Main_root_cause", "Reason"]]
                     .rename(columns={"Main_root_cause": "Main root cause"}),
                     width="stretch", hide_index=True,
                     column_config={"Wording": st.column_config.TextColumn(width="large"),
                                    "Reason": st.column_config.TextColumn(width="large")})
        st.markdown("**How the grouping works**")
        st.markdown(
            "1. **Same text after normalisation**: case, punctuation and the spellings of *stock-out* and *pull-in* "
            "are ignored (`PX13` = `px13`), whatever the root cause.\n\n"
            "2. **One wording contained in the other** (at most 3 more words) **and the same main root cause**: "
            "`px13` and `px13 - No stockout detected`.\n\n"
            "3. **Similar wording or meaning** (above the threshold) **and the same main root cause**, or no root "
            "cause entered: `Request for Pull-in` and `Request to pull in the LT`.\n\n"
            "A wording joins the group of the most frequent similar wording (no chains). Two ways to measure the "
            "similarity: **wording** (character sequences: same words, typos) or **meaning** (a small English "
            "language model, *all-MiniLM-L6-v2*, run on this machine: it also groups paraphrases such as *No stockout "
            "detected* / *No stockout apparent*, 0.89). Neither knows the supply-chain jargon: *expedite* and "
            "*pull-in* stay apart (0.20 with the language model). The groups are made on the English translation.")


# ------------------------------------------------------------------ main page: root cause trends
def _trend_chart(weekly, by, selection_name):
    """One line per root cause (or stakeholder): items per week."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = plt.cm.tab20.colors if len(weekly.columns) > 8 else None
    for i, name in enumerate(weekly.columns):
        ax.plot(weekly.index, weekly[name], marker="o", markersize=3, label=name,
                color=colors[i % 20] if colors else None)
    ax.set_xlim(weekly.index.min() - pd.Timedelta(days=3), weekly.index.max() + pd.Timedelta(days=3))
    ax.set_ylim(bottom=0)
    ax.set_title(ui.chart_title(f"Items per {by.lower()} and week", selection_name, 110))
    ax.set_ylabel("Items with this root cause during the week")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    return fig


def page_trends(df, selections, selection_name):
    panel = options("trends", "Root Cause Trends — " + selection_name)
    _, period = timeframe_option(panel, "tf_trends")
    tracking = _tracking(df, selections, period)[1]
    explain("How many items (a material from a vendor) carried each root cause, week by week, over the timeframe "
            "chosen in the Options. Every snapshot row counts, also the ones without a comment.\n\n"
            "A rising line = a cause that concerns more and more items.")
    if tracking.empty:
        st.info("No data for the current filters and timeframe.")
        return
    c1, c2 = panel.columns([1, 3])
    by = c1.radio("Group by:", ["Root cause", "Stakeholder"], key="lab_trend_by",
                  help="Stakeholder = who is responsible, from the prefix of the root cause (S = Supplier, "
                       "PL = Planning, P = Procurement, B = Buying...; legend of Johannes Höttler).")
    weekly = ins.weekly_items(tracking, "Root_Cause" if by == "Root cause" else "Stakeholder")
    if weekly.empty:
        st.info("No root cause for the current filters.")
    else:
        ranking = weekly.sum().sort_values(ascending=False).index.tolist()
        shown = c2.multiselect("Lines to display (default: the 6 largest):", ranking, default=ranking[:6],
                               key=f"lab_lines_{by}")
        if shown:
            ui.chart(_trend_chart, weekly[shown], by, selection_name, name="root_cause_trends")
            first, last = weekly.index[0], weekly.index[-1]
            change = pd.DataFrame({f"Week of {first:%d.%m}": weekly.loc[first, shown],
                                   f"Week of {last:%d.%m}": weekly.loc[last, shown],
                                   "Peak": weekly[shown].max(), "Peak week": weekly[shown].idxmax().dt.strftime("%d.%m")})
            change["Change"] = change.iloc[:, 1] - change.iloc[:, 0]
            st.dataframe(change.reset_index().rename(columns={"index": by, "Root_Cause": by, "Stakeholder": by}),
                         width="stretch", hide_index=True,
                         column_config={"Change": st.column_config.NumberColumn(
                             help="Items in the last week minus items in the first week of the timeframe.")})

    with fold("What changed between two weeks?"):
        _status_change(tracking)


def _status_change(tracking):
    explain("Why are there more (or fewer) stock-outs this week? The items in the status at the end of each week, "
            "and the root causes of the items that **entered** or **left** it.\n\n"
            "Read the table from the top: the root causes with the largest net change explain the difference.")
    ends = ins.week_end_days(tracking)
    weeks = sorted(ends)
    if len(weeks) < 2:
        st.info("Not enough weeks in the timeframe.")
        return
    c1, c2, c3 = st.columns([1.6, 1, 1])
    statuses = c1.multiselect("Status:", ["Actual Stock Out", "Potential Stock Out"], default=["Actual Stock Out"],
                              key="lab_status")
    label = lambda w: f"Week of {w:%d.%m.%Y} (snapshot {ends[w]:%d.%m})"
    week_a = c2.selectbox("From:", weeks, index=len(weeks) - 2, format_func=label, key="lab_week_a")
    week_b = c3.selectbox("To:", weeks, index=len(weeks) - 1, format_func=label, key="lab_week_b")
    if not statuses:
        return
    summary, entered, n_a, n_b = ins.status_change(tracking, ends[week_a], ends[week_b], tuple(statuses))
    st.metric(f"Items in {' / '.join(statuses)}", f"{n_a} → {n_b}", delta=f"{n_b - n_a:+d}", delta_color="inverse",
              help="Number of items in the status on the last snapshot of each week, and the difference.")
    st.dataframe(ui.signed_colours(summary, "Net change"), width="stretch", hide_index=True, column_config={
        "Entered": st.column_config.NumberColumn(help="Items in the status in the second week but not in the first."),
        "Left": st.column_config.NumberColumn(help="Items in the status in the first week but no longer in the second."),
        "Net change": st.column_config.NumberColumn(
            help="Entered minus left: the contribution of this root cause. Red = more items in the status, "
                 "green = fewer; the stronger the colour, the larger the change.")})
    if len(entered):
        st.markdown("**Items that entered the status, with their root cause and comment**")
        comments_en = st.session_state.get("_comments_en", {})
        shown = entered.assign(Comment_EN=entered["Comment"].map(comments_en).fillna(entered["Comment"]))
        ui.picked_table("t_entered", shown[["Root_Cause", "Comment_EN", "VENDOR_NAME", "MRP_CONTROLLER", "MATERIAL_LABEL"]]
                     .rename(columns={"Root_Cause": "Root cause", "Comment_EN": "Comment (EN)", "VENDOR_NAME": "Vendor",
                                      "MRP_CONTROLLER": "MRP Controller", "MATERIAL_LABEL": "Material"}),
                     width="stretch", hide_index=True,
                     column_config={"Comment (EN)": st.column_config.TextColumn(width="large")})


# ------------------------------------------------------------------ page 3
def page_cause_impact(df, selections, selection_name):
    panel = options("cause-impact", "Impact by Root Cause — " + selection_name)
    _, period = timeframe_option(panel, "tf_impact")
    tracking = _tracking(df, selections, period)[1]
    explain("**Which root causes lead to which impact?** For every item (a material from a vendor) of the timeframe: "
            "its root cause(s), the most severe impact planning communicated for it, and its material status.\n\n"
            "Impact levels, from the most severe: **Market** > **Production** > **Logistics** > **None** (planning "
            "entered that there is no impact). Items without any impact entered are **not communicated**: their "
            "impact is unknown, so they are left out of the impact chart.")
    by = panel.radio("Show by:", ["Root cause", "Stakeholder"], horizontal=True, key="lab_impact_by",
                  help="Stakeholder = who is responsible, from the prefix of the root cause (S = Supplier, "
                       "PL = Planning, P = Procurement, B = Buying...).")
    data = tracking if by == "Root cause" else tracking.assign(
        Root_Cause=tracking["Root_Cause"].map(lambda c: ins.stakeholder(c) if pd.notna(c) else c))
    try:
        table, per_cause, test = ins.root_cause_impact(data)
    except ValueError:
        st.info("Not enough items with a communicated impact for the current filters and timeframe.")
        return
    min_items = panel.slider(f"Show the {by.lower()}s with at least ... items with a communicated impact:", 1, 30, 5,
                          key="lab_min_items", help="Small numbers give unstable percentages: 5 or more is a good minimum.")
    max_bars = panel.slider("Bars in the charts (at most):", 6, 40, 16, 2, key="lab_max_bars",
                            help="Beyond this number, the charts keep the root causes that most often and least often "
                                 "lead to a severe impact (half each), so that a chart fits on the screen. The table "
                                 "under the charts lists them all.")
    shown = per_cause[per_cause["Impact communicated"] >= min_items]
    if shown.empty:   # a narrow selection (e.g. after a click on a chart): show what there is
        shown = per_cause[per_cause["Impact communicated"] > 0]
    shown = shown.sort_values("% severe", ascending=False)

    # --- main findings in plain words (shown under the impact chart)
    base = test["severe_base_pct"]
    lines = [f"Across the selection, **{base:.0f}%** of the items with a communicated impact had a **Production or "
             "Market** impact."]
    worse = shown[(shown["Lift"] > 1) & (shown["p-value"] < SIGNIFICANT)]
    better = shown[(shown["Lift"] < 1) & (shown["p-value"] < SIGNIFICANT)]
    for _, r in worse.head(3).iterrows():
        lines.append(f"**{r['Root cause']}** leads to a severe impact more often: **{r['% severe']:.0f}%** of its "
                     f"{r['Impact communicated']} items ({r['Lift']:.1f} times the {base:.0f}% of the selection).")
    for _, r in better.head(2).iterrows():
        lines.append(f"**{r['Root cause']}** rarely leads to a severe impact: **{r['% severe']:.0f}%** of its "
                     f"{r['Impact communicated']} items.")
    if worse.empty and better.empty:
        lines.append(f"No {by.lower()} differs clearly from the {base:.0f}% of the selection (the differences may be "
                     "chance).")

    # --- one chart at a time: impact (default) or material status, chosen in a two-part switch
    charted = shown if len(shown) <= max_bars else pd.concat([shown.head(max_bars // 2), shown.tail(max_bars // 2)])
    order = charted["Root cause"].tolist()
    impact_label = f"Impact communicated per {by.lower()}"
    status_label = f"Material status per {by.lower()}"
    cause_filter = "f_root_cause" if by == "Root cause" else "f_stakeholder"
    if len(charted) < len(shown):
        st.caption(f"The chart shows the {max_bars // 2} {by.lower()}s that most often and the {max_bars // 2} that "
                   f"least often lead to a severe impact, out of {len(shown)}; the statistical details list them all "
                   "(Options: more bars).")
    # both charts are in the page; the switch shows one of them in the browser
    ui.client_switch("impact_chart", [impact_label, status_label])
    counts = table.loc[order, list(IMPACT_COLORS)]
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    notes = pd.DataFrame({"Cause": order, "Note": [f"{n} items{' ★' if p < SIGNIFICANT else ''}" for n, p in
                                                   zip(counts.sum(axis=1), charted["p-value"])]})
    chart_id = hash((by, selection_name, tuple(order), tuple(counts.to_numpy().ravel())))
    with ui.pane("impact_chart", 0):
        key = f"impact_{chart_id}"
        ui.clickable_stacked(_stacked_rows(shares, counts, IMPACT_COLORS), notes, order, IMPACT_COLORS,
                             f"{impact_label} (sorted by share of Production + Market)",
                             "% of the items with a communicated impact", key,
                             partial(_pick_segment, key, cause_filter, "f_impact"))
        st.caption(f"Click a segment to filter the dashboard on this {by.lower()} and this impact.")
        if len(shown) > 1:   # one bar left (e.g. after a click): nothing to compare it with
            insights_box(lines)
        explain("Each bar = 100% of the items of a root cause whose impact was communicated. **Dark red "
                "and red** = severe impacts (Market, Production); **green** = no impact.\n\n"
                "The root causes at the **top** most often lead to a severe impact. **★** = the difference "
                f"with the {base:.0f}% of the whole selection is unlikely to be chance.")
    status = ins.root_cause_status(data).reindex(order).fillna(0)
    status_shares = status.div(status.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    status_notes = pd.DataFrame({"Cause": order, "Note": [f"{int(n)} items" for n in status.sum(axis=1)]})
    with ui.pane("impact_chart", 1):
        key = f"status_{chart_id}"
        ui.clickable_stacked(_stacked_rows(status_shares, status, ui.STATUS_COLORS), status_notes, order,
                             {c: ui.STATUS_COLORS[c] for c in status.columns}, status_label,
                             "% of the items with this root cause", key,
                             partial(_pick_segment, key, cause_filter, "f_status"))
        st.caption(f"Click a segment to filter the dashboard on this {by.lower()} and on the materials in this status "
                   "today. The bars count the worst status over the timeframe: the filter can show fewer items.")
        explain("For the same root causes, in the same order: the **worst material status** each item "
                "reached while the root cause was entered (ION colours: red = Actual Stock Out, orange = "
                "Potential, yellow = Below Safety, green = Good).")

    with details():
        # the table of every root cause, with the statistics (for analysts)
        st.dataframe(shown.assign(Significant=np.where(shown["p-value"] < SIGNIFICANT, "Yes", "No"))
                     .rename(columns={"Root cause": by, "Lift": LIFT})
                     [[by, "Stakeholder", "Items", "Impact communicated", "Severe (Prod.+Market)", "% severe", LIFT,
                       "Significant", "p-value"]].drop(columns=["Stakeholder"] if by == "Stakeholder" else []),
                     width="stretch", hide_index=True, column_config={
                         "Items": st.column_config.NumberColumn(help="Items with this root cause in the timeframe."),
                         "Impact communicated": st.column_config.NumberColumn(help=HELP["communicated"]),
                         "% severe": st.column_config.NumberColumn(help=HELP["severe"]),
                         LIFT: st.column_config.NumberColumn(help=HELP["lift"]),
                         "Significant": st.column_config.TextColumn(help=HELP["significant"]),
                         "p-value": st.column_config.NumberColumn(format="%.4f", help=HELP["p"])})
        if test["chi2"] is None:
            st.markdown("**Are root cause and impact linked?** Too few root causes or impacts in this selection for "
                        "the chi-square test: widen the filters.")
            return
        significant = test["p"] < SIGNIFICANT
        strength = "weak" if test["cramers_v"] < 0.2 else "moderate" if test["cramers_v"] < 0.4 else "strong"
        st.markdown(
            f"**Are root cause and impact linked, or is it chance?** A chi-square test of independence compares the "
            f"impacts observed per root cause with what we would see if the root cause made no difference. On the "
            f"{test['n_items']} items with a communicated impact of the {test['n_causes']} root causes with at least "
            f"5 of them: χ² = {test['chi2']}, {test['dof']} degrees of freedom, p = {test['p']:.2g}.\n\n"
            + ("→ **The link is real** (p below 0.05): the root cause tells something about the impact to expect.\n\n"
               if significant else "→ **No clear link** (p above 0.05): the differences may be chance.\n\n")
            + f"**How strong is the link?** Cramér's V = {test['cramers_v']} ({strength}). It goes from 0 (no link) "
            f"to 1 (the root cause fully decides the impact); 0.1 is weak, 0.3 moderate, 0.5 strong.\n\n"
            f"**{LIFT} (lift)** = share of severe impacts of the root cause ÷ the share of the whole selection "
            f"({base}%).\n\n"
            f"**p-value per root cause** (Fisher's exact test): this root cause compared with all the others. Below "
            f"0.05, the difference is unlikely to be chance (★ in the chart).\n\n"
            "**Limits**: an item with two root causes counts for both; the impact is the most severe one entered for "
            "the item over the timeframe; items without impact entered are left out (unknown, not 'no impact').")


# ------------------------------------------------------------------ page 4
@st.cache_data(show_spinner="Looking for weak signals...")
def _weak_signals(_tracking, _comments_en, version):
    """All items with a comment and the prediction tool, trained on the whole history (it needs volume).
    The tables are not hashed at every rerun (143,000 rows): version = what identifies them."""
    tracking, comments_en = _tracking, _comments_en
    items = ins.weak_signal_items(tracking, comments_en)
    model = ins.signal_model(items)
    if model:
        items["Risk"], items["Why"] = model["risk"], model["reasons"]
    return items, model


# Colour of a bar = how many times more often than for all items an impact follows (the further, the warmer);
# solid = unlikely to be chance (p < 0.05, ★), light = could be chance (often few items)
LIFT_BANDS = [(1.0, "#2FA84F", "less often than all items"), (1.5, "#F5C518", "up to 1.5×"),
              (2.5, "#F28C28", "1.5 to 2.5×"), (float("inf"), "#E0443A", "more than 2.5×")]


def _lift_colour(lift):
    return next(colour for limit, colour, _ in LIFT_BANDS if lift < limit)


def _bar_vs_average(frame, label_column, value_column, average, title, highlight):
    """Horizontal bars of a share, with the average as a dashed line. Colour = times the average (LIFT_BANDS);
    solid with ★ = significant (highlight), light = could be chance. A legend above the bars says so."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    frame = frame.iloc[::-1]
    significant = list(highlight)[::-1]
    lifts = frame[value_column] / average if average else frame[value_column] * 0 + 1
    fig, ax = plt.subplots(figsize=(11, 0.27 * len(frame) + 1.6))
    labels = [str(v) if len(str(v)) <= 45 else str(v)[:44] + "…" for v in frame[label_column]]
    colours = [to_rgba(_lift_colour(l), 1.0 if sig else 0.35) for l, sig in zip(lifts, significant)]
    edges = [_lift_colour(l) for l in lifts]
    ax.barh(labels, frame[value_column], color=colours, edgecolor=edges, linewidth=0.8)
    for i, (v, n, sig) in enumerate(zip(frame[value_column], frame["Items"], significant)):
        ax.text(v + 1, i, f"{v:.0f}%  ({n} items){'  ★' if sig else ''}", va="center", fontsize=7.5)
    ax.axvline(average, color="#374151", linestyle="--", linewidth=1)
    ax.text(average, len(frame) - 0.45, f" all items: {average:.0f}%", fontsize=7.5, color="#374151", va="bottom")
    ax.set_ylim(-0.6, len(frame) + 0.3)   # room for the 'average' label, under the legend
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of the items followed by a Production / Market impact")
    ax.set_title(title, pad=24)
    used = {_lift_colour(l) for l in lifts}
    handles = [Patch(color=c, label=name) for _, c, name in LIFT_BANDS if c in used]
    handles += [Patch(facecolor="#9CA3AF", edgecolor="#9CA3AF", label="solid ★ = unlikely to be chance"),
                Patch(facecolor=to_rgba("#9CA3AF", 0.35), edgecolor="#9CA3AF", label="light = could be chance")]
    ax.legend(handles=handles, ncol=len(handles), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              fontsize=7.5, borderaxespad=0.3, handlelength=1.2, columnspacing=1.2)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def weak_signals(tracking, comments_en):
    """_weak_signals for the current tracking table and comments (the page and the Smart Analysis)."""
    version = (os.path.getmtime(TRACKING_PATH), hash(tuple(sorted(comments_en.items(), key=str))))
    return _weak_signals(tracking, comments_en, version)


@st.cache_data(show_spinner="Looking for weak signals in the selection...")
def _selection_signals(items):
    return ins.signal_phrases(items), {c: ins.signal_context(items, c) for c in ins.CONTEXT}


def _top_share(items, share=0.1):
    """Of the items the tool ranks highest (top 10%), the share that later had an impact, and the share of all
    impacts they contain."""
    n_top = max(1, int(round(len(items) * share)))
    top = items.sort_values("Risk", ascending=False).head(n_top)
    found = top["Severe_later"].sum() / max(items["Severe_later"].sum(), 1) * 100
    return n_top, top["Severe_later"].mean() * 100, found


def page_watchlist(df, selections, selection_name):
    settings, columns_panel = options_with_columns("watchlist", "Watchlist — " + selection_name)
    _, period = timeframe_option(settings, "tf_watchlist")
    tracking, tracking_filtered = _tracking(df, selections, period)
    comments_en = st.session_state["_comments_en"]
    all_items, model = weak_signals(tracking, comments_en)
    in_scope = set(map(tuple, tracking_filtered[ins.ITEM].drop_duplicates().to_numpy()))
    items = all_items[[(m, v) in in_scope for m, v in zip(all_items["MATERIAL_NUMBER"], all_items["VENDOR_ACCOUNT_NUMBER"])]]
    if period:
        items = items[(items["First_comment"] >= period[0]) & (items["First_comment"] <= period[1])]
    items = items.reset_index(drop=True)
    if items.empty:
        st.info("No item with a comment for the current filters and timeframe.")
        return
    _watch_list(items, tracking, model, columns_panel)
    with details():
        _watchlist_details(items, model)


def _watchlist_details(items, model):
    """What the ranking of the watchlist rests on, for the analysts: one page (tab) per question."""
    explain("**Can we see an impact coming?** For every item (a material from a vendor), we look at its **first "
            "comment** and its context at that moment (vendor, root cause, material status), and check whether a "
            "**Production or Market impact** was communicated **later**.\n\n"
            "A weak signal = a phrase or a context after which such an impact comes more often than usual. "
            "These pages follow the sidebar filters and the timeframe (items first commented in it); only the "
            "prediction tool is trained on the whole history, because it needs volume.")
    severe = items[items["Severe_later"]]
    base = len(severe) / len(items) * 100
    enough = len(items) >= 30 and len(severe) >= 3
    phrases, context = _selection_signals(items) if enough else (pd.DataFrame(), {c: pd.DataFrame() for c in ins.CONTEXT})
    m1, m2, m3 = st.columns(3)
    m1.metric("Items with a comment", f"{len(items):,}",
              help="Items whose first comment came before any severe impact (items already in Production / Market "
                   "impact when first commented are left out: nothing to anticipate).")
    m2.metric("Impact communicated later", f"{len(severe):,}", delta=f"{base:.1f}% of them", delta_color="off",
              help="Items that received a Production or Market impact after their first comment. The others had no "
                   "such impact communicated (which does not prove there was none).")
    m3.metric("Median warning time", f"{severe['Days_before_impact'].median():.0f} days" if len(severe) else "–",
              help="Half of the impacts came less than this many days after the first comment, half came later: the "
                   "time there was to react.")
    if not enough:
        st.warning(f"{len(items)} item{'s' if len(items) > 1 else ''} with a comment ({len(severe)} with an impact "
                   "later) in this selection: too few to find reliable signals (at least 30 items and 3 impacts are "
                   "needed). Widen the filters or the timeframe. The watchlist still works.")

    # --- main findings
    lines = []
    if not phrases.empty:
        top = phrases.head(3)
        lines.append("First entries most often followed by an impact: " + "; ".join(
            f"**\"{r['Phrase']}\"** ({r['% impact later']:.0f}% of {r['Items']} items, {r['× average']:.1f} times "
            f"the {base:.0f}% of all items)"
            for _, r in top.iterrows()) + ".")
    vendors = context["VENDOR_NAME"]
    if not vendors.empty:
        top_v = vendors[vendors["p-value"] < SIGNIFICANT].head(3)
        if len(top_v):
            lines.append("Vendors whose items most often end with an impact: " + "; ".join(
                f"**{r['VENDOR_NAME']}** ({r['% impact later']:.0f}% of {r['Items']} items)" for _, r in top_v.iterrows()) + ".")
    status = context["MATERIAL_STATUS"]
    if not status.empty:
        aso = status[status["MATERIAL_STATUS"] == "Actual Stock Out"]
        if len(aso):
            lines.append(f"Items already in **Actual Stock Out** when first commented: **{aso['% impact later'].iloc[0]:.0f}%** "
                         f"later had an impact, against **{base:.0f}%** of all the items of the selection.")
    insights_box(lines)

    tab_entry, tab_context, tab_weights, tab_tool = st.tabs(["Comment in the first entry", "Context of the first comment",
                                                             "Elements that weigh most", "How reliable is the tool"])
    with tab_entry:
        explain("Phrases of 2 to 3 words (common words like 'the', 'for' are skipped, so `request pull-in` stands for "
                "\"Request for pull-in\"). The **colour** says how many times more often an impact follows than for all "
                "the items of the selection (green = less often, yellow, orange, **red** = more than 2.5×); **solid bars with ★** are "
                "unlikely to be chance, **light** bars could be chance (often few items). The "
                "dashed line is the rate of all the items of the selection.\n\n"
                "The table gives the context: the usual root cause and vendors of these comments, and an example.")
        if phrases.empty:
            st.info("No phrase is followed by an impact clearly more often than for all the items of the selection.")
        else:
            top = phrases.head(12)
            ui.chart(_bar_vs_average, top, "Phrase", "% impact later", base, "Comment in the first entry",
                     (top["p-value"] < SIGNIFICANT).tolist())
            ui.picked_table("t_phrases", phrases.assign(Significant=np.where(phrases["p-value"] < SIGNIFICANT, "Yes", "No"))
                         .rename(columns={"× average": LIFT})
                         [["Phrase", "Items", "Impact later", "% impact later", LIFT, "Significant", "Main root cause",
                           "Main vendors", "Example", "p-value"]],
                         width="stretch", hide_index=True, column_config={
                             "Items": st.column_config.NumberColumn(help="Items whose first comment contains the phrase."),
                             "Impact later": st.column_config.NumberColumn(help="Of these, items with a Production / Market "
                                                                                "impact communicated later."),
                             "% impact later": st.column_config.NumberColumn(help=HELP["impact_later"]),
                             LIFT: st.column_config.NumberColumn(help=HELP["lift"]),
                             "Significant": st.column_config.TextColumn(help=HELP["significant"]),
                             "Example": st.column_config.TextColumn(width="large", help="A first comment with this phrase."),
                             "p-value": st.column_config.NumberColumn(format="%.4f", help=HELP["p"])})

    with tab_context:
        explain("The same question for the context of the first comment: after which vendors, root causes and material "
                "statuses does an impact come more often? Only values with at least 3 items are shown.")
        names = {"VENDOR_NAME": "Vendor", "Root_Cause": "Root cause", "MATERIAL_STATUS": "Material status"}
        for tab, column in zip(st.tabs([names[c] for c in ["VENDOR_NAME", "Root_Cause", "MATERIAL_STATUS"]]),
                               ["VENDOR_NAME", "Root_Cause", "MATERIAL_STATUS"]):
            with tab:
                frame = context[column]
                if frame.empty:
                    st.info("Not enough items.")
                    continue
                top = frame.head(12)
                ui.chart(_bar_vs_average, top, column, "% impact later", base, f"{names[column]} at the first comment",
                         (top["p-value"] < SIGNIFICANT).tolist())
                st.dataframe(frame.rename(columns={column: names[column], "× average": LIFT}).assign(
                    Significant=lambda d: np.where(d["p-value"] < SIGNIFICANT, "Yes", "No")),
                    width="stretch", hide_index=True, column_config={
                        "% impact later": st.column_config.NumberColumn(help=HELP["impact_later"]),
                        LIFT: st.column_config.NumberColumn(help=HELP["lift"]),
                        "Significant": st.column_config.TextColumn(help=HELP["significant"]),
                        "p-value": st.column_config.NumberColumn(format="%.4f", help=HELP["p"])})

    with tab_weights:
        if not model:
            st.info("Not enough items in the history to train the tool.")
        else:
            weights = model["weights"]
            st.markdown("Positive = raises the risk, negative = lowers it. Learned on the whole history.")
            st.dataframe(pd.concat([weights.head(15).assign(Effect="raises the risk"),
                                    weights.tail(10).iloc[::-1].assign(Effect="lowers the risk")]),
                         width="stretch", hide_index=True)

    # --- the prediction tool: how it works, then how well it does
    with tab_tool:
        if not model:
            st.info("Not enough items in the history to train the tool.")
            return
        scores = model["scores"]
        st.markdown(
            f"**Model**: logistic regression, classes balanced, on {scores.attrs['n_items']} items "
            f"({scores.attrs['n_severe']} with an impact later). Inputs: phrases of 2-3 words of the first comment "
            "(TF-IDF), root cause, vendor and material status at the first comment.\n\n"
            "**Honest evaluation**: 5-fold cross-validation. The items are split in 5 parts; each part is scored "
            "by a model trained on the 4 others, so every item is scored by a model that never saw it.\n\n"
            "**Top 10%**: the items are ranked by their risk; we look at the 10% at the top, as a planner would "
            "check them first.\n\n"
            "**MCC** (Matthews correlation coefficient): overall quality of the yes/no prediction, from -1 to 1; "
            "0 = no better than chance; 0.2-0.3 = modest but real; it stays fair when impacts are rare. "
            "**ROC AUC**: the chance that an item with an impact is ranked above one without; 0.5 = chance, 1 = perfect.\n\n"
            "Method after Usuga-Cadavid et al. (classic text model, MCC for imbalanced classes, contributions of "
            "the words to explain the predictions).")
        st.dataframe(scores, width="stretch", hide_index=True, column_config={
            "MCC": st.column_config.NumberColumn(help="0 = chance, 1 = perfect."),
            "ROC AUC": st.column_config.NumberColumn(help="0.5 = chance, 1 = perfect.")})

        st.markdown("**How well can the tool anticipate an impact?**")
        n_top, hit, found = _top_share(items)
        lines = [f"In this selection, if you check first the **{n_top} items** (10%) the tool ranks most at risk, "
                 f"**{hit:.0f}%** of them later had a Production / Market impact, against **{base:.0f}%** of all "
                 "items" + (f": **{hit / base:.1f}× better than checking at random**." if base else ".")]
        if len(severe):
            lines.append(f"These {n_top} items contain **{found:.0f}%** of the impacts that came later in the "
                         "selection.")
        lines.append("Over all items, using the comment and its context (vendor, root cause, status) works better "
                     f"than the root cause alone: {scores.iloc[-1]['Impact rate in the top 10%']:.0f}% vs "
                     f"{scores.iloc[0]['Impact rate in the top 10%']:.0f}% in the top 10% (chart below).")
        insights_box(lines)
        all_base = scores.attrs["n_severe"] / scores.attrs["n_items"] * 100
        ui.chart(_model_chart, scores[["Model", "Impact rate in the top 10%"]], all_base, name="weak_signals_model")
        explain("Each bar is a version of the tool, using different information. The further a bar goes beyond "
                "the dashed line (all items), the better that version spots the items that will get an impact.")


def _model_chart(scores, all_base):
    """Impact rate in the top 10% of each version of the tool, against the average."""
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.barh(scores["Model"][::-1], scores["Impact rate in the top 10%"][::-1], color=["#1A73E8", "#7B61FF", "#0B41CD"])
    for i, v in enumerate(scores["Impact rate in the top 10%"][::-1]):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8)
    ax.axvline(all_base, color="#374151", linestyle="--", linewidth=1)
    ax.text(all_base, 2.45, f" all items: {all_base:.0f}%", fontsize=7.5, color="#374151")
    ax.set_xlabel("All items: % of the 10% highest-ranked items that later had a Production / Market impact")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _watch_list(items, tracking, model, columns_panel):
    """Open items of the selection, ranked by the risk the tool gives them, with the reasons."""
    explain("Items **still visible on the last snapshot**, with no Production / Market impact communicated yet, ranked "
            "by their risk. The risk comes from a prediction tool (see the statistical details below): it compares the item's first comment and context with "
            "all the items of the past.\n\n"
            "**Why** gives the 3 elements that raise the risk of this item most. A list of items to look at first, "
            "not a certainty.")
    if not model:
        st.info("Not enough items in the history to train the tool.")
        return
    last_day = tracking["Snapshot_Date"].max()
    watch = items[(items["Last_seen"] == last_day) & ~items["Severe_later"]].copy()
    watch["Days since first comment"] = (last_day - watch["First_comment"]).dt.days
    watch = watch.sort_values("Risk", ascending=False).head(25)
    watch["Risk"] = (watch["Risk"] * 100).round(0)
    watch["Material number"] = watch["MATERIAL_NUMBER"].astype("int64").astype(str)
    watch["Why"] = watch["Why"].str.replace(r", (?=vendor: |root cause: |status: |phrase )", "\n", regex=True)   # one per line
    table = watch[["Risk", "MATERIAL_DESC", "Material number", "Why", "Text", "VENDOR_NAME", "Root_Cause", "Status_now",
                   "Days since first comment", "MRP_CONTROLLER"]].rename(columns={
        "Risk": "Risk score", "MATERIAL_DESC": "Material", "Text": "First comment (EN)", "Root_Cause": "Root cause", "VENDOR_NAME": "Vendor",
        "Status_now": "Status now", "MRP_CONTROLLER": "MRP Controller"})
    # In HTML for the Copy button of the material number
    shown = ui.table_toolbar("t_watchlist", list(table.columns), list(table.columns), panel=columns_panel)
    ui.copy_table(table[shown], "Material number", bar_column="Risk score", wide=("Why", "First comment (EN)"),
                  nowrap=("Material", "Status now", "MRP Controller"))
    st.caption(f"Items of the selection still visible on {last_day:%d.%m.%Y} with no Production / Market impact "
               "communicated. Risk score (0-100): it ranks the items from the most to the least at risk; it is not a "
               "probability of impact. "
               "Why = the 3 elements that raise the risk most. Copy = the material number, to paste into SAP.")


_FILTERED = {}   # (filters, timeframe) -> filtered tracking table: the sections of one rerun share it


def filtered_tracking(selections, period):
    """filter_tracking of the shared table, kept for the last filters used (3 sections and the Smart Analysis
    ask for the same one)."""
    tracking = load_tracking()
    if tracking is None:
        return None
    key = repr((sorted((k, tuple(v)) for k, v in selections.items() if v), period))
    if key not in _FILTERED:
        if len(_FILTERED) >= 20:
            _FILTERED.pop(next(iter(_FILTERED)))
        _FILTERED[key] = filter_tracking(tracking, selections, period)
    return _FILTERED[key]


def _tracking(df, selections, period):
    """(whole tracking table, tracking table of the sidebar filters and timeframe); the English comments for
    the tables. (None, None) when the table was not built."""
    st.session_state["_comments_en"] = dict(zip(df["Comment"], df["Comment_EN"]))
    return load_tracking(), filtered_tracking(selections, period)


# ------------------------------------------------------------------ context for the Smart Analysis (LLM)
FILTER_NAMES = {"MRP_CONTROLLER": "MRP controller", "PURCHASING_GROUP": "purchasing group", "VENDOR_LABEL": "vendor",
                "MATERIAL_LABEL": "material", "Stakeholder": "stakeholder", "Root_Cause": "root cause",
                "Impact_Level": "impact", "Material_Status": "material status today"}


def _short(text, n=160):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


def smart_context(df, filtered_df, selections, period=None, timeframe=""):
    """Everything the Insights Lab knows about the current selection, as compact facts for the Smart Analysis:
    the filters (so that the model does not 'discover' them), how the selection differs from all comments,
    root cause -> impact, weak signals and the watch list, long-open comments, recurring issues (grouped),
    what changed in the last week and the root causes rising over the last 4 weeks.
    All figures are computed here; the model only interprets them."""
    facts = {"selection": {"filters": {FILTER_NAMES[k]: [v.split(" — ")[0] for v in vals]
                                       for k, vals in selections.items() if vals and k in FILTER_NAMES},
                           "timeframe": timeframe or "all snapshots",
                           "comments": int(len(filtered_df)), "all_comments": int(len(df))}}
    filtered_on = set(facts["selection"]["filters"])

    # 1. how the selection differs from all comments (root causes over / under-represented)
    if filtered_on and len(filtered_df):
        sel = filtered_df["Root_Cause"].fillna(ins.NO_ROOT_CAUSE).value_counts(normalize=True) * 100
        everyone = df["Root_Cause"].fillna(ins.NO_ROOT_CAUSE).value_counts(normalize=True) * 100
        diff = (sel - everyone.reindex(sel.index).fillna(0)).sort_values(key=abs, ascending=False).head(5)
        facts["root_causes_vs_all_comments_pct_points"] = {
            c: {"selection_pct": round(sel[c], 1), "all_comments_pct": round(everyone.get(c, 0), 1),
                "difference_pct_points": round(d, 1)} for c, d in diff.items()}

    # 2. recurring issues (near-identical comments grouped) and comments open for long
    entries = ins.comment_entries(filtered_df)[["Entry", "Root_Cause", "MATERIAL_NUMBER", "VENDOR_NAME", "Days_Active"]]
    if len(entries):
        groups, _ = _groups(entries.reset_index(drop=True), ins.METHODS["chars"], "chars")
        facts["recurring_issues"] = [
            {"issue": _short(g["Comment"], 120), "entries": int(g["Occurrences"]), "materials": int(g["Materials"]),
             "vendors": int(g["Vendors"]), "main_root_cause": g["Main_root_cause"]}
            for _, g in groups.head(6).iterrows()]
    last = df["Last_Seen"].max()
    still_open = (filtered_df[filtered_df["Last_Seen"] == last].sort_values("Days_Active", ascending=False)
                  .drop_duplicates("Comment_EN").head(4))
    facts["comments_open_longest"] = [
        {"comment": _short(r["Comment_EN"]), "days_visible": int(r["Days_Active"]),
         "root_cause": r["Root_Cause"] if pd.notna(r["Root_Cause"]) else ins.NO_ROOT_CAUSE,
         "impact": r["Impact_Level"], "material_status": r["MATERIAL_STATUS"]} for _, r in still_open.iterrows()]

    tracking = load_tracking()
    if tracking is None:
        return facts
    scope = filtered_tracking(selections, period)

    # 3. root cause -> impact (items, impact communicated only)
    try:
        table, per_cause, test = ins.root_cause_impact(scope, min_items=3)
        rows = per_cause[per_cause["Impact communicated"] >= 3].head(6)
        facts["root_cause_to_impact"] = {
            "items_with_impact_communicated": test["n_items"],
            "severe_impact_pct_in_selection": test["severe_base_pct"],
            "per_root_cause": {r["Root cause"]: {"items_with_impact_communicated": int(r["Impact communicated"]),
                                                 "severe_pct": r["% severe"], "times_average": r["Lift"],
                                                 "significant": bool(r["p-value"] < SIGNIFICANT)}
                               for _, r in rows.iterrows()}}
    except ValueError:
        pass
    items_scope = scope.groupby(ins.ITEM)["Impact_Level"].agg(lambda s: s.dropna().nunique() > 0)
    facts["items_impact_not_communicated"] = int((~items_scope).sum())
    facts["items_in_selection"] = int(len(items_scope))

    # 4. weak signals of the selection and watch list
    all_items, model = weak_signals(tracking, st.session_state.get("_comments_en") or dict(zip(df["Comment"], df["Comment_EN"])))
    keys = set(map(tuple, scope[ins.ITEM].drop_duplicates().to_numpy()))
    items = all_items[[(m, v) in keys for m, v in zip(all_items["MATERIAL_NUMBER"], all_items["VENDOR_ACCOUNT_NUMBER"])]]
    if len(items):
        severe = items["Severe_later"]
        weak = {"items_with_first_comment": int(len(items)),
                "share_of_these_items_with_impact_later_pct": round(severe.mean() * 100, 1),
                "median_days_between_first_comment_and_impact":
                    float(items.loc[severe, "Days_before_impact"].median()) if severe.any() else None}
        if len(items) >= 30 and severe.sum() >= 3:
            phrases, context = _selection_signals(items.reset_index(drop=True))
            if len(phrases):
                weak["phrases_followed_by_impact"] = [
                    {"phrase": r["Phrase"], "items_with_phrase": int(r["Items"]),
                     "of_these_impact_later_pct": r["% impact later"],
                     "times_average": r["× average"], "significant": bool(r["p-value"] < SIGNIFICANT),
                     "example": _short(r["Example"], 120)} for _, r in phrases.head(5).iterrows()]
            for column, label in (("VENDOR_NAME", "vendor"), ("Root_Cause", "root cause"),
                                  ("MATERIAL_STATUS", "material status")):
                frame = context[column]
                if label in filtered_on or frame.empty:
                    continue
                top = frame[(frame["× average"] > 1) & (frame["p-value"] < 0.1)].head(3)
                if len(top):
                    weak[f"{label}_followed_by_impact"] = {
                        r[column]: {"items": int(r["Items"]), "of_these_impact_later_pct": r["% impact later"]}
                        for _, r in top.iterrows()}
        else:
            weak["note"] = "too few items for reliable weak signals"
        facts["weak_signals"] = weak
        if model:
            watch = items[(items["Last_seen"] == tracking["Snapshot_Date"].max()) & ~items["Severe_later"]]
            facts["watch_list_open_items_most_at_risk"] = [
                {"risk_score": int(round(r["Risk"] * 100)), "why": r["Why"], "first_comment": _short(r["Text"], 120),
                 "root_cause": r["Root_Cause"], "status_now": r["Status_now"],
                 "days_since_first_comment": int((tracking["Snapshot_Date"].max() - r["First_comment"]).days)}
                for _, r in watch.sort_values("Risk", ascending=False).head(5).iterrows()]

    # 5. last week: items that entered / left Actual Stock Out, by root cause
    ends = ins.week_end_days(scope)
    weeks = sorted(ends)
    if len(weeks) >= 2:
        summary, _, n_a, n_b = ins.status_change(scope, ends[weeks[-2]], ends[weeks[-1]])
        facts["actual_stock_out_last_week"] = {
            "items_previous_week": n_a, "items_last_week": n_b,
            "root_causes_of_the_change": {r["Root cause"]: {"entered": int(r["Entered"]), "left": int(r["Left"])}
                                          for _, r in summary.head(4).iterrows() if r["Entered"] or r["Left"]}}
    # 6. root causes rising / falling: last 4 weeks vs the 4 weeks before (items per week, all snapshot rows)
    weekly = ins.weekly_items(scope)
    if len(weekly) >= 8:
        change = (weekly.iloc[-4:].mean() - weekly.iloc[-8:-4].mean()).round(1)
        facts["root_causes_items_per_week_change_last_4_weeks"] = {
            "rising": {c: v for c, v in change[change > 0].sort_values(ascending=False).head(3).items()},
            "falling": {c: v for c, v in change[change < 0].sort_values().head(3).items()}}
    return facts
