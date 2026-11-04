"""
Streamlit dashboard "Material Availability – Insights Lab" (run: venv/bin/streamlit run scripts/app.py).

One page, in the order of the menu:
  Overview (status tiles, Top Root Causes) · Filtered Comments · Most Frequent Comments* · Root Cause Trends* ·
  Key Statistics · Impact by Root Cause* · Weak Signals* · Smart Analysis
  (* drawn by scripts/insights_pages.py from data/item_tracking.csv; the others here, with scripts/stats_engine.py)
Also: password page (APP_PASSWORD in .env), sidebar filters + timeframe + saved views, which apply to every section.
Look and components (menu, switches, charts, tables, loading screen) are in scripts/ui.py.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import sys
import threading
import time

# Make scripts/ importable so we can reuse our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm_summary import (stream_summary, stream_summary_cloud, tidy, is_server_available,
                         is_cloud_configured, MODEL_NAME)
from pseudonymise import PromptPseudonymiser
from entries import ENTRY_SEPARATOR
from stats_engine import compute_stats
import config
import ui
import insights
import insights_pages



# Browser tab icon: the Roche logo in blue (assets/favicon.png), Streamlit's default if the file is missing
FAVICON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "favicon.png")

st.set_page_config(
    page_title="Roche AI PoC - NLP Insights",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    layout="wide",
)
ui.apply_style()  # Look of the Roche ION Material Availability dashboard (scripts/ui.py)


# Password gate: active when APP_PASSWORD is set (.env locally, the secrets once deployed; required there)
def check_password():
    password = config.app_password()   # re-read from .env: a new password applies without a restart
    if not password and config.ON_CLOUD:   # deployed copy: never open without a password
        st.error("Set APP_PASSWORD in the secrets of the app (Streamlit Cloud → Settings → Secrets).")
        st.stop()
    if not password or st.session_state.get("authenticated"):
        return
    st.title("Material Availability – Insights Lab")
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == password:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()

check_password()

# Deployed copy: the data comes encrypted with the code (deploy/data.enc, scripts/data_bundle.py)
if config.ON_CLOUD:
    import data_bundle
    data_bundle.unpack_if_needed(os.getenv("DATA_KEY", ""))


# Load cleaned data (built by scripts/data_cleaning.py, translated by scripts/translation.py)
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_comments.csv")

df = load_data().copy()

# --- Create combined display labels for filters ---

# Material: description + number (e.g., "Syringe 10ml - 3000123456")
df["MATERIAL_LABEL"] = (
    df["MATERIAL_DESC"].fillna("No description")
    + " — "
    + df["MATERIAL_NUMBER"].astype("int64").astype(str)
)

# Vendor: name + account number (e.g., "B. Braun — 50012345")
df["VENDOR_LABEL"] = (
    df["VENDOR_NAME"].fillna("Unknown vendor")
    + " — "
    + df["VENDOR_ACCOUNT_NUMBER"].astype("Int64").astype(str).replace("<NA>", "no vendor")
)

df["MRP_CONTROLLER"] = df["MRP_CONTROLLER"].fillna("Unknown")
df["PURCHASING_GROUP"] = df["PURCHASING_GROUP"].fillna("Unknown")

# Impact filter: the data holds two spellings per level ("Production" and
# "Production (Stop&Go/...)"), so the filter works on the four official levels
# (ION: None / Logistics / Production / Market) plus the comments whose impact was not communicated
# (nothing entered: this does NOT mean that there is no impact).
IMPACT_FILTER_OPTIONS = ["Market", "Production", "Logistics", "None (no impact)", "Not communicated"]


def impact_level(value):
    """Map a raw Impact label to one of the filter options."""
    if pd.isna(value):
        return "Not communicated"
    label = str(value).strip().lower()
    for prefix, level in (("market", "Market"), ("production", "Production"),
                          ("logistic", "Logistics"), ("none", "None (no impact)")):
        if label.startswith(prefix):
            return level
    return str(value).strip()


df["Impact_Level"] = df["Impact"].map(impact_level)

# Convert dates to datetime for proper sorting and filtering
for col in ["Snapshot_Date", "First_Seen", "Last_Seen"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")



# Title + navigation menu (each entry scrolls down to its section; all sections follow each other)
SECTIONS = [
    ("overview", "Overview"),
    ("filtered-comments", "Filtered Comments"),
    ("frequent-comments", "Most Frequent Comments"),
    ("trends", "Root Cause Trends"),
    ("key-statistics", "Key Statistics"),
    ("cause-impact", "Impact by Root Cause"),
    ("weak-signals", "Weak Signals"),
    ("ai-summary", "Smart Analysis"),   # always the last one
]
SECTION_TITLES = dict(SECTIONS)
ui.top_bar("Material Availability", "Insights Lab", SECTIONS)
if "page" in st.query_params:   # links of the former separate pages (?page=...): the dashboard is one page now
    del st.query_params["page"]
ui.scroll_spy()  # menu entry of the section on screen in blue

# --- Saved views (like SAP "personalised views"): filters + table columns under a name ---
# Stored on this Mac (data/interim, not pushed to GitHub), so every user of the shared app sees them.
VIEWS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "interim", "saved_views.json")
FILTER_KEYS = ["f_mrp", "f_pgroup", "f_vendor", "f_material", "f_root_cause", "f_impact"]
TABLE_COLUMNS = {  # data column -> displayed name, in the default order of the comments table
    "Comment_EN": "Comment (EN)", "Comment": "Original comment", "First_Seen": "First seen",
    "Last_Seen": "Last seen", "Days_Active": "Days visible", "STOCKOUT_DATE": "Stockout Date",
    "MATERIAL_STATUS": "Material Status", "Root_Cause": "Root Cause", "Impact": "Impact",
    "VENDOR_NAME": "Vendor Name", "PURCHASING_GROUP": "Purchasing Group",
    "MRP_CONTROLLER": "MRP Controller", "MATERIAL_LABEL": "Material"}
if "Comment_Expanded" in df.columns:  # built by data_cleaning.py: abbreviations written out
    TABLE_COLUMNS["Comment_Expanded"] = "Comment (expanded)"
DEFAULT_COLUMNS = [c for c in TABLE_COLUMNS.values() if c not in ("Original comment", "Comment (expanded)")]

# A comment is a journal: its entries are already written one after the other, separated by ENTRY_SEPARATOR
# (" ¦▸ ", scripts/entries.py), so a table cell shows them on one line as they are.
ENTRY_MARK = ENTRY_SEPARATOR


def inline_entries(series):
    return series.astype("string")


def load_views():
    try:
        with open(VIEWS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# Columns of the comments table: their order (col_order) and the visible ones (col_visible).
# Kept outside the checkbox widgets: Streamlit forgets a widget's state while it is not drawn
# (here, while the column panel is closed).
st.session_state.setdefault("col_order", list(TABLE_COLUMNS.values()))
st.session_state.setdefault("col_visible", list(DEFAULT_COLUMNS))


def shown_columns():
    return [c for c in st.session_state["col_order"] if c in st.session_state["col_visible"]]


def set_columns(columns):
    """Shows these columns in this order; the other columns are hidden and placed after them."""
    columns = [c for c in columns if c in TABLE_COLUMNS.values()] or DEFAULT_COLUMNS
    st.session_state["col_order"] = columns + [c for c in TABLE_COLUMNS.values() if c not in columns]
    st.session_state["col_visible"] = list(columns)
    for c in TABLE_COLUMNS.values():
        st.session_state.pop(f"cb_{c}", None)  # checkboxes are redrawn from col_visible


def apply_view():
    """Callback of the view selector: puts the saved values into the widgets before they are drawn."""
    view = load_views().get(st.session_state["view_name"])
    if view:
        for key in FILTER_KEYS:
            st.session_state[key] = view.get(key, [])
        set_columns(view.get("table_columns", DEFAULT_COLUMNS))
        st.session_state["timeframe"] = view.get("timeframe", "All snapshots")


def save_view():
    name = st.session_state["new_view_name"].strip()
    if name:
        views = load_views()
        views[name] = {key: st.session_state.get(key, []) for key in FILTER_KEYS}
        views[name]["table_columns"] = shown_columns()
        views[name]["timeframe"] = st.session_state.get("timeframe", "All snapshots")
        os.makedirs(os.path.dirname(VIEWS_PATH), exist_ok=True)
        with open(VIEWS_PATH, "w") as f:
            json.dump(views, f, indent=2, ensure_ascii=False)
        st.session_state["view_name"] = name
        st.session_state["new_view_name"] = ""


def delete_view():
    views = load_views()
    views.pop(st.session_state["view_name"], None)
    with open(VIEWS_PATH, "w") as f:
        json.dump(views, f, indent=2, ensure_ascii=False)
    st.session_state["view_name"] = "—"


# Sidebar: three menus that fold (Views, Filters, Timeframe), so all filters fit without scrolling.
# Fixed titles: a title that changes would make Streamlit re-create (and close) the menu.
with st.sidebar.expander("Views", expanded=False):
    st.selectbox("Saved view:", ["—"] + sorted(load_views()), key="view_name", on_change=apply_view,
                 help="Applies the saved filters, period and table columns.")
    if st.session_state.get("view_name", "—") != "—":
        st.button("Delete this view", on_click=delete_view)
    st.text_input("Save the current view as:", key="new_view_name", placeholder="e.g. E28 + backups")
    st.button("Save", on_click=save_view)


# Filters, from top to bottom: MRP controller, purchasing group (organisational
# responsibility), vendor, material, root cause, impact.
# Every filter accepts several values (e.g. an MRP controller and the colleagues they back up);
# an empty filter means "All". The filters cascade: each one only offers the values that exist
# within the filters above it (e.g. only the vendors of the selected MRP controllers).
cascade_df = df


def multi_filter(box, label, column, key, order=None):
    global cascade_df
    available = set(cascade_df[column].dropna().unique())
    selected = [v for v in st.session_state.get(key, []) if v in set(df[column].dropna())]
    st.session_state[key] = selected  # drops values of a saved view that no longer exist
    options = sorted(available | set(selected), key=order) if order else sorted(available | set(selected))
    selected = box.multiselect(label, options, key=key, placeholder="All")
    if selected:
        cascade_df = cascade_df[cascade_df[column].isin(selected)]
    return selected


# Impact filter order: severity first (Market, Production, Logistics, None, Not communicated)
def impact_order(value):
    return (IMPACT_FILTER_OPTIONS.index(value), "") if value in IMPACT_FILTER_OPTIONS else (len(IMPACT_FILTER_OPTIONS), value)


filters_box = st.sidebar.expander("Filters", expanded=True)
selected_mrp = multi_filter(filters_box, "MRP Controller", "MRP_CONTROLLER", "f_mrp")
selected_pgroup = multi_filter(filters_box, "Purchasing Group", "PURCHASING_GROUP", "f_pgroup")
selected_vendor_label = multi_filter(filters_box, "Vendor", "VENDOR_LABEL", "f_vendor")          # name or account number
selected_material_label = multi_filter(filters_box, "Material", "MATERIAL_LABEL", "f_material")  # number or description
selected_root_cause = multi_filter(filters_box, "Root Cause", "Root_Cause", "f_root_cause")
selected_impact = multi_filter(filters_box, "Impact", "Impact_Level", "f_impact", order=impact_order)


# --- Timeframe filter (Part A: record the user's choice only) ---
timeframe_box = st.sidebar.expander("Timeframe", expanded=False)
timeframe_option = timeframe_box.selectbox(
    "Select period:",
    ["All snapshots", "Last 4 weeks", "Last 8 weeks", "Last 16 weeks", "Custom period"],
    key="timeframe",
)

# If "Custom period" is selected, show a date range picker
start_ts, end_ts = None, None
if timeframe_option == "Custom period":
    min_date = df["First_Seen"].min().date()
    max_date = df["Last_Seen"].max().date()
    date_range = timeframe_box.date_input(
        "Date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    # date_input returns a tuple only when BOTH dates are selected
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_ts = pd.to_datetime(date_range[0])
        end_ts = pd.to_datetime(date_range[1])
timeframe_box.caption("A comment is in the period if it was visible in the dashboard at least once during it.")


# Filter data
filtered_df = df.copy()
for column, selected in [("MRP_CONTROLLER", selected_mrp), ("PURCHASING_GROUP", selected_pgroup),
                         ("VENDOR_LABEL", selected_vendor_label), ("MATERIAL_LABEL", selected_material_label),
                         ("Root_Cause", selected_root_cause), ("Impact_Level", selected_impact)]:
    if selected:
        filtered_df = filtered_df[filtered_df[column].isin(selected)]

# --- Timeframe filter (Part B: apply the choice to filtered_df) ---
# A comment stays visible for many days: keep it if [First_Seen, Last_Seen] overlaps the period.
# PERIOD = (start, end) of the timeframe, or None for all snapshots: it also limits the weeks of the trend
# charts (a comment visible for months would otherwise stretch the chart beyond the period).
PERIOD = None
if timeframe_option == "Custom period":
    if start_ts is not None:
        PERIOD = (start_ts, end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    if start_ts is not None:
        filtered_df = filtered_df[
            (filtered_df["Last_Seen"] >= start_ts)
            & (filtered_df["First_Seen"] <= end_ts)
        ]
elif timeframe_option.startswith("Last"):
    weeks = int(timeframe_option.split(" ")[1])
    latest_snapshot = df["Last_Seen"].max()
    cutoff = latest_snapshot - pd.Timedelta(weeks=weeks)
    filtered_df = filtered_df[filtered_df["Last_Seen"] >= cutoff]
    PERIOD = (cutoff, latest_snapshot)
# "All snapshots" → no filtering needed


# Name of the current selection, used in chart titles
def selection_label(values, short=False):
    """Chart-title label of one filter: the value, or "first value +N" when several are selected."""
    names = [v.split(" — ")[0] for v in values] if short else list(values)
    return names[0] if len(names) == 1 else f"{names[0]} +{len(names) - 1}" if names else None

selection_parts = [p for p in [selection_label(selected_mrp), selection_label(selected_pgroup),
                               selection_label(selected_vendor_label, short=True),
                               selection_label(selected_material_label, short=True),
                               selection_label(selected_root_cause), selection_label(selected_impact)] if p]
selection_name = " / ".join(selection_parts) if selection_parts else "All comments"

# The sidebar filters, as the Insights Lab and the Smart Analysis read them
LAB_SELECTIONS = {"MRP_CONTROLLER": selected_mrp, "PURCHASING_GROUP": selected_pgroup,
                  "VENDOR_LABEL": selected_vendor_label, "MATERIAL_LABEL": selected_material_label,
                  "Root_Cause": selected_root_cause, "Impact_Level": selected_impact}

# Compute all statistics ONCE (used by all sections and the AI Summary)
stats = compute_stats(filtered_df, df)

# ================= OVERVIEW =================
ui.section("overview", "Overview — " + selection_name)

# --- Headline: where the selection stands among all comments ---
if stats["total_comments"] > 0:
    st.markdown(
        f"**{selection_name}**: {stats['total_comments']} comments "
        f"(**{stats['share_of_all_comments_pct']}%** of all {len(df)} comments) · "
        f"{stats['root_cause_coverage_pct']}% have a root cause · "
        f"{stats['comments_still_open']} still visible on the last snapshot ({stats['latest_snapshot']})"
    )

# --- Material status tiles (ION colour code) ---
# Each comment is written on an MRP element (a material / purchase order line of the ION
# workbench). The tiles count the comments by the ION status of that element.
ui.panel_header("Material status of the commented items")
ui.status_tiles(stats.get("material_status_counts", {}), stats["total_comments"])
st.caption("Each comment is written on an item of the ION workbench (a material / purchase order line). "
           "The tiles show the status of that item the last time the comment was visible, e.g. how many "
           "comments concern items in Actual Stock Out.")
ui.status_definitions()

# --- Sections: all analyses follow each other on one page (menu at the top to jump) ---
# (Top Root Causes, tab2, is part of the Overview: its container comes first)
tab2, tab1, tab3, tab5, tab6, tab_impact, tab_signals, tab7 = [st.container() for _ in range(8)]



# ================= SECTION 1: Filtered Comments =================

with tab1:
    ui.section("filtered-comments", SECTION_TITLES["filtered-comments"])
    # "▥ Columns" (right): a panel floating over the table, under its header row (ui.column_picker).
    # Same state as before (col_order / col_visible), so the saved views keep their columns.
    shown = ui.table_toolbar("col", list(TABLE_COLUMNS.values()), DEFAULT_COLUMNS, order_key="col_order",
                             visible_key="col_visible", checkbox_prefix="cb_") or DEFAULT_COLUMNS
    to_data = {v: k for k, v in TABLE_COLUMNS.items()}
    table_df = filtered_df.sort_values("Last_Seen", ascending=False)[[to_data[c] for c in shown]] \
        .rename(columns=TABLE_COLUMNS)
    for text_col in ["Comment (EN)", "Original comment", "Comment (expanded)"]:
        if text_col in table_df:
            table_df[text_col] = inline_entries(table_df[text_col])
    for date_col in ["First seen", "Last seen", "Stockout Date"]:
        if date_col in table_df:
            table_df[date_col] = pd.to_datetime(table_df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    st.dataframe(
        table_df,
        use_container_width=True,
        height=500,
        column_config={
            "Comment (EN)": st.column_config.TextColumn(width="large"),
            "Original comment": st.column_config.TextColumn(width="large"),
        },
        hide_index=True,
    )
    st.caption("One row = one comment on one material of one vendor. "
               "'Days visible' = for how many days the comment stayed in the dashboard. "
               f"A comment with several entries (journal) shows them one after the other, separated by '{ENTRY_MARK.strip()}'.")


# ================= OVERVIEW (continued): Top Root Causes =================

with tab2:
    # Options (small button on the right of the title): how many, and how to treat the stakeholders
    # (prefix of the root cause: PL_ = Planning, S_ = Supplier...)
    AS_ENTERED, MERGED, ONE_STAKEHOLDER = ("As entered", "Stakeholders merged", "One stakeholder")
    with ui.section_with_options("top-root-causes", "Top Root Causes"):
        top_n_causes = st.selectbox("Select a view:", options=["Top 5", "Top 10", "Top 20"], key="radio_top_causes")
        cause_view = st.radio(
            "Root causes:", [AS_ENTERED, MERGED, ONE_STAKEHOLDER], key="top_causes_view", horizontal=True,
            help="As entered: one bar per root cause, with its stakeholder (PL_Internal Errors and "
                 "P_Internal Error are two bars). Stakeholders merged: the same root cause counts once, whatever "
                 "the stakeholder. One stakeholder: only the root causes of the stakeholder chosen below.")
        stakeholders = filtered_df["Root_Cause"].dropna().map(insights.stakeholder).value_counts().index.tolist()
        chosen_stakeholder = (st.selectbox("Stakeholder:", stakeholders or ["–"], key="top_causes_stakeholder")
                              if cause_view == ONE_STAKEHOLDER else None)
    n_causes = int(top_n_causes.split(" ")[1])

    def causes_of(frame):
        """The root causes of `frame` in the chosen view (NaN = not counted)."""
        if cause_view == MERGED:
            return insights.merge_stakeholders(frame["Root_Cause"])
        if cause_view == ONE_STAKEHOLDER:
            return frame["Root_Cause"].where(frame["Root_Cause"].map(insights.stakeholder) == chosen_stakeholder)
        return frame["Root_Cause"]

    causes, all_causes = causes_of(filtered_df), causes_of(df)
    view_label = {MERGED: " (stakeholders merged)", ONE_STAKEHOLDER: f" ({chosen_stakeholder})"}.get(cause_view, "")

    # 1. Comments without a root cause are counted apart (they are not a cause)
    total_comments = len(filtered_df)
    no_cause_count = int(filtered_df["Root_Cause"].isna().sum())

    # 2. Count root causes in the FILTERED data (respects all filters)
    root_cause_counts = causes.value_counts().head(n_causes)

    if root_cause_counts.empty:
        st.info("No root cause filled in for the current filters.")
    else:
        # 3. Build the color gradient: green (low) -> yellow -> orange -> red (high)
        #    We use a custom colormap and scale each bar by its count relative to the max
        colormap = mcolors.LinearSegmentedColormap.from_list(
            "severity", ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]  # green, yellow, orange, red
        )
        max_count = root_cause_counts.max()
        colors = [colormap(count / max_count) for count in root_cause_counts.values]

        # Bar chart and pie side by side (Streamlit stacks the columns on a small screen).
        # Both figures have the SAME height and widths in the same ratio as the columns (3:2),
        # so they are displayed at the same height and grow / shrink with the screen.
        col_bar, col_pie = st.columns([3, 2], gap="medium")
        chart_height = max(3.6, len(root_cause_counts) * 0.26)

        # 4. Plot (horizontal bars: easier to read long root cause names). A function of its data: ui.chart keeps
        #    the picture, so a rerun with the same data does not draw it again.
        def draw_bars(counts, colors, title, total, height):
            fig, ax = plt.subplots(figsize=(6, height), facecolor="white")
            bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
            fig.suptitle(title, fontsize=12, fontweight="bold")  # centred on the figure, not over the bars only
            ax.set_xlabel("Number of Comments")
            ax.set_xlim(0, counts.max() * 1.22)  # room for the labels at the end of the bars
            # Add the count and the share of the selection's comments at the end of each bar
            for bar, count in zip(bars, counts.values[::-1]):
                ax.text(bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{count} ({count / total * 100:.1f}%)", va="center", fontsize=8)
            fig.tight_layout()
            return fig

        with col_bar:
            # tight=False: keep the exact figure size (same height as the pie)
            ui.chart(draw_bars, root_cause_counts, colors,
                     ui.chart_title(f"Top {len(root_cause_counts)} Root Causes{view_label}", selection_name, 46),
                     total_comments,
                     chart_height, name="top_root_causes", tight=False)
            st.caption(f"% = share of the {total_comments} comments of the selection. "
                       f"{no_cause_count} comments ({no_cause_count / total_comments * 100:.1f}%) have no root cause.")

        # 5. Show the percentage share
        # Split: top N causes as individual slices, rest grouped as "Other"
        all_cause_counts = causes.value_counts()
        top_counts = all_cause_counts.head(n_causes)
        other_count = all_cause_counts.iloc[n_causes:].sum()

        pie_labels = top_counts.index.tolist()
        pie_values = top_counts.values.tolist()
        if other_count > 0:
            pie_labels.append("Other")
            pie_values.append(other_count)

        # Pie on top, legend below (labels on the slices overlap when there are many causes)
        def draw_pie(values, labels, n_top, has_other, title, height):
            fig_pie = plt.figure(figsize=(3.85, height), facecolor="white")  # 3.85: same height as the bars once in the cards
            legend_rows = -(-len(labels) // 2)
            legend_frac = (0.15 + legend_rows * 0.15) / height
            title_frac = (0.35 + 0.17 * title.count("\n")) / height
            ax_pie = fig_pie.add_axes([0.02, legend_frac + 0.02, 0.96, 1 - legend_frac - title_frac - 0.03])
            palette = plt.cm.tab20.colors if n_top > 12 else plt.cm.Set3.colors
            pie_colors = list(palette[:n_top]) + (["#BDBDBD"] if has_other else [])  # "Other" in grey
            wedges, _, _ = ax_pie.pie(
                values,
                autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",  # Show percentage on the large slices
                startangle=90,
                counterclock=False,
                colors=pie_colors,  # Distinct colors
                wedgeprops={"edgecolor": "white", "linewidth": 1},
                textprops={"fontsize": 8},
            )
            fig_pie.suptitle(title, fontsize=10, fontweight="bold", y=1 - 0.08 / height)
            legend_labels = [l if len(l) <= 26 else l[:25] + "…" for l in labels]
            fig_pie.legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                           ncol=2, fontsize=7, frameon=False, handlelength=1, columnspacing=1)
            return fig_pie

        with col_pie:
            ui.chart(draw_pie, [int(v) for v in pie_values], pie_labels, len(top_counts), other_count > 0,
                     ui.chart_title(f"Root Cause Share{view_label}", selection_name, 42), chart_height,
                     name="root_cause_share",
                     tight=False)
            # Keep the caption only for the "Other" context
            st.caption(f"Share among the comments WITH a root cause"
                       + (f" of {chosen_stakeholder}. " if cause_view == ONE_STAKEHOLDER else ". ")
                       + (f"'Other' = all remaining root causes beyond the Top {n_causes} ({other_count} comments)."
                          if other_count else ""))

        # 6. Compare with all comments: is a root cause over-represented in this selection?
        if selection_parts:
            all_total = len(df)
            compare_df = pd.DataFrame({
                "Root Cause": root_cause_counts.index,
                f"% of {selection_name} comments": (root_cause_counts.values / total_comments * 100).round(2),
                "% of all comments": [round((all_causes == c).sum() / all_total * 100, 2)
                                      for c in root_cause_counts.index],
            })
            compare_df["Difference (pts)"] = compare_df.iloc[:, 1] - compare_df.iloc[:, 2]
            st.markdown("**Compared with all comments**")
            st.dataframe(ui.compare_colours(compare_df, compare_df.columns[1], "% of all comments", decimals=2),
                         use_container_width=True, hide_index=True,
                         column_config={compare_df.columns[1]: st.column_config.NumberColumn(help=ui.COMPARE_HELP)})
            st.caption(ui.COMPARE_LEGEND.replace("'% of selection'", "the selection"))


# ================= SECTIONS 3-4: Most Frequent Comments, Root Cause Trends =================
# The validated pages of the Insights Lab (scripts/insights_pages.py) replace the first versions: similar
# wordings grouped; trends on all snapshot rows, with what changed between two weeks.
with tab3:
    insights_pages.frequent_comments_section(filtered_df, selection_name)

with tab5:
    insights_pages.trends_section(df, LAB_SELECTIONS, selection_name, PERIOD)


# ================= SECTION 6: Key Statistics =================
with tab6:
    ui.section("key-statistics", SECTION_TITLES["key-statistics"])

    # Handle the empty case first (avoid crashes on very narrow filters)
    if stats["total_comments"] == 0:
        st.info("No comments match the current filters. Relax some filters to see statistics.")
    else:
        # --- Row 1: headline metrics ---
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Comments", stats["total_comments"],
                      help="One comment on one material of one vendor")
            st.caption(f"{stats['share_of_all_comments_pct']}% of all comments")

        with col2:
            st.metric("With a root cause", f"{stats['root_cause_coverage_pct']}%")
            st.caption(f"Vendors: {stats['vendor_count']} · Purchasing groups: {stats['purchasing_group_count']} "
                       f"· MRP controllers: {stats['mrp_controller_count']} "
                       f"· Materials: {stats['material_count']}")

        with col3:
            st.metric("Median days visible", f"{stats['median_days_active']:.0f}",
                      delta=(f"{stats['median_days_active'] - stats['median_days_active_all']:+.0f} vs all"
                             if stats['median_days_active'] != stats['median_days_active_all'] else None),
                      delta_color="inverse")
            st.caption(f"{stats['comments_still_open']} still visible on {stats['latest_snapshot']} "
                       f"({stats['comments_still_open_pct']}%)")

        with col4:
            if "trend_pct" in stats:
                st.metric(
                    "Weekly comment volume",
                    f"{stats['comments_first_snapshot']} → {stats['comments_last_snapshot']}",
                    delta=f"{stats['trend_pct']:+.1f}%",
                    delta_color="inverse",
                )
            if "german_comments_pct" in stats:
                st.caption(f"{stats['german_comments_pct']}% written in German (translated)")

        # --- Row 2: time window ---
        if "first_snapshot" in stats:
            st.caption(
                f"Period: weeks of {stats['first_snapshot']} → {stats['last_snapshot']}"
            )

        # --- Root cause table with counts AND percentages (vs all comments) ---
        st.markdown("**Top 5 Root Causes**")
        cause_df = pd.DataFrame({
            "Root Cause": list(stats["top_root_causes"].keys()),
            "Comments": list(stats["top_root_causes"].values()),
            "% of selection": list(stats["top_root_causes_pct"].values()),
            "% of all comments": list(stats["top_root_causes_pct_all"].values()),
        })
        st.dataframe(ui.compare_colours(cause_df, "% of selection", "% of all comments"), use_container_width=True,
                     hide_index=True, column_config={"% of selection": st.column_config.NumberColumn(
                         format="%.1f", help=ui.COMPARE_HELP)})
        st.caption(ui.COMPARE_LEGEND)

        # --- Who is concerned: vendors and MRP controllers ---
        for column, (key, label) in zip(st.columns(3), [("top_vendors", "Vendor"),
                                                         ("top_purchasing_groups", "Purchasing Group"),
                                                         ("top_mrp_controllers", "MRP Controller")]):
            with column:
                st.markdown(f"**Top {label}s (share of the comments)**")
                st.dataframe(pd.DataFrame([{label: k, "Comments": v["comments"], "%": v["pct"]}
                                           for k, v in stats[key].items()]),
                             use_container_width=True, hide_index=True)

        # --- Rising vs falling causes, and impact: one line each (the material status is in the Overview) ---
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Root causes, first → last week**")
            for arrow, key in (("↑ Increasing", "causes_increasing"), ("↓ Decreasing", "causes_decreasing")):
                if stats.get(key):
                    st.caption(f"{arrow}: " + " · ".join(f"{cause} **{delta:+d}**" for cause, delta in stats[key].items()))
        with col_b:
            st.markdown("**Impact**")
            counts = filtered_df["Impact_Level"].value_counts().reindex(IMPACT_FILTER_OPTIONS, fill_value=0)
            st.caption(" · ".join(f"{impact} **{count}** ({count / stats['total_comments'] * 100:.0f}%)"
                                  for impact, count in counts.items() if count)
                       + "  \nNot communicated = nothing entered, which does not mean 'no impact'.")



# ================= SECTION 7: Smart Analysis =================
# Two ways to write the same analysis from the same pandas statistics:
#  - Deployed (Streamlit Cloud): Mistral API; names are always pseudonymised before sending.
#  - On this Mac (localhost): LM Studio, the local model; no data leaves the machine.
@st.cache_resource(show_spinner=False)
def pseudonymiser_template(n_rows: int):
    """Entity lists (vendors, materials...) are built once; each request gets a fresh() copy."""
    return PromptPseudonymiser(df)


def render_ai_summary(mode: str):
    """Button + loading screen + streamed answer, for mode 'cloud' or 'local'."""
    cloud = mode == "cloud"
    label = "Smart Analysis" if cloud else "local LLM (LM Studio)"
    session_key = "ai_summary_cloud" if cloud else "ai_summary"

    filter_key = tuple(tuple(v) for v in (selected_mrp, selected_pgroup, selected_vendor_label,
                  selected_material_label, selected_root_cause, selected_impact)) + (timeframe_option,)

    pseudonymise_on = cloud  # names, vendors and material numbers are always replaced before sending
    st.caption("A written review of the current selection: key findings, what drives the issues, early warnings, "
               "risks to escalate and recommended actions, illustrated with planner comments. It uses the findings "
               "of the Insights Lab (root cause → impact, weak signals, watch list, long-open comments, recurring "
               "issues) and knows the filters, so it does not state the obvious. All figures are computed from "
               "the data; the analysis only interprets them."
               + ("" if cloud else " Written by the local model (LM Studio): no data leaves this machine."))

    if stats["total_comments"] == 0:
        st.info("No comments match the current filters.")
        return
    if cloud and not is_cloud_configured():
        st.warning("Smart Analysis is not configured: add MISTRAL_API_KEY to the .env file.")
        return
    if not cloud and not is_server_available():
        st.warning("LM Studio server not reachable. Start it (Developer tab) and load the model "
                   f"({MODEL_NAME}).")
        return

    pseudonymiser = pseudonymiser_template(len(df)).fresh() if (cloud and pseudonymise_on) else None
    if st.button("Run Smart Analysis", type="primary", key=f"generate_{mode}"):
        # The LLM runs in a background thread while the loading screen shows the analysis
        # steps. Every value shown on the screen is a real figure of the current selection.
        # The answer is streamed: it appears word by word as soon as the model writes.
        result = {"parts": [], "done": False}
        # Findings of the Insights Lab for the same selection (filters + timeframe): root cause -> impact, weak
        # signals, watch list, long-open comments, recurring issues, recent changes. Computed here (not in the
        # thread: they use the Streamlit caches); the model only interprets them.
        with st.spinner("Gathering the findings of the deeper analyses..."):
            context = insights_pages.smart_context(df, filtered_df, LAB_SELECTIONS, PERIOD, timeframe_option)

        def call_llm():
            try:
                if cloud:
                    # fresh pseudonymiser: the aliases of this request only
                    pieces = stream_summary_cloud(
                        stats, stats["top_comments"], pseudonymiser_template(len(df)).fresh() if pseudonymise_on else None,
                        context=context)
                else:
                    pieces = stream_summary(stats, stats["top_comments"], context=context)
                for piece in pieces:
                    result["parts"].append(piece)
            except Exception as e:  # shown in the dashboard below
                result["error"] = e
            finally:
                result["done"] = True

        worker = threading.Thread(target=call_llm, daemon=True)
        worker.start()

        top_cause = next(iter(stats.get("top_root_causes_pct", {}).items()), None)
        n_days = (df["Last_Seen"].max() - df["First_Seen"].min()).days + 1
        steps = [
            {"text": "Loading planner comments from the daily snapshots",
             "value": f"{len(df):,} comments · {n_days} days"},
            {"text": "Applying filters", "value": f"{selection_name} → {stats['total_comments']:,} comments"},
            {"text": "Reading translated comments (DE → EN)",
             "value": f"{stats.get('german_comments_pct', 0)}% originally in German"},
            {"text": "Cross-referencing vendors and MRP controllers",
             "value": f"{stats.get('vendor_count', 0)} vendors · {stats.get('mrp_controller_count', 0)} MRP controllers"},
            {"text": "Ranking root causes",
             "value": f"top: {top_cause[0]} ({top_cause[1]}%)" if top_cause else "no root cause filled in"},
            {"text": "Measuring how long issues stay open",
             "value": f"median {stats.get('median_days_active', 0):.0f} days · {stats.get('comments_still_open', 0)} still open"},
            {"text": "Detecting week-by-week trends",
             "value": f"{stats.get('comments_first_snapshot', '–')} → {stats.get('comments_last_snapshot', '–')} comments/week"},
            {"text": "Linking root causes to Production / Market impacts",
             "value": f"{context.get('root_cause_to_impact', {}).get('severe_impact_pct_in_selection', '–')}% severe "
                      "among the communicated impacts"},
            {"text": "Looking for weak signals and open items at risk",
             "value": f"{len(context.get('watch_list_open_items_most_at_risk', []))} items on the watch list"},
            {"text": "Writing the analysis" if cloud else "Writing the summary with the local language model",
             "value": "names pseudonymised" if cloud else "on-device"},
        ]
        sample = (filtered_df["Comment_EN"].dropna().astype(str).str.slice(0, 110)
                  .sample(min(40, filtered_df["Comment_EN"].notna().sum()), random_state=0).tolist())

        step_ms = 850
        min_seconds = step_ms * (len(steps) - 1) / 1000 + 0.6  # let the animation reach the last step
        screen = st.empty()
        with screen.container():
            ui.ai_loading_screen(steps, total=stats["total_comments"],
                                 sub="Statistics computed from the data" if cloud else f"Statistics computed by pandas · {label}",
                                 comments=sample, step_ms=step_ms)
        started = time.time()
        # Loading screen until the model starts writing (and at least the animation length)
        while (not result["parts"] and not result["done"]) or time.time() - started < min_seconds:
            time.sleep(0.2)
        screen.empty()

        # Then the text appears as it is written
        live = st.empty()
        meta = f"{selection_name} · writing…" if cloud else f"{selection_name} · writing with the {label}…"
        while not result["done"]:
            live.markdown(ui.ai_result_html("".join(result["parts"]) + " ▌", meta, tools=False), unsafe_allow_html=True)
            time.sleep(0.15)
        live.empty()

        text = tidy("".join(result["parts"]))  # up to the last complete sentence
        if "error" in result:
            st.error(f"The analysis could not be generated: {result['error']}")
        elif not text:
            st.error("The analysis could not be generated: empty answer.")
        else:
            st.session_state[session_key] = (
                filter_key, text, selection_name, stats["total_comments"],
                time.strftime("%Y-%m-%d %H:%M"), label)

    saved = st.session_state.get(session_key)
    if saved:
        if saved[0] != filter_key:
            st.caption("Note: analysis generated with different filters.")
        ui.ai_result(saved[1], f"{saved[2]} · based on {saved[3]} comments · generated {saved[4]}"
                               + ("" if cloud else f" with {saved[5]} · figures computed by pandas"))


# ================= Impact by Root Cause, Weak Signals (scripts/insights_pages.py) =================
with tab_impact:
    insights_pages.impact_section(df, LAB_SELECTIONS, selection_name, PERIOD)

with tab_signals:
    insights_pages.weak_signals_section(df, LAB_SELECTIONS, selection_name, PERIOD)


with tab7:
    ui.section("ai-summary", SECTION_TITLES["ai-summary"])
    render_ai_summary("cloud" if config.ON_CLOUD else "local")  # localhost: LM Studio; Streamlit Cloud: Mistral API

ui.page_end()  # room under Smart Analysis: the menu brings its title up under the menu too
