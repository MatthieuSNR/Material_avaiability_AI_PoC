"""
Layout and components of the dashboard, close to the Roche ION Material Availability dashboard:
* look: blue top bar (a click brings back to the Overview), menu over the full width (scroll_spy: section on
  screen, section title brought under the menu), blue section titles with an optional "Options" button,
  material status tiles, coloured comparisons (compare_colours);
* in the browser, without a rerun of Streamlit 1.32: "Show explanations" (explain_switch), switches between
  parts of a section (client_switch + pane), PNG export of the charts, copy / download of
  the Smart Analysis;
* charts (chart): a matplotlib figure drawn once for the same data, shown as a PNG card;
* tables: column picker (show / hide, drag and drop with streamlit-sortables);
* the animated loading screen and the result card of the Smart Analysis.
Kept apart from app.py so that app.py only contains the analysis.
"""

import base64
import html
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components

ROCHE_BLUE = "#0B41CD"
ION_BLUE = "#1A73E8"
STATUS_ORDER = ["Actual Stock Out", "Potential Stock Out", "Below Safety", "Good Part"]
STATUS_COLORS = {
    "Actual Stock Out": "#E0443A",
    "Potential Stock Out": "#F28C28",
    "Below Safety": "#F5C518",
    "Good Part": "#2FA84F",
}
STATUS_ICONS = {"Actual Stock Out": "!", "Potential Stock Out": "▲", "Below Safety": "↓", "Good Part": "✓"}
STATUS_DEFINITIONS = {
    "Actual Stock Out": "The current delivery confirmation is later than the SAP requirement date or the delivery is "
                        "overdue. The production demand is not covered: 100% of the safety buffer and GR time are used.",
    "Potential Stock Out": "The current delivery confirmation is later than the SAP requirement date. More than 70% of "
                           "the safety buffer is consumed; GR time is not used.",
    "Below Safety": "The current delivery confirmation is later than the SAP requirement date. Less than 70% of the "
                    "safety buffer is consumed; GR time is not used.",
    "Good Part": "The current delivery confirmation is earlier than the SAP requirement date. The production demand is covered.",
}

RADIUS = "2px"  # ION uses almost square corners
BAR_HEIGHT = 54  # px, height of the blue top bar (fixed, full width)
# Chevron pointing left (same icon family as Streamlit's "open sidebar" arrow)
CHEVRON_LEFT = ("url(\"data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E"
                "%3Cpath d='M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z'/%3E%3C/svg%3E\")")
LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "roche_logo.png")

CSS = f"""
<style>
.stApp {{ background-color: #F3F5F8; }}
[data-testid="stDecoration"] {{ display: none; }}
/* Streamlit's own toolbar (Deploy, menu) sits on the right of the blue bar, in white */
header[data-testid="stHeader"] {{ background: transparent; height: {BAR_HEIGHT}px; z-index: 1000002; }}
/* the transparent Streamlit header lets the clicks through to the blue bar (back to the Overview), except on
   its own buttons (Deploy, menu, "Source file changed") */
header[data-testid="stHeader"] {{ pointer-events: none; }}
header[data-testid="stHeader"] [data-testid="stToolbar"], header[data-testid="stHeader"] [data-testid="stStatusWidget"],
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a, [data-testid="collapsedControl"] {{ pointer-events: auto; }}
header[data-testid="stHeader"] [data-testid="stToolbar"] {{ top: 50%; transform: translateY(-50%); right: 12px; }}
header[data-testid="stHeader"] button, header[data-testid="stHeader"] [data-testid="stToolbar"] * {{ color: #fff !important; }}
.block-container {{ padding-top: {BAR_HEIGHT + 4}px; }}
/* Sidebar starts under the blue bar */
section[data-testid="stSidebar"] {{
  background-color: #FFFFFF; border-right: 1px solid #E1E5EB;
  top: {BAR_HEIGHT}px !important; height: calc(100vh - {BAR_HEIGHT}px) !important;
}}
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {{ top: {BAR_HEIGHT + 8}px !important; }}
/* "Close sidebar" button: a left arrow instead of the cross */
section[data-testid="stSidebar"] button[data-testid="baseButton-header"] svg {{ display: none; }}
section[data-testid="stSidebar"] button[data-testid="baseButton-header"]::before {{
  content: ""; display: block; width: 1.5rem; height: 1.5rem; background-color: currentColor;
  -webkit-mask: {CHEVRON_LEFT} center / contain no-repeat; mask: {CHEVRON_LEFT} center / contain no-repeat;
}}

/* Charts: white cards, centred, same look everywhere */
[data-testid="stImage"] {{ display: flex; justify-content: center; }}
[data-testid="stImage"] img {{
  background: #fff; border: 1px solid #E1E5EB; border-radius: {RADIUS}; padding: 10px;
  box-shadow: 0 1px 2px rgba(0,0,0,.05); max-width: 100%; box-sizing: border-box;
}}
/* Sidebar filters: a chosen value (blue tag) stays on one line and uses the width of the sidebar; when it is
   too long, its beginning is shown, ended by "…" (the filters below do not move down) */
section[data-testid="stSidebar"] [data-baseweb="tag"] {{ max-width: 100%; min-width: 0; }}
section[data-testid="stSidebar"] [data-baseweb="tag"] span[title] {{
  max-width: none; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
section[data-testid="stSidebar"] [data-baseweb="select"] > div > div:first-child {{ min-width: 0; }}

/* Switch in the browser (client_switch): one bar split in equal parts, the first option chosen by default */
.ion-seg {{ display: flex; width: 100%; border: 1px solid {ION_BLUE}; border-radius: {RADIUS}; overflow: hidden;
  margin: 2px 0 10px 0; }}
.ion-seg a {{ flex: 1 1 0; text-align: center; padding: 7px 10px; cursor: pointer; background: #fff;
  color: {ION_BLUE} !important; font-weight: 600; font-size: .9rem; text-decoration: none !important; user-select: none; }}
.ion-seg a + a {{ border-left: 1px solid {ION_BLUE}; }}
.ion-seg a:hover {{ background: #E8F0FE; }}
.ion-seg a[data-option="0"] {{ background: {ION_BLUE}; color: #fff !important; }}
.ion-findings {{ background: rgba(33, 195, 84, .1); color: rgb(23, 114, 51); border-radius: {RADIUS};
  padding: 14px 18px; margin: 4px 0 12px 0; line-height: 1.55; }}
.ion-findings p {{ margin: 0 0 8px 0; }}
.ion-findings p:last-child {{ margin-bottom: 0; }}

/* Charts drawn by ui.chart: a white card, with a small "PNG" download link under it */
.ion-chart {{ margin-bottom: 6px; }}
.ion-chart img {{
  display: block; width: 100%; background: #fff; border: 1px solid #E1E5EB; border-radius: {RADIUS}; padding: 10px;
  box-shadow: 0 1px 2px rgba(0,0,0,.05); box-sizing: border-box;
}}
/* Table with a copy button on a column (copy_table) */
.ion-table {{ max-height: 430px; overflow: auto; border: 1px solid #E5E7EB; border-radius: 6px; background: #fff; }}
.ion-table table {{ border-collapse: collapse; width: 100%; font-size: .82rem; margin: 0; }}
.ion-table th {{ position: sticky; top: 0; z-index: 1; background: #F7F8FA; color: #6B7280; font-weight: 500;
  text-align: left; white-space: nowrap; padding: 7px 10px; border: 0; border-bottom: 1px solid #E5E7EB; }}
.ion-table td {{ padding: 6px 10px; vertical-align: top; border: 0; border-bottom: 1px solid #F0F1F3; min-width: 90px; }}
.ion-table td.wide {{ min-width: 300px; }}
.ion-table td.nowrap {{ white-space: nowrap; }}
.ion-table .bar {{ display: inline-block; width: 60px; height: 6px; margin-right: 6px; vertical-align: middle;
  background: #E5E7EB; border-radius: 3px; overflow: hidden; }}
.ion-table .bar i {{ display: block; height: 100%; background: {ION_BLUE}; }}
.ion-copy {{ margin-left: 6px; padding: 1px 5px; border: 1px solid #D1D5DB; border-radius: 4px; cursor: pointer;
  color: {ION_BLUE} !important; text-decoration: none; font-size: .72rem; }}
.ion-copy:hover {{ background: #EAF1FE; }}
.ion-chart .dl {{ text-align: right; font-size: .78rem; margin-top: 3px; }}
.ion-chart .dl a, .ai-tools a {{ color: {ION_BLUE} !important; text-decoration: none; cursor: pointer; }}
.ion-chart .dl a:hover, .ai-tools a:hover {{ text-decoration: underline; }}

/* "How to read" boxes: hidden until the "Show explanations" switch of their section is on (in the browser,
   no rerun; explain_switch / explain_box) */
.ion-explain {{
  display: none; background: #E8F0FE; color: #0B3A82; border-radius: {RADIUS}; padding: 12px 16px;
  margin: 4px 0 12px 0; font-size: .92rem; line-height: 1.5;
}}
.ion-explain p {{ margin: 0 0 6px 0; }}
.ion-explain p:last-child {{ margin-bottom: 0; }}
.explain-switch {{ display: inline-flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; font-size: .9rem; }}
.explain-switch .knob {{
  width: 30px; height: 16px; border-radius: 8px; background: #C9CED6; position: relative; transition: background .15s;
}}
.explain-switch .knob::after {{
  content: ""; position: absolute; top: 2px; left: 2px; width: 12px; height: 12px; border-radius: 50%;
  background: #fff; transition: left .15s;
}}

/* Room under the last section, so that the menu can bring its title up under the menu as well */
.ion-page-end {{ height: 70vh; }}

/* Smart Analysis: copy / download links */
.ai-tools {{ display: flex; gap: 16px; justify-content: flex-end; font-size: .82rem; margin-top: 8px; }}
html, .main, [data-testid="stAppViewContainer"] {{ scroll-behavior: smooth; }}

/* Square corners on the Streamlit widgets too */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="popover"] ul, .stButton > button,
.stDateInput > div > div, [data-testid="stExpander"] details, [data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"], [data-testid="stNotification"], [data-testid="stAlert"] > div {{
  border-radius: {RADIUS} !important;
}}

.ion-topbar {{
  position: fixed; top: 0; left: 0; right: 0; height: {BAR_HEIGHT}px; z-index: 1000001;
  background: {ION_BLUE}; color: #fff; padding: 0 200px 0 18px;
  display: flex; align-items: center; gap: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.2);
}}
.ion-topbar {{ cursor: pointer; }}   /* a click brings back to the Overview */
.ion-topbar .brand {{ display: flex; align-items: center; gap: 14px; font-size: 1.15rem; font-weight: 600; }}
.ion-topbar .brand img {{ height: 30px; width: auto; display: block; }}
.ion-topbar .hex {{
  border: 2px solid #fff; padding: 1px 10px; font-weight: 700; font-size: .95rem;
  clip-path: polygon(12% 0, 88% 0, 100% 50%, 88% 100%, 12% 100%, 0 50%);
}}
.ion-topbar .sub {{ font-size: .85rem; opacity: .9; }}
.ion-topbar .tag {{ background: rgba(255,255,255,.18); border-radius: {RADIUS}; padding: 2px 8px; font-size: .8rem; }}

/* Navigation menu: stays just under the blue bar while scrolling */
div[data-testid="stVerticalBlock"] > div:has(.ion-nav) {{ position: sticky; top: {BAR_HEIGHT}px; z-index: 990; }}
.ion-nav {{
  display: flex; flex-wrap: wrap; gap: 2px; background: #fff; border: 1px solid #D6DCE5;
  padding: 4px; box-shadow: 0 2px 4px rgba(0,0,0,.06);
}}
.ion-nav a {{
  color: #1F2937 !important; text-decoration: none !important; font-size: .84rem; padding: 6px 8px;
  border-radius: {RADIUS}; white-space: nowrap; flex: 1 1 auto; text-align: center;   /* full width of the menu */
}}
.ion-nav a:hover {{ background: #E8F0FE; }}
.ion-nav a.active {{ background: {ION_BLUE}; color: #fff !important; }}
.emo {{ margin-right: 6px; }}
.ion-anchor {{ scroll-margin-top: {BAR_HEIGHT + 70}px; height: 0; }}
/* the invisible scroll-spy iframe takes no space */
div[data-testid="stElementContainer"]:has( iframe[height="0"]) {{ display: none; }}

.ion-panel {{
  background: {ION_BLUE}; color: #fff; font-weight: 600; padding: 7px 14px; border-radius: {RADIUS};
  margin: 18px 0 8px 0; font-size: .98rem;
}}
/* Folded panels (insights_pages.fold): a blue header like .ion-panel, the arrow on the right opens it */
[data-testid="stExpander"]:has(.ion-fold) {{ margin: 18px 0 8px 0; }}
[data-testid="stExpander"]:has(.ion-fold) details {{ border: 1px solid {ION_BLUE}; }}
[data-testid="stExpander"]:has(.ion-fold) summary {{ background: {ION_BLUE}; padding: 7px 14px; }}
[data-testid="stExpander"]:has(.ion-fold) summary p {{ color: #fff; font-weight: 600; font-size: .98rem; }}
[data-testid="stExpander"]:has(.ion-fold) summary svg {{ color: #fff; fill: #fff; }}
[data-testid="stExpander"]:has(.ion-fold) summary:hover p, [data-testid="stExpander"]:has(.ion-fold) summary:hover svg {{
  color: #fff; fill: #fff; opacity: .85;
}}
[data-testid="stMarkdownContainer"]:has(.ion-fold) {{ display: none; }}
.ion-section {{ margin-top: 26px; }}
.ion-section .ion-panel {{ font-size: 1.05rem; padding: 9px 14px; }}
/* Section with options: a small "Options" button on the right of its blue title (section_with_options) */
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .ion-section.has-options) {{
  position: relative; gap: 0;
}}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .ion-section.has-options)
  > div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stPopover"]) {{
  position: absolute; top: 31px; right: 8px; width: auto !important; z-index: 5;
}}
.ion-section.has-options .ion-panel {{ padding-right: 120px; }}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .ion-section.has-options)
  [data-testid="stPopover"] button {{
  min-height: 0; padding: 2px 10px; background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.6);
  color: #fff;
}}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .ion-section.has-options)
  [data-testid="stPopover"] button p {{ font-size: .8rem; color: #fff; }}
div[data-baseweb="popover"]:has(.section-options) {{ z-index: 1000005 !important; }}
div[data-testid="stPopoverBody"]:has(.section-options) {{ width: 640px; max-width: 92vw; }}
.section-options {{ display: none; }}

.ion-tiles {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 4px 0 6px 0; }}
.ion-tile {{
  position: relative; color: #fff; border-radius: {RADIUS}; padding: 12px 16px; min-height: 86px; overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,.15);
}}
.ion-tile .num {{ font-size: 2rem; font-weight: 700; line-height: 1.1; }}
.ion-tile .lbl {{ font-size: .88rem; margin-top: 2px; }}
.ion-tile .pct {{ font-size: .78rem; opacity: .9; }}
.ion-tile .icon {{
  position: absolute; right: 14px; top: 50%; transform: translateY(-50%); font-size: 2.6rem; font-weight: 700;
  opacity: .35;
}}
.ion-tile.light {{ color: #3A3A3A; }}
@media (max-width: 900px) {{ .ion-tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}

.ai-result {{
  background: #fff; border-left: 4px solid {ROCHE_BLUE}; border-radius: {RADIUS}; padding: 16px 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,.12); line-height: 1.6;
}}
.ai-result .meta {{ color: #6B7280; font-size: .8rem; margin-top: 10px; }}
.ai-result h4 {{
  color: {ION_BLUE}; font-size: 1rem; font-weight: 700; margin: 16px 0 6px 0; padding-bottom: 4px;
  border-bottom: 1px solid #E1E5EB;
}}
.ai-result h4:first-child {{ margin-top: 0; }}
.ai-result ul {{ margin: 0 0 4px 0; padding-left: 22px; }}
.ai-result li {{ margin: 4px 0; }}
.ai-result p {{ margin: 6px 0; }}
</style>
"""


def apply_style():
    """Inject the CSS and align the matplotlib charts with the dashboard colours."""
    st.markdown(CSS, unsafe_allow_html=True)
    plt.rcParams.update({
        "axes.prop_cycle": plt.cycler(color=[ION_BLUE, "#E0443A", "#F28C28", "#2FA84F", "#7B61FF",
                                             "#00A3AD", "#F5C518", "#8C8C8C"]),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
    })


def _logo_html():
    """Roche logo from assets/roche_logo.png (kept local, not on GitHub); text badge if missing."""
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{data}" alt="Roche">'
    return '<span class="hex">Roche</span>'


def _label(text):
    """Escape a label; a leading emoji gets its own span so it keeps a gap with the text
    (Streamlit's markdown drops the space)."""
    first, _, rest = text.partition(" ")
    if rest and not first.isalnum():
        return f'<span class="emo">{html.escape(first)}</span>{html.escape(rest)}'
    return html.escape(text)


def top_bar(title, subtitle, sections, tag=""):
    """Blue title bar + navigation menu, one row over the full width. sections = [(anchor_id, label)]: clicking a
    label scrolls to that section, its blue title just under the menu; the menu stays visible at the top."""
    st.markdown(
        f'<div class="ion-topbar"><div class="brand">{_logo_html()}'
        f'<span>{html.escape(title)}</span><span class="sub">{html.escape(subtitle)}</span>'
        + (f'<span class="tag">{html.escape(tag)}</span>' if tag else "") + '</div></div>',
        unsafe_allow_html=True,
    )
    links = "".join(f'<a href="#{a}" target="_self" data-target="{a}">{_label(label)}</a>' for a, label in sections)
    st.markdown(f'<div class="ion-nav-wrap"><div class="ion-nav">{links}</div></div>', unsafe_allow_html=True)


SCROLL_SPY_JS = """
<script>
// Highlights in the menu the section currently on screen. The code is installed once in the
// dashboard page itself (not in this iframe), so it survives Streamlit reruns.
(function () {
  const doc = window.parent.document;
  if (doc.getElementById("ion-scroll-spy")) return;
  const script = doc.createElement("script");
  script.id = "ion-scroll-spy";
  script.textContent = `
    (function () {
      const MIN_OFFSET = __OFFSET__;
      let ticking = false, scroller = null;
      function update() {
        ticking = false;
        const links = [...document.querySelectorAll(".ion-nav a[data-target]")]
          .filter(a => document.getElementById(a.dataset.target));
        if (!links.length) {
          document.querySelectorAll(".ion-nav a[data-target].active").forEach(a => a.classList.remove("active"));
          return;
        }
        // a section is "current" once its title is in the upper third of the screen
        const offset = Math.max(MIN_OFFSET, window.innerHeight * 0.35);
        let current = links[0].dataset.target;
        links.forEach(a => {
          const el = document.getElementById(a.dataset.target);
          if (el && el.getBoundingClientRect().top <= offset) current = a.dataset.target;
        });
        // at the very bottom of the page, the last section is the current one
        if (scroller && scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 4) {
          current = links[links.length - 1].dataset.target;
        }
        links.forEach(a => a.classList.toggle("active", a.dataset.target === current));
      }
      function request(e) {
        const t = e && (e.target === document ? document.scrollingElement : e.target);
        if (t && t.scrollHeight > t.clientHeight) scroller = t;
        if (!ticking) { ticking = true; requestAnimationFrame(update); }
      }
      // A menu entry: its section's blue title lands just under the menu (measured, the menu may wrap);
      // corrected once more after the charts above it have finished drawing.
      document.addEventListener("click", e => {
        const a = e.target.closest(".ion-nav a[data-target]");
        const anchor = a && document.getElementById(a.dataset.target);
        const el = anchor && (anchor.nextElementSibling || anchor);   // the blue title
        if (!el) return;
        e.preventDefault();
        const go = (behavior) => {
          const nav = document.querySelector(".ion-nav");
          // the menu sticks under the blue bar: bar height + menu height (it may wrap on two lines)
          el.style.scrollMarginTop = (__BAR__ + (nav ? nav.offsetHeight : 40) + 8) + "px";
          el.scrollIntoView({behavior: behavior, block: "start"});
        };
        go("smooth");
        setTimeout(() => go("instant"), 1200);   // "auto" would be smooth too (CSS scroll-behavior)
      }, true);
      // The blue bar at the top: back to the beginning of the dashboard (Overview)
      document.addEventListener("click", e => {
        if (!e.target.closest(".ion-topbar")) return;
        const first = document.querySelector(".ion-nav a[data-target]");
        if (first) first.click();
      }, true);
      // "Show explanations" of a section: a class on the page (kept across reruns), no rerun
      document.addEventListener("click", e => {
        const sw = e.target.closest(".explain-switch");
        if (!sw) return;
        document.body.classList.toggle("explain-" + sw.dataset.section);
      }, true);
      // Switch of panes (client_switch): a class on the page shows the chosen pane, no rerun
      document.addEventListener("click", e => {
        const a = e.target.closest(".ion-seg a[data-group]");
        if (!a) return;
        e.preventDefault();
        const prefix = "switch-" + a.dataset.group + "-";
        [...document.body.classList].filter(c => c.startsWith(prefix)).forEach(c => document.body.classList.remove(c));
        document.body.classList.add(prefix + a.dataset.option);
      }, true);
      // "PNG" under a chart: saves the picture already in the page
      document.addEventListener("click", e => {
        const a = e.target.closest(".ion-chart .dl a");
        if (!a) return;
        e.preventDefault();
        const img = a.closest(".ion-chart").querySelector("img");
        const link = document.createElement("a");
        link.href = img.src; link.download = a.dataset.name + ".png";
        document.body.appendChild(link); link.click(); link.remove();
      }, true);
      // Copy button of a table cell (copy_table), e.g. a material number to paste into SAP
      document.addEventListener("click", async e => {
        const a = e.target.closest("a.ion-copy");
        if (!a) return;
        e.preventDefault();
        try {
          await navigator.clipboard.writeText(a.dataset.copy);
        } catch (err) {   // no clipboard API here: copy through a hidden field
          const field = document.createElement("textarea");
          field.value = a.dataset.copy; document.body.appendChild(field); field.select();
          document.execCommand("copy"); field.remove();
        }
        a.textContent = "✓ Copied";
        setTimeout(() => { a.textContent = "Copy"; }, 1200);
      }, true);
      // Smart Analysis: copy (formatted, for Word / e-mail, and plain text) or download
      document.addEventListener("click", async e => {
        const a = e.target.closest(".ai-tools a");
        if (!a) return;
        e.preventDefault();
        const box = a.closest(".ai-result").querySelector(".ai-body");
        if (a.dataset.action === "copy") {
          const htmlText = box.innerHTML, plain = box.innerText;
          try {
            await navigator.clipboard.write([new ClipboardItem({
              "text/html": new Blob([htmlText], {type: "text/html"}),
              "text/plain": new Blob([plain], {type: "text/plain"})})]);
          } catch (err) {   // no clipboard API (e.g. http on another machine): copy the selection
            const range = document.createRange(); range.selectNodeContents(box);
            const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
            document.execCommand("copy"); sel.removeAllRanges();
          }
          const label = a.textContent; a.textContent = "Copied";
          setTimeout(() => { a.textContent = label; }, 1500);
        } else {
          const doc = "<html><head><meta charset='utf-8'></head><body>" + box.innerHTML + "</body></html>";
          const blob = a.dataset.action === "word" ? new Blob(["\ufeff" + doc], {type: "application/msword"})
                                                   : new Blob([box.innerText], {type: "text/plain"});
          const link = document.createElement("a");
          link.href = URL.createObjectURL(blob);
          link.download = "smart_analysis." + (a.dataset.action === "word" ? "doc" : "txt");
          document.body.appendChild(link); link.click(); link.remove();
        }
      }, true);
      document.addEventListener("scroll", request, true);
      window.addEventListener("resize", request);
      setInterval(update, 700);  // after Streamlit re-renders the menu
      update();
    })();`;
  doc.body.appendChild(script);
})();
</script>
"""


def scroll_spy():
    """Invisible helper: colours in blue the menu entry of the section on screen."""
    components.html(SCROLL_SPY_JS.replace("__OFFSET__", str(BAR_HEIGHT + 90)).replace("__BAR__", str(BAR_HEIGHT)),
                    height=0)


def section(anchor_id, title, options=False):
    """Start of a page section: anchor for the menu + blue header. options=True: draws in the current container
    a small 'Options' button on the right of the header and returns its panel (see section_with_options)."""
    extra = " has-options" if options else ""
    st.markdown(
        f'<div class="ion-section{extra}"><div id="{anchor_id}" class="ion-anchor"></div>'
        f'<div class="ion-panel">{_label(title)}</div></div>',
        unsafe_allow_html=True,
    )
    if options:
        panel = st.popover("Options")
        panel.markdown('<div class="section-options"></div>', unsafe_allow_html=True)
        return panel


def section_with_options(anchor_id, title):
    """Blue section header with a small 'Options' button on its right; returns the panel of the button
    (settings of the section, 'Show explanations'). Header and button share a container (for the CSS)."""
    with st.container():
        return section(anchor_id, title, options=True)


def explain_switch(section_id):
    """'Show explanations' switch of a section. It works in the browser (a class on the page, set by the
    script of scroll_spy): showing the explanations does not rerun the dashboard."""
    st.markdown(
        f'<style>body.explain-{section_id} .ion-explain[data-section="{section_id}"] {{ display: block; }}'
        f'body.explain-{section_id} .explain-switch[data-section="{section_id}"] .knob {{ background: {ION_BLUE}; }}'
        f'body.explain-{section_id} .explain-switch[data-section="{section_id}"] .knob::after {{ left: 16px; }}</style>'
        f'<span class="explain-switch" data-section="{section_id}"><span class="knob"></span>Show explanations</span>',
        unsafe_allow_html=True)


def _markdown_html(text):
    """**bold**, *italic*, `code` and paragraphs (blank line) of a short text, as HTML."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return "".join(f"<p>{p.strip()}</p>" for p in text.split("\n\n") if p.strip())


def explain_box(section_id, text):
    """A 'how to read' box (markdown text), shown when the switch of its section is on."""
    st.markdown(explain_html(section_id, text), unsafe_allow_html=True)


def explain_html(section_id, text):
    return f'<div class="ion-explain" data-section="{section_id}">{_markdown_html(text)}</div>'


def findings_html(lines):
    """The green box of the main findings (markdown lines), as HTML."""
    if not lines:
        return ""
    paragraphs = "\n\n".join(lines)
    return f'<div class="ion-findings">{_markdown_html(paragraphs)}</div>'


def client_switch(group, options):
    """A bar split in equal parts over the full width, one per option; the chosen one in blue. It only shows one
    of the panes (pane(group, i, html)) and hides the others, in the browser: no rerun of the dashboard."""
    def show(option, visible, on=""):
        """CSS showing (or hiding) the pane(s) `option` (None = all) of the group; on = condition on the page."""
        o = "" if option is None else f'[data-option="{option}"]'
        return f'{on} .ion-pane[data-group="{group}"]{o} {{ display: {"block" if visible else "none"}; }}'

    rules = [show(i, False) for i in range(1, len(options))]          # by default: the first option
    for i in range(1, len(options)):
        on = f"body.switch-{group}-{i}"
        rules += [show(None, False, on), show(i, True, on),
                  f'{on} .ion-seg[data-group="{group}"] a[data-option="0"] {{ background: #fff; color: {ION_BLUE} !important; }}',
                  f'{on} .ion-seg[data-group="{group}"] a[data-option="{i}"] {{ background: {ION_BLUE}; color: #fff !important; }}']
    links = "".join(f'<a data-group="{group}" data-option="{i}">{html.escape(o)}</a>' for i, o in enumerate(options))
    st.markdown(f'<style>{"".join(rules)}</style><div class="ion-seg" data-group="{group}">{links}</div>',
                unsafe_allow_html=True)


def pane(group, index, content_html):
    """One pane of client_switch: shown when its option is chosen."""
    st.markdown(f'<div class="ion-pane" data-group="{group}" data-option="{index}">{content_html}</div>',
                unsafe_allow_html=True)


_PNG_CACHE = {}          # fingerprint of a chart -> (png, name); shared by all users, the oldest dropped
_PNG_CACHE_SIZE = 300


def _fingerprint(value):
    """A stable text for the inputs of a chart (data frames hashed on their content)."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        content = int(pd.util.hash_pandas_object(value, index=True).sum())
        names = tuple(value.columns) if isinstance(value, pd.DataFrame) else value.name
        return f"{type(value).__name__}{value.shape}{names}{content}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fingerprint(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k!r}:{_fingerprint(v)}" for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))) + "}"
    if isinstance(value, np.ndarray):
        return f"array{value.shape}{hash(value.tobytes())}"
    return repr(value)


def _png(fig, tight):
    """The figure as a PNG, base64 (put directly in the page: no processing by Streamlit at every rerun)."""
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight" if tight else None, facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _chart_name(fig):
    title = (fig._suptitle.get_text() if fig._suptitle else "") or next(
        (ax.get_title() for ax in fig.axes if ax.get_title()), "chart")
    return re.sub(r"[^A-Za-z0-9]+", "_", title.split(" — ")[0]).strip("_").lower() or "chart"


def chart_title(head, selection, width):
    """'head — selection' when it fits on `width` characters, otherwise the selection on its own lines under the
    head (a long vendor or several filters would be cut at the edge of the chart)."""
    import textwrap
    if len(head) + 3 + len(selection) <= width:
        return f"{head} — {selection}"
    return "\n".join([head] + textwrap.wrap(selection, width))


def chart(draw, *args, name=None, tight=True, **kwargs):
    """Draw a chart as a white card, with a 'PNG' link under it that saves the picture (in the browser: no
    rerun). draw(*args, **kwargs) returns a matplotlib figure; the picture is kept for the same inputs, so a
    rerun of the dashboard (e.g. an option of another section) does not draw the unchanged charts again.
    draw may also be a figure (drawn every time). name = file name (default: from the title)."""
    st.markdown(chart_html(draw, *args, name=name, tight=tight, **kwargs), unsafe_allow_html=True)


def chart_html(draw, *args, name=None, tight=True, **kwargs):
    """The card of `chart` as HTML (to put it in a pane of client_switch)."""
    if isinstance(draw, plt.Figure):   # drawn every time
        png, name = _png(draw, tight), name or _chart_name(draw)
    else:
        key = getattr(draw, "__qualname__", repr(draw)) + _fingerprint((args, kwargs, tight))
        if key not in _PNG_CACHE:
            fig = draw(*args, **kwargs)
            if len(_PNG_CACHE) >= _PNG_CACHE_SIZE:
                _PNG_CACHE.pop(next(iter(_PNG_CACHE)))
            _PNG_CACHE[key] = (_png(fig, tight), _chart_name(fig))
        png, default_name = _PNG_CACHE[key]
        name = name or default_name
    return (f'<div class="ion-chart"><img src="data:image/png;base64,{png}" alt="{html.escape(name)}">'
            f'<div class="dl"><a href="#" data-name="{html.escape(name)}">⬇ PNG</a></div></div>')


CHART_PADDING = {"left": 12, "right": 12, "top": 12, "bottom": 12}


def clickable_bars(data, title, height, key, on_pick):
    """Horizontal bars of `data` (columns Cause, Materials, Colour, Label) as a white card; on_pick runs when a bar
    is clicked and reads the choice from st.session_state[key] (selection 'pick')."""
    axis_y = alt.Y("Cause:N", sort=None, title=None, axis=alt.Axis(labelLimit=240, labelFontSize=11, ticks=False))
    axis_x = alt.X("Materials:Q", title="Number of materials", scale=alt.Scale(domain=[0, data["Materials"].max() * 1.25]),
                   axis=alt.Axis(tickMinStep=1, format="d", grid=False))
    base = alt.Chart(data).encode(y=axis_y, x=axis_x)
    pick = alt.selection_point(name="pick", fields=["Cause"], on="click")
    bars = base.mark_bar(cursor="pointer").encode(color=alt.Color("Colour:N", scale=None),
                                                  tooltip=["Cause", "Materials"]).add_params(pick)
    labels = base.mark_text(align="left", dx=4, fontSize=11, color="#374151").encode(text="Label:N")
    chart = alt.layer(bars, labels).properties(title=_card_title(title), height=height, background="white", padding=CHART_PADDING)
    st.altair_chart(chart.configure_view(stroke=None), use_container_width=True, theme=None, key=key,
                    on_select=on_pick, selection_mode="pick")


def clickable_pie(data, title, height, key, on_pick):
    """Pie of `data` (columns Cause, Materials, Colour, Share), clickable like clickable_bars."""
    data = data.assign(Order=range(len(data)))
    base = alt.Chart(data).encode(theta=alt.Theta("Materials:Q", stack=True), order=alt.Order("Order:Q"))
    pick = alt.selection_point(name="pick", fields=["Cause"], on="click")
    slices = base.mark_arc(outerRadius=95, stroke="white", cursor="pointer").encode(
        color=alt.Color("Cause:N", sort=data["Cause"].tolist(),
                        scale=alt.Scale(domain=data["Cause"].tolist(), range=data["Colour"].tolist()),
                        legend=alt.Legend(orient="bottom", columns=2, title=None, labelLimit=150, labelFontSize=10)),
        tooltip=["Cause", "Materials", "Share"]).add_params(pick)
    shares = base.mark_text(radius=70, fontSize=10, color="#1F2937").encode(text="Share:N")
    chart = alt.layer(slices, shares).properties(title=_card_title(title, 12), height=height, background="white", padding=CHART_PADDING)
    st.altair_chart(chart.configure_view(stroke=None), use_container_width=True, theme=None, key=key,
                    on_select=on_pick, selection_mode="pick")


def _card_title(title, size=13):
    return alt.TitleParams(title.split("\n"), fontSize=size, fontWeight="bold", anchor="middle", color="#111827")


def page_end():
    """Empty room at the bottom of the page: the menu can bring the last section's title up under it."""
    st.markdown('<div class="ion-page-end"></div>', unsafe_allow_html=True)


def panel_header(title):
    """Blue sub-header inside a section, like the panels of the ION dashboard."""
    st.markdown(f'<div class="ion-panel">{html.escape(title)}</div>', unsafe_allow_html=True)


def status_tiles(status_counts: dict, total: int, unit="materials"):
    """The four ION material status tiles, counting the materials of the selection."""
    tiles = []
    for status in STATUS_ORDER:
        count = int(status_counts.get(status, 0))
        if status == "Good Part":  # ION shows "Unknown Status (Good Part)" with the good parts
            count += int(status_counts.get("Unknown Status (Good Part)", 0))
        pct = count / total * 100 if total else 0
        light = " light" if status == "Below Safety" else ""
        tiles.append(
            f'<div class="ion-tile{light}" style="background:{STATUS_COLORS[status]}">'
            f'<div class="num">{count}</div><div class="lbl">{status}</div>'
            f'<div class="pct">{pct:.1f}% of the {unit}</div>'
            f'<div class="icon">{STATUS_ICONS[status]}</div></div>'
        )
    st.markdown(f'<div class="ion-tiles">{"".join(tiles)}</div>', unsafe_allow_html=True)


# Heat map of a share compared with its reference (e.g. % of the selection vs % of all comments)
COMPARE_HELP = ("Coloured by comparison with '% of all comments': green = less often than overall, no colour = "
                "about the same, yellow = a bit more often, red = much more often.")
COMPARE_LEGEND = ("Colour of '% of selection' compared with '% of all comments': green = less frequent here · "
                  "no colour = about the same · yellow = a bit more frequent · red = much more frequent.")


def compare_colour(value, reference):
    """Transparent background of a share against its reference: ratio below 0.9 green, 0.9-1.1 none,
    1.1-1.5 yellow, above 1.5 red; the further from the reference, the stronger the colour."""
    if pd.isna(value) or pd.isna(reference) or reference <= 0:
        return ""
    ratio = value / reference
    if ratio < 0.9:
        rgb, strength = "47, 168, 79", min(1, (1 - ratio) / 0.6)
    elif ratio <= 1.1:
        return ""
    elif ratio <= 1.5:
        rgb, strength = "245, 197, 24", (ratio - 1.1) / 0.4
    else:
        rgb, strength = "224, 68, 58", min(1, (ratio - 1.5) / 1.5)
    return f"background-color: rgba({rgb}, {0.18 + 0.27 * strength:.2f})"


def compare_colours(frame, column, reference_column, decimals=1):
    """The frame as a Styler, `column` coloured against `reference_column` (compare_colour). The Styler decides
    how the numbers are written: every other number column gets `decimals` too (with a sign if 'Difference')."""
    colours = [compare_colour(v, r) for v, r in zip(frame[column], frame[reference_column])]
    numbers = frame.select_dtypes("float").columns   # counts stay whole numbers
    return (frame.style.apply(lambda _: colours, subset=[column])
            .format({c: ("{:+.%df}" if c.startswith("Difference") else "{:.%df}") % decimals for c in numbers}))


def signed_colours(frame, column):
    """The frame as a Styler, `column` shaded by its sign: positive red, negative green, zero no colour; the
    larger the value against the column's largest one, the stronger (square root, so small values still show)."""
    largest = frame[column].abs().max()

    def colour(value):
        if pd.isna(value) or value == 0 or not largest:
            return ""
        rgb = "224, 68, 58" if value > 0 else "47, 168, 79"
        return f"background-color: rgba({rgb}, {0.12 + 0.43 * (abs(value) / largest) ** 0.5:.2f})"

    return frame.style.map(colour, subset=[column])


def status_definitions():
    with st.expander("Material Status Definitions"):
        for status in STATUS_ORDER:
            st.markdown(f"- **{status}**: {STATUS_DEFINITIONS[status]}")


# ---------------------------------------------------------------------------
# AI summary: animated loading screen
# ---------------------------------------------------------------------------

LOADING_HTML = """
<div id="ai" class="ai">
  <div class="left">
    <svg viewBox="0 0 200 200" class="brain" aria-hidden="true">
      <circle cx="100" cy="100" r="86" class="ring r1"/>
      <circle cx="100" cy="100" r="62" class="ring r2"/>
      <circle cx="100" cy="100" r="38" class="ring r3"/>
      <g class="orbit o1"><circle cx="100" cy="14" r="5" class="dot"/></g>
      <g class="orbit o2"><circle cx="100" cy="38" r="4" class="dot"/></g>
      <g class="orbit o3"><circle cx="100" cy="62" r="3.5" class="dot"/></g>
      <circle cx="100" cy="100" r="16" class="core"/>
    </svg>
    <div class="counter"><span id="cnt">0</span><small>comments analysed</small></div>
    <div class="sub" id="sub"></div>
  </div>
  <div class="right">
    <div class="title">Analysing planner comments<span class="blink">_</span></div>
    <ul id="steps"></ul>
    <div class="bar"><div id="fill"></div></div>
    <div class="stream" id="stream"></div>
  </div>
</div>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  .ai { display: flex; gap: 22px; padding: 22px; border-radius: 2px; color: #fff; height: 100vh;
        background: radial-gradient(circle at 15% 20%, #2D6BFF 0%, #0B41CD 40%, #021E66 100%); overflow: hidden; }
  .left { width: 210px; flex-shrink: 0; text-align: center; }
  .brain { width: 170px; height: 170px; }
  .ring { fill: none; stroke: rgba(255,255,255,.22); stroke-width: 1.2; stroke-dasharray: 4 6; }
  .r1 { animation: spin 14s linear infinite; transform-origin: 100px 100px; }
  .r2 { animation: spin 9s linear infinite reverse; transform-origin: 100px 100px; }
  .r3 { animation: spin 6s linear infinite; transform-origin: 100px 100px; }
  .orbit { transform-origin: 100px 100px; }
  .o1 { animation: spin 3.2s linear infinite; } .o2 { animation: spin 2.4s linear infinite reverse; }
  .o3 { animation: spin 1.7s linear infinite; }
  .dot { fill: #7FD3FF; filter: drop-shadow(0 0 4px #7FD3FF); }
  .core { fill: #fff; animation: pulse 1.4s ease-in-out infinite; transform-origin: 100px 100px; }
  .counter { font-size: 30px; font-weight: 700; margin-top: 6px; font-variant-numeric: tabular-nums; }
  .counter small { display: block; font-size: 12px; font-weight: 400; opacity: .8; }
  .left .sub { font-size: 11px; opacity: .7; margin-top: 8px; line-height: 1.5; }
  .right { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  .title { font-size: 17px; font-weight: 600; margin-bottom: 10px; letter-spacing: .2px; }
  .blink { animation: blink 1s step-end infinite; }
  ul { list-style: none; margin: 0; padding: 0; font-size: 13px; }
  li { padding: 4px 0; display: flex; gap: 10px; align-items: baseline; opacity: .35; transition: opacity .3s; }
  li.on, li.done { opacity: 1; }
  li .ic { width: 16px; flex-shrink: 0; text-align: center; }
  li.done .ic { color: #6CFFB0; }
  li.on .ic { display: inline-block; animation: spin 1s linear infinite; }
  li.on { color: #BFE6FF; font-weight: 600; }
  li .v { color: #BFE6FF; }
  .bar { height: 5px; background: rgba(255,255,255,.18); border-radius: 3px; margin: 12px 0 10px; overflow: hidden; }
  #fill { height: 100%; width: 0; border-radius: 3px; transition: width .4s ease;
          background: linear-gradient(90deg, #7FD3FF, #6CFFB0, #7FD3FF); background-size: 200% 100%;
          animation: shine 1.6s linear infinite; }
  .stream { flex: 1; overflow: hidden; font-family: Menlo, Consolas, monospace; font-size: 11px;
            opacity: .75; mask-image: linear-gradient(to bottom, transparent, #000 25%, #000 75%, transparent); }
  .stream div { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding: 1px 0;
                animation: fadein .4s ease; }
  .stream b { color: #7FD3FF; font-weight: 400; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes pulse { 0%,100% { transform: scale(.85); opacity: .85; } 50% { transform: scale(1.1); opacity: 1; } }
  @keyframes blink { 50% { opacity: 0; } }
  @keyframes shine { to { background-position: -200% 0; } }
  @keyframes fadein { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; } }
</style>
<script>
  const D = __DATA__;
  const steps = document.getElementById("steps"), fill = document.getElementById("fill");
  const cnt = document.getElementById("cnt"), stream = document.getElementById("stream");
  document.getElementById("sub").textContent = D.sub;
  D.steps.forEach(s => {
    const li = document.createElement("li");
    li.innerHTML = '<span class="ic">○</span><span><span class="t"></span> <span class="v"></span></span>';
    li.querySelector(".t").textContent = s.text;
    li.querySelector(".v").textContent = s.value ? "— " + s.value : "";
    li.querySelector(".v").style.visibility = "hidden";
    steps.appendChild(li);
  });
  const items = [...steps.children];
  let i = 0;
  function next() {
    if (i > 0) { const p = items[i - 1]; p.className = "done"; p.querySelector(".ic").textContent = "✓";
                 p.querySelector(".v").style.visibility = "visible"; }
    if (i >= items.length) return;
    const li = items[i]; li.className = "on"; li.querySelector(".ic").textContent = "◌";
    // the last step (LLM writing) stays active until Python replaces this screen
    fill.style.width = Math.round((i + 1) / (items.length + 1) * 100) + "%";
    if (i < items.length - 1) setTimeout(next, D.stepMs);
    i++;
  }
  next();
  // counter: comments "analysed", reaches the total during the first steps
  const t0 = performance.now(), dur = D.stepMs * 3;
  (function tick(now) {
    const k = Math.min(1, (now - t0) / dur);
    cnt.textContent = Math.round(D.total * (1 - Math.pow(1 - k, 3))).toLocaleString("en-US");
    if (k < 1) requestAnimationFrame(tick);
  })(t0);
  // stream of real (translated) comments scrolling by
  let j = 0;
  setInterval(() => {
    if (!D.comments.length) return;
    const c = D.comments[j++ % D.comments.length];
    const div = document.createElement("div");
    div.innerHTML = "<b>›</b> ";
    div.appendChild(document.createTextNode(c));
    stream.appendChild(div);
    while (stream.children.length > 9) stream.removeChild(stream.firstChild);
  }, 260);
</script>
"""


def ai_loading_screen(steps, total, sub, comments, step_ms=900, height=380):
    """Render the animated loading screen. steps = [{"text": ..., "value": ...}], the last one
    stays active until the caller replaces the screen (e.g. when the LLM has answered)."""
    data = {"steps": steps, "total": int(total), "sub": sub, "comments": comments, "stepMs": step_ms}
    payload = json.dumps(data).replace("</", "<\\/")
    components.html(LOADING_HTML.replace("__DATA__", payload), height=height)


SECTION_NAMES = {"key findings", "what drives the issues", "early warnings", "risks to escalate",
                 "recommended actions", "trend over time", "where the issues come from"}


def _inline(text, max_bold=1):
    """Escape a line and keep **bold**: only the first max_bold bold phrases of a line, the others become plain
    text (the model tends to bold half of the words, which makes nothing stand out)."""
    count = 0

    def bold(match):
        nonlocal count
        count += 1
        return f"<b>{match.group(1)}</b>" if count <= max_bold else match.group(1)
    text = re.sub(r"\*\*(.+?)\*\*", bold, html.escape(text))
    # *single* asterisks (italic in markdown) and lone ones: plain text, never shown as stars
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    return text.replace("*", "")


def summary_to_html(text):
    """Render the LLM answer (### headings, '- ' bullets, **bold**) as HTML sections.
    Robust to small format variations of the model ("**Key findings:**", "1. ...")."""
    out, in_list = [], False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or set(line) <= set("-*_ "):   # empty line or horizontal rule ('---')
            continue
        name = re.sub(r"^#+\s*|\*\*|:$", "", line).strip().rstrip(":").strip()
        if line.startswith("#") or name.lower() in SECTION_NAMES:
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f'<h4>{html.escape(name)}</h4>')
            continue
        bullet = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)", line)
        if bullet:
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{_inline(bullet.group(1))}</li>")
        else:
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<p>{_inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def ai_result_html(text, meta, tools=True):
    """The analysis as a card; tools = links to copy it (formatted) or download it (Word, text), in the browser."""
    links = ('<div class="ai-tools"><a data-action="copy">Copy</a><a data-action="word">Download Word</a>'
             '<a data-action="text">Download text</a></div>') if tools else ""
    return (f'<div class="ai-result"><div class="ai-body">{summary_to_html(text)}</div>'
            f'<div class="meta">{html.escape(meta)}</div>{links}</div>')


def ai_result(text, meta):
    st.markdown(ai_result_html(text, meta), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Column picker of the tables: a floating panel ("popover") over the table
# ---------------------------------------------------------------------------
# The panel opens in the browser (no rerun, so it appears at once) and floats OVER the table instead of
# pushing it down; it starts under the header row of the table, so the effect of every change on the
# columns is visible straight away. Checkboxes show / hide a column; drag and drop changes the order.
COLUMN_PICKER_CSS = """
<style>
/* the panel starts ~1 header row lower than the button: the column titles of the table stay visible;
   compact (scrolls inside) so that it fits under the button, and above the fixed blue bar of the page */
div[data-baseweb="popover"]:has(.col-picker) { z-index: 1000005 !important; }
div[data-testid="stPopoverBody"]:has(.col-picker) {
  margin-top: 38px; width: 460px; max-width: 92vw; max-height: 40vh; overflow-y: auto; padding-top: 6px;
}
div[data-testid="stPopoverBody"]:has(.col-picker) [data-testid="stCheckbox"] { margin-bottom: -10px; }
.col-picker { display: none; }
</style>
"""


def column_picker(key, columns, default=None, order_key=None, visible_key=None, checkbox_prefix=None):
    """Panel '▥ Columns' for one table. columns = all displayed names, in their default order; default = the
    columns shown at first (all if None). Returns the columns to show, in the chosen order. The state lives in
    st.session_state (order_key, visible_key), so it survives reruns and can be saved in a view."""
    try:
        from streamlit_sortables import sort_items
    except ImportError:   # app started outside the project environment (venv): no drag and drop, the rest works
        sort_items = None

    order_key = order_key or f"{key}_order"
    visible_key = visible_key or f"{key}_visible"
    prefix = checkbox_prefix or f"{key}_cb_"
    st.session_state.setdefault(order_key, list(columns))
    st.session_state.setdefault(visible_key, list(default if default is not None else columns))
    # columns that appeared or disappeared since the state was created (e.g. another version of the table)
    order = [c for c in st.session_state[order_key] if c in columns] + [c for c in columns if c not in st.session_state[order_key]]
    st.session_state[order_key] = order
    st.session_state[visible_key] = [c for c in st.session_state[visible_key] if c in columns]

    def toggle(name):
        visible = st.session_state[visible_key]
        if st.session_state[f"{prefix}{name}"]:
            if name not in visible:
                visible.append(name)
        elif name in visible:
            visible.remove(name)

    st.markdown(COLUMN_PICKER_CSS, unsafe_allow_html=True)
    with st.popover("▥ Columns"):
        st.markdown('<div class="col-picker"></div>', unsafe_allow_html=True)
        c_check, c_order = st.columns(2)
        c_check.caption("Show / hide")
        for name in order:
            c_check.checkbox(name, value=name in st.session_state[visible_key], key=f"{prefix}{name}",
                             on_change=toggle, args=(name,))
        with c_order:
            st.caption("Order (drag and drop)")
            shown = [c for c in order if c in st.session_state[visible_key]]
            if sort_items is None:
                st.caption("Not available: start the dashboard with venv/bin/streamlit (package streamlit-sortables).")
            elif shown:
                ordered = sort_items(shown, direction="vertical", key=f"{key}_sort_" + "|".join(shown))
                if ordered and ordered != shown:
                    st.session_state[order_key] = ordered + [c for c in order if c not in ordered]
                    st.rerun()
    result = [c for c in st.session_state[order_key] if c in st.session_state[visible_key]]
    return result or list(columns[:1])


def table_toolbar(key, columns, default=None, **state_keys):
    """Right-aligned '▥ Columns' button above a table; returns the columns to show."""
    _, right = st.columns([6, 1])
    with right:
        return column_picker(key, columns, default, **state_keys)


def copy_table(frame, copy_column, bar_column=None, wide=(), nowrap=()):
    """A table in HTML (st.dataframe cannot hold a button): each cell of copy_column has a 'Copy' button that puts
    its value in the clipboard, in the browser (no rerun). bar_column = a 0-100 score drawn as a bar; wide = long
    texts, wrapped in a wider column; nowrap = columns on one line."""
    def cell(column, value):
        text = "" if pd.isna(value) else html.escape(" ".join(str(value).split()))
        css = "wide" if column in wide else "nowrap" if column in (*nowrap, copy_column, bar_column) else ""
        if column == bar_column and text:
            text = f'<span class="bar"><i style="width:{float(value):.0f}%"></i></span>{float(value):.0f}%'
        elif column == copy_column and text:
            text += f'<a class="ion-copy" href="#" data-copy="{text}">Copy</a>'
        return f'<td class="{css}">{text}</td>'

    head = "".join(f"<th>{html.escape(c)}</th>" for c in frame.columns)
    rows = "".join("<tr>" + "".join(cell(c, v) for c, v in zip(frame.columns, row)) + "</tr>"
                   for row in frame.itertuples(index=False))
    st.markdown(f'<div class="ion-table"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>',
                unsafe_allow_html=True)


def picked_table(key, frame, hidden=(), **dataframe_kwargs):
    """st.dataframe with a '▥ Columns' panel above it: every column of `frame` can be shown, hidden and
    reordered (hidden = columns not shown at first). The choice is kept for the session."""
    columns = list(frame.columns)
    shown = table_toolbar(key, columns, [c for c in columns if c not in hidden])
    dataframe_kwargs.setdefault("use_container_width", True)
    dataframe_kwargs.setdefault("hide_index", True)
    st.dataframe(frame[shown], **dataframe_kwargs)
