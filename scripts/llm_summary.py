"""
LLM Summary (Smart Analysis): a natural-language analysis of the planner comments, written by
the Mistral API (cloud, prompt pseudonymised, used by the dashboard) or by a local model served by
LM Studio (backup: no data leaves the machine). Both are reached through the OpenAI-compatible API.

Architecture: pandas computes the facts (stats dict), the LLM only
interprets them — it never counts or invents figures.

The prompt contains a compact selection of the statistics (no indentation, no
duplicated figures), the most frequent informative comments of the main root
causes, as evidence, the findings of the Insights Lab for the same selection
(context, see insights_pages.smart_context) and the filters, so that the model does
not "discover" them. The model writes a structured analysis in 5 fixed sections
(SECTIONS), which the dashboard renders as headings and bullet points. The answer
is streamed (stream_summary) so the dashboard can show it word by word.
"""



import json
import os
import re
import pandas as pd
from openai import OpenAI

import config  # noqa: F401  (loads the .env file: MISTRAL_API_KEY)

# LM Studio local server — no data leaves the machine
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL") or "http://127.0.0.1:1234/v1"
MODEL_NAME = os.getenv("LM_STUDIO_MODEL") or "mistralai/mistral-7b-instruct-v0.3"
# Cloud: Mistral API (OpenAI-compatible). Same family of models as the local one, but hosted
# by Mistral AI. The key is read from .env or, once deployed, from the Streamlit secrets.
MISTRAL_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "ministral-14b-latest")  # free plan: small/medium often rate-limited (429), large not included (403)
MAX_TOKENS = 1100  # structured analysis of 350-450 words (~650 tokens), with a margin
SECTIONS = ["Key findings", "What drives the issues", "Early warnings",
            "Risks to escalate", "Recommended actions"]
# Statistics that only repeat a filter ("all comments come from the selected vendor") are left out
FILTER_FACETS = {"vendor": "top_vendors_pct_of_comments", "MRP controller": "top_mrp_controllers_pct_of_comments",
                 "purchasing group": "top_purchasing_groups_pct_of_comments"}

def get_client() -> OpenAI:
    """Create an OpenAI-compatible client pointing to the local LM Studio server."""
    return OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio", timeout=120.0)  # key is ignored locally


def is_server_available() -> bool:
    """Check that LM Studio is running (avoids a crash in the dashboard)."""
    try:
        get_client().models.list()
        return True
    except Exception:
        return False



def build_facts(stats: dict, filtered_on=()) -> dict:
    """Compact facts for the prompt. Every key states its unit, so the model does not
    turn a number of comments into a percentage (seen in tests: "+75%" for "+75 comments").
    filtered_on: the filters of the selection; the statistics that only repeat them are left out."""
    facts = {
        "comments_in_selection": stats.get("total_comments"),
        "share_of_all_comments_pct": stats.get("share_of_all_comments_pct"),
        "comments_with_root_cause_pct": stats.get("root_cause_coverage_pct"),
        "top_root_causes_pct_of_comments": stats.get("top_root_causes_pct"),
        "same_root_causes_pct_of_all_comments": (stats.get("top_root_causes_pct_all")
                                                 if stats.get("share_of_all_comments_pct", 100) < 100 else None),
        "top_vendors_pct_of_comments": {k: v["pct"] for k, v in stats.get("top_vendors", {}).items()},
        "top_mrp_controllers_pct_of_comments": {k: v["pct"] for k, v in stats.get("top_mrp_controllers", {}).items()},
        "top_purchasing_groups_pct_of_comments": {k: v["pct"] for k, v in stats.get("top_purchasing_groups", {}).items()},
        "comments_still_open_on_last_snapshot": stats.get("comments_still_open"),
        "comments_still_open_pct": stats.get("comments_still_open_pct"),
        "median_days_a_comment_stays_open": stats.get("median_days_active"),
        "comments_per_week_first_week": stats.get("comments_first_snapshot"),
        "comments_per_week_last_week": stats.get("comments_last_snapshot"),
        "weekly_comment_volume_change_pct": stats.get("trend_pct"),
        "period": f"{stats.get('first_snapshot')} to {stats.get('last_snapshot')}",
        "root_causes_increasing_comments_per_week": stats.get("causes_increasing"),
        "root_causes_decreasing_comments_per_week": stats.get("causes_decreasing"),
        "material_status_number_of_comments": stats.get("material_status_counts"),
        "impact_number_of_comments": stats.get("impact_counts"),
        # nothing entered = impact unknown, not "no impact": stated so that the model does not read it as none
        "impact_not_communicated_number_of_comments": (
            stats["total_comments"] - sum(stats["impact_counts"].values())
            if stats.get("impact_counts") is not None and stats.get("total_comments") else None),
    }
    for name in filtered_on:
        facts.pop(FILTER_FACETS.get(name, ""), None)
    # explicit test: "v not in (None, {}, [])" raises an error when v is a numpy number
    return {k: v for k, v in facts.items()
            if v is not None and not (isinstance(v, (dict, list, str)) and (not v or v == "None to None"))}


def _times(x):
    return f"{x:.1f}× the average ({'more' if x > 1 else 'less'} often than average)"


def describe_context(context: dict, with_sources: bool = False) -> list:
    """The findings of the Insights Lab written as plain sentences, one verified fact per line: the model
    reasons on clear statements instead of reading JSON keys (it misread ratios and counts in tests).
    with_sources=True: (sentence, section of the dashboard it comes from, names in it) for each line."""
    lines = []

    def add(text, section, *names):
        lines.append((text, section, [str(n) for n in names if n]))

    sel = context.get("selection", {})
    if sel.get("filters"):
        add("The user filtered on: " + "; ".join(f"{k} = {', '.join(v)}" for k, v in sel["filters"].items())
            + f" (timeframe: {sel.get('timeframe')}). {sel.get('comments')} of {sel.get('all_comments')} comments.",
            "key-statistics")
    for cause, d in context.get("root_causes_vs_all_comments_pct_points", {}).items():
        add(f"Root cause '{cause}': {d['selection_pct']}% of the comments of the selection vs "
            f"{d['all_comments_pct']}% of all comments ({d['difference_pct_points']:+} percentage points).",
            "key-statistics", cause)
    for g in context.get("recurring_issues", []):
        add(f"Recurring issue \"{g['issue']}\": {g['entries']} entries on {g['materials']} materials, "
            f"{g['vendors']} vendor(s), main root cause '{g['main_root_cause']}'.",
            "frequent-comments", g["issue"], g["main_root_cause"])
    for c in context.get("comments_open_longest", []):
        add(f"Open for {c['days_visible']} days and still visible: \"{c['comment']}\" (root cause "
            f"'{c['root_cause']}', impact {c['impact']}, status {c['material_status']}).",
            "frequent-comments", c["comment"], c["root_cause"])
    rci = context.get("root_cause_to_impact")
    if rci:
        add(f"Of the {rci['items_with_impact_communicated']} items whose impact was communicated, "
            f"{rci['severe_impact_pct_in_selection']}% had a Production or Market impact.", "cause-impact")
        for cause, d in rci.get("per_root_cause", {}).items():
            add(f"Items with root cause '{cause}': {d['severe_pct']}% severe impact "
                f"({d['items_with_impact_communicated']} items), {_times(d['times_average'])}"
                + (", unlikely to be chance." if d["significant"] else ", may be chance."), "cause-impact", cause)
    if "items_impact_not_communicated" in context:
        add(f"{context['items_impact_not_communicated']} of {context.get('items_in_selection')} items have no "
            "impact communicated (unknown, not 'no impact').", "cause-impact")
    weak = context.get("weak_signals", {})
    if weak:
        add(f"Weak signals: of {weak['items_with_first_comment']} items with a first comment, "
            f"{weak['share_of_these_items_with_impact_later_pct']}% later received a Production / Market impact"
            + (f", a median of {weak['median_days_between_first_comment_and_impact']:.0f} days after the first comment."
               if weak.get("median_days_between_first_comment_and_impact") is not None else "."), "watchlist")
        for ph in weak.get("phrases_followed_by_impact", []):
            add(f"First comments containing \"{ph['phrase']}\" ({ph['items_with_phrase']} items): "
                f"{ph['of_these_impact_later_pct']}% later had a severe impact, {_times(ph['times_average'])}"
                + (", unlikely to be chance" if ph["significant"] else "") + f". Example: \"{ph['example']}\"",
                "watchlist", ph["phrase"], ph["example"])
        for label in ("vendor", "root cause", "material status"):
            for value, d in weak.get(f"{label}_followed_by_impact", {}).items():
                add(f"Items whose first comment had {label} '{value}' ({d['items']} items): "
                    f"{d['of_these_impact_later_pct']}% later had a severe impact.", "watchlist", value)
        if weak.get("note"):
            add("Weak signals: " + weak["note"] + ".", "watchlist")
    for w in context.get("watch_list_open_items_most_at_risk", []):
        add(f"Watch list (open, no severe impact yet), risk score {w['risk_score']}/100: \"{w['first_comment']}\" "
            f"— root cause '{w['root_cause']}', status now {w['status_now']}, first commented "
            f"{w['days_since_first_comment']} days ago; raised by: {w['why']}.",
            "watchlist", w["first_comment"], w["root_cause"])
    aso = context.get("actual_stock_out_last_week")
    if aso:
        changes = "; ".join(f"'{c}' +{d['entered']} / -{d['left']}" for c, d in aso["root_causes_of_the_change"].items())
        add(f"Items in Actual Stock Out: {aso['items_previous_week']} the previous week, {aso['items_last_week']} "
            f"last week" + (f" (entered / left by root cause: {changes})." if changes else "."),
            "trends", *aso["root_causes_of_the_change"])
    trend = context.get("root_causes_items_per_week_change_last_4_weeks", {})
    for direction in ("rising", "falling"):
        for cause, v in trend.get(direction, {}).items():
            add(f"Root cause '{cause}' is {direction}: {v:+} items per week on average over the last 4 weeks "
                "compared with the 4 weeks before.", "trends", cause)
    return lines if with_sources else [text for text, _, _ in lines]


# ------------------------------------------------------------------ sources of the analysis (fact checking)
# Each line of the analysis is linked to the facts it rests on: the numbers, names and quoted comments it shares
# with the facts sent to the model. Matched afterwards, so it does not depend on the model citing its sources.
SOURCE_PAGES = {   # section of a fact -> (label of the link, anchor of the section, page (tab) to open in it)
    "key-statistics": ("Smart Analysis → Key statistics used", "ai-summary", "Key statistics used"),
    "frequent-comments": ("Most Frequent Comments", "frequent-comments", None),
    "trends": ("Root Cause Trends", "trends", None),
    "cause-impact": ("Impact by Root Cause", "cause-impact", None),
    "watchlist": ("Watchlist (statistical details)", "watchlist", None),
}
_NUMBER = re.compile(r"(?<![\w.])\d[\d,']*(?:\.\d+)?")


def _numbers(text):
    """The numbers of a text, written the same way whatever the format: '1,221' -> '1221', '18.20' -> '18.2'."""
    found = set()
    for raw in _NUMBER.findall(str(text)):
        try:
            found.add(f"{float(raw.replace(',', '').replace(chr(39), '')):g}")
        except ValueError:
            pass
    return found


def _stat_facts(stats, filtered_on=()):
    """The key statistics sent to the model, one sentence each (dictionaries: one sentence per entry)."""
    facts = []
    for key, value in build_facts(stats, filtered_on).items():
        label = key.replace("_", " ")
        if isinstance(value, dict):
            for name, v in value.items():
                facts.append((f"{label}: {name} = {v}", "key-statistics", [str(name)]))
        else:
            facts.append((f"{label}: {value}", "key-statistics", []))
    return facts


def fact_index(stats: dict, context: dict | None = None) -> list[dict]:
    """Every fact the model received, with where it comes from in the dashboard (for match_sources)."""
    context = context or {}
    facts = _stat_facts(stats, list(context.get("selection", {}).get("filters", {})))
    facts += describe_context(context, with_sources=True)
    comments = {c for group in stats.get("comments_by_root_cause", {}).values() for c in group}
    comments |= set(stats.get("top_comments", []))
    facts += [(f"Planner comment: \"{c}\"", "frequent-comments", [c]) for c in sorted(comments)]
    index = []
    for text, section, names in facts:
        label, anchor, tab = SOURCE_PAGES[section]
        names = names + [n.split("_", 1)[1] for n in names if "_" in n and len(n.split("_", 1)[1]) >= 6]
        index.append({"text": text, "label": label, "anchor": anchor, "tab": tab,
                      "numbers": _numbers(text), "names": [n.lower() for n in names if len(n) >= 4]})
    return index


def match_sources(line: str, facts: list[dict], limit: int = 3) -> list[dict]:
    """The facts a line of the analysis rests on, best first: a quoted comment, then the same numbers, then the
    same names (root cause, vendor, phrase). Small numbers (below 10) only count next to a matching name."""
    low = line.lower()
    quotes = [q.lower() for q in re.findall(r'["“”]([^"“”]{12,})["“”]', line)]
    numbers = _numbers(line)
    scored = []
    for fact in facts:
        names = sum(1 for n in fact["names"] if n in low)
        shared = numbers & fact["numbers"]
        if not names:
            shared = {n for n in shared if float(n) >= 10}
        quoted = any(q in fact["text"].lower() for q in quotes)
        score = 4 * quoted + 2 * len(shared) + names
        if quoted or shared or names:
            scored.append((score, -len(scored), fact))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [fact for score, _, fact in scored[:limit]]


def check_line(line: str, facts: list[dict]) -> tuple[list[dict], list[str]]:
    """The sources of a line (match_sources) and its numbers found in NO fact sent to the model (10 and above;
    smaller ones are often counts the model wrote in words): a figure to check, possibly invented."""
    known = set().union(*(fact["numbers"] for fact in facts)) if facts else set()
    unverified = sorted((n for n in _numbers(line) if float(n) >= 10 and n not in known), key=float)
    return match_sources(line, facts), unverified


def build_prompt(stats: dict, sample_comments: list[str], pseudonymised: bool = False, context: dict | None = None) -> list[dict]:
    """Build the chat message. NOTE: mistral-7b-instruct-v0.3 does not
    support the 'system' role, so instructions are merged into the user message.
    context = the findings of the Insights Lab for the selection (insights_pages.smart_context)."""
    headings = "\n".join(f"### {h}" for h in SECTIONS)
    context = context or {}
    filtered_on = list(context.get("selection", {}).get("filters", {}))
    instructions = (
        "You are a senior supply chain analyst at a pharmaceutical company, preparing the weekly "
        "Supply Risk Meeting. You receive PRECOMPUTED facts about the planner comments of a SELECTION, "
        "and the findings of deeper analyses of the same selection: how it differs from all comments, which "
        "root causes lead to a Production / Market impact, weak signals (first comments after which an impact "
        "came more often than usual), open items most at risk, comments open for a long time, recurring "
        "issues, and what changed recently.\n"
        "Write a SHORT analysis in English: 250 to 320 words in total, using EXACTLY these five headings, in "
        "this order:\n"
        f"{headings}\n"
        "LENGTH BUDGET: 2 bullet points per heading (3 only under 'Recommended actions'), each starting with '- '. "
        "Each bullet is ONE or TWO sentences and at most 35 words. Prefer one strong, specific point over "
        "several general ones; do not repeat a fact in two sections.\n"
        "Example of the expected style (invented content): '- **Late confirmations drive the gaps**: 12 of the 30 "
        "comments mention a missing order confirmation, e.g. \"AB still missing, supplier reminded\"; the "
        "supplier likely lacks capacity.'\n"
        "HOW TO THINK:\n"
        "- The selection is defined by the user's filters (see 'selection'). Everything belongs to it BY "
        "CONSTRUCTION: never write that the issues come from, depend on or are concentrated in the filtered "
        "vendor, MRP controller, purchasing group, material, root cause or impact. That is trivial. Look instead "
        "at what differs INSIDE the selection and compared with all comments.\n"
        "- Reason like an analyst, not like a reader of statistics: connect the root cause, the impact and what "
        "the comments say (what went wrong, what was done, what is still pending), and say why it matters. Use "
        "the figures as evidence for the reasoning, not as the content. Separate facts from interpretation "
        "(\"this suggests\", \"likely\").\n"
        "- 'What drives the issues': the root causes and the recurring issues behind the problems, and which "
        "ones lead to a severe impact more often than average.\n"
        "- 'Early warnings': the weak signals (phrases, context) and the open items most at risk of the watch "
        "list, with why they are at risk; mention comments that stay open for a long time.\n"
        "- 'Risks to escalate': what could turn into a Production or Market impact soon, and recent changes.\n"
        "- 'Recommended actions': concrete next steps for planners and buyers (no new figures).\n"
        "- Quote at least two planner comments word for word in double quotes, copied exactly from the facts.\n"
        "- An impact that is not communicated is UNKNOWN, never 'no impact'.\n"
        "- If the selection holds fewer than 30 comments, say that the conclusions are tentative.\n"
        "DOMAIN NOTES: 'pull-in' = asking the supplier to deliver earlier than confirmed. 'No stockout detected' "
        "or 'no stockout apparent' = the planner checked and the dashboard alert was a FALSE ALARM (often root "
        "cause 'Data error dashboard'). 'Actual Stock Out' is a status computed by the dashboard (the delivery "
        "comes too late), not necessarily a stop of the line. Codes such as px13 are product lines / instrument "
        "series. 'LT' = delivery date or lead time.\n"
        "FORMAT: the five headings exactly as written above, each on its own line starting with '### ', every "
        "heading present once. No horizontal lines ('---'), no other headings. BOLD: exactly one short "
        "**bold** phrase (2 to 5 words) at the start of each bullet, nothing else in bold, never a number. "
        "Stay within 320 words.\n"
        "STRICT RULES: only use the numbers provided, copied exactly — never compute, estimate or invent figures. "
        "A key ending in _pct is a percentage, _pct_points a difference of percentages, times_average a ratio "
        "to the average (write it as '2.5× the average'; ABOVE 1 = more often than average, BELOW 1 = less "
        "often, so a root cause at 0.6× is LESS likely to lead to a severe impact); 'significant': true = unlikely "
        "to be chance. Every other number is a count. Attribute each number to what its key describes. Write nothing before "
        "the first heading.\n"
    )
    if pseudonymised:
        instructions += (
            "Names are replaced by pseudonyms such as VENDOR_1, MRP_2, PGROUP_3 or MATERIAL_4. Use them "
            "exactly as written (same spelling, never translate or describe them) and never try to "
            "guess the real names.\n"
        )
    instructions += "\n"

    evidence = "\n".join(
        f"[{cause}]\n" + "\n".join(f"- {c}" for c in comments)
        for cause, comments in stats.get("comments_by_root_cause", {}).items() if comments
    )
    user_content = (
        instructions
        + "Statistics of the selection (computed by pandas, JSON):\n"
        + json.dumps(build_facts(stats, filtered_on), separators=(",", ":"), default=str)
        + ("\n\nFindings of the deeper analyses of the selection (each line is a verified fact):\n"
           + "\n".join(f"- {line}" for line in describe_context(context)) if context else "")
        + "\n\nMost frequent comments of the main root causes:\n"
        + evidence
        + "\n\nMost frequent comments overall:\n"
        + "\n".join(f"- {c}" for c in sample_comments)
    )

    return [
        {"role": "user", "content": user_content},
    ]


def stream_summary(stats: dict, sample_comments: list[str], context: dict | None = None):
    """Call the local LLM and yield its analysis piece by piece (streaming)."""
    client = get_client()
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=build_prompt(stats, sample_comments, context=context),
        temperature=0.2,   # low temperature: factual, less "creative"
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def get_cloud_client() -> OpenAI:
    """OpenAI-compatible client for the Mistral API (key from MISTRAL_API_KEY)."""
    return OpenAI(base_url=MISTRAL_BASE_URL, api_key=os.getenv("MISTRAL_API_KEY", ""), timeout=120.0)


def is_cloud_configured() -> bool:
    """True when an API key is available (the key itself is never displayed)."""
    return bool(os.getenv("MISTRAL_API_KEY", "").strip())


def stream_summary_cloud(stats: dict, sample_comments: list[str], pseudonymiser=None, context: dict | None = None):
    """Same analysis as stream_summary, written by the Mistral API.

    With a PromptPseudonymiser the request contains aliases (VENDOR_1, MRP_2...) instead of
    names; the aliases of the answer are replaced by the original names before they are yielded."""
    if pseudonymiser is not None:
        stats, sample_comments = pseudonymiser.apply(stats), pseudonymiser.apply(sample_comments)
        context = pseudonymiser.apply(context) if context else context
    messages = build_prompt(stats, sample_comments, pseudonymised=pseudonymiser is not None, context=context)
    stream = get_cloud_client().chat.completions.create(
        model=MISTRAL_MODEL, messages=messages, temperature=0.2, max_tokens=MAX_TOKENS, stream=True)
    pieces = (c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta.content)
    yield from (pseudonymiser.stream_restored(pieces) if pseudonymiser is not None else pieces)


def tidy(text: str) -> str:
    """If the answer was cut by MAX_TOKENS, keep it up to its last complete sentence
    (also when the sentence ends a bullet point, i.e. is followed by a line break)."""
    text = text.strip()
    if text and text[-1] not in ".!?":
        ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", text)]
        if ends:
            text = text[:ends[-1]]
    return text


def generate_summary(stats: dict, sample_comments: list[str], context: dict | None = None) -> str:
    """Call the local LLM and return its analysis as text."""
    text = tidy("".join(stream_summary(stats, sample_comments, context)))
    if not text:
        raise ValueError("No response from the LLM.")
    return text
