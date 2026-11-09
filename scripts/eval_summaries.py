"""
Faithfulness evaluation of the Smart Analysis (thesis, Sections 3.7 and 5.4).

Generates the Smart Analysis for a fixed set of selections, exactly as the deployed dashboard does (same statistics,
same findings of the Insights Lab, same pseudonymised prompt, Mistral API, temperature 0.2), and saves for each
summary the PSEUDONYMISED prompt and answer (aliases such as VENDOR_1 are not restored), so that the summaries can be
checked against the facts they were given without exposing real names.

The selections are chosen by rank (e.g. the 1st, 3rd, 8th... vendor by number of comments), not by name.
Some are deliberately small (< 30 comments) to test the rule "conclusions are tentative".
Three selections are generated three times to measure the variation between generations.

    venv/bin/python scripts/eval_summaries.py            # ~26 API calls, a few minutes
    venv/bin/python scripts/eval_summaries.py --dry-run  # prompts only, no API call

Output: outputs/eval/smart_analysis_eval.json (pseudonymised; no real supplier, material or person names).
"""
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.chdir(ROOT)

import insights  # noqa: E402
import insights_pages  # noqa: E402
import llm_summary  # noqa: E402
from pseudonymise import PromptPseudonymiser  # noqa: E402
from stats_engine import compute_stats  # noqa: E402

DRY = "--dry-run" in sys.argv
OUT = os.path.join(ROOT, "outputs", "eval", "smart_analysis_eval.json")


def impact_level(value):  # as in app.py
    if pd.isna(value):
        return "Not communicated"
    label = str(value).strip().lower()
    for prefix, level in (("market", "Market"), ("production", "Production"),
                          ("logistic", "Logistics"), ("none", "None (no impact)")):
        if label.startswith(prefix):
            return level
    return str(value).strip()


def load():
    df = pd.read_csv("data/cleaned_comments.csv")
    df["MATERIAL_LABEL"] = df["MATERIAL_DESC"].fillna("No description") + " — " + df["MATERIAL_NUMBER"].astype("int64").astype(str)
    df["VENDOR_LABEL"] = (df["VENDOR_NAME"].fillna("Unknown vendor") + " — "
                          + df["VENDOR_ACCOUNT_NUMBER"].astype("Int64").astype(str).replace("<NA>", "no vendor"))
    df["MRP_CONTROLLER"] = df["MRP_CONTROLLER"].fillna("Unknown")
    df["PURCHASING_GROUP"] = df["PURCHASING_GROUP"].fillna("Unknown")
    df["Impact_Level"] = df["Impact"].map(impact_level)
    df["Stakeholder"] = df["Root_Cause"].map(insights.stakeholder).where(df["Root_Cause"].notna())
    for col in ["Snapshot_Date", "First_Seen", "Last_Seen"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["MATERIAL_STATUS"] = insights_pages.status_of_materials(df, insights_pages.load_tracking())
    return df


# (type, column of the filter, ranks by number of comments)
PLAN = [("vendor", "VENDOR_LABEL", [1, 3, 8, 20, 40]),
        ("material", "MATERIAL_LABEL", [1, 5, 20, 60]),
        ("mrp_controller", "MRP_CONTROLLER", [1, 4, 10, 25, 50]),
        ("root_cause", "Root_Cause", [1, 2, 3, 4]),
        ("purchasing_group", "PURCHASING_GROUP", [2, 8])]
REPEATS = {"vendor-1": 3, "mrp_controller-4": 3, "material-5": 3}   # generations for the variation check
EMPTY = {k: [] for k in ["MRP_CONTROLLER", "PURCHASING_GROUP", "VENDOR_LABEL", "MATERIAL_LABEL", "Stakeholder",
                         "Root_Cause", "Impact_Level", "Material_Status"]}


def call(messages):
    client = llm_summary.get_cloud_client()
    for attempt in range(6):
        try:
            t0 = time.time()
            r = client.chat.completions.create(model=llm_summary.MISTRAL_MODEL, messages=messages,
                                               temperature=0.2, max_tokens=llm_summary.MAX_TOKENS)
            return r.choices[0].message.content, round(time.time() - t0, 1), getattr(r, "model", None)
        except Exception as e:  # rate limit of the free plan: wait and retry
            print("  retry after error:", type(e).__name__, str(e)[:120])
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("API not reachable")


def main():
    df = load()
    template = PromptPseudonymiser(df)
    results = []
    for kind, column, ranks in PLAN:
        counts = df[column].dropna().value_counts() if column != "Root_Cause" else df["Root_Cause"].dropna().value_counts()
        for rank in ranks:
            if rank > len(counts):
                continue
            value = counts.index[rank - 1]
            sid = f"{kind}-{rank}"
            selections = dict(EMPTY, **{column: [value]})
            filtered = df[df[column] == value]
            stats = compute_stats(filtered, df)
            context = insights_pages.smart_context(df, filtered, selections, None, "All snapshots")
            p = template.fresh()
            stats_p, comments_p = p.apply(stats), p.apply(stats["top_comments"])
            context_p = p.apply(context)
            messages = llm_summary.build_prompt(stats_p, comments_p, pseudonymised=True, context=context_p)
            entry = {"id": sid, "type": kind, "rank": rank, "comments": int(len(filtered)),
                     "prompt": messages[0]["content"], "generations": []}
            for g in range(REPEATS.get(sid, 1)):
                if DRY:
                    entry["generations"].append({"answer": None, "seconds": None})
                    continue
                answer, seconds, model = call(messages)
                entry["generations"].append({"answer": answer, "seconds": seconds, "model": model})
                print(f"{sid} ({len(filtered)} comments) generation {g + 1}: {seconds} s")
                time.sleep(2)
            results.append(entry)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    meta = {"created": datetime.now().isoformat(timespec="minutes"), "model": llm_summary.MISTRAL_MODEL,
            "temperature": 0.2, "max_tokens": llm_summary.MAX_TOKENS, "dry_run": DRY}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summaries": results}, f, ensure_ascii=False, indent=1)
    print("saved", OUT, len(results), "selections")


if __name__ == "__main__":
    main()
