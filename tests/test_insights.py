"""Insights Lab (insights.py, tracking.py): grouping of similar comments, tracking table, status change,
root cause x impact, weak signals. Synthetic data only, no network."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import insights as ins  # noqa: E402
import tracking as tr  # noqa: E402


def entries(rows):
    return pd.DataFrame([{"Entry": text, "Root_Cause": cause, "MATERIAL_NUMBER": i, "VENDOR_NAME": "V", "Days_Active": 1}
                         for i, (text, cause, n) in enumerate(rows) for _ in range(n)])


def test_grouping_case_containment_similarity_and_root_cause():
    groups, variants = ins.group_comments(entries([
        ("px13", "PL_Internal Errors", 5), ("PX13", "PL_Shortterm demand", 3),          # case only: merged anyway
        ("px13 - No stockout detected", "PL_Internal Errors", 4),                      # contained, same cause
        ("no stockout", "Data error dashboard", 6), ("No stockout detected", "Data error dashboard", 2),
        ("px13 Wave 2", "PE_Design/Material Changes (ECR)", 2),                        # contained, other cause
        ("Request for Pull-in", "PL_Shortterm demand", 5), ("Request to pull in the LT", "PL_Shortterm demand", 3),
    ]))
    group_of = {w.lower(): g for w, g in zip(variants["Wording"], variants["Group"])}
    assert group_of["px13"] == group_of["px13 - no stockout detected"]
    assert group_of["no stockout"] == group_of["no stockout detected"]
    assert group_of["px13 - no stockout detected"] != group_of["no stockout"]       # different root causes
    assert group_of["px13 wave 2"] != group_of["px13"]
    assert group_of["request for pull-in"] == group_of["request to pull in the lt"]
    px13 = groups[groups["Comment"].str.lower() == "px13"].iloc[0]
    assert px13["Occurrences"] == 12 and px13["Main_root_cause"] == "PL_Internal Errors"


def test_opposite_meanings_are_not_grouped():
    groups, variants = ins.group_comments(entries([
        ("5 complaints --> Stock coverage is critical", "S_Quality", 6),
        ("QM lot closed --> Stock coverage non-critical", None, 1),
        ("Stock coverage is critical", "S_Quality", 2),
        ("px13", "D", 3), ("px13 - No stockout detected", "D", 1)]))
    group_of = dict(zip(variants["Wording"], variants["Group"]))
    assert group_of["QM lot closed --> Stock coverage non-critical"] != group_of["5 complaints --> Stock coverage is critical"]
    assert group_of["Stock coverage is critical"] == group_of["5 complaints --> Stock coverage is critical"]
    assert group_of["px13"] == group_of["px13 - No stockout detected"]          # nothing negated in common


def test_journal_problem_and_latest_entries():
    df = pd.DataFrame({"Comment_EN": ["solved ¦▸ supplier asked ¦▸ delay announced"], "Root_Cause": ["X"]})
    assert list(ins.comment_entries(df, "problem")["Entry"]) == ["delay announced"]   # written first, shown last
    assert list(ins.comment_entries(df, "latest")["Entry"]) == ["solved"]
    assert len(ins.comment_entries(df)) == 3


def test_stakeholder_prefix():
    assert ins.stakeholder("PL_Shortterm demand") == "Planning"
    assert ins.stakeholder("S_Lack of Raw Material") == "Supplier"
    assert ins.stakeholder("QM Lot") == "Other"


def merged_snapshots():
    rows = []
    for day, status, cause, impact, comment in [
        ("2026-09-01", "Actual Stock Out", None, None, "abc12, 2026-09-01: Request for pull-in"),
        ("2026-09-08", "Actual Stock Out", "PL_Shortterm demand", "Production (Stop&Go/...)", None),
    ]:
        for line in range(2):   # two order lines of the same item: one row per item and day
            rows.append({"Snapshot_Date": day, "MATERIAL_NUMBER": 1, "VENDOR_ACCOUNT_NUMBER": 10, "VENDOR_NAME": "ACME",
                         "MATERIAL_DESC": "Part", "MRP_CONTROLLER": "E1", "PURCHASING_GROUP": "G1", "PLANT": 1,
                         "MATERIAL_STATUS": status if line else "Good Part", "Root_Cause": cause,
                         "Impact": impact if line else None, "Comment": comment})
    rows.append({"Snapshot_Date": "2026-09-08", "MATERIAL_NUMBER": 2, "VENDOR_ACCOUNT_NUMBER": 20, "VENDOR_NAME": "BOLT",
                 "MATERIAL_DESC": "Screw", "MRP_CONTROLLER": "E1", "PURCHASING_GROUP": "G1", "PLANT": 1,
                 "MATERIAL_STATUS": "Actual Stock Out", "Root_Cause": "S_Lack of Raw Material", "Impact": "None",
                 "Comment": None})
    return pd.DataFrame(rows)


def test_tracking_one_row_per_item_and_day_without_logins():
    t = tr.build_tracking(merged_snapshots())
    assert len(t) == 3
    day2 = t[(t["MATERIAL_NUMBER"] == 1) & (t["Snapshot_Date"] == "2026-09-08")].iloc[0]
    assert day2["MATERIAL_STATUS"] == "Actual Stock Out" and day2["Impact_Level"] == "Production" and day2["N_Lines"] == 2
    assert t["Comment"].dropna().tolist() == ["Request for pull-in"]                   # login removed
    assert tr.impact_level("None (no impact)") == "None (no impact)" and tr.impact_level(None) is None


def test_status_change_and_weak_signal_items():
    t = tr.build_tracking(merged_snapshots())
    summary, entered, n_a, n_b = ins.status_change(t, pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-08"))
    assert (n_a, n_b) == (1, 2) and entered["Root_Cause"].tolist() == ["S_Lack of Raw Material"]
    items = ins.weak_signal_items(t, {"Request for pull-in": "Request for pull-in"})
    assert len(items) == 1 and items["Severe_later"].iloc[0] and items["Days_before_impact"].iloc[0] == 7


def test_root_cause_impact_table():
    t = pd.DataFrame({"MATERIAL_NUMBER": range(40), "VENDOR_ACCOUNT_NUMBER": 1,
                      "Root_Cause": ["A"] * 20 + ["B"] * 20,
                      "Impact_Level": ["Production"] * 16 + ["None (no impact)"] * 4 + ["Production"] * 4 + ["None (no impact)"] * 16})
    table, per_cause, test = ins.root_cause_impact(t)
    assert test["p"] < 0.05 and test["n_items"] == 40
    a = per_cause.set_index("Root cause").loc["A"]
    assert a["% severe"] == 80.0 and a["Lift"] == 1.6


def test_label_of_a_group_whose_dates_vary():
    assert ins.neutral_label(["25 estimated time of arrival 9/7-9/9", "40 estimated time of arrival 7/28"]) \
        == "[n] estimated time of arrival [date]"
    assert ins.neutral_label(["px13", "px13 Pilot"]) == "px13"                    # same numbers: unchanged


def test_grouping_by_meaning_uses_the_given_embeddings():
    vectors = {"no stockout detected": [1, 0], "no stockout apparent": [0.95, 0.31], "request for pull-in": [0, 1]}
    fake = lambda texts: [vectors[t.lower()] for t in texts]               # no model download in the tests
    groups, variants = ins.group_comments(entries([("No stockout detected", "D", 3), ("No stockout apparent", "D", 1),
                                                   ("Request for pull-in", "D", 2)]), method="meaning", embeddings=fake)
    group_of = dict(zip(variants["Wording"].str.lower(), variants["Group"]))
    assert group_of["no stockout detected"] == group_of["no stockout apparent"] != group_of["request for pull-in"]
    assert "similar meaning" in variants.set_index("Wording").loc["No stockout apparent", "Reason"]


def weak_items(n=60):
    """Synthetic first comments: 'escalated to procurement' at vendor A is followed by an impact."""
    rows = []
    for i in range(n):
        signal = i % 3 == 0
        rows.append({"Text": ("Supplier escalated to procurement, partial deliveries" if signal else "Delivery on time, no issue"),
                     "Root_Cause": "S_Lack of Raw Material" if signal else "PL_Shortterm demand",
                     "VENDOR_NAME": "A" if signal else "B", "MATERIAL_STATUS": "Actual Stock Out",
                     "Severe_later": signal and i % 2 == 0})
    return pd.DataFrame(rows)


def test_signal_phrases_have_two_words_or_more_and_context():
    phrases = ins.signal_phrases(weak_items())
    assert len(phrases) and all(len(p.split()) >= 2 for p in phrases["Phrase"])
    top = phrases.iloc[0]
    assert top["Main root cause"].startswith("S_Lack of Raw Material") and top["Main vendors"].startswith("A")
    assert top["× average"] > 1


def test_signal_context_and_model():
    items = weak_items(90)
    vendors = ins.signal_context(items, "VENDOR_NAME")
    assert vendors.iloc[0]["VENDOR_NAME"] == "A" and vendors.iloc[0]["× average"] > 1
    result = ins.signal_model(items)
    assert list(result["scores"]["Model"].str.split().str[0]) == ["Root", "Comment", "Comment"]
    assert result["risk"].notna().all() and len(result["reasons"]) == len(items)
    assert not result["weights"]["Element"].str.contains("VENDOR_NAME_").any()      # readable names


def test_root_cause_status_takes_the_worst_status():
    t = pd.DataFrame({"MATERIAL_NUMBER": [1, 1, 2], "VENDOR_ACCOUNT_NUMBER": [1, 1, 1], "Root_Cause": ["X", "X", "X"],
                      "MATERIAL_STATUS": ["Below Safety", "Actual Stock Out", "Unknown Status (Good Part)"]})
    table = ins.root_cause_status(t)
    assert table.loc["X", "Actual Stock Out"] == 1 and table.loc["X", "Good Part"] == 1


def test_embeddings_are_cached_and_the_model_loaded_only_for_new_texts(tmp_path):
    calls = []

    class FakeModel:
        def encode(self, texts, **_):
            calls.append(list(texts))
            return [[float(len(t)), 1.0] for t in texts]

    path = tmp_path / "cache.npz"
    first = ins.embed(["a", "bb"], load_model=FakeModel, cache_path=str(path))
    again = ins.embed(["bb", "a"], load_model=FakeModel, cache_path=str(path))   # all cached: model not used
    ins.embed(["a", "ccc"], load_model=FakeModel, cache_path=str(path))          # only the new text is encoded
    assert calls == [["a", "bb"], ["ccc"]]
    assert (again[1] == first[0]).all()


def test_lab_filters_vendor_material_and_impact():
    import insights_pages as pages
    t = pd.DataFrame({"Snapshot_Date": pd.to_datetime(["2026-09-01"] * 4), "MATERIAL_NUMBER": [1, 1, 2, 3],
                      "VENDOR_ACCOUNT_NUMBER": [10, 10, 20, 30], "MRP_CONTROLLER": "E1", "PURCHASING_GROUP": "G1",
                      "Root_Cause": ["X", "X", "Y", None], "Impact_Level": ["Production", None, "None (no impact)", None],
                      "VENDOR_LABEL": ["A — 10", "A — 10", "B — 20", "C — 30"],
                      "MATERIAL_LABEL": ["m — 1", "m — 1", "n — 2", "o — 3"]})
    pick = lambda **sel: sorted(pages.filter_tracking(t, sel)["MATERIAL_NUMBER"].unique())
    assert pick(VENDOR_LABEL=["other description — 20"]) == [2]          # matched on the number
    assert pick(MATERIAL_LABEL=["m (old description) — 1"]) == [1]
    assert pick(Impact_Level=["Production"]) == [1]                       # all rows of the item kept
    assert pick(Impact_Level=["Not communicated"]) == [3]                 # nothing entered at all
    assert pick(Impact_Level=["None (no impact)"]) == [2]


def test_merge_stakeholders_keeps_one_name_per_root_cause():
    causes = pd.Series(["PL_Internal Errors", "PL_Internal Errors", "P_Internal Error", "S_Lack of Raw Material",
                        "QM Lot", None])
    merged = ins.merge_stakeholders(causes)
    assert merged.tolist()[:5] == ["Internal Errors"] * 3 + ["Lack of Raw Material", "QM Lot"]
    assert pd.isna(merged.iloc[5])


def test_chart_is_drawn_once_for_the_same_data():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import ui
    calls = []

    def draw(frame, title):
        calls.append(title)
        fig, ax = plt.subplots()
        ax.bar(frame.index, frame["n"])
        ax.set_title(title)
        return fig

    frame = pd.DataFrame({"n": [3, 1]}, index=["a", "b"])
    ui.chart(draw, frame, "Test chart")
    ui.chart(draw, frame.copy(), "Test chart")                      # same content: picture reused
    ui.chart(draw, frame.assign(n=[3, 2]), "Test chart")            # other data: drawn again
    assert calls == ["Test chart", "Test chart"]


def test_last_snapshot_keeps_all_lines_and_a_material_counts_once():
    merged = pd.DataFrame({
        "Snapshot_Date": ["2026-09-19", "2026-09-20", "2026-09-20", "2026-09-20"],
        "MATERIAL_NUMBER": [1, 1, 1, 2], "VENDOR_ACCOUNT_NUMBER": [10, 10, 10, 20],
        "MATERIAL_STATUS": ["Good Part", "Good Part", "Actual Stock Out", "Below Safety"],
        "Root_Cause": [None, "S_Planning Issue", None, None], "Impact": [None, "Production (Stop&Go)", None, None],
        "Comment": [None, "text", None, None], "VENDOR_NAME": ["V", "V", "V", "W"], "MATERIAL_DESC": ["A", "A", "A", "B"],
        "MRP_CONTROLLER": ["C"] * 4, "PURCHASING_GROUP": ["P"] * 4, "PLANT": [4700, 4700, 4700, 4419]})
    current = tr.build_current(merged)
    assert current["PLANT"].tolist() == [4700, 4700, 4419]   # the plant filter also works on the last snapshot
    assert len(current) == 3 and "Comment" not in current and current["Commented"].sum() == 1   # uncommented lines stay
    assert current["Impact_Level"].dropna().tolist() == ["Production"]
    assert tr.material_status(current).to_dict() == {1: "Actual Stock Out", 2: "Below Safety"}


def test_tracking_status_is_the_worst_of_all_the_lines_also_uncommented():
    lines = pd.DataFrame({
        "Snapshot_Date": ["2026-09-20"] * 2, "MATERIAL_NUMBER": [1, 1], "VENDOR_ACCOUNT_NUMBER": [10, 10],
        "MATERIAL_STATUS": ["Good Part", "Below Safety"], "Root_Cause": ["S_Planning Issue", None], "Impact": [None, None],
        "Comment": ["text", None], "VENDOR_NAME": ["V", "V"], "MATERIAL_DESC": ["A", "A"], "MRP_CONTROLLER": ["C", "C"],
        "PURCHASING_GROUP": ["P", "P"], "PLANT": [1, 1]})
    row = tr.build_tracking(lines).iloc[0]
    assert row["MATERIAL_STATUS"] == "Below Safety" and row["N_Lines"] == 1


def test_root_cause_impact_works_with_one_root_cause_selected():
    items = pd.DataFrame({"Snapshot_Date": ["2026-09-20"] * 2, "MATERIAL_NUMBER": [1, 2], "VENDOR_ACCOUNT_NUMBER": [10, 10],
                          "Root_Cause": ["S_OoS Delay"] * 2, "Impact_Level": ["Production"] * 2, "MATERIAL_STATUS": ["Actual Stock Out"] * 2})
    table, per_cause, test = ins.root_cause_impact(items)
    assert test["chi2"] is None and per_cause.iloc[0]["% severe"] == 100
