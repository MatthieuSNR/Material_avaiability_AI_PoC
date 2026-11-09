"""Cloud summary (Mistral API): nothing identifying may leave the machine, and the answer must
come back with the original names. The API is replaced by a fake client (no network, no key)."""
import os
import sys
import types

import pandas as pd

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)
if "openai" not in sys.modules:  # the test must also run where openai is not installed
    try:
        import openai  # noqa: F401
    except ImportError:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=object)
        sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)

import llm_summary  # noqa: E402
from pseudonymise import PromptPseudonymiser  # noqa: E402

DF = pd.DataFrame({
    "VENDOR_NAME": ["Acme Medical GmbH", "Beta AG", None],
    "MRP_CONTROLLER": ["K12", "K13", "K12"],
    "PURCHASING_GROUP": ["P01", "P02", "P01"],
    "MATERIAL_NUMBER": ["10012345", "10099999", "10012345"],
})
STATS = {
    "total_comments": 3,
    "top_vendors": {"Acme Medical GmbH": {"pct": 60.0}, "Beta AG": {"pct": 40.0}},
    "top_mrp_controllers": {"K12": {"pct": 70.0}},
    "top_purchasing_groups": {"P01": {"pct": 70.0}},
    "top_root_causes_pct": {"Supplier delay": 55.0},
    "comments_by_root_cause": {"Supplier delay": ["Acme Medical liefert später, PO 4500123456, mail max.muster@roche.com"]},
    "top_comments": ["Material 10012345 waiting for Beta AG"],
}
SECRETS = ["Acme", "Beta", "K12", "K13", "P01", "10012345", "4500123456", "max.muster", "roche"]


def fake_client(answer_pieces, sent):
    def create(**kwargs):
        sent.append(kwargs["messages"][0]["content"])
        chunk = lambda t: types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=t))])
        return iter(chunk(t) for t in answer_pieces)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def test_prompt_contains_no_real_name():
    p = PromptPseudonymiser(DF)
    prompt = llm_summary.build_prompt(p.apply(STATS), p.apply(STATS["top_comments"]), pseudonymised=True)[0]["content"]
    for secret in SECRETS:
        assert secret.lower() not in prompt.lower(), secret
    assert "VENDOR_1" in prompt and "MRP_1" in prompt and "PGROUP_1" in prompt


def test_same_vendor_same_alias_in_stats_and_comments():
    p = PromptPseudonymiser(DF)
    safe = p.apply(STATS)
    vendor_alias = next(iter(safe["top_vendors"]))
    assert vendor_alias in safe["comments_by_root_cause"]["Supplier delay"][0]  # "Acme Medical" -> same alias
    assert p.restore_text(vendor_alias) == "Acme Medical GmbH"


def test_fresh_pseudonymiser_restarts_numbering():
    base = PromptPseudonymiser(DF)
    base.apply(STATS)
    assert base.fresh().apply({"Beta AG": 1}) == {"VENDOR_1": 1}


def test_cloud_stream_sends_aliases_and_restores_names(monkeypatch):
    sent = []
    p = PromptPseudonymiser(DF)
    alias = next(iter(p.apply(STATS)["top_vendors"]))  # VENDOR_1
    answer = ["- **", alias[:3], alias[3:], "** is late; ", "MRP_", "1 should follow up with ", "VENDOR_2."]
    monkeypatch.setattr(llm_summary, "get_cloud_client", lambda: fake_client(answer, sent))
    text = "".join(llm_summary.stream_summary_cloud(STATS, STATS["top_comments"], PromptPseudonymiser(DF)))
    assert text == "- **Acme Medical GmbH** is late; K12 should follow up with Beta AG."
    for secret in SECRETS:
        assert secret.lower() not in sent[0].lower(), secret


def test_local_prompt_is_unchanged_without_pseudonyms():
    content = llm_summary.build_prompt(STATS, STATS["top_comments"])[0]["content"]
    assert "Acme Medical GmbH" in content and "pseudonyms" not in content


def test_prompt_knows_the_filters_and_drops_trivial_statistics():
    import llm_summary as llm
    stats = {"total_comments": 40, "top_vendors": {"Acme": {"pct": 100.0}}, "top_mrp_controllers": {"E1": {"pct": 60.0}}}
    context = {"selection": {"filters": {"vendor": ["Acme"]}, "timeframe": "Last 8 weeks", "comments": 40, "all_comments": 900},
               "root_cause_to_impact": {"items_with_impact_communicated": 20, "severe_impact_pct_in_selection": 55.0,
                                        "per_root_cause": {"QM Lot": {"items_with_impact_communicated": 6, "severe_pct": 33.3,
                                                                      "times_average": 0.6, "significant": False}}}}
    prompt = llm.build_prompt(stats, [], context=context)[0]["content"]
    assert "top_vendors_pct_of_comments" not in prompt and "top_mrp_controllers_pct_of_comments" in prompt
    assert "The user filtered on: vendor = Acme" in prompt and "never write that the issues come from" in prompt
    assert "0.6× the average (less often than average)" in prompt          # ratios written out, not left to the model


def test_alias_invented_by_the_model_becomes_a_neutral_label():
    from pseudonymise import PromptPseudonymiser
    p = PromptPseudonymiser(pd.DataFrame({"VENDOR_NAME": ["Acme Medical GmbH"], "PURCHASING_GROUP": ["G1 - Team"]}))
    p.apply({"Acme Medical GmbH": 1, "G1 - Team": 2})
    assert p.restore_text("Ask PGROUP_1 and PGROUP_7 about VENDOR_1") == "Ask G1 - Team and a purchasing group about Acme Medical GmbH"


def test_rendered_analysis_shows_no_asterisks():
    import ui
    html_text = ui.summary_to_html("### Key findings\n- **Vendor A** drives *most* stock-outs, **second** one\n- a lone * star")
    assert "*" not in html_text and html_text.count("<b>") == 1 and "most stock-outs" in html_text


def test_a_line_of_the_analysis_is_linked_to_the_facts_it_rests_on():
    import llm_summary
    stats = {"total_comments": 120, "top_root_causes_pct": {"PL_Shortterm demand": 18.2, "S_Lack of Raw Material": 9.7},
             "comments_by_root_cause": {"PL_Shortterm demand": ["Request for pull-in from supplier"]}, "top_comments": []}
    context = {"root_cause_to_impact": {"items_with_impact_communicated": 1221, "severe_impact_pct_in_selection": 60.0,
                                        "per_root_cause": {}}}
    facts = llm_summary.fact_index(stats, context)
    line = "- **Short-term demand dominates**: Shortterm demand is 18.2% of the comments, e.g. \"Request for pull-in from supplier\"."
    found = llm_summary.match_sources(line, facts)
    assert found[0]["label"] == "Most Frequent Comments" and "18.2" in found[1]["text"]
    assert found[1]["tab"] == "Key statistics used"
    severe = llm_summary.match_sources("60% of the 1,221 items with an impact were severe.", facts)
    assert severe[0]["label"] == "Impact by Root Cause"
    assert llm_summary.match_sources("Escalate to the buyers.", facts) == []


def test_a_figure_the_model_did_not_receive_is_flagged():
    import llm_summary
    facts = llm_summary.fact_index({"total_comments": 120, "top_root_causes_pct": {"PL_Shortterm demand": 18.2}}, {})
    sources, unverified = llm_summary.check_line("Shortterm demand: 264 entries, 18.2% of the comments, 3 vendors.", facts)
    assert unverified == ["264"] and sources
