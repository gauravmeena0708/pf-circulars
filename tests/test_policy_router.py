from datetime import date

import pytest

from policy_router import is_jd_question, route_retrieval


@pytest.mark.parametrize("query", ["joint declaration profile update", "JD documents", "member profile correction", "संयुक्त घोषणा"])
def test_jd_route_detection(query):
    assert is_jd_question(query)


@pytest.mark.parametrize("query", ["EPS pension eligibility", "joint declaration for higher pension", "joint declaration under para 26(6)", "joint option for pension"])
def test_unrelated_joint_declarations_are_excluded(query):
    assert not is_jd_question(query)


def test_other_policies_preserve_original_search_results():
    original = [{"text": "EPS pension", "metadata": {}}]
    result = route_retrieval("EPS eligibility", "2025-01-16", lambda: original)
    assert result["chunks"] is original
    assert result["policy_context"] is None


def test_jd_uses_parent_pages_and_includes_relationship_evidence():
    def unexpected_hybrid():
        raise AssertionError("No gap requiring fallback at this date")
    result = route_retrieval("joint declaration closed establishment", "2025-01-16", unexpected_hybrid)
    assert result["route"] == "jd_policy"
    assert any("PRO" in c["text"] for c in result["chunks"])
    assert all("supplementary" not in c["metadata"]["evidence_status"] for c in result["chunks"])
    assert any(c["metadata"]["page_number"] == 2 for c in result["chunks"])
    assert result["policy_context"]["conclusion"] == "complete_applicability_not_established"
    assert all(c["metadata"]["date"] <= "2025-01-16" for c in result["chunks"])


def test_current_jd_fallback_filters_future_unknown_and_wrong_policy():
    def hybrid():
        return [
            {"text": "Joint Declaration transgender identity certificate", "metadata": {"date": "26/12/2025"}},
            {"text": "Joint Declaration future update", "metadata": {"date": "2027-01-01"}},
            {"text": "Joint Declaration undated update", "metadata": {}},
            {"text": "Joint declaration for higher pension", "metadata": {"date": "2025-01-01"}},
        ]
    result = route_retrieval("JD transgender certificate", "2026-10-28", hybrid)
    supporting = [c for c in result["chunks"] if "supplementary" in c["metadata"]["evidence_status"]]
    assert len(supporting) == 1
    assert "transgender" in supporting[0]["text"]
    assert supporting[0]["metadata"]["page_number"] is None
    assert supporting[0]["metadata"]["page_provenance"] == "document_only"
    assert result["policy_context"]["fallback_reason"]


def test_store_failure_keeps_fallback_unverified():
    def missing():
        raise FileNotFoundError()
    result = route_retrieval("JD name correction", "2025-01-16", lambda: [
        {"text": "Joint Declaration name correction", "metadata": {"date": "16/01/2025"}}
    ], history_factory=missing)
    assert result["policy_context"]["documents"] == []
    assert "could not be loaded" in result["policy_context"]["fallback_reason"]
    assert "unverified" in result["chunks"][0]["metadata"]["evidence_status"]


def test_hybrid_failure_does_not_drop_curated_pages():
    def failed():
        raise RuntimeError("backend offline")
    result = route_retrieval("joint declaration Aadhaar", "2026-10-28", failed)
    assert result["chunks"]
    assert "unavailable" in result["policy_context"]["fallback_reason"]


def test_explicit_modes_support_followups_and_original_pipeline():
    assert route_retrieval("what documents?", "2025-01-16", lambda: [], mode="jd")["route"] == "jd_policy"
    assert route_retrieval("JD documents", "2025-01-16", lambda: [], mode="hybrid")["route"] == "hybrid"


def test_no_early_sources_means_no_invented_answer_context():
    result = route_retrieval("JD documents", "2020-01-01", lambda: [
        {"text": "Joint Declaration", "metadata": {"date": "2025-01-01"}}
    ])
    assert result["chunks"] == []
    assert result["policy_context"]["conclusion"] == "complete_applicability_not_established"
