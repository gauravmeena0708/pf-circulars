import json
from pathlib import Path

import pytest

from policy_history import PolicyHistory
from policy_registry import mode_for_label, registered_policies, select_policy
from policy_router import route_retrieval


@pytest.mark.parametrize("query", [
    "How do I correct wrong Aadhaar in UAN?", "Aadhaar name mismatch", "Aadhar date of birth proof",
    "Member profile correction", "आधार सुधार",
])
def test_aadhaar_scope_detected(query):
    assert select_policy(query).id == "aadhaar"


@pytest.mark.parametrize("query", [
    "Mandatory Aadhaar seeding for ECR filing", "EPS pension eligibility",
    "UIDAI address update", "How can I correct Aadhaar address?",
])
def test_unregistered_scopes_keep_hybrid_fallback(query):
    original = [{"text": "original search result", "metadata": {}}]
    result = route_retrieval(query, "2026-10-28", lambda: original)
    assert result["route"] == "hybrid"
    assert result["chunks"] is original
    assert result["policy_context"] is None


def test_registered_paths_and_explicit_followups():
    policies = registered_policies()
    assert {p.id for p in policies} == {"jd", "aadhaar", "transfers", "higher_pension", "cpps", "advances"}
    for policy in policies:
        assert policy.path.is_file()
        PolicyHistory(policy.path)
        assert mode_for_label(policy.label) == policy.id
    result = route_retrieval("What documents?", "2025-08-13", lambda: [], mode="aadhaar")
    assert result["route"] == "aadhaar_policy"
    assert result["policy_context"]["policy_label"] == "Member profile / Aadhaar"


def test_each_policy_owns_its_timeline_and_shares_sources_without_copying():
    jd = select_policy("JD documents", "jd")
    aadhaar = select_policy("wrong Aadhaar", "aadhaar")
    assert "aadhaar-2016-11-21" not in PolicyHistory(jd.path).documents
    history = PolicyHistory(aadhaar.path)
    assert "aadhaar-2016-11-21" in history.documents
    assert history.documents["jd-2024-07-31"]["version_label"] == "SOP version 3.0"
    assert history.page("jd-2025-08-13", 1)["text"] == PolicyHistory(jd.path).page("jd-2025-08-13", 1)["text"]


def test_campaign_target_not_promoted_to_commencement_and_missing_original_visible():
    history = PolicyHistory(select_policy("", "aadhaar").path)
    snapshot = history.snapshot("2016-11-21")
    docs = {d["id"]: d for d in snapshot["documents"]}
    assert docs["aadhaar-2016-11-21"]["effective_from"] is None
    assert docs["aadhaar-2014-07-25"]["status"] == "source_unverified"
    assert not snapshot["relationships"]


def test_source_checked_reference_does_not_verify_original_or_invent_supersession():
    history = PolicyHistory(select_policy("", "aadhaar").path)
    report = history.snapshot("2025-08-13")
    ref = next(r for r in report["relationships"] if r["id"] == "aadhaar-2025-ref-2014")
    assert ref["type"] == "references"
    assert ref["verification"] == "source_checked"
    assert next(d for d in report["documents"] if d["id"] == ref["target"])["status"] == "source_unverified"
    assert not any(r["target"] == "aadhaar-2020-04-03" for r in report["relationships"])


def test_new_policy_registration_does_not_require_router_changes(tmp_path):
    entry = {
        "id": "example-policy", "label": "Example policy", "scope": "synthetic test only",
        "path": "example/policy.json", "include": "synthetic test", "exclude": "", "route": "example_policy",
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"policies": [entry]}))
    policies = registered_policies(path)
    assert select_policy("synthetic test", policies=policies).path == tmp_path / "example/policy.json"
    path.write_text(json.dumps({"policies": [entry, entry]}))
    with pytest.raises(ValueError):
        registered_policies(path)


def test_aadhaar_store_failure_uses_its_scope_and_unverified_fallback():
    def missing():
        raise FileNotFoundError()
    result = route_retrieval("correct Aadhaar in UAN", "2025-08-13", lambda: [
        {"text": "Wrong Aadhaar correction in UAN", "metadata": {"date": "13/08/2025"}},
        {"text": "JD transgender certificate", "metadata": {"date": "13/08/2025"}},
    ], history_factory=missing)
    assert result["route"] == "aadhaar_policy"
    assert result["policy_context"]["policy_id"] == "aadhaar"
    assert len(result["chunks"]) == 1
    assert "unverified" in result["chunks"][0]["metadata"]["evidence_status"]
