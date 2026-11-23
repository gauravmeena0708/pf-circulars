import copy
import json

import pytest

from citation_provenance import page_label
from policy_history import DEFAULT_POLICY, PolicyHistory


@pytest.fixture
def history():
    return PolicyHistory()


def test_historical_cutoff_has_no_future_documents_or_relationships(history):
    snapshot = history.snapshot("2024-01-15")
    assert [d["id"] for d in snapshot["documents"]] == ["jd-2023-08-22"]
    assert snapshot["relationships"] == []


def test_partial_modification_does_not_become_whole_document_supersession(history):
    snapshot = history.snapshot("2024-03-11")
    assert {r["type"] for r in snapshot["relationships"]} == {"partially_modifies"}
    assert all(r["status"] == "relationship_verified_timing_unresolved" for r in snapshot["relationships"])
    assert next(d for d in snapshot["documents"] if d["id"] == "jd-2023-08-22")["status"] == "effective_date_established"


def test_effective_dates_require_source_evidence_and_uncertain_targets_stay_uncertain(history):
    snapshot = history.snapshot("2024-07-31")
    relations = {r["source"]: r for r in snapshot["relationships"]}
    assert relations["jd-2024-03-26"]["status"] == "effective_relationship"
    assert relations["jd-2024-07-31"]["status"] == "relationship_unverified"


def test_unknown_commencement_is_not_substituted_with_issue_date(history):
    snapshot = history.snapshot("2025-01-16")
    document = next(d for d in snapshot["documents"] if d["id"] == "jd-2025-01-16")
    assert document["effective_from"] is None
    assert document["status"] == "issued_effective_date_unverified"
    assert snapshot["conclusion"] == "evidence_inventory_only"


def test_unfetched_document_never_acquires_page_evidence(history):
    report = history.search("transgender identity certificate", "2026-10-28")
    assert "jd-2025-12-26" in report["coverage"]["documents_without_page_text"]
    assert not any(e["document_id"] == "jd-2025-12-26" for e in report["evidence"])


def test_retrieved_chunk_expands_to_exact_page_and_keeps_offsets(history):
    report = history.search("closed establishments deceased claimant", "2025-01-16")
    evidence = next(e for e in report["evidence"] if e["document_id"] == "jd-2025-01-16" and e["metadata"]["page_number"] == 1)
    parent = history.page("jd-2025-01-16", 1)["text"]
    assert evidence["parent_context"] == parent
    assert parent[evidence["match_start"]:evidence["match_start"] + len(evidence["text"])] == evidence["text"]
    assert evidence["metadata"]["page_provenance"] == "physical_pdf_page"
    assert not report["coverage"]["complete_applicability_established"]


def test_tiny_budget_reports_missing_evidence_without_truncating_parent(history):
    report = history.search("Aadhaar", "2025-01-16", max_characters=1)
    assert report["evidence"] == []
    assert report["coverage"]["omitted_by_budget"]
    assert report["coverage"]["context_characters"] == 0


def test_no_matching_text_does_not_turn_into_a_rule_answer(history):
    report = history.search("qzxqzx", "2025-01-16")
    assert report["evidence"] == []
    assert report["documents"]  # The dated source inventory remains available.


@pytest.mark.parametrize("value", ["16/01/2025", "2025-1-16", "2025-02-30", "2025-01-16T00:00:00"])
def test_reject_ambiguous_or_invalid_dates(history, value):
    with pytest.raises(ValueError):
        history.snapshot(value)


def test_reject_unbacked_relationship_and_effective_date(tmp_path):
    policy = json.loads(DEFAULT_POLICY.read_text())
    for doc in policy["documents"]:
        if doc.get("source_file"):
            doc["source_file"] = str(DEFAULT_POLICY.parent / doc["source_file"])
    for mutate in ("quote", "effective_date_evidence"):
        broken = copy.deepcopy(policy)
        if mutate == "quote":
            broken["relationships"][0]["evidence"]["quote"] = "invented text"
        else:
            broken["documents"][0]["effective_date_evidence"] = None
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            PolicyHistory(path)


def test_citations_never_promote_page_counts_or_missing_pages():
    assert "unverified" in page_label({"page_number": "1-35", "page_provenance": "document_only", "total_pages": 35})
    assert "unverified" in page_label({})
    assert page_label({"page_number": 2, "page_provenance": "physical_pdf_page"}) == "2"


def test_explicit_future_commencement_is_respected_at_boundary(tmp_path):
    source = {"document_id": "scheduled", "total_pdf_pages": 1, "pages": [
        {"page_number": 1, "text": "Commences 2025-03-01", "extraction_method": "native"}
    ]}
    (tmp_path / "source.json").write_text(json.dumps(source))
    policy = {"id": "synthetic-date-boundary", "limitations": [], "relationships": [], "documents": [
        {"id": "scheduled", "issued_on": "2025-01-01", "source_file": "source.json",
         "effective_from": "2025-03-01", "effective_date_evidence": {
             "document_id": "scheduled", "page_number": 1, "quote": "Commences 2025-03-01"}}
    ]}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    history = PolicyHistory(path)
    assert history.snapshot("2025-02-28")["documents"][0]["status"] == "not_yet_effective"
    assert history.snapshot("2025-03-01")["documents"][0]["status"] == "effective_date_established"
