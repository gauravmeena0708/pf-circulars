"""Development source/page recall and history guardrails; no LLM quality score."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from policy_history import PolicyHistory
from policy_registry import registered_policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    policies = registered_policies()
    parser.add_argument("--policy-name", default="jd", choices=[p.id for p in policies])
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args()
    policy = next(p for p in policies if p.id == args.policy_name)
    cases_path = args.cases or Path(__file__).with_name("cases.json")
    if args.policy_name != "jd" and args.cases is None:
        parser.error("Supply --cases for the selected policy's development dataset")
    dataset = json.loads(cases_path.read_text())
    history = PolicyHistory(policy.path)
    failures, source_hits, source_total, page_hits, passed = [], 0, 0, 0, 0
    for case in dataset["cases"]:
        report = history.search(case["question"], case["as_of"])
        docs = {d["id"]: d for d in report["documents"]}
        sources = {e["document_id"] for e in report["evidence"]}
        pages = {(e["document_id"], e["metadata"]["page_number"]) for e in report["evidence"]}
        checks = [all(d["issued_on"] <= case["as_of"] for d in docs.values())]
        for required in case.get("required_evidence", []):
            source_total += 1
            source_hits += required["document_id"] in sources
            page_hits += (required["document_id"], required["page_number"]) in pages
            checks.append((required["document_id"], required["page_number"]) in pages)
        if "expected_document_count" in case:
            checks.append(len(docs) == case["expected_document_count"])
        checks.extend(doc not in docs for doc in case.get("absent_documents", []))
        if "document_status" in case:
            wanted = case["document_status"]
            checks.append(docs[wanted["id"]]["status"] == wanted["status"])
        if "relationship" in case:
            checks.append(any(all(r.get(k) == v for k, v in case["relationship"].items()) for r in report["relationships"]))
        if "expected_conclusion" in case:
            checks.append(report["conclusion"] == case["expected_conclusion"])
        if all(checks):
            passed += 1
        else:
            failures.append(case["id"])
    print(json.dumps({
        "dataset_status": dataset["dataset_status"], "cases": len(dataset["cases"]),
        "passed": passed, "failed_cases": failures,
        "required_source_recall": source_hits / source_total if source_total else None,
        "required_page_recall": page_hits / source_total if source_total else None,
        "not_measured": ["answer accuracy", "unsupported claim rate", "complete amendment correctness", "production hybrid retrieval performance"],
    }, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
