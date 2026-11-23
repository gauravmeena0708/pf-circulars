"""Offline, evidence-first policy history pilot; no applicability inferred from recency.

Run ``python policy_history.py --as-of 2025-01-16 --query 'closed establishment'``.
This research CLI does not replace the application's hybrid retriever.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
import math
from pathlib import Path
import re

DEFAULT_POLICY = Path(__file__).parent / "policy_data/joint_declaration/policy.json"


def _date(value):
    # Reject ambiguous dates and datetimes rather than guessing locale.
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Dates must use YYYY-MM-DD")
    return date.fromisoformat(value)


def _normalise(text):
    return " ".join(text.split()).casefold()


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.casefold())


class UnknownPolicyMatter(ValueError):
    pass


class PolicyHistory:
    def __init__(self, path=DEFAULT_POLICY):
        path = Path(path)
        self.policy = json.loads(path.read_text(encoding="utf-8"))
        topic_ids = [topic['id'] for topic in self.policy.get('topics', [])]
        if len(set(topic_ids)) != len(topic_ids):
            raise ValueError('Duplicate policy matter')
        for topic in self.policy.get('topics', []):
            re.compile(topic['include'])
        self.documents = {}
        self.pages = {}
        for document in self.policy["documents"]:
            if set(document.get('topics', [])) - set(topic_ids):
                raise ValueError('Document refers to an unknown matter')
            doc_id = document["id"]
            if doc_id in self.documents:
                raise ValueError(f"Duplicate document: {doc_id}")
            _date(document["issued_on"])
            if document.get("effective_from"):
                _date(document["effective_from"])
                if not document.get("effective_date_evidence"):
                    raise ValueError(f"Effective date lacks evidence: {doc_id}")
            self.documents[doc_id] = document
            if document.get("source_file"):
                source = json.loads((path.parent / document["source_file"]).read_text(encoding="utf-8"))
                if source["document_id"] != doc_id:
                    raise ValueError(f"Source identity mismatch: {doc_id}")
                self.pages[doc_id] = source["pages"]
                numbers = [p["page_number"] for p in source["pages"]]
                if len(set(numbers)) != len(numbers) or any(
                    type(n) is not int or not 1 <= n <= source["total_pdf_pages"] for n in numbers
                ):
                    raise ValueError(f"Invalid physical page mapping: {doc_id}")
        self.relationships = self.policy["relationships"]
        for doc in self.documents.values():
            if doc.get("effective_from"):
                self._check_evidence(doc["effective_date_evidence"], doc["id"])
        for relation in self.relationships:
            if relation["type"] not in {"supersedes", "partially_modifies", "clarifies", "references"}:
                raise ValueError("Unsupported relationship type")
            if relation["source"] not in self.documents or relation["target"] not in self.documents:
                raise ValueError("Relationship refers to an unknown document")
            if not relation.get("scope"):
                raise ValueError("Relationships require an explicit scope")
            if relation.get("effective_from"):
                _date(relation["effective_from"])
                if not relation.get("effective_date_evidence"):
                    raise ValueError("Relationship effective date lacks evidence")
                self._check_evidence(relation["effective_date_evidence"], relation["source"])
            if relation["verification"] == "source_checked":
                self._check_evidence(relation["evidence"], relation["source"])

    def _check_evidence(self, evidence, source_id):
        if evidence["document_id"] != source_id:
            raise ValueError("Evidence must come from its issuing source")
        page = self.page(source_id, evidence["page_number"])
        if not evidence.get("quote") or _normalise(evidence["quote"]) not in _normalise(page["text"]):
            raise ValueError("Evidence quotation is absent from the cited page")

    def page(self, doc_id, page_number):
        for page in self.pages.get(doc_id, []):
            if page["page_number"] == page_number:
                return page
        raise ValueError(f"No checked physical page {page_number} for {doc_id}")

    def matching_topics(self, query):
        """Return every matching matter; multi-matter questions retain each scope."""
        return [topic['id'] for topic in self.policy.get('topics', [])
                if re.search(topic['include'], query, re.I)]

    def snapshot(self, as_of, topics=None):
        cutoff = _date(as_of)
        topics = topics or []
        known = {t['id'] for t in self.policy.get('topics', [])}
        if any(topic not in known for topic in topics):
            raise UnknownPolicyMatter('Unknown matter in this policy collection')
        available = {
            k: v for k, v in self.documents.items() if _date(v["issued_on"]) <= cutoff
            and (not topics or set(topics).intersection(v.get('topics', [])))
        }
        documents = []
        for doc in sorted(available.values(), key=lambda d: (d["issued_on"], d["id"])):
            if doc["id"] not in self.pages:
                status = "source_unverified"
            elif not doc.get("effective_from"):
                status = "issued_effective_date_unverified"
            elif _date(doc["effective_from"]) > cutoff:
                status = "not_yet_effective"
            else:
                status = "effective_date_established"
            documents.append({**doc, "status": status})
        relationships = []
        for rel in self.relationships:
            if rel["source"] not in available or rel["target"] not in available:
                continue
            if rel["verification"] != "source_checked":
                status = "relationship_unverified"
            elif not rel.get("effective_from"):
                status = "relationship_verified_timing_unresolved"
            elif _date(rel["effective_from"]) > cutoff:
                status = "not_yet_effective"
            else:
                status = "effective_relationship"
            relationships.append({**rel, "status": status})
        return {
            "policy_id": self.policy["id"], "as_of": as_of,
            "conclusion": "evidence_inventory_only",
            "documents": documents, "relationships": relationships,
            "limitations": self.policy["limitations"],
            "topics": self.policy.get('topics', []), "selected_topics": topics,
        }

    def search(self, query, as_of, max_characters=16000, topics=None):
        """Find small passages, then expand to exact physical parent pages.

        Lexical ranking is deliberately local to this pilot. Production callers
        can use the snapshot alongside existing dense/BM25 retrieval instead.
        All dated documents/relationships remain in the inventory even when their
        pages do not match the query. A character budget is not a completeness claim.
        """
        if max_characters < 1:
            raise ValueError("Context budget must be positive")
        report = self.snapshot(as_of, topics=self.matching_topics(query) if topics is None else topics)
        allowed = {d["id"] for d in report["documents"]}
        terms = set(_tokens(query))
        candidates = []
        for doc_id in sorted(allowed):
            document = self.documents[doc_id]
            if document.get('document_role') == 'consultation':
                continue
            for page in self.pages.get(doc_id, []):
                text = page["text"]
                # Keep offsets into unmodified page text for citation auditing.
                for start in range(0, len(text), 650):
                    chunk = text[start:start + 800]
                    candidates.append((doc_id, page, start, chunk, Counter(_tokens(document["title"] + " " + chunk))))
        frequencies = Counter()
        for *_, counts in candidates:
            frequencies.update(counts.keys())
        ranked = []
        navigation_request = bool(re.search(r"\bcontrol\b|effective date|\bversion\b", query, re.I))
        wanted_version = re.search(r"\bversion\s*(\d+(?:\.\d+)?)", query, re.I)
        for doc_id, page, start, chunk, counts in candidates:
            score = sum(
                (1 + math.log(counts[t])) * math.log(1 + len(candidates) / frequencies[t])
                for t in terms if counts[t]
            )
            if score and navigation_request and re.search(r"SOP\s+version", page["text"], re.I) and re.search(r"Effective\s+Date", page["text"], re.I):
                score *= 2
                if wanted_version and re.search(r"version\s*[:\s]*" + re.escape(wanted_version.group(1)) + r"\b", page["text"], re.I):
                    score *= 2
            if score:
                ranked.append((score, doc_id, page, start, chunk))
        ranked.sort(key=lambda x: (-x[0], x[1], x[2]["page_number"], x[3]))
        # First evidence page per document, then further pages: avoid allowing a
        # long historical SOP to consume the whole context budget.
        first, rest, seen_docs = [], [], set()
        for result in ranked:
            if result[1] in seen_docs:
                rest.append(result)
            else:
                first.append(result)
                seen_docs.add(result[1])
        evidence, seen_pages, used = [], set(), 0
        omitted = set()
        for score, doc_id, page, start, chunk in first + rest:
            key = doc_id, page["page_number"]
            if key in seen_pages:
                continue
            seen_pages.add(key)
            parent = page["text"]
            if used + len(parent) > max_characters:
                omitted.add(doc_id)
                continue
            used += len(parent)
            doc = self.documents[doc_id]
            evidence.append({
                "document_id": doc_id, "score": score, "text": chunk,
                "parent_context": parent, "match_start": start,
                "metadata": {"title": doc["title"], "circular_no": doc["circular_no"],
                             "date": doc["issued_on"], "authority": doc.get("authority", "EPFO document"), "page_number": page["page_number"],
                             "page_provenance": "physical_pdf_page", "english_pdf_link": doc["source_url"],
                             "source_host_kind": doc.get("source_host_kind", "issuer_hosted"), "original_source_url": doc.get("original_source_url"),
                             "extraction_method": page["extraction_method"]},
            })
        report.update({
            "query": query, "evidence": evidence,
            "coverage": {
                "matched_documents": sorted({r[1] for r in ranked}),
                "included_documents": sorted({r["document_id"] for r in evidence}),
                "omitted_by_budget": sorted(omitted),
                "documents_without_page_text": sorted(allowed - self.pages.keys()),
                "context_characters": used,
                "complete_applicability_established": False,
            },
        })
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    from policy_registry import registered_policies
    policies = registered_policies()
    parser.add_argument("--policy-name", choices=[p.id for p in policies], help="Use a registered policy history")
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD; issue-date cutoff, not inferred commencement")
    parser.add_argument("--query", default="")
    parser.add_argument("--max-characters", type=int, default=16000)
    args = parser.parse_args()
    try:
        if args.policy_name and args.policy != DEFAULT_POLICY:
            parser.error("Use either --policy or --policy-name")
        path = next(p.path for p in policies if p.id == args.policy_name) if args.policy_name else args.policy
        history = PolicyHistory(path)
        report = history.search(args.query, args.as_of, args.max_characters) if args.query else history.snapshot(args.as_of)
    except (ValueError, KeyError, StopIteration) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
