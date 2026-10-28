"""Registered policy histories with explicit, conservative hybrid fallback."""

from datetime import datetime

from policy_history import PolicyHistory, _date
from policy_registry import registered_policies, select_policy


def is_jd_question(query):
    """Compatibility helper; the registry owns all routing expressions."""
    return next(p for p in registered_policies() if p.id == "jd").matches(query)


def _issued_on(metadata):
    value = metadata.get("date")
    if not value:
        return None
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(value).strip(), pattern).date()
        except ValueError:
            pass
    return None


def route_retrieval(query, as_of, hybrid_search, mode="auto", history_factory=None):
    """Return answer chunks plus a separate policy review report.

    ``hybrid_search`` is lazy so curated evidence works without embedding services.
    Fallback is supplementary evidence, never a legal-currentness determination.
    """
    cutoff = _date(as_of)
    policies = registered_policies()
    if mode not in {"auto", "hybrid", *(p.id for p in policies)}:
        raise ValueError("Unknown retrieval mode")
    policy = select_policy(query, mode, policies)
    if policy is None:
        return {"route": "hybrid", "chunks": hybrid_search(), "policy_context": None, "report": None}

    warning = None
    try:
        history = history_factory() if history_factory else PolicyHistory(policy.path)
        report = history.search(query, as_of)
    except (OSError, ValueError, KeyError) as error:
        # Do not present a failed policy load as a successful verification pass.
        warning = f"Curated {policy.label} evidence could not be loaded; policy verification is unavailable."
        report = {"as_of": as_of, "documents": [], "relationships": [], "evidence": [],
                  "coverage": {}, "limitations": [warning]}

    documents = {d["id"]: d for d in report["documents"]}
    chunks = []
    included_pages = set()
    for evidence in report["evidence"]:
        doc = documents[evidence["document_id"]]
        if doc["status"] == "not_yet_effective":
            continue
        item = {**evidence, "text": evidence["parent_context"], "metadata": {
            **evidence["metadata"], "evidence_status": "curated source page; applicability not certified",
            "policy_document_status": doc["status"],
        }}
        chunks.append(item)
        included_pages.add((evidence["document_id"], evidence["metadata"]["page_number"]))
    # Relationship and commencement pages must reach generation even if the
    # small-window search did not rank them. They carry their own source numbers.
    if warning is None:
        required = [r["evidence"] for r in report["relationships"] if r["verification"] == "source_checked"]
        required += [d["effective_date_evidence"] for d in report["documents"] if d.get("effective_date_evidence")]
        for evidence in required:
            key = evidence["document_id"], evidence["page_number"]
            if key in included_pages:
                continue
            doc = documents[key[0]]
            if doc["status"] == "not_yet_effective":
                continue
            page = history.page(*key)
            chunks.append({"text": page["text"], "metadata": {
                "title": doc["title"], "circular_no": doc["circular_no"], "date": doc["issued_on"],
                "english_pdf_link": doc["source_url"], "page_number": key[1],
                "page_provenance": "physical_pdf_page", "evidence_status": "curated relationship or commencement page",
                "policy_document_status": doc["status"],
            }})
            included_pages.add(key)

    supporting = []
    fallback_reason = ""
    if warning or not report["evidence"] or report["coverage"].get("documents_without_page_text") or report["coverage"].get("omitted_by_budget"):
        fallback_reason = warning or f"Curated {policy.label} coverage is incomplete; supplementary hybrid results remain unverified."
        try:
            candidates = hybrid_search()
        except Exception:
            candidates = []
            fallback_reason += " Supplementary search is unavailable."
        known = {d["circular_no"].casefold().replace(" ", ""): d for d in documents.values()}
        for candidate in candidates:
            meta = candidate.get("metadata", {})
            title = meta.get("title") or ""
            if not policy.matches(title + " " + candidate.get("text", "")):
                continue
            issued = _issued_on(meta)
            # Unknown dates cannot safely enter a historical evidence packet.
            if issued is None or issued > cutoff:
                continue
            doc = known.get(str(meta.get("circular_no", "")).casefold().replace(" ", ""))
            if doc and doc["status"] == "not_yet_effective":
                continue
            supporting_metadata = {
                **meta, "evidence_status": "supplementary hybrid source; amendment and applicability unverified",
            }
            # Legacy OCR imports can contain a document-wide range masquerading
            # as a chunk page. Only explicitly mapped pages keep a page claim.
            if meta.get("page_provenance") != "physical_pdf_page":
                supporting_metadata.update(page_number=None, page_provenance="document_only")
            supporting.append({**candidate, "metadata": supporting_metadata})
        chunks.extend(supporting)

    context = {
        "as_of": as_of, "scope": policy.scope, "policy_id": policy.id, "policy_label": policy.label,
        "date_semantics": "Instructions issued by this date; not a complete retrospective legal applicability determination.",
        "conclusion": "complete_applicability_not_established",
        "documents": [{k: d.get(k) for k in ("id", "title", "version_label", "authority", "notes", "circular_no", "issued_on", "effective_from", "status")} for d in documents.values()],
        "relationships": report["relationships"], "coverage": report["coverage"],
        "limitations": report["limitations"], "fallback_reason": fallback_reason,
        "supporting_source_count": len(supporting),
    }
    return {"route": policy.route, "chunks": chunks, "policy_context": context, "report": report}
