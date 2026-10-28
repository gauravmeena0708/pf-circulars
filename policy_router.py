"""JD-only policy routing with explicit, conservative hybrid fallback."""

from datetime import datetime
import re

from policy_history import PolicyHistory, _date


def is_jd_question(query):
    text = query.casefold()
    # WSU profile correction is distinct from higher-pension joint options.
    if re.search(r"higher[ -]?pension|joint option|26\s*\(6\)|उच्च पेंशन", text):
        return False
    return bool(re.search(
        r"joint\s*declaration|\bjd\b|member\s+profile|संयुक्त घोषणा|जॉइंट डिक्लेरेशन", text
    ))


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


def route_retrieval(query, as_of, hybrid_search, mode="auto", history_factory=PolicyHistory):
    """Return answer chunks plus a separate policy review report.

    ``hybrid_search`` is lazy so JD evidence works without embedding services.
    Fallback is supplementary evidence, never a legal-currentness determination.
    """
    cutoff = _date(as_of)
    if mode not in {"auto", "jd", "hybrid"}:
        raise ValueError("Unknown retrieval mode")
    use_jd = mode == "jd" or (mode == "auto" and is_jd_question(query))
    if not use_jd:
        return {"route": "hybrid", "chunks": hybrid_search(), "policy_context": None, "report": None}

    warning = None
    try:
        history = history_factory()
        report = history.search(query, as_of)
    except (OSError, ValueError, KeyError) as error:
        # Do not present a failed policy load as a successful verification pass.
        warning = "Curated JD evidence could not be loaded; policy verification is unavailable."
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
        fallback_reason = warning or "Curated JD coverage is incomplete; supplementary hybrid results remain unverified."
        try:
            candidates = hybrid_search()
        except Exception:
            candidates = []
            fallback_reason += " Supplementary search is unavailable."
        known = {d["circular_no"].casefold().replace(" ", ""): d for d in documents.values()}
        for candidate in candidates:
            meta = candidate.get("metadata", {})
            title = meta.get("title") or ""
            if not is_jd_question(title + " " + candidate.get("text", "")):
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
        "as_of": as_of, "scope": "WSU Joint Declaration member-profile correction only",
        "date_semantics": "Instructions issued by this date; not a complete retrospective legal applicability determination.",
        "conclusion": "complete_applicability_not_established",
        "documents": [{k: d.get(k) for k in ("id", "circular_no", "issued_on", "effective_from", "status")} for d in documents.values()],
        "relationships": report["relationships"], "coverage": report["coverage"],
        "limitations": report["limitations"], "fallback_reason": fallback_reason,
        "supporting_source_count": len(supporting),
    }
    return {"route": "jd_policy", "chunks": chunks, "policy_context": context, "report": report}
