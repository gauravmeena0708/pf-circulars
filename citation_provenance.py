"""Shared citation labels; a document page count is not passage provenance."""


def page_label(metadata):
    if metadata.get("page_provenance") == "document_only":
        return "unverified (document-level source)"
    value = metadata.get("page_number")
    if value is None or not str(value).strip() or str(value).strip().lower() in {"n/a", "unknown"}:
        return "unverified (document-level source)"
    return str(value)
