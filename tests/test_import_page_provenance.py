"""Exercise the real OCR importer with local data and a tiny in-memory index."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import import_pf_circular_index as importer


def test_ocr_that_kept_its_pages_cites_the_physical_page(tmp_path, monkeypatch):
    url = "https://example.com/circular.pdf"
    first = "The first page sets out the eligibility conditions for the member profile correction. " * 8
    second = "The second page lists the supporting documents that the employer must attest in each case."
    (tmp_path / "index-test.json").write_text(
        json.dumps(
            {url: {"ocr_content": first + second, "pages": [first, "", second], "total_pages": 3, "pages_indexed": 3}}
        )
    )
    model = SimpleNamespace(get_sentence_embedding_dimension=lambda: 2)
    monkeypatch.setattr(importer, "SentenceTransformer", lambda *a, **kw: model)
    monkeypatch.setattr(importer, "backup_existing_index", lambda *a: None)
    monkeypatch.setattr(importer, "load_faiss_index", lambda *a, **kw: (None, [], []))
    monkeypatch.setattr(importer, "encode_normalized", lambda model, texts, **kw: np.tile([1.0, 0.0], (len(texts), 1)))
    monkeypatch.setattr(importer, "load_all_circular_metadata", lambda *a: {})
    saved = []
    monkeypatch.setattr(
        importer, "save_faiss_index", lambda index, texts, metadata, **kw: saved.extend(zip(texts, metadata))
    )
    importer.run_import(str(tmp_path))
    assert all(meta["page_provenance"] == "physical_pdf_page" for _, meta in saved)
    # The blank second page yields no passage; the third page keeps its own number.
    assert sorted({meta["page_number"] for _, meta in saved}) == [1, 3]
    assert len([1 for _, meta in saved if meta["page_number"] == 1]) > 1
    assert all("supporting documents" in text for text, meta in saved if meta["page_number"] == 3)
    assert all(meta["total_pages"] == 3 for _, meta in saved)


@pytest.mark.parametrize("page_info", [{}, {"pages_indexed": 2}, {"total_pages": 35, "pages_indexed": 2}])
def test_flattened_ocr_never_claims_chunk_pages(tmp_path, monkeypatch, page_info):
    url = "https://example.com/circular.pdf"
    (tmp_path / "index-test.json").write_text(
        json.dumps(
            {
                url: {
                    "ocr_content": "A sufficiently long direction about member profile correction and supporting documents.",
                    **page_info,
                }
            }
        )
    )
    model = SimpleNamespace(get_sentence_embedding_dimension=lambda: 2)
    monkeypatch.setattr(importer, "SentenceTransformer", lambda *a, **kw: model)
    monkeypatch.setattr(importer, "backup_existing_index", lambda *a: None)
    monkeypatch.setattr(importer, "load_faiss_index", lambda *a, **kw: (None, [], []))
    # Real FAISS index helper, local deterministic embeddings; no model download.
    monkeypatch.setattr(importer, "encode_normalized", lambda model, texts, **kw: np.tile([1.0, 0.0], (len(texts), 1)))
    monkeypatch.setattr(importer, "load_all_circular_metadata", lambda *a: {url: {"title": "Pilot"}})
    captured = []
    monkeypatch.setattr(importer, "save_faiss_index", lambda index, texts, metadata, **kw: captured.extend(metadata))
    importer.run_import(str(tmp_path))
    assert captured
    for metadata in captured:
        assert metadata["page_number"] is None
        assert metadata["page_provenance"] == "document_only"
        assert metadata["total_pages"] == page_info.get("total_pages")
        assert metadata["pages_indexed"] == page_info.get("pages_indexed")
