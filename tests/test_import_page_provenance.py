"""Exercise the real OCR importer with local data and a tiny in-memory index."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import import_pf_circular_index as importer


@pytest.mark.parametrize("page_info", [{}, {"pages_indexed": 2}, {"total_pages": 35, "pages_indexed": 2}])
def test_flattened_ocr_never_claims_chunk_pages(tmp_path, monkeypatch, page_info):
    url = "https://example.com/circular.pdf"
    (tmp_path / "index-test.json").write_text(json.dumps({
        url: {"ocr_content": "A sufficiently long direction about member profile correction and supporting documents.", **page_info}
    }))
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
