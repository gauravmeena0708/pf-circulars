import gzip
import json
import os

import retriever


def test_warm_bm25_cache_creates_persisted_file(tmp_path):
    texts = [
        "Provident fund withdrawal under para 68",
        "Form 13 transfer of PF account between establishments",
        "Employees Pension Scheme 1995 eligibility criteria",
    ]
    cache_path = os.path.join(str(tmp_path), "faiss_index.bm25.json.gz")

    retriever.warm_bm25_cache(texts, cache_path)

    assert os.path.isfile(cache_path)
    with gzip.open(cache_path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["document_count"] == len(texts)
    assert payload["bm25_state"]["corpus_size"] == len(texts)


def test_warm_bm25_cache_is_reused_without_rebuild(tmp_path, monkeypatch):
    texts = ["Provident fund withdrawal under para 68", "Form 13 transfer request"]
    cache_path = os.path.join(str(tmp_path), "faiss_index.bm25.json.gz")

    retriever.warm_bm25_cache(texts, cache_path)

    # Force the in-memory identity-based cache to miss, so the second call
    # must go through the persisted-file path.
    retriever._BM25_CACHE["corpus"] = None
    retriever._BM25_CACHE["bm25_instance"] = None

    build_calls = []
    original_init = retriever.BM25Okapi.__init__

    def tracking_init(self, *args, **kwargs):
        build_calls.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(retriever.BM25Okapi, "__init__", tracking_init)

    retriever.warm_bm25_cache(texts, cache_path)

    assert build_calls == []  # loaded from the persisted cache, not rebuilt
