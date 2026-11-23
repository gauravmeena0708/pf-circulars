import gzip
import json
import os

import faiss
import numpy as np

import vector_indexer


def test_save_faiss_index_also_writes_bm25_cache(tmp_path):
    texts = ["Provident fund withdrawal under para 68", "Form 13 transfer request"]
    metadata = [{"source_pdf": "a.pdf"}, {"source_pdf": "b.pdf"}]

    dimension = 4
    flat_index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(flat_index)
    vectors = np.eye(len(texts), dimension, dtype="float32")
    index.add_with_ids(vectors, np.arange(len(texts)))

    index_dir = str(tmp_path)
    vector_indexer.save_faiss_index(index, texts, metadata, index_dir, index_name="faiss_index")

    bm25_cache_path = os.path.join(index_dir, "faiss_index.bm25.json.gz")
    assert os.path.isfile(bm25_cache_path)
    with gzip.open(bm25_cache_path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["document_count"] == len(texts)


def test_save_faiss_index_refreshes_stale_cache_when_corpus_grows_in_place(tmp_path):
    texts = ["Provident fund withdrawal under para 68", "Form 13 transfer request"]
    metadata = [{"source_pdf": "a.pdf"}, {"source_pdf": "b.pdf"}]

    def build_index(count, dimension=4):
        flat_index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap(flat_index)
        vectors = np.eye(count, dimension, dtype="float32")
        index.add_with_ids(vectors, np.arange(count))
        return index

    index_dir = str(tmp_path)
    vector_indexer.save_faiss_index(
        build_index(len(texts)), texts, metadata, index_dir, index_name="faiss_index"
    )

    # Simulate a checkpointed ingestion run: same list object mutated in place,
    # exactly like index_manuals.py / import_pf_circular_index.py do every 5000 chunks.
    texts.append("Employees Pension Scheme 1995 eligibility criteria")
    metadata.append({"source_pdf": "c.pdf"})

    vector_indexer.save_faiss_index(
        build_index(len(texts)), texts, metadata, index_dir, index_name="faiss_index"
    )

    bm25_cache_path = os.path.join(index_dir, "faiss_index.bm25.json.gz")
    with gzip.open(bm25_cache_path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["document_count"] == len(texts)
