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
