# tests/test_vector_indexer_passages.py
import os

import faiss
import numpy as np

import vector_indexer


def test_save_and_load_round_trip_through_sqlite(tmp_path):
    texts = ["alpha passage", "beta passage"]
    metadata = [{"source_pdf": "a.pdf"}, {"source_pdf": "b.pdf"}]

    dimension = 4
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
    vectors = np.eye(len(texts), dimension, dtype="float32")
    index.add_with_ids(vectors, np.arange(len(texts)))

    index_dir = str(tmp_path)
    vector_indexer.save_faiss_index(index, texts, metadata, index_dir, index_name="faiss_index")

    db_path = os.path.join(index_dir, "faiss_index.passages.db")
    assert os.path.isfile(db_path)
    assert not os.path.isfile(os.path.join(index_dir, "faiss_index.texts.json"))

    loaded_index, loaded_texts, loaded_metadata = vector_indexer.load_faiss_index(
        index_dir, index_name="faiss_index"
    )

    assert loaded_index.ntotal == 2
    assert loaded_texts == texts
    assert loaded_metadata == metadata


def test_load_faiss_binary_index_works_without_passages_db(tmp_path):
    texts = ["alpha passage"]
    metadata = [{"source_pdf": "a.pdf"}]

    dimension = 4
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
    index.add_with_ids(np.eye(1, dimension, dtype="float32"), np.arange(1))

    index_dir = str(tmp_path)
    vector_indexer.save_faiss_index(index, texts, metadata, index_dir, index_name="faiss_index")

    # Remove the passages db, keep only the FAISS binary index.
    os.remove(os.path.join(index_dir, "faiss_index.passages.db"))

    binary_index = vector_indexer.load_faiss_binary_index(index_dir, index_name="faiss_index")
    assert binary_index is not None
    assert binary_index.ntotal == 1

    # load_faiss_index (which also needs the passages db) must fail gracefully.
    full_index, full_texts, full_metadata = vector_indexer.load_faiss_index(
        index_dir, index_name="faiss_index"
    )
    assert (full_index, full_texts, full_metadata) == (None, None, None)


def test_load_faiss_index_returns_none_when_index_missing(tmp_path):
    index_dir = str(tmp_path)
    result = vector_indexer.load_faiss_index(index_dir, index_name="faiss_index")
    assert result == (None, None, None)
