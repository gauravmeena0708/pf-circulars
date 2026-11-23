# tests/test_passage_store.py
import json
import os
import sqlite3

import pytest

import passage_store


def _write_passages_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE passages (id INTEGER PRIMARY KEY, text TEXT NOT NULL, metadata TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO passages (id, text, metadata) VALUES (?, ?, ?)",
        [(doc_id, text, json.dumps(metadata)) for doc_id, text, metadata in rows],
    )
    conn.commit()
    conn.close()


def test_open_passage_store_returns_none_when_file_missing(tmp_path):
    assert passage_store.open_passage_store(str(tmp_path)) is None


def test_texts_and_metadata_views_support_len_and_getitem(tmp_path):
    db_path = os.path.join(str(tmp_path), "faiss_index.passages.db")
    rows = [
        (0, "alpha passage", {"source_pdf": "a.pdf"}),
        (1, "beta passage", {"source_pdf": "b.pdf"}),
        (2, "gamma passage", {"source_pdf": "c.pdf"}),
    ]
    _write_passages_db(db_path, rows)

    store = passage_store.open_passage_store(str(tmp_path))

    assert len(store.texts) == 3
    assert len(store.metadata) == 3
    assert store.texts[1] == "beta passage"
    assert store.metadata[2] == {"source_pdf": "c.pdf"}


def test_getitem_raises_indexerror_for_unknown_id(tmp_path):
    db_path = os.path.join(str(tmp_path), "faiss_index.passages.db")
    _write_passages_db(db_path, [(0, "alpha passage", {"source_pdf": "a.pdf"})])

    store = passage_store.open_passage_store(str(tmp_path))

    with pytest.raises(IndexError):
        store.texts[99]


def test_iteration_streams_all_rows_in_id_order(tmp_path):
    db_path = os.path.join(str(tmp_path), "faiss_index.passages.db")
    rows = [
        (0, "alpha passage", {"source_pdf": "a.pdf"}),
        (1, "beta passage", {"source_pdf": "b.pdf"}),
        (2, "gamma passage", {"source_pdf": "c.pdf"}),
    ]
    _write_passages_db(db_path, rows)

    store = passage_store.open_passage_store(str(tmp_path))

    assert list(store.texts) == ["alpha passage", "beta passage", "gamma passage"]
    assert list(store.metadata) == [
        {"source_pdf": "a.pdf"},
        {"source_pdf": "b.pdf"},
        {"source_pdf": "c.pdf"},
    ]


def test_retrieve_relevant_chunks_gives_identical_results_for_lists_and_passage_store(tmp_path):
    """Proves retriever.retrieve_relevant_chunks cannot tell the difference
    between real lists and PassageStore views -- the core claim this
    module exists to satisfy."""
    import faiss
    import numpy as np

    import retriever

    texts = [
        "alpha unique statutory reference one",
        "beta unique statutory reference two",
        "gamma unique statutory reference three",
    ]
    metadata = [{"source_pdf": f"doc{i}.pdf"} for i in range(3)]

    db_path = os.path.join(str(tmp_path), "faiss_index.passages.db")
    _write_passages_db(db_path, list(zip(range(3), texts, metadata)))
    store = passage_store.open_passage_store(str(tmp_path))

    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    index.add(np.eye(3, dimension, dtype="float32"))

    class _FixedVectorEmbeddingModel:
        def encode(self, _texts, **_kwargs):
            return np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")

    embedding_model = _FixedVectorEmbeddingModel()

    list_based_results = retriever.retrieve_relevant_chunks(
        "beta", index, texts, metadata, embedding_model, top_n_final=3
    )
    store_based_results = retriever.retrieve_relevant_chunks(
        "beta", index, store.texts, store.metadata, embedding_model, top_n_final=3
    )

    assert list_based_results == store_based_results


def test_retrieve_relevant_chunks_gives_identical_results_with_confidence_gate_and_cross_encoder(tmp_path, monkeypatch):
    """Proves the confidence-gate branch's direct all_indexed_metadata[agreed_doc_id]
    access (used for its log line) also behaves identically for lists vs.
    PassageStore views -- the other cross-check test never exercises this
    branch since it never passes a cross_encoder_model."""
    import faiss
    import numpy as np

    import retriever

    texts = [
        "alpha unique statutory reference one",
        "beta unique statutory reference two",
        "gamma unique statutory reference three",
    ]
    metadata = [{"source_pdf": f"doc{i}.pdf"} for i in range(3)]

    db_path = os.path.join(str(tmp_path), "faiss_index.passages.db")
    _write_passages_db(db_path, list(zip(range(3), texts, metadata)))
    store = passage_store.open_passage_store(str(tmp_path))

    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    index.add(np.eye(3, dimension, dtype="float32"))

    class _FixedVectorEmbeddingModel:
        def encode(self, _texts, **_kwargs):
            return np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")

    class _RecordingCrossEncoder:
        def predict(self, pairs):
            return [0.0] * len(pairs)

    embedding_model = _FixedVectorEmbeddingModel()
    cross_encoder = _RecordingCrossEncoder()

    monkeypatch.setattr(retriever, "_has_high_confidence_agreement", lambda dense, sparse: True)

    list_based_results = retriever.retrieve_relevant_chunks(
        "beta", index, texts, metadata, embedding_model,
        cross_encoder_model=cross_encoder, top_n_final=3,
    )
    store_based_results = retriever.retrieve_relevant_chunks(
        "beta", index, store.texts, store.metadata, embedding_model,
        cross_encoder_model=cross_encoder, top_n_final=3,
    )

    assert list_based_results == store_based_results
