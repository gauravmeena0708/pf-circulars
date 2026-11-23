# tests/test_retriever_confidence_gate.py
import faiss
import numpy as np

import retriever


# --- Unit tests for the pure helper ---

def test_agreement_true_when_same_doc_is_rank_one_in_both():
    dense_rankings = {5: 1, 2: 2, 9: 3}
    sparse_rankings = {5: 1, 7: 2}
    assert retriever._has_high_confidence_agreement(dense_rankings, sparse_rankings) is True


def test_agreement_false_when_rank_one_docs_differ():
    dense_rankings = {5: 1, 2: 2}
    sparse_rankings = {7: 1, 5: 2}
    assert retriever._has_high_confidence_agreement(dense_rankings, sparse_rankings) is False


def test_agreement_false_when_either_ranking_is_empty():
    assert retriever._has_high_confidence_agreement({}, {5: 1}) is False
    assert retriever._has_high_confidence_agreement({5: 1}, {}) is False
    assert retriever._has_high_confidence_agreement({}, {}) is False


# --- End-to-end tests against retrieve_relevant_chunks ---

class _FixedVectorEmbeddingModel:
    """Fake embedding model that always returns a pre-chosen vector,
    so the test controls exactly which document dense search ranks #1."""

    def __init__(self, vector):
        self._vector = vector

    def encode(self, texts, **kwargs):
        return self._vector


class _RecordingCrossEncoder:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, pairs):
        self.predict_calls += 1
        return [0.0] * len(pairs)


def _build_fixture(query_vector):
    # Three docs with orthogonal one-hot vectors: doc0=[1,0,0,0], doc1=[0,1,0,0], doc2=[0,0,1,0].
    # BM25-side, "alpha"/"beta"/"gamma" each appear in exactly one doc's text, so a
    # single-token query deterministically ranks that doc #1 via real BM25Okapi scoring.
    texts = [
        "alpha unique statutory reference one",
        "beta unique statutory reference two",
        "gamma unique statutory reference three",
    ]
    metadata = [{"source_pdf": f"doc{i}.pdf"} for i in range(3)]

    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    vectors = np.eye(3, dimension, dtype="float32")
    index.add(vectors)

    embedding_model = _FixedVectorEmbeddingModel(np.array(query_vector, dtype="float32"))
    return index, texts, metadata, embedding_model


def test_cross_encoder_is_skipped_when_dense_and_sparse_agree_on_top_doc():
    # Query "alpha" -> BM25 top-1 is doc0. Query vector == doc0's vector -> dense top-1 is doc0 too.
    index, texts, metadata, embedding_model = _build_fixture(query_vector=[1.0, 0.0, 0.0, 0.0])
    cross_encoder = _RecordingCrossEncoder()

    results = retriever.retrieve_relevant_chunks(
        "alpha",
        index,
        texts,
        metadata,
        embedding_model,
        cross_encoder_model=cross_encoder,
        top_n_final=3,
    )

    assert cross_encoder.predict_calls == 0
    assert results[0]["text"] == texts[0]


def test_cross_encoder_is_invoked_when_dense_and_sparse_disagree():
    # Query "alpha" -> BM25 top-1 is still doc0. Query vector == doc1's vector -> dense top-1 is doc1.
    index, texts, metadata, embedding_model = _build_fixture(query_vector=[0.0, 1.0, 0.0, 0.0])
    cross_encoder = _RecordingCrossEncoder()

    results = retriever.retrieve_relevant_chunks(
        "alpha",
        index,
        texts,
        metadata,
        embedding_model,
        cross_encoder_model=cross_encoder,
        top_n_final=3,
    )

    assert cross_encoder.predict_calls == 1
    assert len(results) > 0


def test_confidence_gate_can_be_disabled_via_parameter():
    # Same agreement scenario as the "skipped" test above, but with the gate disabled.
    index, texts, metadata, embedding_model = _build_fixture(query_vector=[1.0, 0.0, 0.0, 0.0])
    cross_encoder = _RecordingCrossEncoder()

    retriever.retrieve_relevant_chunks(
        "alpha",
        index,
        texts,
        metadata,
        embedding_model,
        cross_encoder_model=cross_encoder,
        top_n_final=3,
        confidence_gate_enabled=False,
    )

    assert cross_encoder.predict_calls == 1


def test_confidence_gate_never_fires_when_hybrid_disabled():
    # Same agreement scenario as the "skipped" test, but with use_hybrid=False.
    # sparse_rankings will be empty, so the gate must never fire regardless of dense agreement.
    index, texts, metadata, embedding_model = _build_fixture(query_vector=[1.0, 0.0, 0.0, 0.0])
    cross_encoder = _RecordingCrossEncoder()

    retriever.retrieve_relevant_chunks(
        "alpha",
        index,
        texts,
        metadata,
        embedding_model,
        cross_encoder_model=cross_encoder,
        top_n_final=3,
        use_hybrid=False,
    )

    assert cross_encoder.predict_calls == 1


def test_agreed_doc_wins_even_when_a_rival_is_rank_two_in_both_rankings():
    # doc0 is rank 1 in both dense and sparse. doc1 is rank 2 in both. Confirms
    # the RRF-maximum guarantee holds beyond the simple 3-orthogonal-doc case:
    # doc0's RRF score (rank 1 + rank 1) must still beat doc1's (rank 2 + rank 2).
    dense_rankings = {0: 1, 1: 2, 2: 3}
    sparse_rankings = {0: 1, 1: 2, 2: 3}
    rrf_k = 60

    def rrf_score(doc_id):
        return 1.0 / (rrf_k + dense_rankings[doc_id]) + 1.0 / (rrf_k + sparse_rankings[doc_id])

    assert retriever._has_high_confidence_agreement(dense_rankings, sparse_rankings) is True
    assert rrf_score(0) > rrf_score(1) > rrf_score(2)


def test_gate_preserves_rrf_order_for_slots_beyond_the_top_match():
    # Documents this branch's known, accepted tradeoff: when the gate fires, slots
    # 2..N are ordered by RRF score (not re-ranked by CrossEncoder), which can differ
    # from what the CrossEncoder would have chosen. This test locks in that RRF
    # ordering is what's actually returned, so a future change to this behavior is
    # a visible, deliberate diff rather than a silent regression.
    index, texts, metadata, embedding_model = _build_fixture(query_vector=[1.0, 0.0, 0.0, 0.0])
    cross_encoder = _RecordingCrossEncoder()

    results = retriever.retrieve_relevant_chunks(
        "alpha",
        index,
        texts,
        metadata,
        embedding_model,
        cross_encoder_model=cross_encoder,
        top_n_final=3,
    )

    assert cross_encoder.predict_calls == 0
    result_rrf_scores = [r["rrf_score"] for r in results]
    assert result_rrf_scores == sorted(result_rrf_scores, reverse=True)
