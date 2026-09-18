import faiss

import vector_indexer


def test_create_empty_faiss_index_uses_inner_product_metric():
    index = vector_indexer.create_empty_faiss_index(dimension=8)

    assert isinstance(index, faiss.IndexIDMap)
    assert index.index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert index.ntotal == 0


class _FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(kwargs)
        return [[0.0] for _ in texts]


def test_encode_normalized_always_forces_normalize_embeddings_true():
    model = _FakeEmbeddingModel()

    vector_indexer.encode_normalized(model, ["a", "b"], normalize_embeddings=False, batch_size=32)

    assert len(model.calls) == 1
    assert model.calls[0]["normalize_embeddings"] is True
    assert model.calls[0]["batch_size"] == 32


def test_encode_normalized_defaults_convert_to_tensor_false_but_allows_override():
    model = _FakeEmbeddingModel()

    vector_indexer.encode_normalized(model, ["a"])
    assert model.calls[0]["convert_to_tensor"] is False

    vector_indexer.encode_normalized(model, ["a"], convert_to_tensor=True)
    assert model.calls[1]["convert_to_tensor"] is True
