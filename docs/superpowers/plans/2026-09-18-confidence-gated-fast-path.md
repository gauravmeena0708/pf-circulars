# Confidence-Gated Fast Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip the CrossEncoder re-ranking pass when dense (FAISS) and sparse (BM25) retrieval already independently agree on the same single best-matching document, since a deep re-ranking pass over the same top candidate would not change the outcome. This is the "high reciprocal agreement" fast path in the optimization backlog — it's most often triggered by an exact statutory reference or circular number (e.g. `"WSU/2022/1"`, `"Section 17"`) that both a literal BM25 token match and a semantic embedding recognize as the unique best hit.

**Architecture:** `retriever.retrieve_relevant_chunks` already computes `dense_rankings` (from FAISS) and `sparse_rankings` (from BM25) independently before fusing them with RRF. This plan adds one small, pure helper — `_has_high_confidence_agreement(dense_rankings, sparse_rankings)` — that checks whether both rankings put the *same* document at rank 1, and uses it to add a third branch to the existing cross-encoder/RRF-fallback branching, ahead of the CrossEncoder branch. A new `config.CONFIDENCE_GATE_ENABLED` flag (default `True`) allows disabling the fast path without a code change if it's ever found to hurt result quality in practice.

**Tech Stack:** Python 3, `rank_bm25`, `faiss-cpu`, `pytest`.

**Spec:** `TODO.md` → Phase 1 → "CPU Inference & Latency Acceleration" → "Confidence-gated fast path for statutory / exact queries".

## Global Constraints

- The gate is purely score/ranking-based (dense rank 1 == sparse rank 1), not query-text pattern matching. The TODO's examples (`"WSU/2022/1"`, `"Section 17"`) describe *when this naturally fires* (exact-identifier queries that BM25's tokenizer already preserves verbatim, per `_tokenize_for_bm25`'s docstring), not a required precondition to hand-code — hardcoding circular-number regexes into the retrieval path would duplicate and drift from that tokenizer's own patterns.
- When the gate fires, results must still be sorted by `rrf_score` (not returned in raw dense/sparse order) — this is the same fallback sort already used when no cross-encoder is available, so the fast path reuses that exact sort, not a new one.
- `retrieve_relevant_chunks`'s existing public signature/behavior must not change for existing callers — the new `confidence_gate_enabled` parameter must default to `None` and fall back to `config.CONFIDENCE_GATE_ENABLED` (`getattr`-guarded, default `True`), matching the existing pattern for `top_n_initial`/`top_n_final`/`use_hybrid`.
- If `sparse_rankings` or `dense_rankings` is empty (hybrid disabled, BM25 tokenized to nothing, or dense search returned nothing), the gate must never fire — it degrades to existing behavior with zero risk of a crash or false positive.
- Run tests with the project's virtualenv interpreter: `.venv/Scripts/python.exe -m pytest tests/ -v`.

---

## File Structure

- **Modify `config.py`**: add `CONFIDENCE_GATE_ENABLED = True` immediately after the existing `RRF_K = 60` line (in the "Re-ranking & Hybrid Retrieval Model" section).
- **Modify `retriever.py`**: add `_has_high_confidence_agreement(dense_rankings, sparse_rankings)` after `_top_k_score_indices` (before `retrieve_relevant_chunks`); add a `confidence_gate_enabled=None` parameter to `retrieve_relevant_chunks` and a new branch in its re-ranking `if/elif/else`.
- **New `tests/test_retriever_confidence_gate.py`**: unit tests for the helper in isolation, plus end-to-end tests against `retrieve_relevant_chunks` using a small real `faiss.IndexFlatIP`, a fake embedding model (controllable query vector), and real `BM25Okapi` (via real text content) — proving the CrossEncoder is actually skipped/invoked based on real dense+sparse agreement/disagreement, not mocked internals.

---

### Task 1: Add `_has_high_confidence_agreement` and wire it into `retrieve_relevant_chunks`

**Files:**
- Modify: `config.py` (after line 37, `RRF_K = 60`)
- Modify: `retriever.py` (insert helper after `_top_k_score_indices`, i.e. after current line 198, before `def retrieve_relevant_chunks`; modify the function's signature at current line 201-204 and the re-ranking branch at current lines 319-341)
- Test: `tests/test_retriever_confidence_gate.py`

**Interfaces:**
- Produces: `retriever._has_high_confidence_agreement(dense_rankings: dict, sparse_rankings: dict) -> bool` — `True` only when both dicts are non-empty, share a doc_id whose value is `1` in both, and that shared doc_id is each dict's own rank-1 entry.
- Produces: `retrieve_relevant_chunks(..., confidence_gate_enabled=None)` — new keyword-only-by-convention parameter (positional is fine, added at the end to not disturb existing positional callers), defaulting to `getattr(config, "CONFIDENCE_GATE_ENABLED", True)`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retriever_confidence_gate.py -v`
Expected: the three pure-helper tests `FAIL` with `AttributeError: module 'retriever' has no attribute '_has_high_confidence_agreement'`; the end-to-end tests `FAIL` the same way (the function raises before ever getting to `cross_encoder.predict_calls`) or, if you reach that far, `test_cross_encoder_is_skipped_when_dense_and_sparse_agree_on_top_doc` fails on `cross_encoder.predict_calls == 0` because the gate doesn't exist yet and the CrossEncoder always runs.

- [ ] **Step 3: Add the config flag**

In `config.py`, immediately after the existing line `RRF_K = 60`:

```python
CONFIDENCE_GATE_ENABLED = True  # Skip CrossEncoder re-ranking when dense and sparse retrieval already agree on the top document
```

- [ ] **Step 4: Add the helper function to `retriever.py`**

Insert immediately after `_top_k_score_indices` (after current line 198, before `def retrieve_relevant_chunks`):

```python
def _has_high_confidence_agreement(dense_rankings, sparse_rankings):
    """Returns True when dense and sparse retrieval independently agree on
    the same single best-matching document (each ranks it #1) -- a strong
    signal that a CrossEncoder re-ranking pass would not change the outcome.
    Most often true for an exact statutory reference or circular number that
    both a literal BM25 token match and a semantic embedding recognize as
    the unique best hit.
    """
    if not dense_rankings or not sparse_rankings:
        return False
    top_dense_doc_id = min(dense_rankings, key=dense_rankings.get)
    top_sparse_doc_id = min(sparse_rankings, key=sparse_rankings.get)
    return (
        dense_rankings[top_dense_doc_id] == 1
        and sparse_rankings[top_sparse_doc_id] == 1
        and top_dense_doc_id == top_sparse_doc_id
    )
```

- [ ] **Step 5: Wire it into `retrieve_relevant_chunks`**

Change the function signature (current lines 201-204) from:
```python
def retrieve_relevant_chunks(query_text, faiss_index, all_indexed_texts, all_indexed_metadata, 
                             embedding_model, cross_encoder_model=None, 
                             top_n_initial=None, top_n_final=None,
                             use_hybrid=None, bm25_cache_path=None):
```
to:
```python
def retrieve_relevant_chunks(query_text, faiss_index, all_indexed_texts, all_indexed_metadata, 
                             embedding_model, cross_encoder_model=None, 
                             top_n_initial=None, top_n_final=None,
                             use_hybrid=None, bm25_cache_path=None,
                             confidence_gate_enabled=None):
```

Immediately after the existing early-default block (current lines 224-229, ending `use_hybrid = getattr(config, "USE_HYBRID_RETRIEVAL", True)`), add:
```python
    if confidence_gate_enabled is None:
        confidence_gate_enabled = getattr(config, "CONFIDENCE_GATE_ENABLED", True)
```

Change the re-ranking branch (current lines 319-341) from:
```python
        # -------------------------------------------------------------
        # 4. Cross-Encoder Deep Re-ranking
        # -------------------------------------------------------------
        if cross_encoder_model and retrieved_results:
            try:
                logger.info(f"Re-ranking {len(retrieved_results)} hybrid candidates with CrossEncoder...")
                cross_input = [[query_text, res["text"]] for res in retrieved_results]
                cross_scores = cross_encoder_model.predict(cross_input)

                for i, c_score in enumerate(cross_scores):
                    retrieved_results[i]["score"] = float(c_score)

                # Sort by CrossEncoder score descending
                retrieved_results.sort(key=lambda x: x['score'], reverse=True)
            except Exception as cross_encoder_error:
                logger.warning(
                    "Cross-encoder re-ranking failed: %s. Using RRF ranking.",
                    cross_encoder_error,
                )
                retrieved_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        else:
            # Sort by RRF score descending
            retrieved_results.sort(key=lambda x: x['rrf_score'], reverse=True)
```
to:
```python
        # -------------------------------------------------------------
        # 4. Cross-Encoder Deep Re-ranking (skipped on high-confidence agreement)
        # -------------------------------------------------------------
        if confidence_gate_enabled and _has_high_confidence_agreement(dense_rankings, sparse_rankings):
            logger.info(
                "Dense and sparse retrieval agree on the top document; skipping CrossEncoder re-ranking."
            )
            retrieved_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        elif cross_encoder_model and retrieved_results:
            try:
                logger.info(f"Re-ranking {len(retrieved_results)} hybrid candidates with CrossEncoder...")
                cross_input = [[query_text, res["text"]] for res in retrieved_results]
                cross_scores = cross_encoder_model.predict(cross_input)

                for i, c_score in enumerate(cross_scores):
                    retrieved_results[i]["score"] = float(c_score)

                # Sort by CrossEncoder score descending
                retrieved_results.sort(key=lambda x: x['score'], reverse=True)
            except Exception as cross_encoder_error:
                logger.warning(
                    "Cross-encoder re-ranking failed: %s. Using RRF ranking.",
                    cross_encoder_error,
                )
                retrieved_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        else:
            # Sort by RRF score descending
            retrieved_results.sort(key=lambda x: x['rrf_score'], reverse=True)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retriever_confidence_gate.py -v`
Expected: all 6 tests `PASS`.

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests pass except the pre-existing, unrelated `tests/test_pdf_utils.py::TestPDFUtils::test_compress_pdf` failure (PyMuPDF version-compat issue, not touched by this plan).

- [ ] **Step 8: Commit**

```bash
git add config.py retriever.py tests/test_retriever_confidence_gate.py
git commit -m "Add confidence-gated fast path to skip CrossEncoder on dense/sparse agreement"
```
