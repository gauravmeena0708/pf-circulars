# FAISS Metric Alignment & Startup Pre-Warm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize all ingestion scripts on the same inner-product FAISS metric with L2-normalized embeddings that `vector_indexer.py`'s canonical `create_faiss_index` already uses, and eliminate the first-query cold start for embedding-model loading by warming the search pipeline once at Space startup instead of on a user's first request.

**Architecture:** Three ingestion scripts (`update_indexer.py`, `index_manuals.py`, `import_pf_circular_index.py`) each inline their own "create a fresh index" fallback using `faiss.IndexFlatL2` (wrong metric — `vector_indexer.py`'s canonical path uses `IndexFlatIP`) and each call `.encode(...)` without `normalize_embeddings=True` (so freshly-embedded vectors going into `add_with_ids` are never unit-normalized). This plan extracts two small, shared, unit-tested helpers into `vector_indexer.py` — `create_empty_faiss_index` and `encode_normalized` — and switches all three scripts to use them, so the three ingestion paths can no longer independently drift from the canonical one. Separately, `app.py` currently loads the embedding model and FAISS-searches for the first time only when a user submits a query in Tab 1; this plan adds a one-time cache-resource-backed pre-warm call right after the index loads at module scope, so that cost is paid once at container boot instead of on the first real user's request.

**Tech Stack:** Python 3, `faiss-cpu`, `sentence-transformers`, `streamlit` (`st.cache_resource`), `pytest`.

**Spec:** `TODO.md` → Phase 1 → "CPU Inference & Latency Acceleration" → "Pre-warm models on Space startup"; and → "Code Integrity & Ingestion Consistency" → "Align FAISS index metric in `update_indexer.py`".

## Global Constraints

- `save_faiss_index`/`load_faiss_index` public signatures (already used by 4 ingestion scripts and `app.py`) must not change.
- The currently-committed production index (`vector_store/data_index/faiss_index.index`, 74,317 vectors) was empirically verified to already hold L2-normalized vectors (`np.linalg.norm` ≈ 1.0 on sampled ids) but is saved with `metric_type == faiss.METRIC_L2` (confirmed via `faiss.read_index(...).index.metric_type == 1`), not `METRIC_INNER_PRODUCT`. Loading an existing index preserves whatever metric it was saved with — this plan's fixes only change what happens on a **from-scratch** index build (no existing index found) and on **appending new vectors**; they do not retroactively change the type of the already-saved index in place, and this plan does not rebuild it. The `normalize_embeddings=True` fix is the higher-value half of this plan: it's what stops any *future* incremental append (via any of the three scripts) from mixing un-normalized new vectors into the currently-consistent (if metric-mislabeled) existing index.
- Run tests with the project's virtualenv interpreter: `.venv/Scripts/python.exe -m pytest tests/ -v` (has `faiss-cpu`, `sentence-transformers`, `pytest` already installed).
- All new/modified Python files use the project's existing `logger`/`config.LOG_LEVEL` logging pattern, not `print`.
- Don't touch `vector_indexer.py`'s existing `create_faiss_index` function body — it is already correct (uses `IndexFlatIP` + `normalize_embeddings=True`) and is out of scope; only add new helper functions alongside it.

---

## File Structure

- **Modify `vector_indexer.py`**: add two new functions, `create_empty_faiss_index(dimension)` and `encode_normalized(embedding_model, texts, **kwargs)`, placed after the existing `create_faiss_index` function (after current line 251, before `save_faiss_index`).
- **New `tests/test_vector_indexer_helpers.py`**: unit tests for both new helpers, using a small fake embedding-model object (no real `SentenceTransformer` download needed).
- **Modify `update_indexer.py`, `index_manuals.py`, `import_pf_circular_index.py`**: swap each script's inline `faiss.IndexFlatL2(...)` fallback and un-normalized `.encode(...)` call for the two new shared helpers.
- **New `tests/test_ingestion_scripts_use_shared_helpers.py`**: a lightweight regression guard — parametrized over the three scripts, asserts each module's source no longer contains a literal `IndexFlatL2(` call and does call both new helpers. This is deliberately a source-inspection test, not a full pipeline test: these ingestion functions are large, side-effecting CLI entry points (network fetch, OCR, embedding-model download) that aren't unit-testable without disproportionate mocking, and none of the three files have any existing test coverage to build on.
- **Modify `app.py`**: add a `prewarm_search_pipeline` function (mirrors the existing `@st.cache_resource`-backed pattern used by `load_embedding_model`/`load_cached_faiss_index`) and call it once, right after the FAISS index load block (after current line 431), before the tabs are defined. No new source file — Streamlit's script-per-interaction execution model and the project's existing test suite (no `app.py` orchestration is unit-tested today — only the logic modules it imports are) make this task verified functionally (Task 3's steps), not via pytest.

---

### Task 1: Add shared `create_empty_faiss_index` and `encode_normalized` helpers to `vector_indexer.py`

**Files:**
- Modify: `vector_indexer.py` (insert after line 251, i.e. immediately after the existing `create_faiss_index` function and before `save_faiss_index`)
- Test: `tests/test_vector_indexer_helpers.py`

**Interfaces:**
- Produces: `create_empty_faiss_index(dimension: int) -> faiss.IndexIDMap` — returns an ID-mapped, empty inner-product FAISS index (`faiss.IndexIDMap(faiss.IndexFlatIP(dimension))`).
- Produces: `encode_normalized(embedding_model, texts, **kwargs)` — calls `embedding_model.encode(texts, **kwargs)` with `normalize_embeddings` always forced to `True` (regardless of what the caller passes) and `convert_to_tensor` defaulted to `False` if the caller didn't supply it. Task 2 depends on both exact names and signatures.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vector_indexer_helpers.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vector_indexer_helpers.py -v`
Expected: all three tests `FAIL` with `AttributeError: module 'vector_indexer' has no attribute 'create_empty_faiss_index'` (or `encode_normalized`).

- [ ] **Step 3: Implement the two helpers**

In `vector_indexer.py`, insert immediately after the existing `create_faiss_index` function (after line 251, before `def save_faiss_index(...)`):

```python
def create_empty_faiss_index(dimension):
    """Creates an empty, ID-mapped FAISS index using the canonical inner-product
    metric for cosine similarity on L2-normalized vectors.

    Shared by every ingestion script's "no existing index found" fallback so
    they can't independently drift onto a different (wrong) metric type.
    """
    return faiss.IndexIDMap(faiss.IndexFlatIP(dimension))


def encode_normalized(embedding_model, texts, **kwargs):
    """Encodes texts with normalize_embeddings always forced True.

    Shared by every ingestion script's embedding step so a caller can never
    accidentally add un-normalized vectors into an inner-product index.
    """
    kwargs.setdefault("convert_to_tensor", False)
    kwargs["normalize_embeddings"] = True
    return embedding_model.encode(texts, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vector_indexer_helpers.py -v`
Expected: all three tests `PASS`.

- [ ] **Step 5: Commit**

```bash
git add vector_indexer.py tests/test_vector_indexer_helpers.py
git commit -m "Add shared create_empty_faiss_index and encode_normalized helpers"
```

---

### Task 2: Switch the three ingestion scripts to the shared helpers

**Files:**
- Modify: `update_indexer.py` (import line 15-20; fallback-index creation at line 224; encode call at line 344)
- Modify: `index_manuals.py` (import line 28; fallback-index creation at line 137; encode call at lines 213-218)
- Modify: `import_pf_circular_index.py` (import line 23; fallback-index creation at line 147; encode call at lines 259-264)
- Test: `tests/test_ingestion_scripts_use_shared_helpers.py`

**Interfaces:**
- Consumes: `vector_indexer.create_empty_faiss_index(dimension)` and `vector_indexer.encode_normalized(embedding_model, texts, **kwargs)` from Task 1.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingestion_scripts_use_shared_helpers.py
import importlib
import inspect

import pytest

MODULE_NAMES = ["update_indexer", "index_manuals", "import_pf_circular_index"]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_script_no_longer_creates_a_raw_indexflatl2(module_name):
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)

    assert "IndexFlatL2(" not in source, (
        f"{module_name} still creates a raw IndexFlatL2 index directly; "
        "it should call vector_indexer.create_empty_faiss_index instead"
    )


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_script_uses_shared_helpers(module_name):
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)

    assert "create_empty_faiss_index(" in source
    assert "encode_normalized(" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingestion_scripts_use_shared_helpers.py -v`
Expected: all 6 parametrized cases `FAIL` — `test_script_no_longer_creates_a_raw_indexflatl2` fails because each module still contains `IndexFlatL2(`, and `test_script_uses_shared_helpers` fails because none of them call the new helpers yet.

- [ ] **Step 3: Update `update_indexer.py`**

Change the import block (lines 15-20) from:
```python
from vector_indexer import (
    group_extracted_content_to_blocks,
    merge_spanning_table_blocks,
    load_faiss_index,
    save_faiss_index
)
```
to:
```python
from vector_indexer import (
    group_extracted_content_to_blocks,
    merge_spanning_table_blocks,
    load_faiss_index,
    save_faiss_index,
    create_empty_faiss_index,
    encode_normalized,
)
```

Change line 224 from:
```python
        faiss_index = faiss.IndexFlatL2(sbert_model.get_sentence_embedding_dimension())
        faiss_index = faiss.IndexIDMap(faiss_index)
```
to:
```python
        faiss_index = create_empty_faiss_index(sbert_model.get_sentence_embedding_dimension())
```

Change line 344 from:
```python
        new_embeddings = sbert_model.encode(new_texts_for_embedding, convert_to_tensor=False, show_progress_bar=True)
```
to:
```python
        new_embeddings = encode_normalized(sbert_model, new_texts_for_embedding, show_progress_bar=True)
```

- [ ] **Step 4: Update `index_manuals.py`**

Change line 28 from:
```python
from vector_indexer import load_faiss_index, save_faiss_index
```
to:
```python
from vector_indexer import load_faiss_index, save_faiss_index, create_empty_faiss_index, encode_normalized
```

Change line 137 from:
```python
        base_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index = faiss.IndexIDMap(base_index)
```
to:
```python
        faiss_index = create_empty_faiss_index(embedding_dim)
```

Change lines 213-218 from:
```python
        embeddings = embedding_model.encode(
            chunk_slice,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=False
        )
```
to:
```python
        embeddings = encode_normalized(
            embedding_model,
            chunk_slice,
            batch_size=batch_size,
            show_progress_bar=False
        )
```

- [ ] **Step 5: Update `import_pf_circular_index.py`**

Change line 23 from:
```python
from vector_indexer import load_faiss_index, save_faiss_index
```
to:
```python
from vector_indexer import load_faiss_index, save_faiss_index, create_empty_faiss_index, encode_normalized
```

Change line 147 from:
```python
        base_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index = faiss.IndexIDMap(base_index)
```
to:
```python
        faiss_index = create_empty_faiss_index(embedding_dim)
```

Change lines 259-264 from:
```python
        embeddings = embedding_model.encode(
            chunk_slice,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=False
        )
```
to:
```python
        embeddings = encode_normalized(
            embedding_model,
            chunk_slice,
            batch_size=batch_size,
            show_progress_bar=False
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingestion_scripts_use_shared_helpers.py -v`
Expected: all 6 parametrized cases `PASS`.

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests pass except the pre-existing, unrelated `tests/test_pdf_utils.py::TestPDFUtils::test_compress_pdf` failure (PyMuPDF version-compat issue, not touched by this plan).

- [ ] **Step 8: Commit**

```bash
git add update_indexer.py index_manuals.py import_pf_circular_index.py tests/test_ingestion_scripts_use_shared_helpers.py
git commit -m "Standardize ingestion scripts on shared inner-product index + normalized-embedding helpers"
```

---

### Task 3: Pre-warm the search pipeline once at Space startup

**Files:**
- Modify: `app.py` (insert after the existing index-load block, i.e. after current line 431, before the `loaded_retrieval_signature` block at line 433 — or immediately after it, either position is before the tabs are defined at line 446)

**Interfaces:**
- Consumes: `retrieve_relevant_chunks` (already imported in `app.py` at line 32), `load_embedding_model` (already defined in `app.py` at line 85), `faiss_index`/`indexed_texts`/`indexed_metadata`/`index_signature`/`bm25_cache_path` (already computed in `app.py` at lines 422-431).

- [ ] **Step 1: Add the pre-warm function**

In `app.py`, insert this function after `load_cached_faiss_index` (after the existing block ending at current line 184, before `@st.cache_data(...) def retrieve_cached_chunks(...)` at line 187):

```python
@st.cache_resource(max_entries=1)
def prewarm_search_pipeline(
    index_signature,
    embedding_model_name,
    _faiss_index,
    _indexed_texts,
    _indexed_metadata,
    _embedding_model,
    _bm25_cache_path,
):
    """Runs one throwaway retrieval at process startup so the first real user
    query doesn't also pay for first-time embedding-model inference and BM25
    setup. Cached so this only runs once per server process, not on every
    Streamlit script rerun."""
    _ = (index_signature, embedding_model_name)
    try:
        retrieve_relevant_chunks(
            "a",
            _faiss_index,
            _indexed_texts,
            _indexed_metadata,
            _embedding_model,
            top_n_final=1,
            bm25_cache_path=_bm25_cache_path,
        )
        logger.info("Pre-warmed embedding model, FAISS index, and BM25 cache.")
    except Exception as e:
        logger.warning(f"Pre-warm failed (non-fatal, first real query will pay the cost): {e}", exc_info=True)
    return True
```

- [ ] **Step 2: Call it right after the index loads**

Find this block (current lines 421-431):
```python
# --- Load the persisted index; query models are loaded only when Tab 1 searches ---
index_dir = os.path.join(config.DEFAULT_INDEX_DIR, "data_index")
bm25_cache_path = os.path.join(index_dir, f"{config.DEFAULT_INDEX_NAME}.bm25.json.gz")
index_signature = get_index_file_signature(index_dir, config.DEFAULT_INDEX_NAME)
faiss_index, indexed_texts, indexed_metadata = load_cached_faiss_index(
    index_dir,
    config.DEFAULT_INDEX_NAME,
    index_signature,
    config.EMBEDDING_MODEL_NAME,
    None,
)
```

Replace the comment and add the pre-warm call immediately after the `load_cached_faiss_index(...)` call:
```python
# --- Load the persisted index and pre-warm the search pipeline at startup ---
index_dir = os.path.join(config.DEFAULT_INDEX_DIR, "data_index")
bm25_cache_path = os.path.join(index_dir, f"{config.DEFAULT_INDEX_NAME}.bm25.json.gz")
index_signature = get_index_file_signature(index_dir, config.DEFAULT_INDEX_NAME)
faiss_index, indexed_texts, indexed_metadata = load_cached_faiss_index(
    index_dir,
    config.DEFAULT_INDEX_NAME,
    index_signature,
    config.EMBEDDING_MODEL_NAME,
    None,
)

if faiss_index and indexed_texts and indexed_metadata:
    _prewarm_embedding_model = load_embedding_model(config.EMBEDDING_MODEL_NAME, config.EMBEDDING_DEVICE)
    if _prewarm_embedding_model:
        prewarm_search_pipeline(
            index_signature,
            config.EMBEDDING_MODEL_NAME,
            faiss_index,
            indexed_texts,
            indexed_metadata,
            _prewarm_embedding_model,
            bm25_cache_path,
        )
```

- [ ] **Step 3: Verify functionally (no pytest coverage possible for this step)**

This codebase has no existing unit tests for `app.py`'s Streamlit orchestration (only the logic modules it imports — `retriever.py`, `vector_indexer.py`, etc. — have tests), and the other four `@st.cache_resource` functions already in `app.py` are untested the same way. Verify by actually running the app:

Run (from the repo root, using the project venv; log file is repo-relative, not `/tmp`, to avoid platform assumptions):
```bash
"/c/Users/IT/Documents/GitHub/pf-circulars/.venv/Scripts/python.exe" -m streamlit run app.py --server.headless true --server.port 8765 > streamlit_prewarm_check.log 2>&1 &
```
Wait a few seconds for it to bind, then fetch the root page once to trigger the first script execution:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765
```
Expected: HTTP 200, and the log file contains exactly one `Pre-warmed embedding model, FAISS index, and BM25 cache.` line — appearing during startup (before any simulated user interaction), not after. Also confirm the log contains exactly one `Loading embedding model: ...` line total (from inside `load_embedding_model`'s cache-miss branch) — proving the model was loaded once, at pre-warm time, not lazily on a later "first search."

Stop the server afterward:
```bash
kill %1
rm -f streamlit_prewarm_check.log
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "Pre-warm embedding model, FAISS index, and BM25 cache at Space startup"
```
