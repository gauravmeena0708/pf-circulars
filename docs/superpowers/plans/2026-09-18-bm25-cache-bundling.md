# BM25 Cache Bundling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the 45–60s first-query freeze caused by tokenizing 74,317 passages on container boot, by pre-building the compressed BM25 sparse-index cache during ingestion and bundling it via Git LFS so it's already on disk when the app starts.

**Architecture:** `retriever.py` already contains full persist/load machinery for a gzip-compressed BM25 state cache (`_persist_bm25` / `_load_persisted_bm25`), keyed by a content fingerprint of the corpus — but nothing calls the persist path except a runtime rebuild triggered by the first user query. This plan adds one public entry point (`warm_bm25_cache`), wires it into `vector_indexer.save_faiss_index` so every ingestion script produces a fresh cache automatically, backfills the cache for the index already committed to the repo, and confirms it ships correctly under the existing Git LFS pattern.

**Tech Stack:** Python 3, `rank_bm25.BM25Okapi`, `gzip`/`json` for the cache file, `faiss-cpu`, `git-lfs` 3.7.1 (already installed on this machine), `pytest`.

**Spec:** `TODO.md` → Phase 1 → "Storage & Memory Optimization" → "Precompute and bundle the compressed BM25 index in Git LFS"

## Global Constraints

- The persisted cache format is fixed by existing code and must not change: `_BM25_CACHE_VERSION = 1`, top-level keys `version`/`corpus_fingerprint`/`document_count`/`bm25_state`, and `bm25_state` containing exactly `_BM25_STATE_FIELDS` (`k1`, `b`, `epsilon`, `corpus_size`, `avgdl`, `doc_freqs`, `idf`, `doc_len`, `average_idf`). New code must reuse `_load_persisted_bm25`/`_persist_bm25`, not reimplement serialization.
- `save_faiss_index(index, texts_for_retrieval, metadata_for_retrieval, index_dir, index_name=config.DEFAULT_INDEX_NAME)`'s public signature must not change — four ingestion scripts (`main.py`, `update_indexer.py`, `index_manuals.py`, `import_pf_circular_index.py`) call it as-is.
- Cache file path convention: `os.path.join(index_dir, f"{index_name}.bm25.json.gz")` — matches what `app.py:423` already computes for the read side.
- All new/modified Python files use the project's existing `logger`/`config.LOG_LEVEL` logging pattern, not `print`.
- Run tests with the project's virtualenv interpreter: `.venv/Scripts/python.exe -m pytest tests/ -v` (the `.venv` has `rank_bm25`, `faiss-cpu`, `sentence-transformers` already installed; the `miniconda3` `python`/`pip` on PATH do not).

---

## File Structure

- **Modify `retriever.py`**: add `warm_bm25_cache(all_indexed_texts, cache_path)`, a thin public wrapper around the existing private `_get_or_build_bm25_index`, placed directly after it (after current line 165).
- **Modify `vector_indexer.py`**: `save_faiss_index` gains one extra step — after writing `<index_name>.texts.json`, call `retriever.warm_bm25_cache(...)` so the sparse cache is always regenerated alongside the dense index and raw texts.
- **New `tests/test_retriever_bm25_cache.py`**: unit tests for `warm_bm25_cache` — file is created, and a second call reuses the persisted cache instead of rebuilding.
- **New `tests/test_vector_indexer_bm25.py`**: unit test confirming `save_faiss_index` writes the `.bm25.json.gz` file as a side effect.
- **No `.gitattributes` change needed** — `vector_store/** filter=lfs diff=lfs merge=lfs -text` already matches `vector_store/data_index/faiss_index.bm25.json.gz`; Task 3 verifies this instead of editing it.

---

### Task 1: Add `retriever.warm_bm25_cache` public entry point

**Files:**
- Modify: `retriever.py` (insert after line 165, immediately following `_get_or_build_bm25_index`)
- Test: `tests/test_retriever_bm25_cache.py`

**Interfaces:**
- Consumes: existing private helpers `_get_or_build_bm25_index(all_indexed_texts, cache_path)`, `_BM25_CACHE` module dict, `BM25Okapi` — all already defined in `retriever.py`.
- Produces: `warm_bm25_cache(all_indexed_texts: list[str], cache_path: str) -> None`. Guarantees that after it returns, `cache_path` exists on disk and its `corpus_fingerprint`/`document_count` match `all_indexed_texts`. Task 2 depends on this exact name and signature.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retriever_bm25_cache.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retriever_bm25_cache.py -v`
Expected: both tests `FAIL` with `AttributeError: module 'retriever' has no attribute 'warm_bm25_cache'`.

- [ ] **Step 3: Implement `warm_bm25_cache`**

In `retriever.py`, insert immediately after the existing `_get_or_build_bm25_index` function (after line 165):

```python
def warm_bm25_cache(all_indexed_texts, cache_path):
    """Builds the BM25 index for all_indexed_texts and ensures cache_path
    holds a persisted gzip cache matching that corpus.

    Ingestion scripts call this after writing a new index so the sparse
    index ships pre-built instead of being rebuilt on the first query.
    """
    _get_or_build_bm25_index(all_indexed_texts, cache_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retriever_bm25_cache.py -v`
Expected: both tests `PASS`.

- [ ] **Step 5: Commit**

```bash
git add retriever.py tests/test_retriever_bm25_cache.py
git commit -m "Add warm_bm25_cache entry point for pre-building the BM25 sparse index cache"
```

---

### Task 2: Auto-regenerate the BM25 cache whenever the FAISS index is saved

**Files:**
- Modify: `vector_indexer.py` (`save_faiss_index`, currently lines 254–268; add `import retriever` near the top)
- Test: `tests/test_vector_indexer_bm25.py`

**Interfaces:**
- Consumes: `retriever.warm_bm25_cache(all_indexed_texts, cache_path)` from Task 1.
- Produces: `save_faiss_index(...)` now also writes `<index_dir>/<index_name>.bm25.json.gz` as a side effect. Public signature and return value (`None`) are unchanged, so `main.py`, `update_indexer.py`, `index_manuals.py`, and `import_pf_circular_index.py` require no changes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vector_indexer_bm25.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vector_indexer_bm25.py -v`
Expected: `FAIL` — `faiss_index.bm25.json.gz` does not exist (assertion on `os.path.isfile`).

- [ ] **Step 3: Wire it into `save_faiss_index`**

Add near the top of `vector_indexer.py` (with the other local imports, after `import config` on line 9):

```python
import retriever
```

In `save_faiss_index`, after the existing `logger.info(f"Texts and metadata saved to {texts_path}")` line (current line 266), add:

```python
        bm25_cache_path = os.path.join(index_dir, f"{index_name}.bm25.json.gz")
        retriever.warm_bm25_cache(texts_for_retrieval, bm25_cache_path)
        logger.info(f"BM25 sparse index cache refreshed at {bm25_cache_path}")
```

This stays inside the function's existing `try` block, so a BM25-cache failure is caught and logged by the existing `except Exception as e:` handler rather than aborting the FAISS/texts save that already succeeded.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vector_indexer_bm25.py -v`
Expected: `PASS`.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests pass, including Task 1's two tests.

- [ ] **Step 6: Commit**

```bash
git add vector_indexer.py tests/test_vector_indexer_bm25.py
git commit -m "Regenerate the BM25 sparse cache automatically whenever the FAISS index is saved"
```

---

### Task 3: Backfill the cache for the current index and confirm Git LFS bundling

**Files:** none created by code; operates on `vector_store/data_index/` and repo-level git/LFS state.

**Interfaces:**
- Consumes: `vector_indexer.load_faiss_index` (existing) and `retriever.warm_bm25_cache` (Task 1) to backfill the cache for the index already committed to the repo, without re-running any ingestion script.

- [ ] **Step 1: Initialize Git LFS hooks for this repository**

The filter driver (`filter.lfs.clean`/`smudge`/`process`) is already configured globally, but this repo's `.git/hooks/` has no LFS pre-push/post-checkout/post-merge hooks yet.

Run: `git lfs install`
Expected: `Updated git hooks.` / `Git LFS initialized.`

- [ ] **Step 2: Confirm the existing `.gitattributes` pattern already covers the new file**

Run: `git check-attr filter -- vector_store/data_index/faiss_index.bm25.json.gz`
Expected: `vector_store/data_index/faiss_index.bm25.json.gz: filter: lfs` (no `.gitattributes` edit needed — `vector_store/**` already matches).

- [ ] **Step 3: Generate the cache file for the index already on disk**

Run:
```bash
.venv/Scripts/python.exe -c "
import os
import config
import vector_indexer
import retriever

index_dir = os.path.join(config.DEFAULT_INDEX_DIR, 'data_index')
index, texts, metadata = vector_indexer.load_faiss_index(index_dir, index_name=config.DEFAULT_INDEX_NAME)
cache_path = os.path.join(index_dir, f'{config.DEFAULT_INDEX_NAME}.bm25.json.gz')
retriever.warm_bm25_cache(texts, cache_path)
print('done:', os.path.getsize(cache_path), 'bytes for', len(texts), 'passages')
"
```
Expected: prints `done: <15-20MB in bytes> bytes for 74317 passages`. This takes roughly 45–60s — the one-time tokenization cost, now paid here instead of on a user's first request.

- [ ] **Step 4: Confirm the file will be pushed as an LFS object, not a regular blob**

Run: `git lfs status`
Expected: `vector_store/data_index/faiss_index.bm25.json.gz` listed as a new file to be committed, tracked via the LFS filter (not under "Git blobs").

- [ ] **Step 5: Run the full test suite as a final regression check**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 6: Commit the generated cache file**

```bash
git add vector_store/data_index/faiss_index.bm25.json.gz
git commit -m "Bundle precomputed BM25 sparse index cache to avoid cold-start tokenization"
```

Do not push — confirm with the user before pushing, since this updates a large binary asset on a shared branch.

---

## Follow-up plans (not covered here)

Per the spec, Phase 1 also includes: SQLite migration for `passages.db` (Storage & Memory §2 — deliberately excluded from this plan since it's a much larger, higher-risk change to the on-disk format touching `retriever.py`, `vector_indexer.py`, `app.py`, and read paths in four ingestion scripts), ONNX INT8 cross-encoder quantization, the confidence-gated fast path, FAISS metric alignment across `update_indexer.py`/`index_manuals.py`/`import_pf_circular_index.py`, and model pre-warming on Space startup. Each should get its own plan once this one lands.
