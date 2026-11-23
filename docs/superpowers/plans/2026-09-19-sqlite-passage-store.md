# SQLite Passage Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate `app.py`'s ~500-800MB RAM cost and multi-second startup cost from loading `faiss_index.texts.json` (105MB) into a Python list, by replacing it with a lazy SQLite-backed passage store — without changing `retriever.py` or any of the four ingestion scripts.

**Architecture:** `vector_indexer.save_faiss_index`/`load_faiss_index` keep their exact existing public signatures; internally they switch from writing/reading `{index_name}.texts.json` to writing/reading `{index_name}.passages.db` (SQLite, stdlib `sqlite3`, no new dependency). A new `passage_store.py` module provides `PassageStore` plus two list-like proxy views (`.texts`, `.metadata`) supporting `__len__`/`__getitem__`/`__iter__` — the only three operations `retriever.retrieve_relevant_chunks` ever performs on its `all_indexed_texts`/`all_indexed_metadata` arguments. `app.py`'s `load_cached_faiss_index` switches to a new `vector_indexer.load_faiss_binary_index` (FAISS only) plus `passage_store.open_passage_store` (lazy SQLite) instead of the old full-list-returning `load_faiss_index`. Ingestion scripts keep calling `load_faiss_index`/`save_faiss_index` exactly as today, unaware the on-disk format changed.

**Tech Stack:** Python 3, `sqlite3` (stdlib), `faiss-cpu`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-19-sqlite-passage-store-design.md`

## Global Constraints

- `retriever.py` must not be touched — zero lines changed. This is the plan's central claim and is proven, not assumed, by Task 1's cross-check test.
- None of the four ingestion scripts (`main.py`, `update_indexer.py`, `index_manuals.py`, `import_pf_circular_index.py`) may be touched.
- `vector_indexer.save_faiss_index`/`load_faiss_index`'s public signatures and return contracts (`(index, texts_list, metadata_list)` for `load_faiss_index`; `None` return for both on failure) must not change.
- No new third-party dependency — `sqlite3` is Python stdlib.
- `PassageStore`'s SQLite connection is opened with `check_same_thread=False` (safe: runtime access is read-only, shared across a Streamlit process's sessions/threads via `@st.cache_resource`).
- `open_passage_store` and `load_faiss_binary_index` return `None` on a missing file — never raise — matching `load_faiss_index`'s existing "not found" behavior that `app.py` already handles.
- Run tests with the project's virtualenv interpreter: `.venv/Scripts/python.exe -m pytest tests/ -v`.

---

## File Structure

- **New `passage_store.py`**: `PassageStore` class, `_PassageTextsView`/`_PassageMetadataView` proxy classes, `open_passage_store(index_dir, index_name=...)` function.
- **New `tests/test_passage_store.py`**: unit tests for the module against a small hand-built real SQLite db, plus the cross-check test proving `retrieve_relevant_chunks` behaves identically with lists vs. `PassageStore` views.
- **Modify `vector_indexer.py`**: extract `load_faiss_binary_index` from `load_faiss_index` (current lines 297-320); rewrite `save_faiss_index`'s (current lines 276-294) and `load_faiss_index`'s passage-storage internals to use SQLite instead of JSON. Add `import sqlite3`.
- **New `tests/test_vector_indexer_passages.py`**: round-trip integration tests for `save_faiss_index`/`load_faiss_index`/`load_faiss_binary_index` through the new schema.
- **Modify `app.py`**: `get_index_file_signature` (current lines 155-165) and `load_cached_faiss_index` (current lines 168-184); add two imports.
- **Migration** (no new file): one-time backfill of `vector_store/data_index/faiss_index.passages.db` from the currently-committed `faiss_index.texts.json`, followed by deleting `faiss_index.texts.json`.

---

### Task 1: Add `passage_store.py` and prove `retriever.py` needs zero changes

**Files:**
- Create: `passage_store.py`
- Test: `tests/test_passage_store.py`

**Interfaces:**
- Produces: `passage_store.open_passage_store(index_dir, index_name=config.DEFAULT_INDEX_NAME) -> PassageStore | None`.
- Produces: `PassageStore` with `.texts` and `.metadata` attributes, each supporting `__len__`, `__getitem__(doc_id)` (raises `IndexError` for an unknown id), and `__iter__` (streams in ascending id order).
- Task 3 depends on `open_passage_store`'s exact name/signature and the `.texts`/`.metadata` attribute names.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_passage_store.py -v`
Expected: all 5 tests `FAIL` — the first four with `ModuleNotFoundError: No module named 'passage_store'`, the last one the same way.

- [ ] **Step 3: Implement `passage_store.py`**

```python
# passage_store.py

import json
import logging
import os
import sqlite3

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


class _PassageTextsView:
    """List-like, read-only view over a PassageStore's text column."""

    def __init__(self, store):
        self._store = store

    def __len__(self):
        return len(self._store)

    def __getitem__(self, doc_id):
        return self._store.fetch_text(doc_id)

    def __iter__(self):
        return self._store.iter_texts()


class _PassageMetadataView:
    """List-like, read-only view over a PassageStore's metadata column."""

    def __init__(self, store):
        self._store = store

    def __len__(self):
        return len(self._store)

    def __getitem__(self, doc_id):
        return self._store.fetch_metadata(doc_id)

    def __iter__(self):
        return self._store.iter_metadata()


class PassageStore:
    """Lazy, low-memory read access to a passages SQLite database.

    Exposes .texts and .metadata, two list-like views (__len__,
    __getitem__, __iter__) that retriever.retrieve_relevant_chunks can use
    exactly as it would a plain list, without loading the whole corpus
    into memory. Intended for app.py's always-running process; ingestion
    scripts continue to use vector_indexer.load_faiss_index's full
    in-memory lists instead.
    """

    def __init__(self, db_path):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._total = self._conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        self.texts = _PassageTextsView(self)
        self.metadata = _PassageMetadataView(self)

    def __len__(self):
        return self._total

    def fetch_text(self, doc_id):
        row = self._conn.execute(
            "SELECT text FROM passages WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            raise IndexError(f"No passage with id {doc_id}")
        return row[0]

    def fetch_metadata(self, doc_id):
        row = self._conn.execute(
            "SELECT metadata FROM passages WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            raise IndexError(f"No passage with id {doc_id}")
        return json.loads(row[0])

    def iter_texts(self):
        cursor = self._conn.execute("SELECT text FROM passages ORDER BY id")
        for (text,) in cursor:
            yield text

    def iter_metadata(self):
        cursor = self._conn.execute("SELECT metadata FROM passages ORDER BY id")
        for (metadata_json,) in cursor:
            yield json.loads(metadata_json)


def open_passage_store(index_dir, index_name=config.DEFAULT_INDEX_NAME):
    """Opens the passages SQLite database for index_name in index_dir.

    Returns None (never raises) if the database file doesn't exist, so
    callers can use the same "index not found" handling they already use
    for a missing FAISS index.
    """
    db_path = os.path.join(index_dir, f"{index_name}.passages.db")
    if not os.path.isfile(db_path):
        logger.warning(f"Passage store '{db_path}' not found.")
        return None
    try:
        return PassageStore(db_path)
    except sqlite3.Error as error:
        logger.error(f"Could not open passage store '{db_path}': {error}", exc_info=True)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_passage_store.py -v`
Expected: all 5 tests `PASS`.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests pass except the pre-existing, unrelated `tests/test_pdf_utils.py::TestPDFUtils::test_compress_pdf` failure (PyMuPDF version-compat issue, not touched by this plan).

- [ ] **Step 6: Commit**

```bash
git add passage_store.py tests/test_passage_store.py
git commit -m "Add lazy SQLite-backed passage store module"
```

---

### Task 2: Switch `vector_indexer.py`'s passage storage from JSON to SQLite

**Files:**
- Modify: `vector_indexer.py` (add `import sqlite3`; extract `load_faiss_binary_index` from `load_faiss_index`, current lines 297-320; rewrite `save_faiss_index`'s passage-storage internals, current lines 276-294)
- Test: `tests/test_vector_indexer_passages.py`

**Interfaces:**
- Consumes: nothing from Task 1 (this task only touches `vector_indexer.py`).
- Produces: `vector_indexer.load_faiss_binary_index(index_dir, embedding_model_for_dim_check=None, index_name=config.DEFAULT_INDEX_NAME) -> faiss.Index | None`. Task 3 depends on this exact name/signature.
- `load_faiss_index`'s and `save_faiss_index`'s existing public signatures are unchanged (see Global Constraints).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vector_indexer_passages.py -v`
Expected: `test_save_and_load_round_trip_through_sqlite` and `test_load_faiss_binary_index_works_without_passages_db` `FAIL` (`faiss_index.passages.db` doesn't exist; `AttributeError: module 'vector_indexer' has no attribute 'load_faiss_binary_index'`). `test_load_faiss_index_returns_none_when_index_missing` may already `PASS` against the current code (this is fine — it locks in existing behavior that must survive the refactor).

- [ ] **Step 3: Add the `sqlite3` import**

In `vector_indexer.py`, add to the top-of-file imports (after `import json` on line 4):
```python
import sqlite3
```

- [ ] **Step 4: Extract `load_faiss_binary_index` and rewrite `load_faiss_index`**

Replace the current `load_faiss_index` function (current lines 297-320):
```python
def load_faiss_index(index_dir, embedding_model_for_dim_check=None, index_name=config.DEFAULT_INDEX_NAME):
    # Loads the FAISS index, corresponding texts, and metadata from disk.
    index_path = os.path.join(index_dir, f"{index_name}.index")
    texts_path = os.path.join(index_dir, f"{index_name}.texts.json")
    if not os.path.exists(index_path) or not os.path.exists(texts_path):
        logger.warning(f"Index file '{index_path}' or texts file '{texts_path}' not found.")
        return None, None, None
    try:
        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        if embedding_model_for_dim_check:
            expected_dim = embedding_model_for_dim_check.get_sentence_embedding_dimension()
            if index.d != expected_dim:
                logger.error(f"Loaded index dimension ({index.d}) does not match embedding model dimension ({expected_dim}).")
                return None, None, None
        with open(texts_path, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
        texts_for_retrieval = retrieval_data.get("texts", [])
        metadata_for_retrieval = retrieval_data.get("metadata", [])
        logger.info(f"FAISS index and {len(texts_for_retrieval)} text blocks with metadata loaded successfully.")
        return index, texts_for_retrieval, metadata_for_retrieval
    except Exception as e:
        logger.error(f"Error loading FAISS index or associated data: {e}", exc_info=True)
        return None, None, None
```
with:
```python
def load_faiss_binary_index(index_dir, embedding_model_for_dim_check=None, index_name=config.DEFAULT_INDEX_NAME):
    # Loads just the FAISS binary index from disk, without touching passage text/metadata.
    index_path = os.path.join(index_dir, f"{index_name}.index")
    if not os.path.exists(index_path):
        logger.warning(f"Index file '{index_path}' not found.")
        return None
    try:
        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        if embedding_model_for_dim_check:
            expected_dim = embedding_model_for_dim_check.get_sentence_embedding_dimension()
            if index.d != expected_dim:
                logger.error(f"Loaded index dimension ({index.d}) does not match embedding model dimension ({expected_dim}).")
                return None
        return index
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}", exc_info=True)
        return None


def load_faiss_index(index_dir, embedding_model_for_dim_check=None, index_name=config.DEFAULT_INDEX_NAME):
    # Loads the FAISS index, corresponding texts, and metadata from disk.
    index = load_faiss_binary_index(index_dir, embedding_model_for_dim_check, index_name)
    if index is None:
        return None, None, None

    db_path = os.path.join(index_dir, f"{index_name}.passages.db")
    if not os.path.exists(db_path):
        logger.warning(f"Passages database '{db_path}' not found.")
        return None, None, None
    try:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT text, metadata FROM passages ORDER BY id").fetchall()
        finally:
            conn.close()
        texts_for_retrieval = [row[0] for row in rows]
        metadata_for_retrieval = [json.loads(row[1]) for row in rows]
        logger.info(f"FAISS index and {len(texts_for_retrieval)} text blocks with metadata loaded successfully.")
        return index, texts_for_retrieval, metadata_for_retrieval
    except Exception as e:
        logger.error(f"Error loading passages database: {e}", exc_info=True)
        return None, None, None
```

- [ ] **Step 5: Rewrite `save_faiss_index`'s passage-storage internals**

Replace the current `save_faiss_index` function (current lines 276-294):
```python
def save_faiss_index(index, texts_for_retrieval, metadata_for_retrieval, index_dir, index_name=config.DEFAULT_INDEX_NAME):
    # Saves the FAISS index, corresponding texts, and metadata to disk.
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    index_path = os.path.join(index_dir, f"{index_name}.index")
    texts_path = os.path.join(index_dir, f"{index_name}.texts.json") 
    try:
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, index_path)
        retrieval_data = {"texts": texts_for_retrieval, "metadata": metadata_for_retrieval}
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(retrieval_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Texts and metadata saved to {texts_path}")
        bm25_cache_path = os.path.join(index_dir, f"{index_name}.bm25.json.gz")
        retriever.warm_bm25_cache(texts_for_retrieval, bm25_cache_path)
        if texts_for_retrieval:
            logger.info(f"BM25 sparse index cache refreshed at {bm25_cache_path}")
    except Exception as e:
        logger.error(f"Error saving FAISS index or associated data: {e}", exc_info=True)
```
with:
```python
def save_faiss_index(index, texts_for_retrieval, metadata_for_retrieval, index_dir, index_name=config.DEFAULT_INDEX_NAME):
    # Saves the FAISS index, corresponding texts, and metadata to disk.
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    index_path = os.path.join(index_dir, f"{index_name}.index")
    db_path = os.path.join(index_dir, f"{index_name}.passages.db")
    try:
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, index_path)

        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "CREATE TABLE passages (id INTEGER PRIMARY KEY, text TEXT NOT NULL, metadata TEXT NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO passages (id, text, metadata) VALUES (?, ?, ?)",
                (
                    (doc_id, text, json.dumps(meta, ensure_ascii=False))
                    for doc_id, (text, meta) in enumerate(
                        zip(texts_for_retrieval, metadata_for_retrieval)
                    )
                ),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Texts and metadata saved to {db_path}")

        bm25_cache_path = os.path.join(index_dir, f"{index_name}.bm25.json.gz")
        retriever.warm_bm25_cache(texts_for_retrieval, bm25_cache_path)
        if texts_for_retrieval:
            logger.info(f"BM25 sparse index cache refreshed at {bm25_cache_path}")
    except Exception as e:
        logger.error(f"Error saving FAISS index or associated data: {e}", exc_info=True)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vector_indexer_passages.py -v`
Expected: all 3 tests `PASS`.

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests pass except the pre-existing, unrelated `test_compress_pdf` failure. In particular, `tests/test_vector_indexer_bm25.py` and `tests/test_vector_indexer_helpers.py` (from earlier plans) must still pass unchanged — they exercise `save_faiss_index`'s BM25 side-effect and the FAISS-metric helpers, both of which this task's changes must not disturb.

- [ ] **Step 8: Commit**

```bash
git add vector_indexer.py tests/test_vector_indexer_passages.py
git commit -m "Switch vector_indexer's passage storage from JSON to SQLite"
```

---

### Task 3: Wire `app.py` to the lazy passage store

**Files:**
- Modify: `app.py` (imports at current line 31; `get_index_file_signature` at current lines 155-165; `load_cached_faiss_index` at current lines 168-184)

**Interfaces:**
- Consumes: `vector_indexer.load_faiss_binary_index` (Task 2) and `passage_store.open_passage_store` (Task 1).
- No automated test for this task — see rationale in Step 4 below (same reasoning as the earlier pre-warm plan: `app.py`'s Streamlit orchestration has zero pre-existing unit-test coverage, and this task doesn't change that boundary).

- [ ] **Step 1: Update the imports**

Change (current line 31):
```python
from vector_indexer import load_faiss_index
```
to:
```python
from vector_indexer import load_faiss_binary_index
from passage_store import open_passage_store
```

- [ ] **Step 2: Update `get_index_file_signature`'s tracked file suffixes**

Change (current lines 155-165):
```python
def get_index_file_signature(index_dir, index_name):
    """Returns a lightweight signature that changes when either index file changes."""
    signature = []
    for suffix in ("index", "texts.json"):
        path = os.path.abspath(os.path.join(index_dir, f"{index_name}.{suffix}"))
        try:
            stat_result = os.stat(path)
            signature.append((path, stat_result.st_size, stat_result.st_mtime_ns))
        except OSError:
            signature.append((path, None, None))
    return tuple(signature)
```
to:
```python
def get_index_file_signature(index_dir, index_name):
    """Returns a lightweight signature that changes when either index file changes."""
    signature = []
    for suffix in ("index", "passages.db"):
        path = os.path.abspath(os.path.join(index_dir, f"{index_name}.{suffix}"))
        try:
            stat_result = os.stat(path)
            signature.append((path, stat_result.st_size, stat_result.st_mtime_ns))
        except OSError:
            signature.append((path, None, None))
    return tuple(signature)
```

- [ ] **Step 3: Rewrite `load_cached_faiss_index`**

Change (current lines 168-184):
```python
@st.cache_resource(max_entries=1)
def load_cached_faiss_index(
    index_dir,
    index_name,
    index_signature,
    embedding_model_name,
    _embedding_model,
):
    """Loads and caches the persistent FAISS index and metadata in memory."""
    # These values are intentionally part of the cache key.
    _ = (index_signature, embedding_model_name)
    index, texts, metadata = load_faiss_index(
        index_dir,
        _embedding_model,
        index_name=index_name,
    )
    return index, texts, metadata
```
to:
```python
@st.cache_resource(max_entries=1)
def load_cached_faiss_index(
    index_dir,
    index_name,
    index_signature,
    embedding_model_name,
    _embedding_model,
):
    """Loads and caches the persistent FAISS index, and opens a lazy
    SQLite-backed passage store instead of loading all passage text and
    metadata into memory."""
    # These values are intentionally part of the cache key.
    _ = (index_signature, embedding_model_name)
    index = load_faiss_binary_index(
        index_dir,
        _embedding_model,
        index_name=index_name,
    )
    if index is None:
        return None, None, None
    store = open_passage_store(index_dir, index_name=index_name)
    if store is None:
        return None, None, None
    return index, store.texts, store.metadata
```

- [ ] **Step 4: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests pass except the pre-existing, unrelated `test_compress_pdf` failure.

No new automated test is added for this step: `app.py`'s Streamlit orchestration has zero pre-existing unit-test coverage (the other `@st.cache_resource` functions in the file are equally untested, for the same structural reason — they need a live Streamlit runtime). Task 4's migration step includes a direct functional check that this code path actually works end-to-end against the real committed data, which is the proportionate verification here.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "Wire app.py to the lazy SQLite passage store instead of loading all texts into memory"
```

---

### Task 4: Migrate the committed index to SQLite

**Files:** none created by code; operates on `vector_store/data_index/` and repo-level git state.

**Interfaces:**
- Consumes: `vector_indexer.load_faiss_binary_index` (Task 2), `vector_indexer.save_faiss_index` (Task 2), `passage_store.open_passage_store` (Task 1) — all already implemented and merged by this point in the plan.

- [ ] **Step 1: Generate `passages.db` from the currently-committed `texts.json`**

Run:
```bash
.venv/Scripts/python.exe -c "
import json
import os

import config
import vector_indexer

index_dir = os.path.join(config.DEFAULT_INDEX_DIR, 'data_index')
index_name = config.DEFAULT_INDEX_NAME

texts_path = os.path.join(index_dir, f'{index_name}.texts.json')
with open(texts_path, 'r', encoding='utf-8') as f:
    retrieval_data = json.load(f)
texts = retrieval_data['texts']
metadata = retrieval_data['metadata']

index = vector_indexer.load_faiss_binary_index(index_dir, index_name=index_name)
assert index is not None, 'Failed to load the existing FAISS binary index'
assert index.ntotal == len(texts) == len(metadata), (
    f'Row count mismatch: index.ntotal={index.ntotal}, texts={len(texts)}, metadata={len(metadata)}'
)

vector_indexer.save_faiss_index(index, texts, metadata, index_dir, index_name=index_name)
print(f'done: {len(texts)} passages migrated to SQLite')
"
```
Expected: prints `done: 74317 passages migrated to SQLite` and takes roughly the time of a single `save_faiss_index` call (a few seconds for the SQLite write; the BM25 cache regeneration this triggers should be near-instant since the corpus is unchanged and the existing persisted BM25 cache's fingerprint will still match).

- [ ] **Step 2: Verify the migrated data loads correctly through both paths**

Run:
```bash
.venv/Scripts/python.exe -c "
import json
import os

import config
import passage_store
import vector_indexer

index_dir = os.path.join(config.DEFAULT_INDEX_DIR, 'data_index')
index_name = config.DEFAULT_INDEX_NAME

# Reference: what the OLD texts.json still on disk says (about to be deleted).
texts_path = os.path.join(index_dir, f'{index_name}.texts.json')
with open(texts_path, 'r', encoding='utf-8') as f:
    reference = json.load(f)

# Ingestion-facing path (Task 2): must match the reference exactly.
_, loaded_texts, loaded_metadata = vector_indexer.load_faiss_index(index_dir, index_name=index_name)
assert loaded_texts == reference['texts'], 'load_faiss_index texts mismatch'
assert loaded_metadata == reference['metadata'], 'load_faiss_index metadata mismatch'

# App-facing lazy path (Task 1/3): spot-check a handful of ids across the range.
store = passage_store.open_passage_store(index_dir, index_name=index_name)
assert store is not None, 'open_passage_store returned None'
assert len(store.texts) == len(reference['texts'])
for doc_id in (0, 1000, len(reference['texts']) // 2, len(reference['texts']) - 1):
    assert store.texts[doc_id] == reference['texts'][doc_id], f'lazy text mismatch at id={doc_id}'
    assert store.metadata[doc_id] == reference['metadata'][doc_id], f'lazy metadata mismatch at id={doc_id}'

print('done: both loading paths verified against the reference JSON')
"
```
Expected: prints `done: both loading paths verified against the reference JSON`.

- [ ] **Step 3: Run the full test suite as a final regression check**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests pass except the pre-existing, unrelated `test_compress_pdf` failure.

- [ ] **Step 4: Delete `texts.json` and commit**

```bash
git add vector_store/data_index/faiss_index.passages.db
git rm vector_store/data_index/faiss_index.texts.json
git commit -m "Migrate committed passage data from texts.json to SQLite passages.db"
```

Do not push — confirm with the user before pushing, since this updates a large binary asset on a shared branch (same convention as the earlier BM25 backfill).
