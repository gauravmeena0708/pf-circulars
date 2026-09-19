# SQLite Passage Store — Design

## Problem

`app.py` (Streamlit) loads `vector_store/data_index/faiss_index.texts.json` (105MB on disk) into a single Python list at process startup, purely so `retriever.retrieve_relevant_chunks` can do `all_indexed_texts[doc_id]` lookups for a handful of RRF-fused candidates per query (typically ~20-40 out of 74,317 passages). Loading and holding this JSON blob costs 500-800MB of Python heap RAM and multiple seconds of `json.load()` time on every container boot, on a CPU/RAM-constrained free-tier Hugging Face Space.

The earlier BM25 cache-bundling work already means `retriever._get_or_build_bm25_index` doesn't need the raw corpus text in memory when a valid persisted BM25 cache exists (`_load_persisted_bm25` deserializes aggregate statistics — `doc_freqs`, `idf`, `doc_len`, etc. — directly). The only place that still needs to touch every row is `_fingerprint_corpus`'s staleness check, which can be satisfied by streaming rather than by holding a full list.

## Goal

Eliminate `app.py`'s multi-hundred-MB RAM cost and multi-second startup cost for passage text/metadata, **without changing how any of the four ingestion scripts work**. This is a runtime (`app.py`) fix only — ingestion scripts (`main.py`, `update_indexer.py`, `index_manuals.py`, `import_pf_circular_index.py`) are batch CLI jobs, not RAM-constrained the way a long-running Streamlit process is, and continue to load the full corpus into memory exactly as they do today.

## Non-Goals

- Reducing ingestion scripts' memory usage. Explicitly out of scope — see Goal above.
- Changing `retriever.py`'s retrieval logic, ranking, or any behavior added by prior plans this session (BM25 cache, confidence gate, FAISS metric fixes). This plan touches zero lines of `retriever.py`.
- Batching SQL reads into a single query per search (considered as Approach B, rejected — see Approaches below).

## Chosen Approach: Lazy SQLite Proxy Objects

Two other approaches were considered and rejected:
- **Batch-fetch inside `retriever.py`**: restructure `retrieve_relevant_chunks` to collect candidate doc_ids, then issue one batched `SELECT ... WHERE id IN (...)` query. Marginally more SQL-efficient (one query vs. ~20-40 tiny point-lookups per search), but requires touching `retriever.py`, which has already accumulated real complexity this session (BM25 gate, confidence gate) and was deliberately left alone. SQLite point-lookups on an indexed primary key are microseconds each; the difference is not expected to be measurable.
- **Memory-mapped/streaming JSON**: doesn't solve random access well — JSON isn't designed for it, and a custom index-within-the-index would end up reinventing what SQLite already provides, for more custom code and less benefit.

**Chosen**: replace `faiss_index.texts.json` with `faiss_index.passages.db` (SQLite, stdlib `sqlite3`, no new dependency). `vector_indexer.save_faiss_index`/`load_faiss_index` keep their exact existing public signatures and return contracts — every ingestion script is unaffected. `app.py` gets a new lightweight loading path that opens the SQLite db and returns two proxy objects, `.texts` and `.metadata`, that duck-type as list-like (`__len__`, `__getitem__(doc_id)`, `__iter__`). Since `retriever.py` only ever does `all_indexed_texts[doc_id]`, `len(all_indexed_texts)`, and `for text in all_indexed_texts`, it cannot tell the difference between a real list and this proxy — **zero lines of `retriever.py` change**.

## Components

### New module: `passage_store.py`

```
PassageStore
  - wraps one read-only sqlite3.Connection (check_same_thread=False; safe
    because access is read-only and shared read-only access across threads
    is a supported SQLite usage)
  - holds row count (queried once at open time)
  - .texts   -> _PassageTextsView
  - .metadata -> _PassageMetadataView

_PassageTextsView / _PassageMetadataView
  - __len__      -> delegates to PassageStore's row count
  - __getitem__(doc_id) -> single-row SQL lookup by primary key
  - __iter__     -> streams rows in id order via a cursor; never
                    materializes the full corpus as a Python list
                    (used only by BM25's fingerprint hash and the
                    cold-rebuild fallback path)

open_passage_store(index_dir, index_name=config.DEFAULT_INDEX_NAME) -> PassageStore | None
  - returns None if the db file doesn't exist (never raises) --
    mirrors load_faiss_index's existing "not found" behavior
```

SQLite schema:
```sql
CREATE TABLE passages (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL  -- JSON-encoded dict
);
```
`id` matches the FAISS vector id (0..N-1), the same convention `all_indexed_texts[doc_id]` positional indexing already relies on.

### `vector_indexer.py` changes

- Extract the FAISS-binary-loading + embedding-dimension-check logic (currently the first half of `load_faiss_index`) into a new function, `load_faiss_binary_index(index_dir, embedding_model_for_dim_check=None, index_name=config.DEFAULT_INDEX_NAME) -> faiss.Index | None`.
- `load_faiss_index(...)` — **public signature and return contract unchanged** (`(index, texts_list, metadata_list)`). Internally: calls `load_faiss_binary_index(...)`, then fully materializes `passages.db` into `texts_list`/`metadata_list` by reading all rows in id order. This is what all four ingestion scripts keep calling, completely unmodified.
- `save_faiss_index(...)` — **public signature unchanged**. Internally: writes the FAISS index (unchanged) and writes `passages.db` (new) instead of `.texts.json` (removed). BM25 cache regeneration (from the earlier plan) is unaffected — it consumes the same `texts_for_retrieval` list `save_faiss_index` already receives as a parameter, not the new db file.

### `app.py` changes

`load_cached_faiss_index` (the `@st.cache_resource`-backed function at module scope) changes internally from one call (`load_faiss_index` → full lists) to two calls: `vector_indexer.load_faiss_binary_index(...)` for the FAISS index, and `passage_store.open_passage_store(...)` for the lazy `.texts`/`.metadata` pair. Every downstream use in `app.py` — the truthy check (`if not faiss_index or not indexed_texts or not indexed_metadata:`), passing `indexed_texts`/`indexed_metadata` into `retrieve_relevant_chunks` and `prewarm_search_pipeline`, the `@st.cache_data` wrapper around `retrieve_relevant_chunks` — is **byte-for-byte unchanged**, since the proxy objects are truthy/falsy and indexable exactly like a list.

## Data Flow Summary

- **Ingestion** (unchanged): `load_faiss_index` → full lists in memory → append/dedupe/modify → `save_faiss_index` → writes FAISS index + `passages.db` (+ BM25 cache, from the earlier plan).
- **App runtime** (new): `load_faiss_binary_index` → FAISS index only; `open_passage_store` → lazy `PassageStore`; `retrieve_relevant_chunks` called exactly as before, unaware its `all_indexed_texts`/`all_indexed_metadata` arguments are now proxies instead of lists.

## Migration of Existing Data

One-time backfill (same shape as the earlier BM25 cache backfill): load the currently-committed `faiss_index.texts.json` via the existing JSON-reading code one last time, write its contents into a fresh `faiss_index.passages.db`, verify the row count matches the committed FAISS index's `ntotal` (74,317), then delete `faiss_index.texts.json` from the working tree and git history going forward (not rewriting past history — just not carrying it forward) and commit `passages.db` under the existing `vector_store/** filter=lfs` Git LFS pattern (no `.gitattributes` change needed, same as the BM25 cache backfill).

`faiss_index.texts.json` is deleted, not kept as a fallback — full replacement, single source of truth. Keeping both would double LFS storage for zero runtime benefit, since nothing would read `texts.json` after this migration.

## Error Handling

- `open_passage_store` returns `None` on a missing or corrupt db file, never raises — `app.py`'s existing "index not found" warning path (`st.warning(...)`) handles this exactly as it already does for a missing FAISS index.
- The SQLite connection is opened read-only at application runtime; ingestion scripts are the only writers, and they write by rebuilding the whole db file fresh via `save_faiss_index` (not incremental SQL writes) — there is no concurrent-writer/locking scenario to handle.
- `check_same_thread=False` is required because the connection is created once inside a `@st.cache_resource` function and shared across all of a Streamlit process's sessions/threads; this is safe because all runtime access is read-only.

## Testing Strategy

- Unit tests for `PassageStore`/its views against a small, hand-built real SQLite db (no mocks): `__len__`, `__getitem__` (including an out-of-range id), `__iter__` (order and completeness), and `open_passage_store`'s `None`-on-missing-file behavior.
- An integration test for `vector_indexer.save_faiss_index` → `load_faiss_index` round-tripping through the new schema, extending the existing test pattern from `tests/test_vector_indexer_bm25.py`.
- A test proving `retrieve_relevant_chunks` produces **identical results** whether given real Python lists or `PassageStore` views for the same small fixture — this is the test that actually proves "zero `retriever.py` changes needed" is true, not merely assumed.

## Scope Boundary (for the implementation plan)

This spec covers exactly: the new `passage_store.py` module, the `vector_indexer.py` refactor (`load_faiss_binary_index` extraction, `save_faiss_index`/`load_faiss_index` internals), the `app.py` loading-path change, and the one-time migration/backfill of the committed data. It does not touch `retriever.py`, any of the four ingestion scripts, or `.gitattributes`.
