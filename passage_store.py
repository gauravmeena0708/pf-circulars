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

    def close(self):
        """Closes the underlying SQLite connection. Safe to call more than once."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def fetch_text(self, doc_id):
        # retriever.py indexes with doc ids taken straight from FAISS/BM25
        # results (numpy.int64), which plain Python lists accept via
        # __index__ but sqlite3 does not: it silently binds a numpy int as
        # a BLOB via the buffer protocol instead of matching the INTEGER
        # column, so the query returns no row rather than raising. Coerce
        # to a native int first so a SQLite-backed store behaves exactly
        # like a list here too.
        row = self._conn.execute(
            "SELECT text FROM passages WHERE id = ?", (int(doc_id),)
        ).fetchone()
        if row is None:
            raise IndexError(f"No passage with id {doc_id}")
        return row[0]

    def fetch_metadata(self, doc_id):
        row = self._conn.execute(
            "SELECT metadata FROM passages WHERE id = ?", (int(doc_id),)
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
