"""Shared test configuration."""

from pathlib import Path

import pytest

PASSAGES = Path(__file__).parents[1] / 'vector_store/data_index/faiss_index.passages.db'


def index_available():
    """True when the search index is a real file rather than a Git LFS pointer."""
    try:
        with PASSAGES.open('rb') as source:
            return source.read(15) == b'SQLite format 3'
    except OSError:
        return False


def pytest_configure(config):
    config.addinivalue_line('markers', 'needs_index: reads the real search index (run "git lfs pull" first)')


def pytest_collection_modifyitems(config, items):
    if index_available():
        return
    skip = pytest.mark.skip(reason='the search index is not checked out (git lfs pull)')
    for item in items:
        if 'needs_index' in item.keywords:
            item.add_marker(skip)
