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
