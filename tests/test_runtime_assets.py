import hashlib
import io
from pathlib import Path
import pytest
from scripts import fetch_runtime_assets as assets


def test_manifest_rejects_mutable_revision_traversal_duplicates_and_wrong_hash():
    manifest={'repo':assets.REPO,'revision':'d'*40,'files':[{'path':'vector_store/data_index/test.db','size':3,'sha256':'a'*64}]}
    assets.validate(manifest)
    manifest['revision']='main'
    with pytest.raises(ValueError):assets.validate(manifest)
    manifest['revision']='d'*40;manifest['files'][0]['path']='../escape.pdf'
    with pytest.raises(ValueError):assets.validate(manifest)


def test_download_hash_is_required_before_replacing_an_existing_asset(tmp_path, monkeypatch):
    target=tmp_path/'vector_store/data_index/test.db';target.parent.mkdir(parents=True);target.write_bytes(b'old')
    entry={'path':'vector_store/data_index/test.db','size':3,'sha256':hashlib.sha256(b'new').hexdigest()}
    monkeypatch.setattr(assets.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'bad'))
    monkeypatch.setattr(assets.time,'sleep',lambda *a:None)
    with pytest.raises(ValueError):assets.fetch(entry,'d'*40,tmp_path)
    assert target.read_bytes()==b'old' and len(list(target.parent.iterdir()))==1
    monkeypatch.setattr(assets.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'new'))
    assets.fetch(entry,'d'*40,tmp_path)
    assert target.read_bytes()==b'new'
    monkeypatch.setattr(assets.urllib.request,'urlopen',lambda *a,**k:pytest.fail('Matching cached asset must not download again'))
    assets.fetch(entry,'d'*40,tmp_path)


def test_ci_refresh_advances_revision_only_when_runtime_bytes_change(tmp_path):
    index = tmp_path/'vector_store/data_index'
    index.mkdir(parents=True)
    for name in ('faiss_index.index','faiss_index.passages.db','faiss_index.bm25.json.gz'):
        (index/name).write_bytes(b'initial')
    path = tmp_path/'manifest.json'
    assets.record('d'*40, root=tmp_path, path=path)
    same = assets.record('e'*40, root=tmp_path, path=path, preserve_revision=True)
    assert same['revision'] == 'd'*40
    (index/'faiss_index.index').write_bytes(b'updated')
    changed = assets.record('e'*40, root=tmp_path, path=path, preserve_revision=True)
    assert changed['revision'] == 'e'*40
