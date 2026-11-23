"""Append auditable policy pages to a staged hybrid index; never overwrite the input.

Run with --index-dir vector_store/data_index --output-dir /tmp/policy-index.
Existing passages are retained byte-for-byte. Page hashes/offsets make reruns idempotent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '4')


def evidence_chunks(policies):
    from policy_history import PolicyHistory
    seen = set()
    for policy in policies:
        history = PolicyHistory(policy.path)
        for doc in history.documents.values():
            if not doc.get('source_file'):
                continue
            source = json.loads((policy.path.parent / doc['source_file']).read_text())
            for page in source['pages']:
                for offset in range(0, len(page['text']), 650):
                    body = page['text'][offset:offset + 800]
                    if len(body.strip()) < 25:
                        continue
                    key = hashlib.sha256(json.dumps([doc['id'], source['pdf_sha256'], page['page_number'], offset, body], ensure_ascii=False).encode()).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    text = doc['title'] + '\n' + body
                    meta = dict(title=doc['title'], circular_no=doc['circular_no'], date=doc['issued_on'],
                                policy_document_id=doc['id'], policy_evidence_key=key,
                                english_pdf_link=doc['source_url'], source_pdf=doc['source_url'],
                                original_source_url=doc.get('original_source_url'),
                                source_host_kind=doc.get('source_host_kind', 'issuer_hosted'),
                                authority=doc.get('authority'), pdf_sha256=source['pdf_sha256'],
                                page_number=page['page_number'], page_provenance='physical_pdf_page',
                                extraction_method=page['extraction_method'], match_start=offset,
                                parent_context=page['text'], evidence_status='source page checked; applicability unverified')
                    yield text, meta


def build(index_dir, output_dir, model=None):
    import config
    from policy_registry import registered_policies
    from vector_indexer import load_faiss_index, save_faiss_index, encode_normalized
    import faiss
    import numpy as np
    if Path(index_dir).resolve() == Path(output_dir).resolve():
        raise ValueError('Use a separate output directory and verify it before publishing')
    index, texts, metadata = load_faiss_index(str(index_dir))
    if index is None or index.ntotal != len(texts) or len(texts) != len(metadata):
        raise ValueError('Input vector/passage alignment is invalid')
    before = index.ntotal
    existing = {m.get('policy_evidence_key') for m in metadata}
    additions = [(t, m) for t, m in evidence_chunks(registered_policies()) if m['policy_evidence_key'] not in existing]
    if additions:
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device='cpu', local_files_only=True)
        if model.get_sentence_embedding_dimension() != index.d:
            raise ValueError('Embedding dimension mismatch')
        embeddings = np.asarray(encode_normalized(model, [t for t, _ in additions], batch_size=64, show_progress_bar=True), dtype='float32')
        if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4):
            raise ValueError('Non-finite or unnormalised embeddings')
        if hasattr(index, 'id_map'):
            if not np.array_equal(faiss.vector_to_array(index.id_map), np.arange(before)):
                raise ValueError('Input ID map is not aligned with passage IDs')
            index.add_with_ids(embeddings, np.arange(before, before + len(additions), dtype='int64'))
        else:
            index.add(embeddings)
        texts.extend(t for t, _ in additions)
        metadata.extend(m for _, m in additions)
    save_faiss_index(index, texts, metadata, str(output_dir))
    # The legacy save helper logs exceptions; verify artifacts rather than trusting return.
    stored = faiss.read_index(str(Path(output_dir) / 'faiss_index.index'))
    with sqlite3.connect(Path(output_dir) / 'faiss_index.passages.db') as db:
        count = db.execute('select count(*) from passages').fetchone()[0]
        db.execute('attach database ? as original', (str(Path(index_dir) / 'faiss_index.passages.db'),))
        changed = db.execute('select count(*) from original.passages o left join main.passages n on o.id=n.id where n.id is null or n.text != o.text or n.metadata != o.metadata').fetchone()[0]
    if stored.ntotal != len(texts) or count != len(texts) or changed:
        raise ValueError('Staged index failed alignment or original-passage preservation check')
    import gzip
    with gzip.open(Path(output_dir) / 'faiss_index.bm25.json.gz', 'rt') as f:
        sparse = json.load(f)
    if sparse.get('document_count') != count or sparse.get('bm25_state', {}).get('corpus_size') != count:
        raise ValueError('Sparse index count does not match dense/passages')
    report = dict(original_passages=before, added_passages=len(additions), final_passages=count,
                  original_passages_changed=changed, model=config.EMBEDDING_MODEL_NAME,
                  dimension=index.d, source_documents=len({m['policy_document_id'] for _, m in additions}),
                  new_page_count=len({(m['pdf_sha256'], m['page_number']) for _, m in additions}),
                  extraction_methods=sorted({m['extraction_method'] for _, m in additions}),
                  sparse_cache_written=bool(sparse), complete_applicability_established=False)
    (Path(output_dir) / 'policy_ingestion_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.index_dir, args.output_dir), indent=2))
