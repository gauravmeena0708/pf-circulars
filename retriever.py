# retriever.py

import gzip
import hashlib
import json
import logging
import os
import re
import tempfile
import numpy as np
from rank_bm25 import BM25Okapi
import config

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

# In-memory cache for BM25 index to avoid re-tokenizing corpus on every query
_BM25_CACHE = {
    "corpus": None,
    "corpus_fingerprint": None,
    "bm25_instance": None
}

_BM25_CACHE_VERSION = 1
_BM25_STATE_FIELDS = (
    "k1",
    "b",
    "epsilon",
    "corpus_size",
    "avgdl",
    "doc_freqs",
    "idf",
    "doc_len",
    "average_idf",
)


def _tokenize_for_bm25(text):
    """
    Tokenizes text for BM25 while preserving statutory references, circular numbers,
    forms, and sub-clauses (e.g. 'para 26(6)', 'form 13', 'wsu/2022/1', '1952').
    """
    if not text:
        return []
    # Match alphanumeric words, including dashes, slashes, and parenthetical clauses
    tokens = re.findall(r'[a-zA-Z0-9]+(?:[\-\/\.][a-zA-Z0-9]+)*(?:\([a-zA-Z0-9]+\))*', text.lower())
    return tokens


def _fingerprint_corpus(all_indexed_texts):
    """Returns a stable content hash used to reject stale persisted indexes."""
    digest = hashlib.sha256()
    digest.update(len(all_indexed_texts).to_bytes(8, byteorder="big"))
    for text in all_indexed_texts:
        encoded_text = text.encode("utf-8", errors="replace")
        digest.update(len(encoded_text).to_bytes(8, byteorder="big"))
        digest.update(encoded_text)
    return digest.hexdigest()


def _load_persisted_bm25(cache_path, corpus_fingerprint, document_count):
    if not cache_path or not os.path.isfile(cache_path):
        return None

    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
        if (
            payload.get("version") != _BM25_CACHE_VERSION
            or payload.get("corpus_fingerprint") != corpus_fingerprint
            or payload.get("document_count") != document_count
        ):
            logger.info("Ignoring stale BM25 cache at %s.", cache_path)
            return None

        state = payload.get("bm25_state", {})
        if not all(field in state for field in _BM25_STATE_FIELDS):
            logger.warning("BM25 cache at %s is missing required fields.", cache_path)
            return None

        bm25_instance = BM25Okapi.__new__(BM25Okapi)
        bm25_instance.__dict__.update(state)
        bm25_instance.tokenizer = None
        logger.info("Loaded persisted BM25 index from %s.", cache_path)
        return bm25_instance
    except (
        OSError,
        EOFError,
        UnicodeError,
        ValueError,
        AttributeError,
        TypeError,
    ) as error:
        logger.warning("Could not load BM25 cache '%s': %s", cache_path, error)
        return None


def _persist_bm25(cache_path, corpus_fingerprint, document_count, bm25_instance):
    if not cache_path:
        return

    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    temp_path = None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        file_descriptor, temp_path = tempfile.mkstemp(
            prefix=".bm25-",
            suffix=".tmp",
            dir=cache_dir,
        )
        os.close(file_descriptor)
        payload = {
            "version": _BM25_CACHE_VERSION,
            "corpus_fingerprint": corpus_fingerprint,
            "document_count": document_count,
            "bm25_state": {
                field: getattr(bm25_instance, field)
                for field in _BM25_STATE_FIELDS
            },
        }
        with gzip.open(temp_path, "wt", encoding="utf-8", compresslevel=5) as cache_file:
            json.dump(payload, cache_file, ensure_ascii=True, separators=(",", ":"))
        os.replace(temp_path, cache_path)
        temp_path = None
        logger.info("Persisted BM25 index to %s.", cache_path)
    except (OSError, TypeError, ValueError) as error:
        logger.warning("Could not persist BM25 cache '%s': %s", cache_path, error)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _get_or_build_bm25_index(all_indexed_texts, cache_path=None):
    """Loads BM25 from memory or disk, rebuilding it when the corpus changes."""
    if (
        _BM25_CACHE["corpus"] is all_indexed_texts
        and _BM25_CACHE["bm25_instance"] is not None
    ):
        return _BM25_CACHE["bm25_instance"]

    corpus_fingerprint = _fingerprint_corpus(all_indexed_texts)
    bm25 = _load_persisted_bm25(
        cache_path,
        corpus_fingerprint,
        len(all_indexed_texts),
    )
    if bm25 is None:
        logger.info(f"Building BM25 index for {len(all_indexed_texts)} documents...")
        tokenized_corpus = [_tokenize_for_bm25(text) for text in all_indexed_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        _persist_bm25(
            cache_path,
            corpus_fingerprint,
            len(all_indexed_texts),
            bm25,
        )

    _BM25_CACHE["corpus"] = all_indexed_texts
    _BM25_CACHE["corpus_fingerprint"] = corpus_fingerprint
    _BM25_CACHE["bm25_instance"] = bm25
    return bm25


def _top_k_score_indices(scores, top_k):
    """Returns score indices in descending order without sorting the full array."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or scores.size == 0 or top_k <= 0:
        return np.empty(0, dtype=np.intp)

    top_k = min(top_k, scores.size)
    if top_k == scores.size:
        return np.argsort(scores)[::-1]

    candidate_indices = np.argpartition(scores, scores.size - top_k)[-top_k:]
    candidate_order = np.argsort(scores[candidate_indices])[::-1]
    return candidate_indices[candidate_order]


def retrieve_relevant_chunks(query_text, faiss_index, all_indexed_texts, all_indexed_metadata, 
                             embedding_model, cross_encoder_model=None, 
                             top_n_initial=None, top_n_final=None,
                             use_hybrid=None, bm25_cache_path=None):
    """
    Retrieves the most relevant text chunks using Hybrid Retrieval (BM25 + Dense FAISS)
    fused with Reciprocal Rank Fusion (RRF) and re-ranked with a CrossEncoder.

    Args:
        query_text (str): The query string.
        faiss_index: Loaded FAISS IndexFlatIP.
        all_indexed_texts (list): List of text chunks corresponding to FAISS vector IDs.
        all_indexed_metadata (list): List of metadata dicts corresponding to FAISS vector IDs.
        embedding_model (SentenceTransformer): Model to encode the query.
        cross_encoder_model (CrossEncoder, optional): Model for deep semantic re-ranking.
        top_n_initial (int, optional): Number of candidates to retrieve initially.
        top_n_final (int, optional): Number of top results to return.
        use_hybrid (bool, optional): Whether to use hybrid BM25 + Dense search.
        bm25_cache_path (str, optional): Path used to persist the sparse index.

    Returns:
        list: List of dicts with keys 'text', 'metadata', 'score'.
    """
    if top_n_initial is None:
        top_n_initial = getattr(config, "TOP_N_INITIAL_RETRIEVAL", 25)
    if top_n_final is None:
        top_n_final = getattr(config, "TOP_N_RETRIEVAL", 5)
    if use_hybrid is None:
        use_hybrid = getattr(config, "USE_HYBRID_RETRIEVAL", True)

    if not query_text or not faiss_index or embedding_model is None:
        logger.warning("Missing required query, index, or embedding model.")
        return []
    if not all_indexed_texts or not all_indexed_metadata:
        logger.warning("Indexed texts or metadata are missing. Cannot retrieve.")
        return []
    if len(all_indexed_texts) != len(all_indexed_metadata) or len(all_indexed_texts) != faiss_index.ntotal:
        logger.warning(f"Mismatch in lengths: texts ({len(all_indexed_texts)}), "
                       f"metadata ({len(all_indexed_metadata)}), index ({faiss_index.ntotal}).")
        return []

    try:
        total_docs = len(all_indexed_texts)
        dense_k = min(getattr(config, "DENSE_TOP_K", 40), total_docs)
        sparse_k = min(getattr(config, "BM25_TOP_K", 40), total_docs)
        rrf_constant = getattr(config, "RRF_K", 60)

        # -------------------------------------------------------------
        # 1. Dense Semantic Search (FAISS IndexFlatIP with normalized vectors)
        # -------------------------------------------------------------
        logger.info(f"Generating normalized embedding for query: '{query_text[:80]}...'")
        query_embedding = embedding_model.encode(
            query_text, 
            convert_to_tensor=False, 
            normalize_embeddings=True
        )
        query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)

        distances, indices = faiss_index.search(query_embedding_np, k=dense_k)
        
        dense_rankings = {}
        dense_scores = {}
        if indices.size > 0:
            for rank, (doc_idx, dist) in enumerate(zip(indices[0], distances[0])):
                if 0 <= doc_idx < total_docs:
                    dense_rankings[doc_idx] = rank + 1
                    # With IndexFlatIP and unit vectors, distance directly equals cosine similarity
                    dense_scores[doc_idx] = float(dist)

        # -------------------------------------------------------------
        # 2. Sparse Lexical Search (BM25Okapi)
        # -------------------------------------------------------------
        sparse_rankings = {}
        sparse_scores = {}
        
        if use_hybrid:
            try:
                bm25 = _get_or_build_bm25_index(all_indexed_texts, bm25_cache_path)
                tokenized_query = _tokenize_for_bm25(query_text)
                if tokenized_query:
                    doc_scores = bm25.get_scores(tokenized_query)
                    top_bm25_indices = _top_k_score_indices(doc_scores, sparse_k)
                    
                    for rank, doc_idx in enumerate(top_bm25_indices):
                        score = doc_scores[doc_idx]
                        if score > 0 and 0 <= doc_idx < total_docs:
                            sparse_rankings[int(doc_idx)] = rank + 1
                            sparse_scores[int(doc_idx)] = float(score)
            except Exception as bm25_err:
                logger.warning(f"BM25 retrieval encountered an error: {bm25_err}. Continuing with dense-only.")

        # -------------------------------------------------------------
        # 3. Reciprocal Rank Fusion (RRF)
        # -------------------------------------------------------------
        candidate_ids = set(dense_rankings.keys()).union(set(sparse_rankings.keys()))
        rrf_scores = {}

        for doc_id in candidate_ids:
            score = 0.0
            if doc_id in dense_rankings:
                score += 1.0 / (rrf_constant + dense_rankings[doc_id])
            if doc_id in sparse_rankings:
                score += 1.0 / (rrf_constant + sparse_rankings[doc_id])
            rrf_scores[doc_id] = score

        # Sort candidate doc IDs by RRF score descending
        sorted_candidates = sorted(candidate_ids, key=lambda d: rrf_scores[d], reverse=True)
        top_candidates = sorted_candidates[:top_n_initial]

        retrieved_results = []
        for doc_id in top_candidates:
            retrieved_results.append({
                "text": all_indexed_texts[doc_id],
                "metadata": all_indexed_metadata[doc_id],
                "score": float(dense_scores.get(doc_id, rrf_scores[doc_id])),
                "rrf_score": float(rrf_scores[doc_id])
            })

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

        # Slice to final requested count
        final_results = retrieved_results[:top_n_final]
        logger.info(f"Retrieved {len(final_results)} top chunks for query: '{query_text[:50]}'.")
        return final_results

    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        return []
