# retriever.py

import logging
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import config

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

# In-memory cache for BM25 index to avoid re-tokenizing corpus on every query
_BM25_CACHE = {
    "corpus_hash": None,
    "bm25_instance": None
}


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


def _get_or_build_bm25_index(all_indexed_texts):
    """Retrieves cached BM25 instance or builds a new one for the corpus."""
    corpus_id = (len(all_indexed_texts), hash(tuple(t[:40] for t in all_indexed_texts[:min(20, len(all_indexed_texts))])))
    
    if _BM25_CACHE["corpus_hash"] == corpus_id and _BM25_CACHE["bm25_instance"] is not None:
        return _BM25_CACHE["bm25_instance"]
    
    logger.info(f"Building BM25 index for {len(all_indexed_texts)} documents...")
    tokenized_corpus = [_tokenize_for_bm25(text) for text in all_indexed_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    _BM25_CACHE["corpus_hash"] = corpus_id
    _BM25_CACHE["bm25_instance"] = bm25
    return bm25


def retrieve_relevant_chunks(query_text, faiss_index, all_indexed_texts, all_indexed_metadata, 
                             embedding_model, cross_encoder_model=None, 
                             top_n_initial=None, top_n_final=None,
                             use_hybrid=None):
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
                bm25 = _get_or_build_bm25_index(all_indexed_texts)
                tokenized_query = _tokenize_for_bm25(query_text)
                if tokenized_query:
                    doc_scores = bm25.get_scores(tokenized_query)
                    top_bm25_indices = np.argsort(doc_scores)[::-1][:sparse_k]
                    
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
            logger.info(f"Re-ranking {len(retrieved_results)} hybrid candidates with CrossEncoder...")
            cross_input = [[query_text, res["text"]] for res in retrieved_results]
            cross_scores = cross_encoder_model.predict(cross_input)
            
            for i, c_score in enumerate(cross_scores):
                retrieved_results[i]["score"] = float(c_score)
                
            # Sort by CrossEncoder score descending
            retrieved_results.sort(key=lambda x: x['score'], reverse=True)
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
