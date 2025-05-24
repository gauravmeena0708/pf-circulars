# retriever.py

import numpy as np
import faiss # Only if direct FAISS search is used, otherwise SentenceTransformer util might be enough for cosine sim
from sentence_transformers import SentenceTransformer, util # util.semantic_search is handy
import logging

# Assuming your config.py is in the same directory or accessible in PYTHONPATH
import config

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

def retrieve_relevant_chunks(query_text,
                             faiss_index,
                             all_indexed_texts,
                             all_indexed_metadata,
                             embedding_model,
                             top_n=config.TOP_N_RETRIEVAL):
    """
    Retrieves the top_n most relevant text chunks from the FAISS index for a given query.

    Args:
        query_text (str): The user's query.
        faiss_index (faiss.IndexIDMap): The loaded FAISS index.
        all_indexed_texts (list): The list of all text chunks that were indexed.
                                  The order must correspond to the IDs used in faiss_index (0 to n-1).
        all_indexed_metadata (list): The list of metadata corresponding to all_indexed_texts.
        embedding_model (SentenceTransformer): The initialized sentence transformer model.
        top_n (int): The number of top relevant chunks to retrieve.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              'text' (the retrieved text chunk),
              'metadata' (the associated metadata),
              'score' (the similarity score).
              Returns an empty list if an error occurs or no chunks are found.
    """
    if not query_text:
        logger.warning("Query text is empty. Cannot retrieve.")
        return []
    if faiss_index is None or faiss_index.ntotal == 0:
        logger.warning("FAISS index is not loaded or is empty. Cannot retrieve.")
        return []
    if not all_indexed_texts or not all_indexed_metadata:
        logger.warning("Indexed texts or metadata are missing. Cannot retrieve.")
        return []
    if len(all_indexed_texts) != len(all_indexed_metadata) or len(all_indexed_texts) != faiss_index.ntotal:
        logger.warning(f"Mismatch in lengths: texts ({len(all_indexed_texts)}), "
                       f"metadata ({len(all_indexed_metadata)}), index ({faiss_index.ntotal}). Cannot retrieve reliably.")
        return []

    try:
        logger.info(f"Embedding query: '{query_text[:100]}...'")
        query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
        query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)

        logger.info(f"Searching FAISS index for top {top_n} chunks...")
        # D are distances (L2 distance for IndexFlatL2), I are the indices/IDs of the vectors
        distances, indices = faiss_index.search(query_embedding_np, k=top_n)

        retrieved_results = []
        if indices.size > 0: # Check if any results were found
            for i in range(indices.shape[1]): # Iterate through the k results
                retrieved_id = indices[0, i]
                distance = distances[0, i]

                if retrieved_id < 0 or retrieved_id >= len(all_indexed_texts):
                    logger.warning(f"Retrieved invalid ID {retrieved_id} from FAISS index. Skipping.")
                    continue
                
                # Convert L2 distance to a similarity score (e.g., 1 / (1 + L2_distance))
                # Or, if you used normalized embeddings and IndexFlatIP (Inner Product), distances would be cosine similarities.
                # For L2 distance with normalized embeddings, similarity can also be derived.
                # A simple approach for L2 distance: smaller is better.
                # For SentenceTransformer embeddings, they are typically normalized.
                # If embeddings are normalized, L2 distance D and cosine similarity S are related: D^2 = 2 - 2*S
                # So, S = 1 - (D^2 / 2)
                similarity_score = 1 - (distance**2 / 2) # Assuming normalized embeddings

                retrieved_results.append({
                    "text": all_indexed_texts[retrieved_id],
                    "metadata": all_indexed_metadata[retrieved_id],
                    "score": float(similarity_score) # Ensure score is a standard float
                })
        
        # Sort by score descending (higher similarity is better)
        retrieved_results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Retrieved {len(retrieved_results)} relevant chunks.")
        return retrieved_results

    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        return []


if __name__ == '__main__':
    logger.info("Starting Retriever example...")

    # --- Prerequisites (simulate loading these as they would be in main.py) ---
    try:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        sbert_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
        exit()

    # Simulate loaded index and data from vector_indexer.py
    # In a real scenario, these would be loaded using vector_indexer.load_faiss_index()
    
    # Create a dummy index for example purposes
    sample_texts = [
        "The weather is sunny and warm today.",
        "Machine learning models require a lot of data.",
        "Climate change is a pressing global issue.",
        "Deep learning is a subfield of machine learning.",
        "Renewable energy sources are crucial for the future."
    ]
    sample_metadata = [
        {"source_pdf": "doc1.pdf", "page_number": 1, "id": 0},
        {"source_pdf": "doc1.pdf", "page_number": 2, "id": 1},
        {"source_pdf": "doc2.pdf", "page_number": 1, "id": 2},
        {"source_pdf": "doc2.pdf", "page_number": 2, "id": 3},
        {"source_pdf": "doc3.pdf", "page_number": 1, "id": 4}
    ]
    
    if not sample_texts:
        logger.warning("No sample texts to create a dummy index. Exiting example.")
        exit()

    try:
        logger.info("Creating dummy FAISS index for example...")
        sample_embeddings = sbert_model.encode(sample_texts, convert_to_tensor=False)
        sample_embeddings_np = np.array(sample_embeddings).astype('float32')
        dimension = sample_embeddings_np.shape[1]
        
        # Using IndexFlatL2 and IndexIDMap
        _index = faiss.IndexFlatL2(dimension)
        dummy_faiss_index = faiss.IndexIDMap(_index)
        ids = np.arange(len(sample_texts))
        dummy_faiss_index.add_with_ids(sample_embeddings_np, ids)
        logger.info(f"Dummy FAISS index created with {dummy_faiss_index.ntotal} vectors.")
    except Exception as e:
        logger.error(f"Could not create dummy FAISS index: {e}", exc_info=True)
        exit()

    # --- Test Query ---
    test_query = "What is deep learning?"
    top_k = 3

    logger.info(f"Performing retrieval for query: '{test_query}' with top_n={top_k}")
    retrieved_chunks = retrieve_relevant_chunks(test_query,
                                               dummy_faiss_index,
                                               sample_texts,
                                               sample_metadata,
                                               sbert_model,
                                               top_n=top_k)

    if retrieved_chunks:
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks:")
        for i, chunk_data in enumerate(retrieved_chunks):
            logger.info(f"  Rank {i+1}: Score: {chunk_data['score']:.4f}")
            logger.info(f"    Text: {chunk_data['text'][:150]}...")
            logger.info(f"    Metadata: {chunk_data['metadata']}")
    else:
        logger.warning("No chunks retrieved for the query.")

    # Test with an empty query
    logger.info("Performing retrieval for an empty query...")
    retrieved_empty = retrieve_relevant_chunks("", dummy_faiss_index, sample_texts, sample_metadata, sbert_model)
    assert len(retrieved_empty) == 0
    logger.info("Empty query test passed.")
    
    # Test with empty index (simulate)
    logger.info("Performing retrieval with an empty index (simulated by passing ntotal=0)...")
    empty_faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension)) # empty
    # To truly test, one might pass an index object known to be empty or has 0 ntotal
    # For this example, let's assume retrieve_relevant_chunks handles faiss_index.ntotal == 0 check
    retrieved_empty_idx = retrieve_relevant_chunks(test_query, empty_faiss_index, [], [], sbert_model)
    assert len(retrieved_empty_idx) == 0
    logger.info("Empty index test passed.")


    logger.info("Retriever example finished.")