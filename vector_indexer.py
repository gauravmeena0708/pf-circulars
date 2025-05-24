# vector_indexer.py

import os
import faiss
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

# Assuming your config.py is in the same directory or accessible in PYTHONPATH
import config

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


def group_extracted_content_to_blocks(extracted_pages_data):
    """
    Groups plain text and nearby tables from extracted PDF pages into contextual blocks.
    This function processes data from potentially multiple PDFs.

    Args:
        extracted_pages_data (list): A list of page data dictionaries.
            Each page dictionary is expected to have 'source_pdf', 'page_number',
            and 'content' (a list of text/table blocks as output by pdf_parser.py).

    Returns:
        list: A list of dictionaries, where each dictionary represents a contextual block
              and contains 'source_pdf', 'page_number', and 'group_content' (a list of items).
    """
    grouped_blocks = []
    for page_data in extracted_pages_data:
        page_number = page_data['page_number']
        source_pdf = page_data['source_pdf']
        page_content = page_data['content'] # Content items are already sorted by y-position

        if not page_content:
            continue

        current_group = []
        for i, content_item in enumerate(page_content):
            current_group.append(content_item)

            # Decision points for finalizing a group:
            # 1. If this is the last item on the page.
            # 2. If the current item is plain text and the next item is also plain text
            #    (meaning we don't want to break up consecutive text paragraphs unnecessarily,
            #     but also want to group text with an immediately following/preceding table).
            #    This logic is a bit different from the original script and aims to create
            #    more coherent text chunks, possibly including a table if it's adjacent.

            is_last_item_on_page = (i == len(page_content) - 1)
            
            # A simpler grouping: one group per item unless it's text followed by text.
            # Or a more complex one: group text with adjacent tables.
            # Let's refine the original logic: group consecutive text blocks, and if a table
            # is next, include it. If text follows a table, start a new group.

            # Original logic adapted:
            # Keep adding to current_group. Finalize if:
            # - It's the last block.
            # - The current block is NOT a table, AND the NEXT block is NOT a table.
            #   (This means group text with text, or text with a following table).

            finalize_group = False
            if is_last_item_on_page:
                finalize_group = True
            else:
                current_is_table = content_item["type"] == "table"
                next_is_table = (not is_last_item_on_page and page_content[i + 1]["type"] == "table")
                
                # If current is text and next is also text, continue grouping
                if not current_is_table and not next_is_table:
                    pass # Continue accumulating text
                else: # current is table, or next is table, or current is text and next is table
                    finalize_group = True
            
            if finalize_group and current_group:
                grouped_blocks.append({
                    "source_pdf": source_pdf,
                    "page_number": page_number,
                    "group_content": list(current_group), # Store a copy
                })
                current_group = []
        
        # Catch any trailing group for the page
        if current_group:
            grouped_blocks.append({
                "source_pdf": source_pdf,
                "page_number": page_number,
                "group_content": list(current_group),
            })
            current_group = []
            
    return grouped_blocks


def convert_grouped_blocks_to_texts_and_metadata(grouped_blocks):
    """
    Converts grouped blocks into plain text strings for embedding and extracts metadata.

    Args:
        grouped_blocks (list): Output from group_extracted_content_to_blocks.

    Returns:
        tuple: (list_of_texts, list_of_metadata)
               - list_of_texts: strings ready for embedding.
               - list_of_metadata: dictionaries with 'source_pdf', 'page_number',
                                   and 'original_group_content_snippet'.
    """
    texts_for_embedding = []
    corresponding_metadata = []

    for block in grouped_blocks:
        block_text_parts = []
        for content_item in block["group_content"]:
            if content_item["type"] == "plain_text":
                block_text_parts.append(content_item["text"])
            elif content_item["type"] == "table":
                # Represent table content as a structured string
                table_str = "Table Content: " + "; ".join(
                    [f"[{cell_text}]" for cell_text in content_item.get("extracted_text_list", [])]
                )
                block_text_parts.append(table_str)
        
        full_block_text = " ".join(block_text_parts).strip()
        if full_block_text: # Only add if there's actual text
            texts_for_embedding.append(full_block_text)
            corresponding_metadata.append({
                "source_pdf": block["source_pdf"],
                "page_number": block["page_number"],
                "original_group_content_snippet": full_block_text[:200] # For quick reference
                # You can add more detailed metadata here, like exact bounding boxes of the group if needed
            })
            
    return texts_for_embedding, corresponding_metadata


def create_faiss_index(texts_for_embedding, list_of_metadata, embedding_model):
    """
    Creates a FAISS index from text chunks and their metadata.

    Args:
        texts_for_embedding (list): List of text strings.
        list_of_metadata (list): List of metadata dictionaries corresponding to each text.
        embedding_model (SentenceTransformer): Initialized sentence transformer model.

    Returns:
        faiss.IndexIDMap or None: The created FAISS index, or None if an error occurs.
    """
    if not texts_for_embedding:
        logger.warning("No texts provided for embedding. Cannot create FAISS index.")
        return None
    try:
        logger.info(f"Generating embeddings for {len(texts_for_embedding)} text blocks...")
        embeddings = embedding_model.encode(texts_for_embedding, convert_to_tensor=False, show_progress_bar=True)
        embeddings_np = np.array(embeddings).astype('float32') # FAISS expects float32
        
        if embeddings_np.ndim == 1: # If only one text block, reshape
            embeddings_np = embeddings_np.reshape(1, -1)

        if embeddings_np.shape[0] != len(list_of_metadata):
            logger.error("Mismatch between number of embeddings and metadata entries. Aborting index creation.")
            return None

        dimension = embeddings_np.shape[1]
        
        # Using IndexFlatL2 as a simple example. For larger datasets, consider more advanced indexing like IndexIVFFlat.
        # IndexIDMap allows us to map FAISS's internal IDs back to our original document IDs/indices if needed,
        # but langchain's FAISS wrapper handles this internally when using `FAISS.from_texts` or `FAISS.from_embeddings`.
        # Here, we are building it more manually to show the steps, but FAISS.from_embeddings is often simpler.
        
        # For simplicity and compatibility with Langchain's typical FAISS usage (which often stores texts directly or uses IndexFlatL2 with metadata)
        # Let's consider using FAISS.from_embeddings which handles some of this abstraction.
        # However, if we want to save/load manually as planned, we might build it like this:

        index = faiss.IndexFlatL2(dimension)
        index_id_map = faiss.IndexIDMap(index) # Maps custom IDs (0 to n-1) to vectors
        ids = np.arange(len(texts_for_embedding))
        index_id_map.add_with_ids(embeddings_np, ids)

        logger.info(f"FAISS index created with {index_id_map.ntotal} vectors.")
        
        # Note: When using this index, you'll get back `ids`. You'll then use these `ids`
        # to look up the corresponding metadata and text from your `list_of_metadata` and `texts_for_embedding`.
        # Langchain's FAISS class abstracts this by having a `docstore` and `index_to_docstore_id`.
        return index_id_map

    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}", exc_info=True)
        return None

def save_faiss_index(index, texts_for_retrieval, metadata_for_retrieval, index_dir, index_name=config.DEFAULT_INDEX_NAME):
    """
    Saves the FAISS index, corresponding texts, and metadata to disk.

    Args:
        index (faiss.Index): The FAISS index object.
        texts_for_retrieval (list): The original text chunks.
        metadata_for_retrieval (list): The metadata for each chunk.
        index_dir (str): Directory to save the index files.
        index_name (str): Base name for the index files.
    """
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    index_path = os.path.join(index_dir, f"{index_name}.index")
    texts_path = os.path.join(index_dir, f"{index_name}.texts.json") # Using json for texts & metadata
    
    try:
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, index_path)

        # Save texts and metadata
        # We need to save these separately as the raw FAISS index only stores vectors.
        # Langchain's FAISS.save_local saves a .pkl for docstore and index_to_docstore_id.
        import json
        retrieval_data = {
            "texts": texts_for_retrieval,
            "metadata": metadata_for_retrieval
        }
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(retrieval_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Texts and metadata saved to {texts_path}")

    except Exception as e:
        logger.error(f"Error saving FAISS index or associated data: {e}", exc_info=True)


def load_faiss_index(index_dir, embedding_model_for_dim_check=None, index_name=config.DEFAULT_INDEX_NAME):
    """
    Loads the FAISS index, corresponding texts, and metadata from disk.

    Args:
        index_dir (str): Directory where the index files are saved.
        embedding_model_for_dim_check (SentenceTransformer, optional): Used to verify index dimension.
        index_name (str): Base name of the index files.

    Returns:
        tuple: (faiss.Index, list_of_texts, list_of_metadata) or (None, None, None) if loading fails.
    """
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
                logger.error(f"Loaded index dimension ({index.d}) does not match "
                             f"embedding model dimension ({expected_dim}).")
                return None, None, None

        import json
        with open(texts_path, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
        
        texts_for_retrieval = retrieval_data.get("texts", [])
        metadata_for_retrieval = retrieval_data.get("metadata", [])

        logger.info(f"FAISS index and {len(texts_for_retrieval)} text blocks with metadata loaded successfully.")
        return index, texts_for_retrieval, metadata_for_retrieval

    except Exception as e:
        logger.error(f"Error loading FAISS index or associated data: {e}", exc_info=True)
        return None, None, None


if __name__ == '__main__':
    logger.info("Starting Vector Indexer example...")

    # --- Initialize Embedding Model (ideally done once in your main orchestrator) ---
    try:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        # Specify trust_remote_code=True if required by the specific SentenceTransformer model
        sbert_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
        exit()

    # --- Example Extracted Data (simulate output from pdf_parser.py) ---
    sample_extracted_pages = [
        {
            "source_pdf": "doc1.pdf",
            "page_number": 1,
            "content": [
                {"type": "plain_text", "text": "This is the first paragraph of document 1.", "bbox_pil": []},
                {"type": "plain_text", "text": "It is followed by a summary.", "bbox_pil": []},
                {"type": "table", "extracted_text_list": ["Header A", "Header B", "Cell 1A", "Cell 1B"], "bbox_pil": []}
            ]
        },
        {
            "source_pdf": "doc1.pdf",
            "page_number": 2,
            "content": [
                {"type": "plain_text", "text": "Page two starts here.", "bbox_pil": []},
            ]
        },
        {
            "source_pdf": "doc2.pdf",
            "page_number": 1,
            "content": [
                {"type": "plain_text", "text": "Document 2 discusses other topics.", "bbox_pil": []},
                {"type": "plain_text", "text": "It has an important conclusion.", "bbox_pil": []}
            ]
        }
    ]

    logger.info("Grouping extracted content into blocks...")
    grouped_content_blocks = group_extracted_content_to_blocks(sample_extracted_pages)
    for i, block in enumerate(grouped_content_blocks):
        logger.debug(f"Group {i}: Source: {block['source_pdf']}, Page: {block['page_number']}, Content items: {len(block['group_content'])}")

    logger.info("Converting grouped blocks to texts and metadata...")
    texts, metadata = convert_grouped_blocks_to_texts_and_metadata(grouped_content_blocks)
    if texts:
        for i in range(min(3, len(texts))): # Log first few
            logger.debug(f"Text {i}: {texts[i][:100]}... | Metadata: {metadata[i]}")
    else:
        logger.warning("No texts were generated from grouped blocks.")


    if texts:
        # --- Create and Save Index ---
        logger.info("Creating FAISS index...")
        faiss_index_object = create_faiss_index(texts, metadata, sbert_model)

        if faiss_index_object:
            index_storage_dir = os.path.join(config.DEFAULT_INDEX_DIR, "my_test_index")
            logger.info(f"Saving index to directory: {index_storage_dir}")
            save_faiss_index(faiss_index_object, texts, metadata, index_storage_dir, index_name="document_index")

            # --- Load Index (for testing) ---
            logger.info("Attempting to load the saved index...")
            loaded_index, loaded_texts, loaded_metadata = load_faiss_index(index_storage_dir, 
                                                                          embedding_model_for_dim_check=sbert_model,
                                                                          index_name="document_index")
            if loaded_index and loaded_texts and loaded_metadata:
                logger.info(f"Successfully loaded index with {loaded_index.ntotal} vectors and {len(loaded_texts)} text entries.")
                assert loaded_index.ntotal == faiss_index_object.ntotal
                assert len(loaded_texts) == len(texts)
                assert loaded_texts[0] == texts[0] if texts else True
                assert loaded_metadata[0]['source_pdf'] == metadata[0]['source_pdf'] if metadata else True
                logger.info("Index load test passed.")
            else:
                logger.error("Failed to load or verify the saved index.")
        else:
            logger.error("FAISS index creation failed.")
    else:
        logger.warning("No text blocks to index.")
    logger.info("Vector Indexer example finished.")