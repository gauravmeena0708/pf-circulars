# vector_indexer.py

import os
import json
import faiss
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import config

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


def group_extracted_content_to_blocks(extracted_pages_data):
    # Groups plain text and nearby tables from extracted PDF pages into contextual blocks.
    grouped_blocks = []
    for page_data in extracted_pages_data:
        page_number = page_data['page_number']
        source_pdf = page_data['source_pdf']
        page_content = page_data['content'] 

        if not page_content:
            continue

        current_group = []
        for i, content_item in enumerate(page_content):
            current_group.append(content_item)
            is_last_item_on_page = (i == len(page_content) - 1)
            finalize_group = False
            if is_last_item_on_page:
                finalize_group = True
            else:
                current_is_table = content_item["type"] == "table"
                next_is_table = (page_content[i + 1]["type"] == "table")
                if not current_is_table and not next_is_table:
                    pass  # Continue accumulating text
                else: 
                    finalize_group = True
            
            if finalize_group and current_group:
                grouped_blocks.append({
                    "source_pdf": source_pdf,
                    "page_number": page_number,
                    "group_content": list(current_group), 
                })
                current_group = []
        
        # Catch any trailing group for the page
        if current_group: 
            grouped_blocks.append({
                "source_pdf": source_pdf,
                "page_number": page_number,
                "group_content": list(current_group),
            })
            
    return grouped_blocks


def get_start_page_num_from_block(block):
    # Helper to get the integer start page number from a block's page_number metadata.
    page_meta = block.get('page_number')
    if isinstance(page_meta, str) and '-' in page_meta:
        try:
            return int(page_meta.split('-')[0])
        except ValueError:
            logger.warning(f"Could not parse start page from string: '{page_meta}'")
            return float('inf') 
    elif isinstance(page_meta, (int, float)):
        return int(page_meta)
    logger.warning(f"Unexpected page_meta type or value: {page_meta}. Assigning high page number for sort stability.")
    return float('inf')


def merge_spanning_table_blocks(grouped_blocks_input):
    # Merges blocks that represent parts of the same table spanning across pages.
    if not grouped_blocks_input:
        return []

    try:
        processing_list = sorted(list(grouped_blocks_input), 
                                 key=lambda b: (b.get('source_pdf', ''), get_start_page_num_from_block(b)))
    except Exception as e:
        logger.error(f"Error during initial sort in merge_spanning_table_blocks: {e}")
        return list(grouped_blocks_input)

    final_merged_blocks = []
    i = 0
    while i < len(processing_list):
        current_block = processing_list[i]
        merged_anything_for_current_block = False

        if current_block.get('group_content') and current_block['group_content'][-1].get('type') == 'table':
            last_item_in_current = current_block['group_content'][-1]
            if last_item_in_current.get('is_at_page_bottom', False):
                merge_target_table_item = last_item_in_current
                current_doc_pdf = current_block.get('source_pdf')
                current_start_page = get_start_page_num_from_block(current_block)

                j = i + 1
                indices_of_blocks_to_consume = []
                while j < len(processing_list):
                    next_block = processing_list[j]
                    if next_block.get('source_pdf') != current_doc_pdf:
                        break

                    next_start_page = get_start_page_num_from_block(next_block)
                    if next_start_page <= current_start_page:
                        j += 1
                        continue

                    if next_start_page == current_start_page + 1:
                        if next_block.get('group_content') and next_block['group_content'][0].get('type') == 'table':
                            first_item_in_next = next_block['group_content'][0]
                            if first_item_in_next.get('is_at_page_top', False):
                                # Merge table data
                                merge_target_table_item['extracted_text_list'].extend(first_item_in_next.get('extracted_text_list', []))
                                merge_target_table_item['raw_cells'].extend(first_item_in_next.get('raw_cells', []))
                                merge_target_table_item['is_at_page_bottom'] = first_item_in_next.get('is_at_page_bottom', False)

                                # Update page range
                                curr_page_meta = str(current_block.get('page_number'))
                                next_page_meta = str(next_block.get('page_number'))
                                start_p = curr_page_meta.split('-')[0]
                                end_p = next_page_meta.split('-')[-1]
                                current_block['page_number'] = f"{start_p}-{end_p}"
                                current_block['metadata_is_merged'] = True

                                remaining_items_in_next = next_block['group_content'][1:]
                                if remaining_items_in_next:
                                    current_block['group_content'].extend(remaining_items_in_next)

                                indices_of_blocks_to_consume.append(j)
                                merged_anything_for_current_block = True
                                current_start_page = next_start_page
                                j += 1
                                continue
                    break

                if indices_of_blocks_to_consume:
                    for idx_to_pop in sorted(indices_of_blocks_to_consume, reverse=True):
                        processing_list.pop(idx_to_pop)
                    continue

        final_merged_blocks.append(current_block)
        i += 1

    return final_merged_blocks


def get_recursive_text_splitter_class():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter


def convert_grouped_blocks_to_texts_and_metadata(grouped_blocks):
    """
    Converts grouped blocks into document-aware semantic chunks with rich metadata.
    Uses configurable chunk sizes and retains parent context for small-to-big retrieval.
    """
    RecursiveCharacterTextSplitter = get_recursive_text_splitter_class()
    texts_for_embedding = []
    corresponding_metadata = []

    chunk_size = getattr(config, "CHUNK_SIZE", 800)
    chunk_overlap = getattr(config, "CHUNK_OVERLAP", 150)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        is_separator_regex=False,
    )

    for block in grouped_blocks:
        block_text_parts = []
        source_doc = block.get("source_pdf", "Unknown")
        page_info_str = f"Page(s): {block.get('page_number', 'N/A')}"

        for content_item in block["group_content"]:
            if content_item["type"] == "plain_text":
                block_text_parts.append(content_item["text"])
            elif content_item["type"] == "table":
                table_str = f"Table Content ({page_info_str}): " + "; ".join(
                    [f"[{cell_text}]" for cell_text in content_item.get("extracted_text_list", [])]
                )
                block_text_parts.append(table_str)
        
        full_block_text = "\n".join(block_text_parts).strip()
        if full_block_text:
            chunks = text_splitter.split_text(full_block_text)
            
            for chunk_idx, chunk in enumerate(chunks):
                texts_for_embedding.append(chunk)
                metadata_item = {
                    "source_pdf": source_doc,
                    "page_number": str(block["page_number"]),
                    "chunk_id": f"{source_doc}_p{block['page_number']}_c{chunk_idx}",
                    "parent_context": full_block_text[:1200],  # Surrounding parent section for LLM synthesis
                    "original_group_content_snippet": chunk[:200]
                }
                if block.get('metadata_is_merged'):
                    metadata_item['is_merged_table'] = True
                corresponding_metadata.append(metadata_item)
            
    return texts_for_embedding, corresponding_metadata


def create_faiss_index(texts_for_embedding, list_of_metadata, embedding_model):
    """
    Creates an L2-normalized FAISS IndexFlatIP index for mathematically sound cosine similarity.
    """
    if not texts_for_embedding:
        logger.warning("No texts provided for embedding. Cannot create FAISS index.")
        return None
    try:
        logger.info(f"Generating normalized embeddings for {len(texts_for_embedding)} text blocks...")
        # Explicit normalization ensures inner product equals cosine similarity
        embeddings = embedding_model.encode(
            texts_for_embedding, 
            convert_to_tensor=False, 
            normalize_embeddings=True, 
            show_progress_bar=True
        )
        embeddings_np = np.array(embeddings).astype('float32') 
        
        if embeddings_np.ndim == 1: 
            embeddings_np = embeddings_np.reshape(1, -1)

        if embeddings_np.shape[0] != len(list_of_metadata):
            logger.error(f"Mismatch between number of embeddings ({embeddings_np.shape[0]}) and metadata entries ({len(list_of_metadata)}). Aborting index creation.")
            return None

        dimension = embeddings_np.shape[1]
        
        # Using IndexFlatIP for Cosine Similarity on unit-normalized vectors
        index = faiss.IndexFlatIP(dimension)
        index_id_map = faiss.IndexIDMap(index) 
        ids = np.arange(len(texts_for_embedding))
        index_id_map.add_with_ids(embeddings_np, ids)
        logger.info(f"FAISS IndexFlatIP created with {index_id_map.ntotal} normalized vectors (dim={dimension}).")
        return index_id_map

    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}", exc_info=True)
        return None


def save_faiss_index(index, texts_for_retrieval, metadata_for_retrieval, index_dir, index_name=config.DEFAULT_INDEX_NAME):
    # Saves the FAISS index, corresponding texts, and metadata to disk.
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    index_path = os.path.join(index_dir, f"{index_name}.index")
    texts_path = os.path.join(index_dir, f"{index_name}.texts.json") 
    try:
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, index_path)
        retrieval_data = {"texts": texts_for_retrieval, "metadata": metadata_for_retrieval}
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(retrieval_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Texts and metadata saved to {texts_path}")
    except Exception as e:
        logger.error(f"Error saving FAISS index or associated data: {e}", exc_info=True)


def load_faiss_index(index_dir, embedding_model_for_dim_check=None, index_name=config.DEFAULT_INDEX_NAME):
    # Loads the FAISS index, corresponding texts, and metadata from disk.
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
                logger.error(f"Loaded index dimension ({index.d}) does not match embedding model dimension ({expected_dim}).")
                return None, None, None
        with open(texts_path, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
        texts_for_retrieval = retrieval_data.get("texts", [])
        metadata_for_retrieval = retrieval_data.get("metadata", [])
        logger.info(f"FAISS index and {len(texts_for_retrieval)} text blocks with metadata loaded successfully.")
        return index, texts_for_retrieval, metadata_for_retrieval
    except Exception as e:
        logger.error(f"Error loading FAISS index or associated data: {e}", exc_info=True)
        return None, None, None
