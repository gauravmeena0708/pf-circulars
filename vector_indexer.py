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
                    pass 
                else: 
                    finalize_group = True
            
            if finalize_group and current_group:
                grouped_blocks.append({
                    "source_pdf": source_pdf,
                    "page_number": page_number,
                    "group_content": list(current_group), 
                })
                current_group = []
        
        if current_group: # Should not happen if logic above is correct, but as a safeguard
            grouped_blocks.append({
                "source_pdf": source_pdf,
                "page_number": page_number,
                "group_content": list(current_group),
            })
            
    return grouped_blocks

# In vector_indexer.py

def get_start_page_num_from_block(block):
    """Helper to get the integer start page number from a block's page_number metadata."""
    page_meta = block.get('page_number')
    if isinstance(page_meta, str) and '-' in page_meta:
        try:
            return int(page_meta.split('-')[0])
        except ValueError:
            logger.warning(f"Could not parse start page from string: '{page_meta}'")
            return float('inf') 
    elif isinstance(page_meta, (int, float)):
        return int(page_meta)
    logger.warning(f"Unexpected page_meta type or value: {page_meta}. Assigning high page number.")
    return float('inf')

def merge_spanning_table_blocks(grouped_blocks_input):
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
        
        merged_current_block_in_this_pass = False
        last_item_curr = current_block['group_content'][-1] if current_block.get('group_content') else None

        # Only attempt to initiate a merge if current_block ends with a table
        if last_item_curr and last_item_curr.get('type') == 'table':
            # Use is_at_page_bottom if available and True, otherwise be more lenient if it's a table.
            # This leniency might be needed if footers prevent accurate is_at_page_bottom.
            is_candidate_for_merge_start = last_item_curr.get('is_at_page_bottom') is True 
            # One could add more heuristics here if is_at_page_bottom is unreliable.
            # For now, we rely on it. If it's False for page 1 table, merging will fail.

            if is_candidate_for_merge_start:
                current_block_last_page_val = get_start_page_num_from_block(current_block)
                if isinstance(current_block.get('page_number'), str) and '-' in current_block.get('page_number'):
                    try:
                        current_block_last_page_val = int(str(current_block.get('page_number')).split('-')[-1])
                    except ValueError:
                        logger.warning(f"Could not parse last page for already merged block: {current_block.get('page_number')}")
                
                # Look ahead for a suitable table continuation
                indices_to_pop = []
                for j in range(i + 1, len(processing_list)):
                    potential_next_block = processing_list[j]

                    if potential_next_block.get('source_pdf') != current_block.get('source_pdf'):
                        break 
                    
                    potential_next_block_start_page = get_start_page_num_from_block(potential_next_block)
                    first_item_potential_next = potential_next_block['group_content'][0] if potential_next_block.get('group_content') else None

                    # Check if potential_next_block is the direct continuation
                    if (first_item_potential_next and 
                        first_item_potential_next.get('type') == 'table' and 
                        first_item_potential_next.get('is_at_page_top') is True and
                        current_block_last_page_val + 1 == potential_next_block_start_page):
                        
                        logger.debug(f"Attempting to merge: {current_block.get('source_pdf')} pages '{current_block.get('page_number')}' (last item type: {last_item_curr.get('type')}, bottom: {last_item_curr.get('is_at_page_bottom')}) with pages '{potential_next_block.get('page_number')}' (first item type: {first_item_potential_next.get('type')}, top: {first_item_potential_next.get('is_at_page_top')})")

                        # Merge content
                        current_block['group_content'].extend(potential_next_block['group_content'])
                        
                        # Update page_number metadata
                        current_start_page_str = str(current_block.get('page_number')).split('-')[0]
                        next_end_page_str = str(potential_next_block.get('page_number')).split('-')[-1]
                        current_block['page_number'] = f"{current_start_page_str}-{next_end_page_str}"
                        current_block['metadata_is_merged'] = True
                        
                        indices_to_pop.append(j) # Mark this block to be removed from processing_list
                        merged_current_block_in_this_pass = True
                        
                        # Update current_block_last_page_val for potential further merges in this same pass
                        try:
                            current_block_last_page_val = int(next_end_page_str)
                        except ValueError:
                            logger.error(f"Could not update current_block_last_page_val from {next_end_page_str}")
                            break # Stop trying to merge this chain further if page number is corrupt
                        
                        # Continue checking with the *next* potential_next_block (j+1) against the *now extended* current_block
                    
                    # If it's not an immediate table continuation, but a small intermediate block,
                    # we might allow skipping it.
                    elif not (first_item_potential_next and first_item_potential_next.get('type') == 'table'):
                        # This is an intermediate non-table block. If it's "small", allow lookahead to continue.
                        # Define "small": e.g., 1-2 plain_text items, total text length < 50 chars.
                        is_small_ignorable = True
                        if not potential_next_block.get('group_content') or len(potential_next_block['group_content']) > 2 :
                            is_small_ignorable = False
                        if is_small_ignorable:
                            total_text_len = sum(len(item.get('text','')) for item in potential_next_block['group_content'] if item.get('type') == 'plain_text')
                            if total_text_len > 50: # Arbitrary threshold
                                is_small_ignorable = False
                        
                        if not is_small_ignorable:
                            break # Intermediate block is too significant, stop lookahead for current_block
                        else:
                            logger.debug(f"Skipping ignorable intermediate block: PDF {potential_next_block.get('source_pdf')} Page {potential_next_block.get('page_number')}")
                            indices_to_pop.append(j) # Mark ignorable block for removal
                    else:
                        # It's a table, but not meeting merge criteria (e.g. wrong page, not is_at_page_top)
                        break # Stop lookahead

                # Remove merged/skipped blocks from processing_list in reverse order of index
                for idx_to_pop in sorted(indices_to_pop, reverse=True):
                    processing_list.pop(idx_to_pop)
                
                if merged_current_block_in_this_pass:
                    # current_block (processing_list[i]) was modified and subsequent blocks popped.
                    # The outer loop should re-evaluate this modified current_block from its current position 'i'.
                    # So, we don't increment 'i' yet and let the while loop continue.
                    # To ensure 'i' effectively stays on the modified current_block for the next main iteration:
                    i -=1 # Decrement i, so that i++ at the end of the loop brings it back to current block index
                          # This is because elements were popped, shifting indices. Or better, just don't increment.
                          # The cleaner way is to append current_block and advance i, and if merged,
                          # replace processing_list[i] and manage popping correctly, then restart from i.
                          # For now, the i-- strategy is simpler if list is rebuilt or i is reset.

                          # Let's stick to: if a merge happened, the current_block is now bigger.
                          # We don't add it to final_merged_blocks yet. 'i' is not incremented.
                          # The loop `while i < len(processing_list)` will run again with the modified current_block.
                    pass # Let the loop re-evaluate from the same 'i' with a potentially modified current_block

        # If no merge was initiated or completed for current_block with any subsequent blocks
        if not merged_current_block_in_this_pass:
            final_merged_blocks.append(current_block)
        
        i += 1 # Move to the next item in the (potentially modified) processing_list
            
    return final_merged_blocks


def convert_grouped_blocks_to_texts_and_metadata(grouped_blocks):
    """
    Converts grouped blocks into plain text strings for embedding and extracts metadata.
    """
    texts_for_embedding = []
    corresponding_metadata = []

    for block in grouped_blocks:
        block_text_parts = []
        # Determine the page range string for this block
        page_info_str = f"Page(s): {block.get('page_number', 'N/A')}"

        for content_item in block["group_content"]:
            if content_item["type"] == "plain_text":
                block_text_parts.append(content_item["text"])
            elif content_item["type"] == "table":
                # Include page info in the table representation
                table_str = f"Table Content ({page_info_str}): " + "; ".join(
                    [f"[{cell_text}]" for cell_text in content_item.get("extracted_text_list", [])]
                )
                block_text_parts.append(table_str)
        
        full_block_text = " ".join(block_text_parts).strip()
        if full_block_text: 
            texts_for_embedding.append(full_block_text)
            metadata = {
                "source_pdf": block["source_pdf"],
                "page_number": str(block["page_number"]), # Ensure page_number is a string for consistency
                "original_group_content_snippet": full_block_text[:200]
            }
            if block.get('metadata_is_merged'):
                metadata['is_merged_table'] = True
            corresponding_metadata.append(metadata)
            
    return texts_for_embedding, corresponding_metadata


def create_faiss_index(texts_for_embedding, list_of_metadata, embedding_model):
    if not texts_for_embedding:
        logger.warning("No texts provided for embedding. Cannot create FAISS index.")
        return None
    try:
        logger.info(f"Generating embeddings for {len(texts_for_embedding)} text blocks...")
        embeddings = embedding_model.encode(texts_for_embedding, convert_to_tensor=False, show_progress_bar=True)
        embeddings_np = np.array(embeddings).astype('float32') 
        
        if embeddings_np.ndim == 1: 
            embeddings_np = embeddings_np.reshape(1, -1)

        if embeddings_np.shape[0] != len(list_of_metadata):
            logger.error(f"Mismatch between number of embeddings ({embeddings_np.shape[0]}) and metadata entries ({len(list_of_metadata)}). Aborting index creation.")
            return None

        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index_id_map = faiss.IndexIDMap(index) 
        ids = np.arange(len(texts_for_embedding))
        index_id_map.add_with_ids(embeddings_np, ids)
        logger.info(f"FAISS index created with {index_id_map.ntotal} vectors.")
        return index_id_map

    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}", exc_info=True)
        return None

def save_faiss_index(index, texts_for_retrieval, metadata_for_retrieval, index_dir, index_name=config.DEFAULT_INDEX_NAME):
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    index_path = os.path.join(index_dir, f"{index_name}.index")
    texts_path = os.path.join(index_dir, f"{index_name}.texts.json") 
    try:
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, index_path)
        import json
        retrieval_data = {"texts": texts_for_retrieval, "metadata": metadata_for_retrieval}
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(retrieval_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Texts and metadata saved to {texts_path}")
    except Exception as e:
        logger.error(f"Error saving FAISS index or associated data: {e}", exc_info=True)


def load_faiss_index(index_dir, embedding_model_for_dim_check=None, index_name=config.DEFAULT_INDEX_NAME):
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
    try:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        sbert_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
        exit()

    # --- Example Extracted Data (simulate output from pdf_parser.py) ---
    # This example should be updated to include is_at_page_top/bottom flags for testing merge
    sample_extracted_pages = [
        { # PDF 1, Page 1
            "source_pdf": "doc1.pdf", "page_number": 1, "content": [
                {"type": "plain_text", "text": "Text before table on page 1."},
                {"type": "table", "extracted_text_list": ["Table1 PartA Row1"], "is_at_page_top": False, "is_at_page_bottom": True}
            ]
        },
        { # PDF 1, Page 2
            "source_pdf": "doc1.pdf", "page_number": 2, "content": [
                {"type": "table", "extracted_text_list": ["Table1 PartB Row1"], "is_at_page_top": True, "is_at_page_bottom": False},
                {"type": "plain_text", "text": "Text after table on page 2."}
            ]
        },
        { # PDF 1, Page 3 - another table not connected
            "source_pdf": "doc1.pdf", "page_number": 3, "content": [
                 {"type": "table", "extracted_text_list": ["Separate Table"], "is_at_page_top": True, "is_at_page_bottom": True}
            ]
        },
        { # PDF 2, Page 1
            "source_pdf": "doc2.pdf", "page_number": 1, "content": [
                {"type": "plain_text", "text": "Document 2 discusses other topics."}
            ]
        }
    ]

    logger.info("Grouping extracted content into blocks...")
    grouped_content_blocks = group_extracted_content_to_blocks(sample_extracted_pages)
    logger.info(f"Number of blocks after initial grouping: {len(grouped_content_blocks)}")
    for i, block in enumerate(grouped_content_blocks):
        logger.debug(f"Initial Group {i}: PDF: {block['source_pdf']}, Page: {block['page_number']}, Items: {len(block['group_content'])}")

    logger.info("Merging spanning table blocks...")
    merged_final_blocks = merge_spanning_table_blocks(grouped_content_blocks)
    logger.info(f"Number of blocks after merging: {len(merged_final_blocks)}")
    for i, block in enumerate(merged_final_blocks):
        logger.debug(f"Merged Group {i}: PDF: {block['source_pdf']}, Page(s): {block['page_number']}, Items: {len(block['group_content'])}")
        if block.get('metadata_is_merged'):
            logger.debug(f"  ^-- This block contains a merged table.")


    logger.info("Converting merged blocks to texts and metadata...")
    texts, metadata = convert_grouped_blocks_to_texts_and_metadata(merged_final_blocks)
    if texts:
        for i in range(min(3, len(texts))): 
            logger.debug(f"Text {i}: {texts[i][:150]}... | Metadata: {metadata[i]}")
    else:
        logger.warning("No texts were generated from merged blocks.")


    if texts:
        logger.info("Creating FAISS index...")
        faiss_index_object = create_faiss_index(texts, metadata, sbert_model)

        if faiss_index_object:
            index_storage_dir = os.path.join(config.DEFAULT_INDEX_DIR, "my_test_index_merged")
            logger.info(f"Saving index to directory: {index_storage_dir}")
            save_faiss_index(faiss_index_object, texts, metadata, index_storage_dir, index_name="document_index_merged")

            logger.info("Attempting to load the saved index...")
            loaded_index, loaded_texts, loaded_metadata = load_faiss_index(index_storage_dir, 
                                                                          embedding_model_for_dim_check=sbert_model,
                                                                          index_name="document_index_merged")
            if loaded_index and loaded_texts and loaded_metadata:
                logger.info(f"Successfully loaded index with {loaded_index.ntotal} vectors and {len(loaded_texts)} text entries.")
                # Add more specific assertions for merged content if possible
            else:
                logger.error("Failed to load or verify the saved index.")
        else:
            logger.error("FAISS index creation failed.")
    else:
        logger.warning("No text blocks to index.")
    logger.info("Vector Indexer example finished.")