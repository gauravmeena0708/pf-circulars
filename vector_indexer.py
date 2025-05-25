# vector_indexer.py

import os
import faiss
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import config # Assuming your config.py is in the same directory or accessible in PYTHONPATH

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
                # Finalize group if current is table, or next is table,
                # or current is text and next is also table.
                # Continue grouping only if current is text and next is text.
                if not current_is_table and not next_is_table:
                    pass # Continue accumulating text
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
    elif isinstance(page_meta, (int, float)): # Allow float for inf case in sorting
        return int(page_meta)
    logger.warning(f"Unexpected page_meta type or value: {page_meta}. Assigning high page number for sort stability.")
    return float('inf')


def merge_spanning_table_blocks(grouped_blocks_input):
    # Merges blocks that represent parts of the same table spanning across pages.
    if not grouped_blocks_input:
        return []

    # Uncomment for detailed debugging of input flags if needed
    # logger.debug("--- Entering merge_spanning_table_blocks ---")
    # for idx, block_to_inspect in enumerate(grouped_blocks_input):
    #     if block_to_inspect.get('source_pdf') == "Circular_RevisedLeaseAccommodation_24042025.pdf":
    #         logger.debug(f"Inspecting Input Block {idx} for '{block_to_inspect.get('source_pdf')}' - Page '{block_to_inspect.get('page_number')}':")
    #         if block_to_inspect.get('group_content'):
    #             for item_idx, item in enumerate(block_to_inspect['group_content']):
    #                 if item.get('type') == 'table':
    #                     logger.debug(f"  Item {item_idx} (Table): is_at_page_top: {item.get('is_at_page_top')}, is_at_page_bottom: {item.get('is_at_page_bottom')}")
    # logger.debug("--- Finished initial inspection of relevant blocks ---")

    try:
        # Initial sort is critical for the lookahead logic.
        processing_list = sorted(list(grouped_blocks_input), 
                                 key=lambda b: (b.get('source_pdf', ''), get_start_page_num_from_block(b)))
    except Exception as e:
        logger.error(f"Error during initial sort in merge_spanning_table_blocks: {e}")
        return list(grouped_blocks_input) # Return original if sort fails

    final_merged_blocks = []
    i = 0
    while i < len(processing_list):
        current_block = processing_list[i]
        
        last_item_curr = current_block['group_content'][-1] if current_block.get('group_content') else None

        # Check if current_block is a candidate to start or continue a merge
        # It must end with a table that is marked as being at the bottom of its page.
        if last_item_curr and last_item_curr.get('type') == 'table' and \
           last_item_curr.get('is_at_page_bottom') is True:
            
            current_block_effective_last_page = get_start_page_num_from_block(current_block)
            if isinstance(current_block.get('page_number'), str) and '-' in current_block.get('page_number'):
                try: # If current_block is already a merged span, get its actual last page
                    current_block_effective_last_page = int(str(current_block.get('page_number')).split('-')[-1])
                except ValueError:
                    logger.warning(f"Could not parse last page for (already) merged block: {current_block.get('page_number')}")
            
            indices_of_blocks_to_consume = [] # To hold indices from processing_list that get merged into current_block

            # Look ahead for continuations or ignorable blocks
            for j in range(i + 1, len(processing_list)):
                potential_block_for_merge_or_skip = processing_list[j]

                # Stop if we've moved to a different PDF
                if potential_block_for_merge_or_skip.get('source_pdf') != current_block.get('source_pdf'):
                    break 
                
                potential_block_start_page = get_start_page_num_from_block(potential_block_for_merge_or_skip)
                first_item_potential = potential_block_for_merge_or_skip['group_content'][0] if potential_block_for_merge_or_skip.get('group_content') else None

                # Condition 1: Direct table continuation
                if (first_item_potential and 
                    first_item_potential.get('type') == 'table' and 
                    first_item_potential.get('is_at_page_top') is True and
                    current_block_effective_last_page + 1 == potential_block_start_page):
                    
                    logger.debug(f"Merging: PDF '{current_block.get('source_pdf')}' current_block_pages '{current_block.get('page_number')}' with next_block_pages '{potential_block_for_merge_or_skip.get('page_number')}'")
                    
                    current_block['group_content'].extend(potential_block_for_merge_or_skip['group_content'])
                    
                    current_start_page_str = str(current_block.get('page_number')).split('-')[0] # Start of current span
                    next_end_page_str = str(potential_block_for_merge_or_skip.get('page_number')).split('-')[-1] # End of block being merged
                    current_block['page_number'] = f"{current_start_page_str}-{next_end_page_str}"
                    current_block['metadata_is_merged'] = True # Mark that this block is a result of merging
                    
                    indices_of_blocks_to_consume.append(j) # Mark this block as consumed
                    
                    try: # Update effective last page of current_block for further chained merges
                        current_block_effective_last_page = int(next_end_page_str)
                    except ValueError:
                        logger.error(f"Could not update current_block_effective_last_page from {next_end_page_str}")
                        break # Stop this chain of merges if page number becomes unparsable
                
                # Condition 2: Intermediate ignorable block (must be on the same page as current_block_effective_last_page OR the immediately next page)
                elif not (first_item_potential and first_item_potential.get('type') == 'table') and \
                     (potential_block_start_page == current_block_effective_last_page or \
                      potential_block_start_page == current_block_effective_last_page + 1):
                    
                    is_small_ignorable = True
                    if not potential_block_for_merge_or_skip.get('group_content') or len(potential_block_for_merge_or_skip['group_content']) > 2 : # Example: more than 2 items
                        is_small_ignorable = False
                    if is_small_ignorable: # Check text length if it has few items
                        total_text_len = sum(len(item.get('text','')) for item in potential_block_for_merge_or_skip['group_content'] if item.get('type') == 'plain_text')
                        if total_text_len > 50: # Arbitrary threshold, e.g., for short footers/headers
                            is_small_ignorable = False
                    
                    if is_small_ignorable:
                        logger.debug(f"Skipping ignorable intermediate block: PDF {potential_block_for_merge_or_skip.get('source_pdf')} Page {potential_block_for_merge_or_skip.get('page_number')}")
                        indices_of_blocks_to_consume.append(j) # Mark this ignorable block as consumed
                        # Do NOT update current_block_effective_last_page here, just continue lookahead
                    else:
                        break # Intermediate block is too significant, stop lookahead for this current_block
                else:
                    # Not a direct table continuation and not an ignorable intermediate block on the expected page path
                    break # Stop lookahead for current_block

            # After trying to merge current_block with all subsequent candidates (j loop)
            if indices_of_blocks_to_consume:
                # Remove the consumed blocks from processing_list (in reverse order of index to avoid shifting issues)
                for idx_to_pop in sorted(indices_of_blocks_to_consume, reverse=True):
                    processing_list.pop(idx_to_pop)
                # current_block (at index i) was modified.
                # The outer loop should re-evaluate this modified current_block against what's now processing_list[i+1].
                # By not incrementing 'i' here, and letting the 'while i < len(processing_list)' re-evaluate,
                # and since items were popped, we effectively restart the attempt to merge the *now extended* current_block.
                # So, we `continue` to the next iteration of the `while i` loop.
                continue # Re-process current_block (which is processing_list[i]) as it has been modified
        
        # If current_block didn't end with a table, or wasn't a candidate for merge_start,
        # or if it was a candidate but no merge/skip occurred with subsequent blocks in the j-loop.
        final_merged_blocks.append(current_block)
        i += 1
            
    return final_merged_blocks


def convert_grouped_blocks_to_texts_and_metadata(grouped_blocks):
    # Converts grouped blocks into plain text strings for embedding and extracts metadata.
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
            metadata_item = {
                "source_pdf": block["source_pdf"],
                "page_number": str(block["page_number"]), # Ensure page_number is a string for consistency
                "original_group_content_snippet": full_block_text[:200]
            }
            if block.get('metadata_is_merged'): # Check if this flag was set during merge
                metadata_item['is_merged_table'] = True
            corresponding_metadata.append(metadata_item)
            
    return texts_for_embedding, corresponding_metadata


def create_faiss_index(texts_for_embedding, list_of_metadata, embedding_model):
    # Creates a FAISS index from text chunks and their metadata.
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
    # Saves the FAISS index, corresponding texts, and metadata to disk.
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

    sample_extracted_pages = [
        { 
            "source_pdf": "doc1.pdf", "page_number": 1, "content": [
                {"type": "plain_text", "text": "Text on page 1 before table."},
                {"type": "table", "extracted_text_list": ["Table1 Page1 Header", "Table1 Page1 Data"], "is_at_page_top": False, "is_at_page_bottom": True}
            ]
        },
        { # Intermediate small block (e.g., footer) that should be on page 1 to be skipped correctly
            "source_pdf": "doc1.pdf", "page_number": 1, "content": [ 
                {"type": "plain_text", "text": "Page 1 footer"}
            ]
        },
        { 
            "source_pdf": "doc1.pdf", "page_number": 2, "content": [
                {"type": "table", "extracted_text_list": ["Table1 Page2 Header", "Table1 Page2 Data"], "is_at_page_top": True, "is_at_page_bottom": False},
                {"type": "plain_text", "text": "Text on page 2 after table."}
            ]
        },
        { 
            "source_pdf": "doc1.pdf", "page_number": 3, "content": [
                 {"type": "table", "extracted_text_list": ["Separate Table on Page 3"], "is_at_page_top": True, "is_at_page_bottom": True}
            ]
        },
        { 
            "source_pdf": "doc2.pdf", "page_number": 1, "content": [
                {"type": "plain_text", "text": "Document 2 content."}
            ]
        }
    ]

    logger.info("Grouping extracted content into blocks...")
    grouped_content_blocks = group_extracted_content_to_blocks(sample_extracted_pages)
    logger.info(f"Number of blocks after initial grouping: {len(grouped_content_blocks)}")
    # for i, block in enumerate(grouped_content_blocks):
    #     logger.debug(f"Initial Group {i}: PDF: {block['source_pdf']}, Page: {block['page_number']}, Items: {len(block['group_content'])}")

    logger.info("Merging spanning table blocks...")
    merged_final_blocks = merge_spanning_table_blocks(grouped_content_blocks) 
    logger.info(f"Number of blocks after merging: {len(merged_final_blocks)}")
    for i, block in enumerate(merged_final_blocks):
        page_num_meta = block.get('page_number', 'N/A')
        logger.debug(f"Final Block {i}: PDF: {block['source_pdf']}, Page(s): {page_num_meta}, Items: {len(block['group_content'])}")
        if block.get('metadata_is_merged'):
            logger.debug(f"  ^-- This block contains a merged table spanning {page_num_meta}.")

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
            else:
                logger.error("Failed to load or verify the saved index.")
        else:
            logger.error("FAISS index creation failed.")
    else:
        logger.warning("No text blocks to index.")
    logger.info("Vector Indexer example finished.")