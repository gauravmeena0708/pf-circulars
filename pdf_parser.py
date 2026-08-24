# pdf_parser.py
import os
import re
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


def convert_pdf_page_to_image(pdf_path, page_num, dpi=config.PDF_TO_IMAGE_DPI):
    """Converts a single page of a PDF to a PIL Image."""
    try:
        images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num + 1, last_page=page_num + 1)
        if images:
            return images[0]
    except Exception as e:
        logger.error(f"Error converting page {page_num} of PDF '{pdf_path}' to image: {e}")
    return None


def pil_to_cv2(pil_image):
    """Converts a PIL Image to an OpenCV image (BGR format)."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _extract_native_page_text(doc, page_num, source_pdf):
    """
    Attempts native text extraction from a PDF page using PyMuPDF.
    Includes a safety net to verify text sufficiency and quality.
    
    Returns:
        tuple: (success (bool), page_data (dict or None))
    """
    try:
        page = doc[page_num]
        raw_text = page.get_text("text") or ""
        words = raw_text.split()
        
        # Safety Net Checks:
        # 1. Minimum word count threshold
        if len(words) < config.NATIVE_TEXT_MIN_WORDS:
            return False, None
            
        # 2. Printable character ratio (detect broken/garbled font mappings)
        printable_count = sum(1 for c in raw_text if c.isprintable() or c.isspace())
        if len(raw_text) > 0 and (printable_count / len(raw_text)) < 0.80:
            logger.info(f"Page {page_num + 1} of '{source_pdf}' failed printable character ratio ({printable_count}/{len(raw_text)}). Falling back to OCR.")
            return False, None
            
        # Extract structured blocks from PyMuPDF
        # format: (x0, y0, x1, y1, "lines of text", block_no, block_type)
        blocks = page.get_text("blocks")
        content_items = []
        
        # Check if PyMuPDF table extraction is available
        tables = []
        try:
            if hasattr(page, "find_tables"):
                tab_finder = page.find_tables()
                if tab_finder and tab_finder.tables:
                    for tab in tab_finder.tables:
                        tab_bbox = [int(v) for v in tab.bbox]
                        extracted_rows = tab.extract()
                        flat_cells = [str(c).strip() for row in extracted_rows for c in row if c is not None and str(c).strip()]
                        if flat_cells:
                            tables.append({
                                "bbox": tab_bbox,
                                "cells": flat_cells
                            })
        except Exception as tab_e:
            logger.debug(f"PyMuPDF find_tables not available or failed: {tab_e}")

        for b in blocks:
            if b[6] == 0:  # Text block
                b_text = b[4].strip()
                if not b_text:
                    continue
                x0, y0, x1, y1 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                
                # Check if this text is already inside an extracted table
                inside_table = False
                for tbl in tables:
                    tx0, ty0, tx1, ty1 = tbl["bbox"]
                    if x0 >= tx0 - 5 and y0 >= ty0 - 5 and x1 <= tx1 + 5 and y1 <= ty1 + 5:
                        inside_table = True
                        break
                        
                if not inside_table:
                    content_items.append((
                        y0,
                        {
                            "type": "plain_text",
                            "text": b_text,
                            "bbox_pil": [x0, y0, x1, y1]
                        }
                    ))
                    
        for tbl in tables:
            x0, y0, x1, y1 = tbl["bbox"]
            is_at_top = (y0 <= 100)
            is_at_bottom = (y1 >= page.rect.height - 100)
            content_items.append((
                y0,
                {
                    "type": "table",
                    "bbox_pil": [x0, y0, x1, y1],
                    "extracted_text_list": tbl["cells"],
                    "raw_cells": [{"text": c, "bbox_pil": [x0, y0, x1, y1]} for c in tbl["cells"]],
                    "is_at_page_top": is_at_top,
                    "is_at_page_bottom": is_at_bottom
                }
            ))

        content_items.sort(key=lambda x: x[0])
        page_data = {
            "page_number": page_num + 1,
            "source_pdf": source_pdf,
            "content": [item[1] for item in content_items]
        }
        return True, page_data

    except Exception as e:
        logger.warning(f"Error during native text extraction on page {page_num + 1} of '{source_pdf}': {e}. Falling back to OCR.")
        return False, None


def _process_page_ocr(pdf_path, page_num, table_detector_pipeline, ocr_reader):
    """Helper function to process a single page using Table Transformer + EasyOCR."""
    page_content_blocks = []
    current_page_data = {"page_number": page_num + 1, "source_pdf": os.path.basename(pdf_path), "content": []}

    try:
        pil_image = convert_pdf_page_to_image(pdf_path, page_num, dpi=config.PDF_TO_IMAGE_DPI)
        if not pil_image:
            logger.warning(f"Could not convert page {page_num} of '{pdf_path}' to image. Skipping page.")
            return current_page_data

        table_boxes_pil = []
        if table_detector_pipeline:
            table_detections = table_detector_pipeline(pil_image)
            for detection in table_detections:
                if detection['label'] == 'table':
                    box = detection['box']
                    x0 = max(0, int(box['xmin']))
                    y0 = max(0, int(box['ymin']))
                    x1 = min(pil_image.width, int(box['xmax']))
                    y1 = min(pil_image.height, int(box['ymax']))
                    if x1 > x0 and y1 > y0:
                        table_boxes_pil.append((x0, y0, x1, y1))

        img_np_rgb = np.array(pil_image)
        non_table_mask = np.ones(img_np_rgb.shape[:2], dtype=np.uint8) * 255
        for x0, y0, x1, y1 in table_boxes_pil:
            non_table_mask[y0:y1, x0:x1] = 0
        
        non_table_img_np = cv2.bitwise_and(img_np_rgb, img_np_rgb, mask=non_table_mask)
        non_table_ocr_results = ocr_reader.readtext(non_table_img_np, paragraph=True) if ocr_reader else []

        for ocr_result in non_table_ocr_results:
            if len(ocr_result) == 2:
                bbox, text = ocr_result
            elif len(ocr_result) == 3:
                bbox, text, _ = ocr_result
            else:
                continue

            if bbox and isinstance(bbox, list) and len(bbox) > 0 and \
               isinstance(bbox[0], (list, tuple)) and len(bbox[0]) == 2:
                pos_y = int(bbox[0][1])
                page_content_blocks.append(
                    (pos_y, {"type": "plain_text", "text": text, "bbox_pil": [int(c) for pt in bbox for c in pt]})
                )

        page_height = pil_image.height
        tolerance = 100  # Pixels tolerance for checking if table is at edge

        for x0_tbl, y0_tbl, x1_tbl, y1_tbl in table_boxes_pil:
            table_pil_image_crop = pil_image.crop((x0_tbl, y0_tbl, x1_tbl, y1_tbl))
            table_ocr_results = ocr_reader.readtext(np.array(table_pil_image_crop)) if ocr_reader else []

            table_cells_text = []
            raw_table_cells = []
            for item in table_ocr_results:
                if len(item) == 3:
                    bbox_cell, text_cell, prob_cell = item
                elif len(item) == 2:
                    bbox_cell, text_cell = item
                else:
                    continue
                table_cells_text.append(text_cell)
                adjusted_bbox_cell = [[int(pt[0] + x0_tbl), int(pt[1] + y0_tbl)] for pt in bbox_cell]
                raw_table_cells.append({"text": text_cell, "bbox_pil": [int(c) for pt in adjusted_bbox_cell for c in pt]})

            if table_cells_text:
                is_at_top = (y0_tbl <= tolerance)
                is_at_bottom = (y1_tbl >= page_height - tolerance)
                page_content_blocks.append(
                    (y0_tbl, {
                        "type": "table",
                        "bbox_pil": [x0_tbl, y0_tbl, x1_tbl, y1_tbl],
                        "extracted_text_list": table_cells_text,
                        "raw_cells": raw_table_cells,
                        "is_at_page_top": is_at_top,
                        "is_at_page_bottom": is_at_bottom
                    })
                )
        
        page_content_blocks.sort(key=lambda x: x[0])
        current_page_data["content"] = [block[1] for block in page_content_blocks]

    except Exception as e:
        logger.error(f"Error OCR processing page {page_num} of PDF '{pdf_path}': {e}", exc_info=True)
    
    return current_page_data


def _process_single_page_hybrid(pdf_path, page_num, doc, table_detector_pipeline, ocr_reader):
    """
    Executes native PyMuPDF text extraction first with safety net;
    falls back to OCR if word density is low or page is scanned.
    """
    source_pdf = os.path.basename(pdf_path)
    
    if getattr(config, "USE_NATIVE_PDF_TEXT", True) and doc is not None:
        success, native_data = _extract_native_page_text(doc, page_num, source_pdf)
        if success and native_data and native_data.get("content"):
            logger.debug(f"Native extraction successful for page {page_num + 1} of '{source_pdf}'.")
            return native_data
            
    logger.info(f"Using OCR fallback for page {page_num + 1} of '{source_pdf}'.")
    return _process_page_ocr(pdf_path, page_num, table_detector_pipeline, ocr_reader)


def extract_content_from_pdf(pdf_path, table_detector_pipeline=None, ocr_reader=None, max_workers=4):
    """
    Extracts structured content (text and tables) from a PDF file.
    Uses native text extraction where available and falls back to OCR.

    Args:
        pdf_path (str): Path to the PDF file.
        table_detector_pipeline: Initialized table detection pipeline (optional for native-only).
        ocr_reader: Initialized EasyOCR reader (optional for native-only).
        max_workers (int): Maximum number of threads for parallel processing.

    Returns:
        list: A list of dicts with 'page_number' and 'content' list.
    """
    extracted_pdf_data = []
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
    except Exception as e:
        logger.error(f"Error opening PDF '{pdf_path}': {e}")
        return extracted_pdf_data

    # Check how many pages can be natively parsed
    all_pages_native = True
    page_results = [None] * num_pages
    
    if getattr(config, "USE_NATIVE_PDF_TEXT", True):
        for page_num in range(num_pages):
            success, native_data = _extract_native_page_text(doc, page_num, os.path.basename(pdf_path))
            if success and native_data and native_data.get("content"):
                page_results[page_num] = native_data
            else:
                all_pages_native = False

    if all_pages_native and all(p is not None for p in page_results):
        doc.close()
        logger.info(f"All {num_pages} pages of '{os.path.basename(pdf_path)}' extracted natively via PyMuPDF.")
        return sorted(page_results, key=lambda x: x['page_number'])

    # For pages requiring OCR, run OCR in thread pool
    missing_page_nums = [i for i, r in enumerate(page_results) if r is None]
    logger.info(f"PDF '{os.path.basename(pdf_path)}': {len(page_results) - len(missing_page_nums)}/{num_pages} pages parsed natively. Running OCR fallback on {len(missing_page_nums)} pages.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_page_ocr, pdf_path, page_num, table_detector_pipeline, ocr_reader): page_num 
            for page_num in missing_page_nums
        }
        for future in as_completed(futures):
            p_num = futures[future]
            page_results[p_num] = future.result()

    doc.close()
    extracted_pdf_data = [p for p in page_results if p is not None]
    extracted_pdf_data = sorted(extracted_pdf_data, key=lambda x: x['page_number'])
    return extracted_pdf_data