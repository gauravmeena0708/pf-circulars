# pdf_parser.py

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2
import logging

# Assuming your config.py is in the same directory or accessible in PYTHONPATH
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

def extract_content_from_pdf(pdf_path, table_detector_pipeline, ocr_reader):
    """
    Extracts structured content (text and tables) from a single PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
        table_detector_pipeline (transformers.pipeline): Initialized table detection pipeline.
        ocr_reader (easyocr.Reader): Initialized EasyOCR reader.

    Returns:
        list: A list of dictionaries, where each dictionary represents a page
              and contains 'page_number' and 'content' (a list of text/table blocks).
              Returns an empty list if the PDF cannot be processed.
    """
    extracted_pdf_data = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Error opening PDF '{pdf_path}': {e}")
        return extracted_pdf_data

    for page_num in range(len(doc)):
        page_content_blocks = []
        current_page_data = {"page_number": page_num + 1, "source_pdf": os.path.basename(pdf_path), "content": []}

        try:
            # 1. Convert PDF page to image
            pil_image = convert_pdf_page_to_image(pdf_path, page_num, dpi=config.PDF_TO_IMAGE_DPI)
            if not pil_image:
                logger.warning(f"Could not convert page {page_num} of '{pdf_path}' to image. Skipping page.")
                extracted_pdf_data.append(current_page_data) # Add page with empty content
                continue

            # 2. Detect tables using the transformer model
            # The table detector expects PIL images
            table_detections = table_detector_pipeline(pil_image)
            table_boxes_pil = [] # Store table boxes in PIL image coordinates (x0, y0, x1, y1)

            for detection in table_detections:
                if detection['label'] == 'table': # Or use a more specific label if your model provides it
                    box = detection['box']
                    # Ensure coordinates are within image bounds
                    x0 = max(0, int(box['xmin']))
                    y0 = max(0, int(box['ymin']))
                    x1 = min(pil_image.width, int(box['xmax']))
                    y1 = min(pil_image.height, int(box['ymax']))
                    if x1 > x0 and y1 > y0: # Ensure valid box
                         table_boxes_pil.append((x0, y0, x1, y1))


            # 3. Create a mask for table regions to exclude them from general text OCR
            # Work with a NumPy array representation of the image for masking
            img_np_rgb = np.array(pil_image) # PIL Image is RGB
            
            # Create a mask for non-table regions. Initialize with all ones (process everything).
            non_table_mask = np.ones(img_np_rgb.shape[:2], dtype=np.uint8) * 255

            for x0, y0, x1, y1 in table_boxes_pil:
                # Black out table areas in the non_table_mask
                non_table_mask[y0:y1, x0:x1] = 0
            
            # Apply mask to get non-table parts of the image
            # EasyOCR can take a numpy array (RGB)
            non_table_img_np = cv2.bitwise_and(img_np_rgb, img_np_rgb, mask=non_table_mask)

            # 4. OCR on non-table regions
            # EasyOCR expects an image path or a numpy array (RGB) or a byte stream.
            non_table_ocr_results = ocr_reader.readtext(non_table_img_np, paragraph=True)

            for (bbox, text, prob) in non_table_ocr_results:
                # bbox is [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
                # We need a consistent y-coordinate for sorting, e.g., top-left y
                # Ensure bbox is valid list of lists/tuples with at least one point
                if bbox and isinstance(bbox, list) and len(bbox) > 0 and \
                   isinstance(bbox[0], (list, tuple)) and len(bbox[0]) == 2:
                    pos_y = int(bbox[0][1]) 
                    page_content_blocks.append(
                        (pos_y, {"type": "plain_text", "text": text, "bbox_pil": [int(c) for pt in bbox for c in pt]})
                    )
                else:
                    logger.warning(f"Invalid bbox format for non-table text: {text[:50]} on page {page_num+1} of {pdf_path}")


            # 5. OCR on table regions
            for x0_tbl, y0_tbl, x1_tbl, y1_tbl in table_boxes_pil:
                table_pil_image = pil_image.crop((x0_tbl, y0_tbl, x1_tbl, y1_tbl))
                
                # EasyOCR expects an image path or a numpy array (RGB) or a byte stream.
                table_ocr_results = ocr_reader.readtext(np.array(table_pil_image)) # Convert cropped PIL to numpy

                table_cells_text = []
                raw_table_cells = [] # To store text with their individual bboxes relative to table crop

                for (bbox_cell, text_cell, prob_cell) in table_ocr_results:
                    table_cells_text.append(text_cell)
                    # Bbox relative to cropped table image
                    # Adjust bbox to be relative to the full page
                    # bbox_cell is [[x_rel,y_rel],[x1_rel,y_rel],[x1_rel,y1_rel],[x0_rel,y1_rel]]
                    adjusted_bbox_cell = [
                        [int(pt[0] + x0_tbl), int(pt[1] + y0_tbl)] for pt in bbox_cell
                    ]
                    raw_table_cells.append({
                        "text": text_cell,
                        "bbox_pil": [int(c) for pt in adjusted_bbox_cell for c in pt]
                    })

                if table_cells_text:
                    page_content_blocks.append(
                        (y0_tbl, { # Use table's top y-coordinate for sorting
                            "type": "table",
                            "bbox_pil": [x0_tbl, y0_tbl, x1_tbl, y1_tbl], # Bbox of the whole table
                            "extracted_text_list": table_cells_text, # Simple list of cell texts
                            "raw_cells": raw_table_cells # More detailed cell info
                        })
                    )
            
            # Sort all content blocks by their vertical position (y-coordinate of top-left corner)
            page_content_blocks.sort(key=lambda x: x[0])
            current_page_data["content"] = [block[1] for block in page_content_blocks]

        except Exception as e:
            logger.error(f"Error processing page {page_num} of PDF '{pdf_path}': {e}", exc_info=True)
            # Add page with empty content if an error occurs during its processing
        
        extracted_pdf_data.append(current_page_data)

    doc.close()
    return extracted_pdf_data

if __name__ == '__main__':
    # This is an example of how to use this module.
    # You would typically call extract_content_from_pdf from your main.py or app.py.
    import os
    from transformers import pipeline
    import easyocr

    logger.info("Starting PDF parser example...")

    # --- Initialize models (ideally done once in your main orchestrator) ---
    try:
        logger.info(f"Loading table detection model: {config.TABLE_DETECTION_MODEL}")
        table_detector = pipeline("object-detection", model=config.TABLE_DETECTION_MODEL, device=config.EMBEDDING_DEVICE)
        logger.info("Table detection model loaded.")

        logger.info(f"Loading OCR reader for languages: {config.OCR_LANGUAGES}")
        ocr_reader_instance = easyocr.Reader(config.OCR_LANGUAGES, gpu=(config.EMBEDDING_DEVICE == "cuda"))
        logger.info("OCR reader loaded.")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}", exc_info=True)
        exit()
    
    # --- Example PDF ---
    # Create a dummy PDF for testing if you don't have one readily available
    # For this example, assume a PDF exists at 'dummy.pdf'
    # You should replace 'dummy.pdf' with an actual PDF path for testing.
    
    # Ensure current working directory is where the script is to find dummy.pdf if it's relative
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_pdf_path = os.path.join(script_dir, "example.pdf") # Put an example.pdf in the same directory

    if not os.path.exists(example_pdf_path):
        logger.warning(f"Example PDF '{example_pdf_path}' not found. Please create it for testing.")
        # Create a simple dummy PDF for testing purposes
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(example_pdf_path)
            c.drawString(100, 750, "Hello World. This is a test PDF.")
            c.drawString(100, 700, "This page contains no tables.")
            c.showPage()
            c.drawString(100, 750, "Page 2 with a pseudo-table:")
            c.drawString(100, 700, "Header1 | Header2")
            c.drawString(100, 680, "Data1   | Data2")
            c.save()
            logger.info(f"Created dummy PDF: {example_pdf_path}")
        except ImportError:
            logger.error("ReportLab not installed. Cannot create dummy PDF. Skipping example run.")
            exit()
        except Exception as e:
            logger.error(f"Could not create dummy PDF: {e}")
            exit()


    if os.path.exists(example_pdf_path):
        logger.info(f"Processing PDF: {example_pdf_path}")
        structured_data = extract_content_from_pdf(example_pdf_path, table_detector, ocr_reader_instance)

        if structured_data:
            logger.info(f"Successfully extracted data from {example_pdf_path}.")
            for page_data in structured_data:
                logger.info(f"  Page Number: {page_data['page_number']} from {page_data['source_pdf']}")
                for i, content_item in enumerate(page_data['content']):
                    logger.info(f"    Item {i+1}: Type: {content_item['type']}")
                    if content_item['type'] == 'plain_text':
                        logger.info(f"      Text: {content_item['text'][:100]}...") # Log snippet
                    elif content_item['type'] == 'table':
                        logger.info(f"      Table Text List: {content_item['extracted_text_list']}")
                        logger.info(f"      Table BBox (PIL): {content_item['bbox_pil']}")
        else:
            logger.error(f"No data extracted from {example_pdf_path}.")
    else:
        logger.error(f"Test PDF '{example_pdf_path}' not found. Cannot run example.")