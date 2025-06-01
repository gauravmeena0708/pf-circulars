# pdf_parser.py
import os
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
    """
    logger.info(f"Opening PDF file: {pdf_path}")
    extracted_pdf_data = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Error opening PDF '{pdf_path}': {e}")
        return extracted_pdf_data

    for page_num in range(len(doc)):
        logger.info(f"Processing page {page_num + 1}/{len(doc)} of PDF: {pdf_path}")
        current_page_data = {"page_number": page_num + 1, "source_pdf": os.path.basename(pdf_path), "content": []}

        try:
            pil_image = convert_pdf_page_to_image(pdf_path, page_num, dpi=config.PDF_TO_IMAGE_DPI)
            if not pil_image:
                logger.warning(f"Could not convert page {page_num + 1} of '{pdf_path}' to image. Skipping page.")
                extracted_pdf_data.append(current_page_data)
                continue

            logger.info(f"Running table detection on page {page_num + 1}...")
            table_detections = table_detector_pipeline(pil_image)
            logger.info(f"Detected {len(table_detections)} potential tables on page {page_num + 1}.")

            logger.info(f"Running OCR on non-table regions of page {page_num + 1}...")
            img_np_rgb = np.array(pil_image)
            non_table_mask = np.ones(img_np_rgb.shape[:2], dtype=np.uint8) * 255
            for detection in table_detections:
                if detection['label'] == 'table':
                    box = detection['box']
                    x0 = max(0, int(box['xmin']))
                    y0 = max(0, int(box['ymin']))
                    x1 = min(pil_image.width, int(box['xmax']))
                    y1 = min(pil_image.height, int(box['ymax']))
                    non_table_mask[y0:y1, x0:x1] = 0

            non_table_img_np = cv2.bitwise_and(img_np_rgb, img_np_rgb, mask=non_table_mask)
            non_table_ocr_results = ocr_reader.readtext(non_table_img_np, paragraph=True)
            logger.info(f"Extracted {len(non_table_ocr_results)} text blocks from non-table regions on page {page_num + 1}.")

            for ocr_result in non_table_ocr_results:
                if len(ocr_result) >= 2:
                    _, text = ocr_result[:2]
                    current_page_data["content"].append({"type": "plain_text", "text": text})

            logger.info(f"Running OCR on detected tables on page {page_num + 1}...")
            for detection in table_detections:
                if detection['label'] == 'table':
                    box = detection['box']
                    x0, y0, x1, y1 = map(int, [box['xmin'], box['ymin'], box['xmax'], box['ymax']])
                    table_image = pil_image.crop((x0, y0, x1, y1))
                    table_ocr_results = ocr_reader.readtext(np.array(table_image))
                    table_text = [result[1] for result in table_ocr_results]
                    current_page_data["content"].append({
                        "type": "table",
                        "extracted_text_list": table_text,
                        "is_at_page_top": False,
                        "is_at_page_bottom": False
                    })

        except Exception as e:
            logger.error(f"Error processing page {page_num + 1} of PDF '{pdf_path}': {e}", exc_info=True)

        extracted_pdf_data.append(current_page_data)
        logger.info(f"Page {page_num + 1} content items: {len(current_page_data['content'])}")
    
    doc.close()
    logger.info(f"Finished processing PDF: {pdf_path}")
    return extracted_pdf_data

if __name__ == '__main__':
    import os
    from transformers import pipeline
    import easyocr

    logger.info("Starting PDF parser example...")

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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_pdf_path = os.path.join(script_dir, "example.pdf") 

    if not os.path.exists(example_pdf_path):
        logger.warning(f"Example PDF '{example_pdf_path}' not found. Please create it for testing.")
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
                        logger.info(f"      Text: {content_item['text'][:100]}...")
                    elif content_item['type'] == 'table':
                        logger.info(f"      Table Extracted Text List: {content_item['extracted_text_list']}")
                        logger.info(f"      Is at page top: {content_item['is_at_page_top']}")
                        logger.info(f"      Is at page bottom: {content_item['is_at_page_bottom']}")
        else:
            logger.error(f"No data extracted from {example_pdf_path}.")
    else:
        logger.error(f"Test PDF '{example_pdf_path}' not found. Cannot run example.")