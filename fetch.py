import io
from bs4 import BeautifulSoup as bs, NavigableString
import requests
from urllib.parse import urljoin
import json
from datetime import datetime, timezone
import os
import argparse # For command-line arguments

# --- Configuration ---
# If tesseract is not in your PATH, you might need to specify its location
# For example, on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

CIRCULAR_DATA_FILE = "circular-data.json"
INDEX_DATA_FILE = "index-data.json"
MAX_URLS_TO_INDEX_PER_RUN = 100

HEADERS = {
    'Host': 'www.epfindia.gov.in',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0',
    'Accept': 'text/html, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': 'https://www.epfindia.gov.in/site_en/Contact_office_wise.php?id=MHEM',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

YEAR_PARAMS = [
    "yr=2025-2026", "yr=2024-2025", "yr=2023-2024", "yr=2022-2023",
    "yr=2021-2022", "yr=2020-2021", "yr=2019-2020", "yr=2018-2019",
    "yr=2017-2018", "yr=2016-2017", "yr=2015-2016", "yr=2014-2015",
    "yr=2013-2014", "yr=2012-2013", "yr=2011-2012", "yr=2010-2011",
    "yr=2009-2010", "yr=Old+Circulars",
]

# --- Utility Functions ---
def load_json_file(filepath):
    """Loads a JSON file if it exists, otherwise returns an empty dictionary."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filepath}. Starting fresh.")
            return {}
    return {}

def save_json_file(data, filepath):
    """Saves data to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {filepath}")

# --- PDF Processing and OCR Function ---
def get_first_page_ocr_text_from_url(pdf_url):
    """
    Downloads a PDF from a URL, extracts the first page,
    and performs OCR on it.
    Returns the OCRed text or None if an error occurs.
    """
    if not pdf_url:
        return None
    try:
        print(f"    Downloading PDF: {pdf_url}")
        pdf_response = requests.get(pdf_url, headers=HEADERS, timeout=45) # Increased timeout
        pdf_response.raise_for_status()
        pdf_bytes = pdf_response.content

        print(f"    Opening PDF with PyMuPDF...")
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        if len(pdf_document) > 0:
            first_page = pdf_document.load_page(0)
            pix = first_page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            print(f"    Performing OCR on the first page...")
            ocr_text = pytesseract.image_to_string(img, lang='eng')
            pdf_document.close()
            print(f"    OCR successful for {pdf_url.split('/')[-1]}.")
            return ocr_text.strip()
        else:
            print(f"    PDF is empty: {pdf_url}")
            pdf_document.close()
            return None
    except requests.exceptions.RequestException as e:
        print(f"    Error downloading PDF {pdf_url}: {e}")
        return None
    except pytesseract.TesseractNotFoundError:
        print("    Tesseract OCR not found. Please ensure it's installed and in your PATH.")
        return "OCR_ERROR: Tesseract not found" # So we know it was attempted but failed due to setup
    except Exception as e:
        print(f"    Error processing PDF {pdf_url} for OCR: {e}")
        return None

# --- Main Data Fetching Logic ---
def fetch_circular_metadata():
    """Fetches circular metadata from EPFO website and saves to circular-data.json."""
    parsed_circulars_data = []
    print("Starting to fetch circular metadata...")

    for q_param in YEAR_PARAMS:
        current_page_url = f'https://www.epfindia.gov.in/site_en/get_cir_content.php?{q_param}'
        print(f"  Requesting URL: {current_page_url}")

        try:
            r = requests.get(current_page_url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            soup = bs(r.text, 'html.parser')
            print(f"  Status Code: {r.status_code} for {q_param}")

            table_rows = soup.find_all('tr')
            data_rows = table_rows[1:] if table_rows and table_rows[0].find('th') else table_rows

            for row_idx, row in enumerate(data_rows):
                cells = row.find_all('td')
                if len(cells) < 4:
                    # print(f"    Skipping row {row_idx+1} in {q_param} due to insufficient cells ({len(cells)}).")
                    continue

                serial_no = cells[0].get_text(strip=True)
                subject_cell = cells[1]
                title_parts = [content.strip() for content in subject_cell.contents if isinstance(content, NavigableString) and content.strip()]
                title = " ".join(title_parts).split('Circular No.')[0].split('No.')[0].strip() # Basic title cleaning

                circular_no_date_raw = ""
                after_first_br = False
                temp_circular_parts = []
                for content in subject_cell.contents:
                    if content.name == 'br':
                        if not after_first_br:
                            after_first_br = True
                            continue
                        else: # Second br or end of relevant part
                            break
                    if after_first_br:
                        if isinstance(content, NavigableString):
                            text_content = content.strip()
                            if text_content:
                                temp_circular_parts.append(text_content)
                        elif content.name == 'a' and temp_circular_parts: # Link after some text
                            break
                circular_no_date_raw = " ".join(filter(None, temp_circular_parts))


                circular_no = ""
                date_of_circular = ""
                delimiter_dated = " dated "
                delimiter_date = " date " # some entries use "date" instead of "dated"
                
                actual_delimiter = None
                if delimiter_dated in circular_no_date_raw.lower(): # Check lower case
                    actual_delimiter = delimiter_dated
                elif delimiter_date in circular_no_date_raw.lower():
                    actual_delimiter = delimiter_date

                if actual_delimiter:
                    # Find the actual delimiter with original casing for split
                    delimiter_pos = circular_no_date_raw.lower().find(actual_delimiter)
                    original_delimiter = circular_no_date_raw[delimiter_pos : delimiter_pos + len(actual_delimiter)]
                    
                    parts = circular_no_date_raw.split(original_delimiter, 1)
                    circular_no = parts[0].strip()
                    if len(parts) > 1:
                        date_of_circular = parts[1].strip()
                else:
                    circular_no = circular_no_date_raw.strip()
                
                # Further clean title from circular number if any residue
                if circular_no and title.endswith(circular_no): # simple check
                    title = title[:-len(circular_no)].strip()


                def get_pdf_link(cell, base_url):
                    link_tag = cell.find('a')
                    if link_tag and link_tag.has_attr('href'):
                        relative_link = link_tag['href']
                        return urljoin(base_url, relative_link)
                    return None

                hindi_pdf_link = get_pdf_link(cells[2], current_page_url)
                english_pdf_link = get_pdf_link(cells[3], current_page_url)

                circular_data = {
                    "serial_no": serial_no,
                    "title": title,
                    "circular_no": circular_no,
                    "date": date_of_circular,
                    "hindi_pdf_link": hindi_pdf_link,
                    "english_pdf_link": english_pdf_link
                }
                parsed_circulars_data.append(circular_data)
            print(f"  Successfully processed {q_param}")

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {q_param}: {e}")
        except Exception as e:
            print(f"  An error occurred during parsing for {q_param}: {e}")

    if parsed_circulars_data:
        save_json_file(parsed_circulars_data, CIRCULAR_DATA_FILE)
    else:
        print("No circular metadata was extracted.")
    print("Finished fetching circular metadata.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch EPFO circular data and/or update PDF index.")
    parser.add_argument(
        "--action",
        choices=['fetch', 'index', 'all'],
        default='all',
        help="Specify action: 'fetch' metadata, 'index' PDFs, or 'all' (default)."
    )
    args = parser.parse_args()

    if args.action == 'fetch' or args.action == 'all':
        fetch_circular_metadata()

    print("\nScript finished.")
