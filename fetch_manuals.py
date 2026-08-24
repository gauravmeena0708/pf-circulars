import os
import re
import requests
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MANUALS_DIR = os.path.join(DATA_DIR, "manuals")
os.makedirs(MANUALS_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

PDF_URLS = {
    # 5-Part Manual of Accounting Procedure (MAP)
    "MAP_PartI_Complete.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/MAP_PartI_Complete.pdf",
    "MAP_PartIIA.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/MAP_PartIIA.pdf",
    "MAP_PartIIB.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/MAP_PartIIB.pdf",
    "MAP_PartIII.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/MAP_PartIII.pdf",
    "MAP_PartIV.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/MAP_PartIV.pdf",
    
    # Core manuals
    "ComplianceManual.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/ComplianceManual.pdf",
    "Pension_Manual.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/Pension_Manual.pdf",
    "EDLI_Manual.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/EDLI_Manual.pdf",
    "Recovery_Manual.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/Recovery_Manual.pdf",
    "Exemption_Manual.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/Exemption_Manual_08122023.pdf",
    "Audit_Manual.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/Audit_Manual_08122023.pdf",
    "Manual_SOP_ClaimSettlement.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/Manual_SOP_ClaimSettlement.pdf",
    "EPFServicesManual1972.pdf": "https://www.epfo.gov.in/wp-content/uploads/2025/11/EPFServicesManual1972.pdf",
}

SCHEME_URLS = {
    "EPF_and_MP_Act_1952.pdf": "https://www.epfo.gov.in/epf-mp-act-1952",
    "EPF_Scheme_1952.pdf": "https://www.epfo.gov.in/epf-scheme/",
    "Employees_Pension_Scheme_1995.pdf": "https://epfo.gov.in/eps-scheme/",
    "Employees_Deposit_Linked_Insurance_Scheme_1976.pdf": "https://epfo.gov.in/edli-scheme/"
}

def download_pdf(name, url):
    dest_path = os.path.join(MANUALS_DIR, name)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[SKIP] {name} already exists ({os.path.getsize(dest_path)} bytes).")
        return dest_path
    
    print(f"[DOWNLOADING] {name} from {url}...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            print(f"[SUCCESS] Saved {name} ({os.path.getsize(dest_path)} bytes)")
            return dest_path
        else:
            print(f"[ERROR] Failed {name}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"[ERROR] Exception downloading {name}: {e}")
    return None

def scrape_and_create_pdf_from_html(name, url):
    dest_path = os.path.join(MANUALS_DIR, name)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[SKIP] {name} already exists ({os.path.getsize(dest_path)} bytes).")
        return dest_path
    
    print(f"[SCRAPING] {name} from {url}...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"[ERROR] Failed fetching {url}: HTTP {resp.status_code}")
            return None
        
        soup = BeautifulSoup(resp.content, "html.parser")
        main_content = soup.find("main") or soup.find("article") or soup.find("div", class_="content") or soup.body
        
        for elem in main_content.find_all(["nav", "header", "footer", "script", "style"]):
            elem.decompose()
            
        doc = SimpleDocTemplate(dest_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        title_style = styles["Heading1"]
        h2_style = styles["Heading2"]
        h3_style = styles["Heading3"]
        normal_style = ParagraphStyle(
            name="NormalText",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6
        )
        
        story = []
        doc_title = name.replace(".pdf", "").replace("_", " ")
        story.append(Paragraph(doc_title, title_style))
        story.append(Spacer(1, 12))
        
        for heading in main_content.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            txt = heading.get_text().strip()
            if not txt or len(txt) < 3:
                continue
            safe_text = (txt.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;"))
            
            if heading.name in ["h1", "h2"]:
                story.append(Spacer(1, 8))
                story.append(Paragraph(safe_text, h2_style))
                story.append(Spacer(1, 4))
            elif heading.name in ["h3", "h4"]:
                story.append(Spacer(1, 6))
                story.append(Paragraph(safe_text, h3_style))
                story.append(Spacer(1, 2))
            else:
                story.append(Paragraph(safe_text, normal_style))
                
        doc.build(story)
        print(f"[SUCCESS] Created {name} ({os.path.getsize(dest_path)} bytes)")
        return dest_path
    except Exception as e:
        print(f"[ERROR] Failed to scrape/build PDF for {name}: {e}")
    return None

if __name__ == "__main__":
    print("=== Downloading EPFO Manuals & Accounting Procedures ===")
    for pdf_name, pdf_url in PDF_URLS.items():
        download_pdf(pdf_name, pdf_url)
        
    print("\n=== Fetching EPFO Acts and Schemes ===")
    for act_name, act_url in SCHEME_URLS.items():
        scrape_and_create_pdf_from_html(act_name, act_url)
        
    print("\nDone fetching all manuals, acts, and schemes!")
