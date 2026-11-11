"""Find the Tesseract program on machines where it is installed but not on PATH."""

import shutil
from pathlib import Path

WINDOWS_DEFAULT = Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')


def configure():
    """Point pytesseract at the default Windows install when `tesseract` is not on PATH.

    Leaves pytesseract alone everywhere else, so Linux and the Docker image keep using PATH.
    """
    import pytesseract

    if shutil.which('tesseract') is None and WINDOWS_DEFAULT.exists():
        pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_DEFAULT)
