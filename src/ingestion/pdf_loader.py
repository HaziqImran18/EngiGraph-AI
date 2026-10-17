"""Extract text from PDFs while keeping document name and page number.

Page numbers are the foundation of citations: every chunk created later
must remember which file and which page it came from.
"""
from pathlib import Path
import pymupdf
from pydantic import BaseModel


class PageContent(BaseModel):
    source: str  # file name, e.g. "motor_manual.pdf"
    page: int    # 1-based page number (matches what a human sees)
    text: str


def load_pdf(pdf_path: str | Path) -> list[PageContent]:
    """Read one PDF and return a PageContent per non-empty page."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[PageContent] = []
    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text:
                # Empty page or scanned image (needs OCR). Skip, don't crash.
                continue
            pages.append(
                PageContent(source=pdf_path.name, page=index + 1, text=text)
            )
    return pages


def load_all_pdfs(folder: str | Path = "data/raw_pdfs") -> list[PageContent]:
    """Load every PDF inside a folder."""
    folder = Path(folder)
    all_pages: list[PageContent] = []
    for pdf_file in sorted(folder.glob("*.pdf")):
        all_pages.extend(load_pdf(pdf_file))
    return all_pages