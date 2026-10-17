"""Split page text into small overlapping chunks, keeping source + page.

Why chunk? Embeddings and LLM prompts work best on small, focused passages.
Chunks are created PER PAGE, so every chunk knows its exact page -> citations.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from src.ingestion.pdf_loader import PageContent

CHUNK_SIZE = 1000      # characters per chunk (~200 words)
CHUNK_OVERLAP = 150    # shared characters between neighbouring chunks
MIN_CHUNK_CHARS = 50   # drop tiny fragments (page numbers, headers)


class Chunk(BaseModel):
    chunk_id: str  # unique and stable: "file.pdf::p12::c0"
    source: str
    page: int
    text: str


def chunk_pages(
    pages: list[PageContent],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer natural boundaries
    )

    chunks: list[Chunk] = []
    for page in pages:
        index = 0
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{page.source}::p{page.page}::c{index}",
                    source=page.source,
                    page=page.page,
                    text=piece,
                )
            )
            index += 1
    return chunks