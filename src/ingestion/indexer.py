"""Embed chunks and store them in ChromaDB (persistent, local).

Safe to re-run: chunks already in the database are skipped (resumable),
so a crash or rate-limit never forces you to start from zero.
"""
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from langchain_chroma import Chroma

from src.ingestion.chunker import Chunk, chunk_pages
from src.ingestion.pdf_loader import load_pdf
from src.services.llm import get_embeddings

CHROMA_DIR = "data/chroma_db"
COLLECTION = "engigraph"
BATCH_SIZE = 20          # chunks per embedding API call
PAUSE_SECONDS = 1.0      # gentle pause between calls (free-tier friendly)
MAX_RETRIES = 5


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # cosine similarity
    )


def reset_index() -> None:
    """Delete everything in the collection."""
    get_vectorstore().delete_collection()


def _add_batch(store: Chroma, batch: list[Chunk]) -> None:
    for attempt in range(MAX_RETRIES):
        try:
            store.add_texts(
                texts=[c.text for c in batch],
                metadatas=[{"source": c.source, "page": c.page} for c in batch],
                ids=[c.chunk_id for c in batch],
            )
            return
        except Exception as error:  # noqa: BLE001
            message = str(error)
            retryable = any(t in message for t in ("429", "RESOURCE_EXHAUSTED", "503", "500"))
            if not retryable or attempt == MAX_RETRIES - 1:
                raise  # bad key, wrong model name, etc. -> fail fast
            wait = 5 * 2**attempt
            print(f"  rate limit / server busy, retrying in {wait}s ...")
            time.sleep(wait)


def index_pdf(
    pdf_path: str | Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Index one PDF. Returns number of NEW chunks added.

    on_progress(done, total) is called after every batch (used by the web UI).
    """
    pdf_path = Path(pdf_path)
    chunks = chunk_pages(load_pdf(pdf_path))
    if not chunks:
        raise ValueError(
            f"No readable text found in {pdf_path.name}. It may be a scanned PDF (needs OCR)."
        )
    print(f"{pdf_path.name}: {len(chunks)} chunks")

    store = get_vectorstore()
    added = 0
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        existing = set(store.get(ids=[c.chunk_id for c in batch])["ids"])
        new = [c for c in batch if c.chunk_id not in existing]
        if new:
            _add_batch(store, new)
            added += len(new)
            time.sleep(PAUSE_SECONDS)
        done = min(start + BATCH_SIZE, len(chunks))
        print(f"  {done}/{len(chunks)} done", end="\r")
        if on_progress:
            on_progress(done, len(chunks))
    print(f"\n  added {added} new chunks")
    return added


def count_chunks() -> int:
    return get_vectorstore()._collection.count()


def list_documents() -> dict[str, int]:
    """Return {file name: number of indexed chunks}."""
    data = get_vectorstore()._collection.get(include=["metadatas"])
    return dict(Counter(m["source"] for m in data["metadatas"]))


def delete_document(source: str) -> None:
    """Remove every chunk that belongs to one PDF."""
    get_vectorstore()._collection.delete(where={"source": source})