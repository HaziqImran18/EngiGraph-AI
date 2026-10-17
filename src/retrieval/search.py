"""Retrieve the most relevant chunks for a question, with source metadata."""
from pydantic import BaseModel

from src.ingestion.indexer import get_vectorstore


class RetrievedChunk(BaseModel):
    source: str
    page: int
    text: str
    score: float  # similarity: closer to 1.0 = more relevant


def search(query: str, k: int = 5, source: str | None = None) -> list[RetrievedChunk]:
    """Return top-k chunks. Optionally restrict to one PDF via `source`."""
    store = get_vectorstore()
    if store._collection.count() == 0:
        raise RuntimeError("The index is empty. Run: python main.py index")

    results = store.similarity_search_with_score(
        query, k=k, filter={"source": source} if source else None
    )
    # Chroma returns cosine DISTANCE (lower = closer); convert to similarity.
    return [
        RetrievedChunk(
            source=doc.metadata["source"],
            page=int(doc.metadata["page"]),
            text=doc.page_content,
            score=round(1 - distance, 4),
        )
        for doc, distance in results
    ]