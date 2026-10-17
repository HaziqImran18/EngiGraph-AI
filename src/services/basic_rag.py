"""Baseline RAG: retrieve -> prompt -> answer with citations.

This is the single-LLM-call baseline. In the evaluation phase the
multi-agent workflow is compared against this.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.retrieval.search import RetrievedChunk, search
from src.services.llm import get_llm

SYSTEM_PROMPT = """You are an engineering documentation assistant.

Rules:
- Use ONLY the numbered context passages provided by the user message.
- After every factual statement, cite its source like (file.pdf, p.12).
- If the passages do not contain the answer, reply exactly:
  "Not found in the provided documents."
- Never invent steps, values, or specifications. If you are unsure, say so.
- This is decision support only. Do not present unverified procedures as safe instructions."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context passages:\n{context}\n\nQuestion: {question}"),
    ]
)


class RAGAnswer(BaseModel):
    question: str
    answer: str
    sources: list[RetrievedChunk]


def format_context(chunks: list[RetrievedChunk], max_chars: int = 800) -> str:
    """Numbered passages for the prompt. Each is trimmed to save tokens."""
    return "\n\n".join(
        f"[{i}] ({c.source}, p.{c.page})\n{c.text[:max_chars]}"
        for i, c in enumerate(chunks, start=1)
    )


def answer_question(question: str, k: int = 5) -> RAGAnswer:
    chunks = search(question, k=k)
    chain = prompt | get_llm() | StrOutputParser()  # LangChain "pipe" (LCEL)
    answer = chain.invoke({"context": format_context(chunks), "question": question})
    return RAGAnswer(question=question, answer=answer, sources=chunks)