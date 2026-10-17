"""Investigator: retrieve evidence from the manuals and assess possible causes."""
from src.retrieval.search import RetrievedChunk, search
from src.services.basic_rag import format_context
from src.services.llm import invoke_structured
from src.workflow.state import EngiState, Finding, FindingsOutput

TOP_K = 4  # fewer passages = fewer tokens per call (free-tier TPM limit)

SYSTEM_PROMPT = """You are the Investigator in an engineering diagnostic team.

Using ONLY the numbered passages, list possible causes or explanations relevant
to the reported problem.

Rules:
- Each finding needs: cause, evidence (short paraphrase of what the passage says),
  source (exact file name from the passage header), page (exact page number from
  the passage header) and confidence.
- confidence: "high" only if the passage directly states it, "medium" if indirect,
  "low" if weak.
- Do NOT use outside knowledge. If the passages do not address the problem,
  return an empty findings list."""


def _extract_findings(query: str, chunks: list[RetrievedChunk]) -> list[Finding]:
    result = invoke_structured(
        FindingsOutput,
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Problem: {query}\n\nPassages:\n{format_context(chunks)}"),
        ]
    )
    return result.findings if result else []


def _drop_invented_citations(findings: list[Finding], chunks: list[RetrievedChunk]) -> list[Finding]:
    """Guardrail: a finding may only cite a (file, page) that was really retrieved."""
    allowed = {(c.source, c.page) for c in chunks}
    return [f for f in findings if (f.source, f.page) in allowed]


def investigator_node(state: EngiState) -> dict:
    chunks = search(state["user_query"], k=TOP_K)
    if not chunks:
        return {"retrieved_chunks": [], "findings": []}
    findings = _drop_invented_citations(_extract_findings(state["user_query"], chunks), chunks)
    return {"retrieved_chunks": chunks, "findings": findings}