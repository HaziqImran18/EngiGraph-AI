"""Verifier: check every claim against the retrieved evidence."""
from src.retrieval.search import RetrievedChunk
from src.services.basic_rag import format_context
from src.services.llm import invoke_structured
from src.workflow.state import (
    DiagnosticPlan,
    EngiState,
    Finding,
    VerificationResult,
    VerifierOutput,
)

SYSTEM_PROMPT = """You are the Verifier in an engineering diagnostic team.

Check each numbered finding and plan step against the passages.
A claim is SUPPORTED only if the passage it cites directly supports it.

Output:
- verified: true only if EVERY finding and step is supported.
- unsupported_claims: short description of each unsupported claim.
- feedback: concrete instructions for the planner to fix the problems
  (empty string if verified).
Keep each unsupported claim and the feedback short (one or two sentences, plain text,
no line breaks). Be strict. Do not give credit for plausible-sounding claims."""


def _check_claims(
    findings: list[Finding], plan: DiagnosticPlan, chunks: list[RetrievedChunk]
) -> VerifierOutput:
    findings_text = "\n".join(
        f"F{i}. {f.cause} -- {f.evidence} [{f.source}, p.{f.page}]"
        for i, f in enumerate(findings, 1)
    )
    steps_text = "\n".join(
        f"S{s.order}. {s.action} -- {s.rationale} [{s.source or 'no source'}, p.{s.page}]"
        for s in plan.steps
    )
    human = (
        f"Findings:\n{findings_text}\n\nPlan steps:\n{steps_text}\n\n"
        f"Passages:\n{format_context(chunks)}"
    )
    return invoke_structured(VerifierOutput, [("system", SYSTEM_PROMPT), ("human", human)])


def verifier_node(state: EngiState) -> dict:
    previous = state.get("verification_result")
    attempts = (previous.attempts if previous else 0) + 1
    findings = state.get("findings", [])
    plan = state["diagnostic_plan"]

    if not findings:  # nothing was claimed, so nothing to verify
        return {
            "verification_result": VerificationResult(
                verified=True,
                unsupported_claims=[],
                feedback="No findings; insufficient evidence in the documents.",
                attempts=attempts,
            )
        }

    out = _check_claims(findings, plan, state.get("retrieved_chunks", []))
    return {
        "verification_result": VerificationResult(
            verified=out.verified and not out.unsupported_claims,
            unsupported_claims=out.unsupported_claims,
            feedback=out.feedback,
            attempts=attempts,
        )
    }