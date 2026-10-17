"""Format the final Markdown report. Pure Python: no LLM, fully deterministic."""
from src.workflow.state import EngiState

SAFETY_NOTES = [
    "EngiGraph is decision support only. It never controls equipment.",
    "Verify every suggestion against the original manual before acting.",
    "Follow manufacturer procedures and site safety rules (e.g. lockout/tagout).",
]


def _cite(source: str, page: int) -> str:
    return f"({source}, p.{page})" if source else "(no manual citation)"


def build_report(state: EngiState) -> str:
    findings = state.get("findings", [])
    plan = state.get("diagnostic_plan")
    verification = state.get("verification_result")
    chunks = state.get("retrieved_chunks", [])

    lines = ["# EngiGraph Diagnostic Report", "", f"**Problem:** {state['user_query']}", ""]

    if not findings:
        status = "No evidence found"
    elif verification and verification.verified:
        status = "Verified against retrieved passages"
    else:
        status = "Partially verified - some claims are UNCERTAIN (see section 3)"
    lines += [f"**Status:** {status}", ""]

    lines += ["## 1. Findings"]
    if findings:
        for f in findings:
            lines.append(f"- **{f.cause}** [{f.confidence} confidence]: {f.evidence} {_cite(f.source, f.page)}")
    else:
        lines.append("- The provided documents do not contain enough evidence to answer this.")
    lines.append("")

    lines += ["## 2. Suggested diagnostic checks"]
    if plan and plan.steps:
        for s in sorted(plan.steps, key=lambda x: x.order):
            note = f" Safety: {s.safety_note}" if s.safety_note else ""
            lines.append(f"{s.order}. {s.action} - {s.rationale} {_cite(s.source, s.page)}{note}")
    else:
        lines.append("- No checks can be suggested from the provided documents.")
    lines.append("")

    lines += ["## 3. Uncertainty"]
    if verification and verification.unsupported_claims:
        lines += [f"- Unsupported: {claim}" for claim in verification.unsupported_claims]
    else:
        lines.append("- No unsupported claims were flagged. Retrieval may still have missed relevant pages.")
    lines.append("")

    lines += ["## 4. Safety notes"]
    lines += [f"- {n}" for n in SAFETY_NOTES]
    if plan:
        lines += [f"- {n}" for n in plan.safety_notes]
    lines.append("")

    lines += ["## 5. Sources consulted"]
    for source, page in sorted({(c.source, c.page) for c in chunks}):
        lines.append(f"- {source}, p.{page}")
    return "\n".join(lines)