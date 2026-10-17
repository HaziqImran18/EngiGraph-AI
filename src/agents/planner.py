"""Planner: turn findings into an ordered diagnostic plan (checks, not commands)."""
from src.retrieval.search import RetrievedChunk
from src.services.basic_rag import format_context
from src.services.llm import invoke_structured
from src.workflow.state import DiagnosticPlan, EngiState, Finding

UNCERTAIN_TAG = "[Not backed by manuals] "
DEFAULT_SAFETY_NOTE = (
    "Follow the manufacturer's procedures and your site's lockout/tagout rules. "
    "Only qualified personnel should inspect or service equipment."
)

SYSTEM_PROMPT = """You are the Planner in an engineering diagnostic team.

Create an ordered diagnostic plan of NON-INTRUSIVE inspection and verification
checks that a qualified technician could consider, based on the findings and passages.

Rules:
- Never instruct anyone to start, stop, override, or command equipment.
- Each step: order, action, rationale, source, page, safety_note.
- source and page MUST come from the passage headers. If a step is not supported
  by the passages, use source "" and page 0.
- safety_note is "" if there is none.
- safety_notes (plan level): cautions found in the passages, if any.
- Keep every text field short (one or two sentences) and use plain text only.
- If REVISION FEEDBACK is given, fix exactly those issues: remove or re-ground
  unsupported steps."""


def _make_plan(
    query: str,
    findings: list[Finding],
    chunks: list[RetrievedChunk],
    feedback: str,
) -> DiagnosticPlan:
    findings_text = "\n".join(
        f"- {f.cause} ({f.source}, p.{f.page}): {f.evidence}" for f in findings
    )
    human = (
        f"Problem: {query}\n\nFindings:\n{findings_text}\n\n"
        f"Passages:\n{format_context(chunks)}"
    )
    if feedback:
        human += f"\n\nREVISION FEEDBACK:\n{feedback}"
    return invoke_structured(DiagnosticPlan, [("system", SYSTEM_PROMPT), ("human", human)])


def _ground_steps(plan: DiagnosticPlan, chunks: list[RetrievedChunk]) -> DiagnosticPlan:
    """Guardrail: steps citing pages we never retrieved are flagged as uncertain."""
    allowed = {(c.source, c.page) for c in chunks}
    for step in plan.steps:
        if (step.source, step.page) not in allowed:
            step.source, step.page = "", 0
            if not step.action.startswith(UNCERTAIN_TAG):
                step.action = UNCERTAIN_TAG + step.action
    if DEFAULT_SAFETY_NOTE not in plan.safety_notes:
        plan.safety_notes.append(DEFAULT_SAFETY_NOTE)
    return plan


def planner_node(state: EngiState) -> dict:
    findings = state.get("findings", [])
    if not findings:
        return {"diagnostic_plan": DiagnosticPlan(steps=[], safety_notes=[DEFAULT_SAFETY_NOTE])}

    previous = state.get("verification_result")
    feedback = previous.feedback if previous and not previous.verified else ""
    chunks = state.get("retrieved_chunks", [])
    plan = _make_plan(state["user_query"], findings, chunks, feedback)
    return {"diagnostic_plan": _ground_steps(plan, chunks)}