"""Shared state + data models passed between agents.

Rule: the shared state holds ONLY six keys. Every node reads some of
them and returns the ones it changes; LangGraph merges the result.
"""
from typing import TypedDict

from pydantic import BaseModel

from src.retrieval.search import RetrievedChunk


# ---------- data models ----------
class Finding(BaseModel):
    cause: str        # possible cause / explanation
    evidence: str     # short paraphrase of what the passage says
    source: str       # exact file name from the passage header
    page: int         # exact page number from the passage header
    confidence: str   # "high" | "medium" | "low"


class PlanStep(BaseModel):
    order: int
    action: str       # what a qualified technician could check
    rationale: str
    source: str       # "" if not backed by the manuals
    page: int         # 0 if not backed by the manuals
    safety_note: str  # "" if none


class DiagnosticPlan(BaseModel):
    steps: list[PlanStep]
    safety_notes: list[str]


class VerificationResult(BaseModel):
    verified: bool
    unsupported_claims: list[str]
    feedback: str     # what the planner must fix
    attempts: int     # how many times the verifier has run (drives the loop)


# ---------- LLM output wrappers (what the agents ask the model to return) ----------
class FindingsOutput(BaseModel):
    findings: list[Finding]


class VerifierOutput(BaseModel):
    verified: bool
    unsupported_claims: list[str]
    feedback: str


# ---------- the shared state ----------
class EngiState(TypedDict, total=False):
    user_query: str
    retrieved_chunks: list[RetrievedChunk]
    findings: list[Finding]
    diagnostic_plan: DiagnosticPlan
    verification_result: VerificationResult
    final_report: str