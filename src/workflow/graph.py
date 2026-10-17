"""LangGraph workflow: investigator -> planner -> verifier -> (revise once) -> report."""
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agents.investigator import investigator_node
from src.agents.planner import planner_node
from src.agents.verifier import verifier_node
from src.services.report import build_report
from src.workflow.state import EngiState

MAX_VERIFICATIONS = 2  # first check + exactly one revision


def report_node(state: EngiState) -> dict:
    return {"final_report": build_report(state)}


def route_after_verifier(state: EngiState) -> Literal["planner", "report"]:
    """Conditional edge: send back to the planner once if verification failed."""
    result = state["verification_result"]
    if not result.verified and result.attempts < MAX_VERIFICATIONS:
        return "planner"
    return "report"


def build_graph():
    graph = StateGraph(EngiState)
    graph.add_node("investigator", investigator_node)
    graph.add_node("planner", planner_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "investigator")
    graph.add_edge("investigator", "planner")
    graph.add_edge("planner", "verifier")
    graph.add_conditional_edges("verifier", route_after_verifier)
    graph.add_edge("report", END)
    return graph.compile()


def run_workflow(query: str, verbose: bool = False, on_step=None) -> EngiState:
    """Run the whole graph and return the final state.

    on_step(node_name, delta) is called after each node (the web UI uses it
    to show live progress).
    """
    app = build_graph()
    state: EngiState = {"user_query": query}
    for update in app.stream(state, stream_mode="updates"):  # one update per node
        for node_name, delta in update.items():
            if verbose:
                print(f"  [done] {node_name}")
            state.update(delta)
            if on_step:
                on_step(node_name, delta)
    return state