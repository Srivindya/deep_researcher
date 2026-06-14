"""
graph/workflow.py — LangGraph Orchestration (the "brain" of the system)

WHAT THIS FILE DOES:
  Defines the multi-agent pipeline as a LangGraph StateGraph.

HOW LANGGRAPH WORKS:
  1. You define a shared state (AgentState) — a TypedDict that every agent can read/write.
  2. You add each agent function as a "node" in the graph.
  3. You add directed "edges" that say "after node A, run node B".
  4. LangGraph handles:
       - Passing the state between nodes
       - Merging each node's return dict back into the shared state
       - Running the graph from START to END

FLOW:
  START
    └→ retriever_node   (web search + RAG)
         └→ analyzer_node   (critical analysis)
              └→ insight_node   (trend & hypothesis generation)
                   └→ report_builder_node   (final markdown report)
                        └→ END

The shared state grows richer at each step:
  After retriever:  state has question + retrieved_chunks + sources
  After analyzer:   state also has analysis
  After insight:    state also has insights
  After report:     state also has final_report
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from agents.retriever import retriever_node
from agents.analyzer import analyzer_node
from agents.insight import insight_node
from agents.report_builder import report_builder_node


# ------------------------------------------------------------------ #
# 1. Define the shared state schema
# ------------------------------------------------------------------ #
# TypedDict is just a typed dictionary — it tells Python (and you)
# exactly which keys exist in the state and what their types are.
class AgentState(TypedDict):
    question: str              # The user's research question (set at the start)
    retrieved_chunks: list     # Top-K relevant text chunks (set by retriever)
    sources: list              # Source URLs (set by retriever)
    analysis: str              # Critical analysis text (set by analyzer)
    insights: str              # Insights and trends text (set by insight agent)
    final_report: str          # Final markdown report (set by report builder)


# ------------------------------------------------------------------ #
# 2. Build the graph
# ------------------------------------------------------------------ #
def build_graph():
    """
    Constructs and compiles the LangGraph StateGraph.

    Returns:
        A compiled LangGraph graph ready to call with .invoke()
    """
    # Create a new graph that uses AgentState as its state schema
    graph = StateGraph(AgentState)

    # Add each agent as a named node
    # The string name ("retriever", etc.) is just an internal label
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("insight", insight_node)
    graph.add_node("report_builder", report_builder_node)

    # Add edges (the arrows in the flow diagram above)
    # START is a special LangGraph constant for the entry point
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "analyzer")
    graph.add_edge("analyzer", "insight")
    graph.add_edge("insight", "report_builder")
    graph.add_edge("report_builder", END)

    # compile() validates the graph and returns a runnable object
    return graph.compile()


# ------------------------------------------------------------------ #
# 3. Convenience function used by app.py
# ------------------------------------------------------------------ #
def run_research(question: str) -> dict:
    """
    Public entry point: run the full multi-agent pipeline.

    Args:
        question: The user's research question string.

    Returns:
        The final AgentState dict with all keys populated,
        including "final_report" which is the markdown output.
    """
    graph = build_graph()

    # .invoke() runs the graph synchronously and returns the final state
    final_state = graph.invoke({"question": question})
    return final_state
