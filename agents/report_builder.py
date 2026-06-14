"""
agents/report_builder.py — Agent 4: Report Builder Agent

WHAT THIS AGENT DOES:
  1. Collects ALL outputs from the previous 3 agents:
       - retrieved_chunks (from Retriever)
       - analysis        (from Analyzer)
       - insights        (from Insight Generator)
       - sources         (from Retriever)
  2. Asks the LLM to synthesize everything into a single, polished
     markdown research report that a non-technical reader could understand.
  3. Writes the final report into the shared state.

OUTPUT FORMAT:
  The report follows a professional research document structure:
  - Executive Summary
  - Key Findings
  - Contradictions / Caveats
  - Insights & Trends
  - Sources
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL


def report_builder_node(state: dict) -> dict:
    """
    LangGraph node function for report generation.

    Args:
        state: Fully populated shared state with keys:
               - "question", "analysis", "insights", "sources"

    Returns:
        Dict with:
        - "final_report": the complete markdown report as a string
    """
    question: str = state["question"]
    analysis: str = state["analysis"]
    insights: str = state["insights"]
    sources: list[str] = state.get("sources", [])

    # Format sources as a numbered markdown list
    sources_block = "\n".join(f"{i+1}. {url}" for i, url in enumerate(sources))

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.2,  # low temp = consistent, professional writing
    )

    system_msg = SystemMessage(content=(
        "You are a professional research report writer. "
        "Your job is to take structured analysis and insights from a research team "
        "and compile them into a single, well-written markdown report. "
        "The report should be clear, precise, and readable by someone without deep domain expertise. "
        "Use markdown formatting: headers, bullet points, and bold for key terms."
    ))

    human_msg = HumanMessage(content=(
        f"Research Question: **{question}**\n\n"
        f"=== ANALYSIS (from Critical Analysis Agent) ===\n{analysis}\n\n"
        f"=== INSIGHTS (from Insight Generation Agent) ===\n{insights}\n\n"
        f"=== SOURCES ===\n{sources_block}\n\n"
        "Write a complete research report in markdown with these exact sections:\n\n"
        "# Research Report: [restate the research question as a title]\n\n"
        "## Executive Summary\n"
        "3-5 sentence overview: what was researched, key conclusion, and why it matters.\n\n"
        "## Key Findings\n"
        "The most important facts from the analysis. Use numbered bullet points.\n\n"
        "## Contradictions & Caveats\n"
        "Where sources disagreed and what limitations exist in the evidence.\n\n"
        "## Insights & Trends\n"
        "The forward-looking insights and implications.\n\n"
        "## Sources\n"
        "List all sources as a numbered markdown list.\n\n"
        "Keep the tone professional but accessible."
    ))

    response = llm.invoke([system_msg, human_msg])
    final_report: str = response.content

    print(f"[ReportBuilder] Report complete ({len(final_report)} chars)")

    return {"final_report": final_report}
