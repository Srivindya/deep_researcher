"""
agents/insight.py — Agent 3: Insight Generation Agent

WHAT THIS AGENT DOES:
  1. Takes the critical analysis from Agent 2.
  2. Asks the LLM to *go beyond* the raw facts and generate:
       - Emerging trends implied by the findings
       - Hypotheses worth investigating
       - Practical implications or recommendations
       - "So what?" — why does this matter?
  3. Writes the insights back into shared state.

WHY A SEPARATE AGENT?
  Analysis = "what do the sources say and where do they disagree"
  Insight   = "what does that MEAN — what patterns, trends, or implications emerge"

  These are cognitively distinct tasks. Separating them gives cleaner,
  more focused outputs from the LLM rather than one jumbled response.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL


def insight_node(state: dict) -> dict:
    """
    LangGraph node function for insight generation.

    Args:
        state: Shared state. At this point it has:
               - "question": original research question
               - "analysis": structured analysis text (from Analyzer)

    Returns:
        Dict with:
        - "insights": bullet-pointed insights and trends string
    """
    question: str = state["question"]
    analysis: str = state["analysis"]

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.6,  # slightly higher temp = more creative/generative thinking
    )

    system_msg = SystemMessage(content=(
        "You are a strategic research synthesizer. Given a critical analysis of sources, "
        "your role is to generate higher-order insights: trends, patterns, hypotheses, "
        "and practical implications that go beyond merely summarizing the facts. "
        "Think like a consultant or senior researcher writing for an executive audience."
    ))

    human_msg = HumanMessage(content=(
        f"Research Question: {question}\n\n"
        f"Critical Analysis from sources:\n{analysis}\n\n"
        "Now generate insights in these exact sections:\n\n"
        "## Emerging Trends\n"
        "What patterns or directions are emerging based on this evidence? "
        "Use bullet points. Each bullet should have a one-line trend + one-line explanation.\n\n"
        "## Hypotheses Worth Investigating\n"
        "What follow-up questions or hypotheses does this analysis suggest? "
        "List 3-5 specific, testable hypotheses.\n\n"
        "## Practical Implications\n"
        "For a practitioner or decision-maker in this field, what does this mean? "
        "What actions or priorities does this suggest?\n\n"
        "## Why This Matters\n"
        "2-3 sentences on the broader significance of these findings."
    ))

    response = llm.invoke([system_msg, human_msg])
    insights: str = response.content

    print(f"[Insight] Insight generation complete ({len(insights)} chars)")

    return {"insights": insights}
