"""
agents/analyzer.py — Agent 2: Critical Analysis Agent

WHAT THIS AGENT DOES:
  1. Takes the retrieved text chunks from the shared state (written by Agent 1).
  2. Sends them to the LLM with a "critical analyst" system prompt.
  3. Asks the LLM to:
       - Summarize the key findings across all sources
       - Spot contradictions between sources
       - Flag low-quality, vague, or unreliable content
       - Identify gaps (what the sources DON'T cover)
  4. Writes the structured analysis back into the shared state.

WHY A SEPARATE AGENT?
  Separation of concerns — the retriever only fetches, the analyzer only thinks.
  This mirrors how a real research team works: a librarian fetches papers,
  an analyst reads and critiques them.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL


def analyzer_node(state: dict) -> dict:
    """
    LangGraph node function for critical analysis.

    Args:
        state: Shared state. At this point it has:
               - "question": original research question
               - "retrieved_chunks": list of relevant text strings (from Retriever)

    Returns:
        Dict with:
        - "analysis": structured analysis string from the LLM
    """
    question: str = state["question"]
    chunks: list[str] = state["retrieved_chunks"]

    # ------------------------------------------------------------------ #
    # Build the LLM client (pointed at OpenRouter)
    # ------------------------------------------------------------------ #
    # ChatOpenAI is LangChain's wrapper for any OpenAI-compatible chat API.
    # By overriding openai_api_base, we redirect traffic to OpenRouter.
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.3,  # low temperature = more factual, less creative
    )

    # ------------------------------------------------------------------ #
    # Assemble the context from retrieved chunks
    # ------------------------------------------------------------------ #
    # We number each chunk so the LLM can reference them by number
    context_block = "\n\n".join(
        f"[Source {i+1}]:\n{chunk}" for i, chunk in enumerate(chunks)
    )

    # ------------------------------------------------------------------ #
    # Build the prompt messages
    # ------------------------------------------------------------------ #
    # SystemMessage = tells the LLM what ROLE to play
    # HumanMessage  = the actual task/question
    system_msg = SystemMessage(content=(
        "You are a critical research analyst. Your job is to rigorously examine "
        "retrieved source material and produce a structured analysis. "
        "Be precise, skeptical, and evidence-based. Do not add information beyond what is in the sources."
    ))

    human_msg = HumanMessage(content=(
        f"Research Question: {question}\n\n"
        f"Retrieved Source Material:\n{context_block}\n\n"
        "Produce a structured analysis with these exact sections:\n"
        "## Key Findings\n"
        "List the most important facts and claims from the sources (with source numbers).\n\n"
        "## Contradictions & Conflicts\n"
        "Identify any places where sources disagree or contradict each other.\n\n"
        "## Gaps & Limitations\n"
        "What important angles does this material NOT cover?\n\n"
        "## Source Reliability Notes\n"
        "Flag any sources that seem vague, biased, or of low quality."
    ))

    # ------------------------------------------------------------------ #
    # Call the LLM
    # ------------------------------------------------------------------ #
    response = llm.invoke([system_msg, human_msg])
    analysis: str = response.content

    print(f"[Analyzer] Analysis complete ({len(analysis)} chars)")

    return {"analysis": analysis}
