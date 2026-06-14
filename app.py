"""
app.py — Streamlit Web UI

WHAT THIS FILE DOES:
  Provides a browser-based interface for the multi-agent researcher.
  The user types a question, clicks "Run Research", and watches the
  4 agents execute one by one with live status updates.

HOW STREAMLIT WORKS:
  - Every time the user interacts (clicks a button, types), Streamlit
    re-runs the entire script from top to bottom.
  - st.session_state persists values between re-runs (like a global store).
  - st.status() creates a collapsible progress panel.

TO RUN:
  streamlit run app.py
"""

import streamlit as st
from graph.workflow import run_research

# ------------------------------------------------------------------ #
# Page configuration (must be the FIRST Streamlit call)
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Multi-Agent Deep Researcher",
    page_icon="🔬",
    layout="wide",
)

# ------------------------------------------------------------------ #
# Header
# ------------------------------------------------------------------ #
st.title("🔬 Multi-Agent AI Deep Researcher")
st.markdown(
    """
    Enter any research question below. Four AI agents will collaborate to produce
    a comprehensive research report:

    | Agent | Role |
    |-------|------|
    | 🌐 **Retriever** | Searches the web via Tavily, chunks & embeds into FAISS |
    | 🔍 **Analyzer** | Critically analyzes findings, spots contradictions & gaps |
    | 💡 **Insight Generator** | Identifies trends, hypotheses & implications |
    | 📝 **Report Builder** | Synthesizes everything into a polished report |
    """
)

st.divider()

# ------------------------------------------------------------------ #
# Input section
# ------------------------------------------------------------------ #
question = st.text_area(
    label="Research Question",
    placeholder="e.g. What are the latest breakthroughs in quantum computing and their practical applications?",
    height=100,
)

run_button = st.button("🚀 Run Research", type="primary", use_container_width=True)

# ------------------------------------------------------------------ #
# Research execution
# ------------------------------------------------------------------ #
if run_button:
    if not question.strip():
        st.warning("Please enter a research question before running.")
        st.stop()

    # Show a live status panel with per-agent progress
    with st.status("Running multi-agent research pipeline...", expanded=True) as status_panel:

        st.write("🌐 **Agent 1 — Retriever:** Searching the web and building vector store...")

        # We run the whole pipeline at once (sequential inside LangGraph).
        # In a production app you could stream per-agent updates using callbacks.
        try:
            final_state = run_research(question)
        except Exception as e:
            status_panel.update(label="Pipeline failed", state="error")
            st.error(f"Error during research: {e}")
            st.stop()

        st.write("🔍 **Agent 2 — Analyzer:** Critical analysis complete.")
        st.write("💡 **Agent 3 — Insight Generator:** Insights generated.")
        st.write("📝 **Agent 4 — Report Builder:** Final report compiled.")

        status_panel.update(label="Research complete!", state="complete", expanded=False)

    st.divider()

    # ------------------------------------------------------------------ #
    # Display the final report
    # ------------------------------------------------------------------ #
    st.subheader("📄 Research Report")
    st.markdown(final_state["final_report"])

    st.divider()

    # ------------------------------------------------------------------ #
    # Expandable sections for intermediate agent outputs
    # ------------------------------------------------------------------ #
    with st.expander("🔍 View Raw Analysis (Agent 2 output)"):
        st.markdown(final_state["analysis"])

    with st.expander("💡 View Raw Insights (Agent 3 output)"):
        st.markdown(final_state["insights"])

    with st.expander("🌐 View Retrieved Sources"):
        sources = final_state.get("sources", [])
        if sources:
            for i, url in enumerate(sources, 1):
                st.markdown(f"{i}. [{url}]({url})")
        else:
            st.write("No sources recorded.")

    with st.expander("📦 View Retrieved Chunks (raw RAG content)"):
        chunks = final_state.get("retrieved_chunks", [])
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"**Chunk {i}:**")
            st.text(chunk)
            st.divider()

    # Download button for the report
    st.download_button(
        label="⬇️ Download Report as Markdown",
        data=final_state["final_report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
