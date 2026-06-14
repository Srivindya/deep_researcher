"""
agents/retriever.py — Agent 1: Contextual Retriever

WHAT THIS AGENT DOES:
  1. Takes the user's research question from the shared state.
  2. Calls Tavily Search API to fetch up-to-date web articles & papers.
  3. Splits the fetched text into small overlapping chunks.
  4. Embeds those chunks and stores them in a FAISS vector store (in-memory).
  5. Retrieves the most relevant chunks using similarity search.
  6. Writes the chunks back into the shared state for downstream agents.

WHY FAISS?
  FAISS (Facebook AI Similarity Search) is a fast, in-memory vector store.
  No database server needed — perfect for a hackathon.

WHY CHUNK + EMBED?
  LLMs have a context window limit. By splitting text into chunks and only
  passing the TOP-K most relevant ones, we stay within the limit and reduce noise.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TAVILY_API_KEY,
    TAVILY_MAX_RESULTS,
    RAG_TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

import os

# Tavily requires its key as an env var (the library reads it automatically)
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


def retriever_node(state: dict) -> dict:
    """
    LangGraph node function.

    Every node receives the current shared `state` dict and returns a dict
    of keys it wants to update. LangGraph merges the return value back into state.

    Args:
        state: The shared AgentState dict. At this point it has:
               - "question": the user's research question (str)

    Returns:
        Dict with updated keys:
        - "retrieved_chunks": list of relevant text strings
        - "sources": list of source URLs found by Tavily
    """
    question: str = state["question"]

    # ------------------------------------------------------------------ #
    # STEP 1: Tavily Web Search
    # ------------------------------------------------------------------ #
    # TavilySearchResults returns a list of dicts like:
    #   [{"url": "...", "content": "..."}, ...]
    search_tool = TavilySearch(max_results=TAVILY_MAX_RESULTS)
    search_output = search_tool.invoke(question)
    # New langchain-tavily returns {"results": [...]} or a plain list
    raw_results: list[dict] = search_output.get("results", search_output) if isinstance(search_output, dict) else search_output

    # Collect URLs for citation in the final report
    sources: list[str] = [r["url"] for r in raw_results if "url" in r]

    # Turn each result into a LangChain Document object
    # Document = a piece of text with optional metadata
    documents: list[Document] = [
        Document(
            page_content=r.get("content", ""),
            metadata={"source": r.get("url", "unknown")},
        )
        for r in raw_results
        if r.get("content")
    ]

    # ------------------------------------------------------------------ #
    # STEP 2: Chunk the documents
    # ------------------------------------------------------------------ #
    # Why chunk? A single article can be 5000+ characters. We split it into
    # smaller pieces so the vector search can find the *exact* relevant part.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,  # overlap avoids cutting a sentence mid-thought
    )
    chunks: list[Document] = splitter.split_documents(documents)

    # ------------------------------------------------------------------ #
    # STEP 3: Embed chunks into FAISS
    # ------------------------------------------------------------------ #
    # OpenAIEmbeddings pointed at OpenRouter converts text → numeric vectors.
    # FAISS stores these vectors and can find the closest ones to a query vector.
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",  # cheap, fast embedding model
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    # ------------------------------------------------------------------ #
    # STEP 4: Retrieve top-K most relevant chunks
    # ------------------------------------------------------------------ #
    # similarity_search converts the question to a vector and finds the
    # nearest chunk vectors in FAISS.
    top_docs: list[Document] = vector_store.similarity_search(question, k=RAG_TOP_K)
    retrieved_chunks: list[str] = [doc.page_content for doc in top_docs]

    print(f"[Retriever] Fetched {len(raw_results)} pages → {len(chunks)} chunks → top {len(retrieved_chunks)} retrieved")

    return {
        "retrieved_chunks": retrieved_chunks,
        "sources": sources,
    }
