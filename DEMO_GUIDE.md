# Multi-Agent AI Deep Researcher — Demo Guide

## What Was Built

A multi-agent AI system where 4 specialized AI agents collaborate to produce a research report on any topic. The user types one question; the agents handle everything else.

---

## System Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                      │
│                                                             │
│  ┌──────────────┐     ┌──────────────┐                      │
│  │  Agent 1     │────▶│  Agent 2     │                      │
│  │  Retriever   │     │  Analyzer    │                      │
│  │              │     │              │                      │
│  │ Tavily Search│     │ Finds:       │                      │
│  │ → FAISS RAG  │     │ • Key facts  │                      │
│  │ → top chunks │     │ • Conflicts  │                      │
│  └──────────────┘     │ • Gaps       │                      │
│                       └──────┬───────┘                      │
│                              │                              │
│  ┌──────────────┐     ┌──────▼───────┐                      │
│  │  Agent 4     │◀────│  Agent 3     │                      │
│  │  Report      │     │  Insight     │                      │
│  │  Builder     │     │  Generator   │                      │
│  │              │     │              │                      │
│  │ Final .md    │     │ • Trends     │                      │
│  │ report       │     │ • Hypotheses │                      │
│  └──────────────┘     │ • Implications│                     │
│                       └──────────────┘                      │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
Streamlit UI (browser)
```

---

## File Structure

```
deep_researcher/
├── agents/
│   ├── __init__.py
│   ├── retriever.py       ← Agent 1: Tavily search + FAISS vector store
│   ├── analyzer.py        ← Agent 2: LLM critical analysis
│   ├── insight.py         ← Agent 3: LLM insight generation
│   └── report_builder.py  ← Agent 4: LLM report synthesis
├── graph/
│   ├── __init__.py
│   └── workflow.py        ← LangGraph StateGraph definition
├── .env.example           ← Template for API keys
├── config.py              ← All settings (keys, model name, chunk sizes)
├── app.py                 ← Streamlit web UI
├── requirements.txt
└── DEMO_GUIDE.md          ← This file
```

---

## Setup Instructions (Step by Step)

### Step 1 — Create a virtual environment

```bash
cd deep_researcher
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Set up your API keys

```bash
# Copy the example file
cp .env.example .env
```

Now open `.env` and fill in your real keys:
```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxx
```

Where to get them:
- **OpenRouter**: https://openrouter.ai → Settings → API Keys → Create Key
- **Tavily**: https://app.tavily.com → API Keys → Create New Key

### Step 4 — Run the app

```bash
streamlit run app.py
```

A browser window will open at `http://localhost:8501`

---

## How to Demo (Judge Walkthrough)

### Opening (30 seconds)

> "This is a multi-agent AI research assistant. Instead of one AI doing everything, we've broken the research process into four specialized agents, each with a single job — just like a real research team."

### Live Demo (2 minutes)

1. Type a research question. Good examples for demos:
   - `"What are the latest breakthroughs in quantum computing for drug discovery?"`
   - `"How is AI changing the future of cybersecurity?"`
   - `"What are the economic impacts of generative AI on the job market?"`

2. Click **Run Research**

3. Point out the live status panel showing each agent running

4. When complete, scroll through the report sections with the judge:
   - **Executive Summary** — the 30-second answer
   - **Key Findings** — the facts with source attribution
   - **Contradictions** — shows the AI is *thinking critically*, not just copy-pasting
   - **Insights** — forward-looking, original reasoning

5. Click **View Raw Analysis** and **View Raw Insights** expandables to show the intermediate agent outputs

6. Click **Download Report as Markdown** to show it's production-ready output

### Technical Talking Points

| Concept | Where it appears | What to say |
|---------|-----------------|-------------|
| **RAG** | Agent 1 (Retriever) | "We don't just send raw search results to the LLM — we embed them into a FAISS vector store and retrieve only the most relevant chunks. This keeps the LLM focused." |
| **LangGraph** | graph/workflow.py | "LangGraph manages the state machine — it passes a shared state dict between agents and ensures each agent's output feeds into the next." |
| **LangChain** | All agent files | "LangChain provides the building blocks: document loaders, text splitters, prompt templates, and the LLM interface." |
| **MCP-ready** | Architecture | "This pipeline can be exposed as an MCP server so any Claude-powered tool can call it as a single tool call." |
| **OpenRouter** | config.py | "OpenRouter gives us model flexibility — we can swap Claude for GPT-4o or Mistral by changing one line in config.py." |
| **Tavily** | retriever.py | "Tavily is purpose-built for AI agents — it returns clean, structured content rather than raw HTML." |

---

## Code Walkthrough (for technical judges)

### How the state machine works

Open `graph/workflow.py`. The key class is `AgentState`:

```python
class AgentState(TypedDict):
    question: str           # set by the user at the start
    retrieved_chunks: list  # set by Agent 1
    sources: list           # set by Agent 1
    analysis: str           # set by Agent 2
    insights: str           # set by Agent 3
    final_report: str       # set by Agent 4
```

Each agent function receives this dict, does its work, and returns *only the keys it changes*. LangGraph merges those keys back into the shared state automatically.

### How the RAG pipeline works

Open `agents/retriever.py`:

1. `TavilySearchResults.invoke(question)` → returns list of `{url, content}` dicts
2. `RecursiveCharacterTextSplitter` → splits long articles into 1000-char chunks with 150-char overlap
3. `OpenAIEmbeddings` → converts each chunk to a vector (list of numbers)
4. `FAISS.from_documents` → stores all vectors in an in-memory index
5. `similarity_search(question, k=5)` → finds the 5 chunks whose vectors are closest to the question vector

This is the core of Retrieval-Augmented Generation (RAG).

### How agents prompt the LLM

Open `agents/analyzer.py`:

```python
system_msg = SystemMessage(content="You are a critical research analyst...")
human_msg = HumanMessage(content=f"Research Question: {question}\n\n{context_block}\n\n...")
response = llm.invoke([system_msg, human_msg])
```

- `SystemMessage` sets the agent's *persona and constraints*
- `HumanMessage` provides the *task and data*
- `llm.invoke()` calls OpenRouter which routes to Claude/GPT-4o

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `KeyError: 'OPENROUTER_API_KEY'` | `.env` file missing or wrong key name | Check `.env` file has both keys exactly as shown |
| `401 Unauthorized` from OpenRouter | Wrong API key | Verify at openrouter.ai → Settings → API Keys |
| `401 Unauthorized` from Tavily | Wrong Tavily key | Verify at app.tavily.com |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` in the venv |
| Embedding model error | OpenRouter may not support the embedding model | Switch to OpenAI embeddings with a real OpenAI key, or use `langchain_community.embeddings.FakeEmbeddings` for testing |
| Slow response | LLM calls are sequential | Normal — 4 LLM calls take 20-60s total |
