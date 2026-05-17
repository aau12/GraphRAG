# GraphRAG Inference Hackathon — Comparison Dashboard

A side-by-side benchmarking system that runs the same query through three different AI pipelines and compares their answers, token usage, latency, and cost.

---

## Overview

| Pipeline | Method | Retrieval |
|----------|--------|-----------|
| Pipeline 1: LLM-Only | Direct Gemini call | None |
| Pipeline 2: Basic RAG | BM25 keyword search + Gemini | Local text index |
| Pipeline 3: GraphRAG | Knowledge graph traversal + Gemini | TigerGraph Savanna |

**Dataset:** 164 Wikipedia articles on AI/ML topics — ~1M tokens covering neural networks, researchers, companies, models, and concepts.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                    │
│              dashboard.py  (localhost:8501)              │
└──────────────┬──────────────┬──────────────┬────────────┘
               │              │              │
    ┌──────────▼───┐  ┌───────▼──────┐  ┌───▼──────────────┐
    │  Pipeline 1  │  │  Pipeline 2  │  │   Pipeline 3      │
    │  LLM-Only    │  │  Basic RAG   │  │   GraphRAG        │
    └──────────┬───┘  └───────┬──────┘  └───┬──────────────┘
               │              │              │
               │        ┌─────▼──────┐  ┌───▼──────────────┐
               │        │  BM25 Index│  │  TigerGraph       │
               │        │rag_index   │  │  Savanna (cloud)  │
               │        │  .pkl      │  │  257 entities     │
               │        └─────┬──────┘  └───┬──────────────┘
               │              │              │
    ┌──────────▼──────────────▼──────────────▼──────────────┐
    │                  Gemini 2.5 Flash API                   │
    │              (google.genai SDK, free tier)              │
    └─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
graphrag-main/
├── dataset/                  # 164 Wikipedia AI/ML articles (.txt)
├── pipelines/
│   ├── __init__.py
│   ├── pipeline1_llm_only.py # Direct Gemini call, no retrieval
│   ├── pipeline2_basic_rag.py# BM25 search + Gemini
│   └── pipeline3_graphrag.py # TigerGraph traversal + Gemini
├── config.py                 # Google API key
├── dashboard.py              # Streamlit comparison UI
├── rag_index.pkl             # Pre-built BM25 index (1015 chunks)
├── LICENSE
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash (`google.genai` SDK) |
| Graph Database | TigerGraph Savanna (cloud, free tier) |
| Graph Client | pyTigerGraph 2.0.3 |
| Vector/Keyword Search | BM25 (pure Python, no dependencies) |
| Dashboard | Streamlit |
| Dataset | Wikipedia API (164 AI/ML articles) |

---

## Setup

### Prerequisites

- Python 3.10+
- Google Gemini API key (free tier: 20 req/day for gemini-2.5-flash)
- TigerGraph Savanna account (free tier available at tgcloud.io)

### Install Dependencies

```bash
pip install google-genai pyTigerGraph streamlit requests
```

### Configure API Key

Edit `config.py`:

```python
GOOGLE_API_KEY = "your-gemini-api-key-here"
```

Get a free key at: https://aistudio.google.com/apikey

---

## Running the Dashboard

```bash
streamlit run dashboard.py
```

Open http://localhost:8501 in your browser.

---

## Pipeline Details

### Pipeline 1 — LLM-Only

Sends the query directly to Gemini with no context. Baseline for comparison.

```
Query → Gemini 2.5 Flash → Answer
```

- **Tokens:** High (model uses training data only)
- **Latency:** ~5–10s
- **Cost:** Based on output tokens only

### Pipeline 2 — Basic RAG

Retrieves the top-5 most relevant chunks from the dataset using BM25 keyword scoring, then passes them as context to Gemini.

```
Query → BM25 Search (1015 chunks) → Top-5 chunks → Gemini → Answer
```

- **Retrieval:** Local, zero API calls, zero cost
- **Scoring:** BM25 (k1=1.5, b=0.75) over tokenized chunks
- **Tokens:** Medium (context-grounded answer)
- **Latency:** ~5–10s (retrieval is instant)

### Pipeline 3 — GraphRAG

Searches the TigerGraph knowledge graph for entities matching the query keywords, traverses relationships to nearby entities, and retrieves linked document chunks as context.

```
Query → Keyword Extraction → TigerGraph Entity Search
      → Relationship Traversal (hop 1 & 2)
      → DocChunk Retrieval
      → Context Assembly → Gemini → Answer
```

- **Graph schema:** Document → DocChunk → Entity → Entity (RELATED_TO)
- **Entities:** 257 extracted via regex from 10 articles
- **Context:** Entity relationships + raw chunk content
- **Tokens:** Low (graph-filtered, precise context)

---

## Graph Schema (TigerGraph)

```
Document  ──HAS_CHUNK──►  DocChunk  ──MENTIONS──►  Entity
                                                      │
                                                 RELATED_TO
                                                      │
                                                    Entity
```

**Vertex types:**
- `Document` — article title and ID
- `DocChunk` — 600-word text chunk with source title
- `Entity` — named entity with type (PERSON/ORG/CONCEPT) and description

**Edge types:**
- `HAS_CHUNK` — Document to its chunks
- `MENTIONS` — Chunk to entities it references
- `RELATED_TO` — Entity co-occurrence relationships

---

## Metrics Compared

| Metric | Description |
|--------|-------------|
| Total Tokens | Prompt + completion tokens sent/received |
| Latency | End-to-end response time in seconds |
| Cost/Query | Estimated USD cost at $0.15/1M tokens |
| Chunks Retrieved | Number of text chunks used as context (Pipeline 2) |
| Graph Facts Used | Number of entity/relationship facts retrieved (Pipeline 3) |

---

## Dataset

164 Wikipedia articles covering:

- **Concepts:** Neural networks, backpropagation, transformers, attention, BERT, GPT, LLMs, RAG, embeddings, gradient descent, reinforcement learning
- **Researchers:** Geoffrey Hinton, Yann LeCun, Yoshua Bengio, Andrej Karpathy, Andrew Ng, Alan Turing
- **Companies:** OpenAI, DeepMind, Anthropic, Google, Meta, NVIDIA, Cohere
- **Models:** ChatGPT, Claude, Gemini, LLaMA, Mistral, AlphaGo, AlphaFold, DALL-E
- **Applications:** Autonomous vehicles, robotics, speech recognition, computer vision

Total: ~1,001,687 tokens

---

## API Quota Notes

Gemini free tier limits:
- `gemini-2.5-flash`: 20 requests/day
- `gemini-2.0-flash`: unavailable in some regions (India)

Each full 3-pipeline comparison = 3 API requests. With 20 req/day, you get ~6 full comparison runs.

**Tip:** The BM25 index (Pipeline 2) and TigerGraph graph traversal (Pipeline 3 entity search) use zero API calls — only the final Gemini answer generation costs a request.

---

## TigerGraph Connection

The graph is hosted on TigerGraph Savanna (free cloud tier). Authentication uses JWT tokens via the `/gsql/v1/tokens` endpoint.

Connection config in `pipeline3_graphrag.py`:
```python
TG_HOST   = "https://tg-<workspace-id>.tgcloud.io"
TG_SECRET = "your-tigergraph-secret"
TG_GRAPH  = "GraphRAG"
```

---

## Hackathon Submission

**Event:** TigerGraph GraphRAG Inference Hackathon  
**Theme:** Compare LLM-Only vs Basic RAG vs GraphRAG on a 1M-token AI/ML knowledge base  
**Key finding:** GraphRAG uses graph-structured context to answer with fewer, more targeted tokens compared to LLM-Only and Basic RAG approaches.
