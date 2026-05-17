"""Pipeline 3: GraphRAG — entity extraction + TigerGraph Savanna + Gemini."""

import os
import re
import time
import json
import glob
import hashlib
import warnings
import requests
from google import genai
from google.genai import types
import pyTigerGraph as tg

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
TG_HOST     = "https://tg-6590283b-f841-4f3e-b8db-0f57dd2a28be.tg-3452941248.i.tgcloud.io"
TG_USER     = "ayushiupadhyay40@gmail.com"
TG_SECRET   = "s9cn2oq0p20ajcorc1kr8vuo0mf0h6j4"
TG_GRAPH    = "GraphRAG"
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
CHUNK_SIZE  = 600
CHUNK_OVERLAP = 80
TOP_K_ENTITIES = 5

# ── Helpers ──────────────────────────────────────────────────────────────────
def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 50]

def _short_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]

# ── TigerGraph Connection ─────────────────────────────────────────────────────
def get_connection() -> tg.TigerGraphConnection:
    # Get JWT token via Savanna's GSQL token endpoint
    r = requests.post(
        f"{TG_HOST}/gsql/v1/tokens",
        json={"secret": TG_SECRET},
        verify=False,
        timeout=30,
    )
    r.raise_for_status()
    token = r.json()["token"]

    conn = tg.TigerGraphConnection(
        host=TG_HOST,
        gsPort="443",
        restppPort="443",
        graphname=TG_GRAPH,
        tgCloud=True,
        useCert=True,
        jwtToken=token,
    )
    return conn

# ── Schema Init ──────────────────────────────────────────────────────────────
SCHEMA_GSQL = f"""
USE GLOBAL
CREATE VERTEX Document   (PRIMARY_ID id STRING, title STRING) WITH STATS="OUTDEGREE_BY_EDGETYPE"
CREATE VERTEX Entity     (PRIMARY_ID id STRING, name STRING, entity_type STRING, description STRING) WITH STATS="OUTDEGREE_BY_EDGETYPE"
CREATE VERTEX DocChunk   (PRIMARY_ID id STRING, content STRING, doc_title STRING) WITH STATS="OUTDEGREE_BY_EDGETYPE"
CREATE DIRECTED EDGE HAS_CHUNK      (FROM Document, TO DocChunk)
CREATE DIRECTED EDGE MENTIONS       (FROM DocChunk, TO Entity)
CREATE DIRECTED EDGE RELATED_TO     (FROM Entity, TO Entity, relationship STRING)
CREATE GRAPH {TG_GRAPH} (Document, Entity, DocChunk, HAS_CHUNK, MENTIONS, RELATED_TO)
"""

def init_schema(conn: tg.TigerGraphConnection):
    try:
        conn.gsql(SCHEMA_GSQL)
        print("Schema created.")
    except Exception as e:
        msg = str(e).lower()
        if "already exists" in msg or "semantic check fails" in msg or "is used by another" in msg:
            print("Schema already exists, skipping.")
        else:
            raise

# ── Entity Extraction (regex-based, zero API calls) ───────────────────────────
_AI_TERMS = {
    "neural network","deep learning","machine learning","backpropagation","gradient descent",
    "transformer","attention","bert","gpt","llm","reinforcement learning","convolutional",
    "lstm","rnn","gan","autoencoder","embedding","tokenization","fine-tuning","pre-training",
    "supervised","unsupervised","overfitting","regularization","dropout","batch normalization",
    "openai","deepmind","anthropic","google","meta","microsoft","nvidia","pytorch","tensorflow",
    "keras","hugging face","langchain","rag","vector database","knowledge graph","tigergraph",
    "geoffrey hinton","yann lecun","yoshua bengio","andrej karpathy","sam altman",
    "natural language processing","computer vision","speech recognition","diffusion model",
    "stable diffusion","dall-e","midjourney","chatgpt","gemini","claude","llama","mistral",
}

def extract_entities(text: str, client=None) -> dict:
    """Rule-based entity extraction — uses zero API calls."""
    entities, seen = [], set()
    lower = text.lower()

    # Match known AI/ML terms
    for term in _AI_TERMS:
        if term in lower and term not in seen:
            seen.add(term)
            etype = "PERSON" if any(c.isupper() for c in term.split()[0]) else "CONCEPT"
            entities.append({"name": term.title(), "type": etype, "description": ""})

    # Match capitalized proper nouns (2-3 word sequences)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text):
        name = m.group(1)
        if name.lower() not in seen and len(name) > 3:
            seen.add(name.lower())
            entities.append({"name": name, "type": "CONCEPT", "description": ""})

    # Simple co-occurrence relationships between found entities
    names = [e["name"] for e in entities[:6]]
    relationships = []
    for i in range(len(names) - 1):
        relationships.append({"source": names[i], "target": names[i+1], "relationship": "related to"})

    return {"entities": entities[:8], "relationships": relationships[:6]}

# ── Ingest ────────────────────────────────────────────────────────────────────
def ingest(api_key: str, max_articles: int = 30):
    """Extract entities from dataset and load into TigerGraph (zero LLM calls)."""
    conn = get_connection()
    init_schema(conn)

    files = glob.glob(os.path.join(DATASET_DIR, "*.txt"))[:max_articles]
    print(f"\nIngesting {len(files)} articles into TigerGraph...")

    total_entities, total_chunks = 0, 0

    for filepath in files:
        text = open(filepath, encoding="utf-8").read()
        title = os.path.basename(filepath).replace(".txt", "").replace("_", " ")
        doc_id = _short_id(title)

        conn.upsertVertex("Document", doc_id, {"title": title})

        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            if i % 3 != 0:
                continue
            chunk_id = _short_id(chunk)
            conn.upsertVertex("DocChunk", chunk_id, {"content": chunk[:500], "doc_title": title})
            conn.upsertEdge("Document", doc_id, "HAS_CHUNK", "DocChunk", chunk_id)
            total_chunks += 1

            extracted = extract_entities(chunk)
            entities = extracted.get("entities", [])
            relationships = extracted.get("relationships", [])

            for ent in entities:
                name = ent.get("name", "").strip()
                if not name:
                    continue
                ent_id = _short_id(name.lower())
                conn.upsertVertex("Entity", ent_id, {
                    "name": name,
                    "entity_type": ent.get("type", "CONCEPT"),
                    "description": ent.get("description", "")[:200],
                })
                conn.upsertEdge("DocChunk", chunk_id, "MENTIONS", "Entity", ent_id)
                total_entities += 1

            for rel in relationships:
                src = rel.get("source", "").strip()
                tgt = rel.get("target", "").strip()
                relationship = rel.get("relationship", "related to")
                if src and tgt:
                    src_id = _short_id(src.lower())
                    tgt_id = _short_id(tgt.lower())
                    try:
                        conn.upsertEdge("Entity", src_id, "RELATED_TO", "Entity", tgt_id,
                                        {"relationship": relationship})
                    except Exception:
                        pass

        print(f"  OK {title[:50]} - {len(chunks)} chunks, {total_entities} entities so far")

    print(f"\nDone! Ingested {total_chunks} chunks, {total_entities} entities.")

# ── Query (Python REST traversal, no GSQL) ───────────────────────────────────
def _graph_search(conn, keyword: str, top_k: int = TOP_K_ENTITIES) -> list[str]:
    """Traverse graph via REST API: find matching entities → related entities → chunks."""
    context_items = []
    seen = set()

    # Step 1: get all entities, filter by keyword in Python
    try:
        all_entities = conn.getVertices("Entity", limit=500)
    except Exception:
        return context_items

    relevant = [
        v for v in all_entities
        if keyword in v["attributes"].get("name", "").lower()
    ][:top_k]

    if not relevant:
        # Fallback: partial match on any word in the keyword
        for word in keyword.split():
            if len(word) < 4:
                continue
            relevant = [
                v for v in all_entities
                if word in v["attributes"].get("name", "").lower()
            ][:top_k]
            if relevant:
                break

    # Step 2: get related entities (hop 1 & 2)
    for ent in relevant:
        eid = ent["v_id"]
        name = ent["attributes"].get("name", "")
        try:
            neighbors = conn.getEdges("Entity", eid, "RELATED_TO")
            for edge in neighbors[:10]:
                rel = edge["attributes"].get("relationship", "related to")
                tgt_id = edge["to_id"]
                try:
                    tgt = conn.getVerticesById("Entity", tgt_id)
                    tgt_name = tgt[0]["attributes"].get("name", tgt_id) if tgt else tgt_id
                except Exception:
                    tgt_name = tgt_id
                fact = f"{name} {rel} {tgt_name}"
                if fact not in seen:
                    seen.add(fact)
                    context_items.append(fact)
        except Exception:
            pass

    # Step 3: get DocChunks that mention these entities
    for ent in relevant:
        eid = ent["v_id"]
        try:
            edges = conn.getEdges("DocChunk", sourceVertexType="DocChunk",
                                  edgeType="MENTIONS", targetVertexType="Entity",
                                  targetVertexId=eid)
            for edge in edges[:3]:
                chunk_id = edge["from_id"]
                try:
                    chunks = conn.getVerticesById("DocChunk", chunk_id)
                    if chunks:
                        content = chunks[0]["attributes"].get("content", "")
                        if content and content not in seen:
                            seen.add(content)
                            context_items.append(content)
                except Exception:
                    pass
        except Exception:
            pass

    return context_items

def run(query: str, api_key: str) -> dict:
    client = genai.Client(api_key=api_key)
    conn = get_connection()
    start = time.time()

    # Extract meaningful keywords from query
    _stop = {"how", "to", "what", "is", "are", "the", "a", "an", "does", "do",
             "why", "when", "where", "who", "which", "was", "were", "can", "could",
             "would", "should", "in", "of", "and", "or", "explain", "describe"}
    words = query.lower().split()
    keywords = [w for w in words if w not in _stop and len(w) > 2]
    keyword = " ".join(keywords[:2]) if keywords else query.lower()

    try:
        context_items = _graph_search(conn, keyword)
        # If no results with multi-word, try first keyword alone
        if not context_items and keywords:
            context_items = _graph_search(conn, keywords[0])
        context = "\n".join(context_items[:30]) if context_items else "No graph context found."
    except Exception as e:
        context = f"Graph traversal error: {e}"
        context_items = []

    prompt = f"""You are answering based on a knowledge graph about AI and machine learning.

Graph Context (entities and relationships):
{context}

Question: {query}

Answer concisely using the graph context:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    latency = time.time() - start

    usage = response.usage_metadata
    prompt_tokens = usage.prompt_token_count or 0
    completion_tokens = usage.candidates_token_count or 0
    total_tokens = usage.total_token_count or 0

    cost = total_tokens * 0.00000015

    return {
        "pipeline": "GraphRAG",
        "answer": response.text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": round(latency, 2),
        "cost_usd": round(cost, 6),
        "context_facts": len(context_items),
        "graph_context_preview": context[:300],
    }
