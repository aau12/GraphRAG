"""Pipeline 2: Basic RAG — BM25-style keyword retrieval + Gemini."""

import os
import re
import time
import glob
import json
import math
import pickle
from collections import defaultdict
from google import genai
from google.genai import types

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(_BASE, "dataset")
INDEX_FILE  = os.path.join(_BASE, "rag_index.pkl")
TOP_K = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ── Text helpers ─────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())

def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 50]

# ── Index build ───────────────────────────────────────────────────────────────
def build_index(api_key: str = None, force_rebuild: bool = False):
    """Build a BM25-style inverted index from dataset files."""
    index_path = INDEX_FILE

    if os.path.exists(index_path) and not force_rebuild:
        print("RAG index already exists, loading.")
        with open(index_path, "rb") as f:
            return pickle.load(f)

    files = glob.glob(os.path.join(DATASET_DIR, "*.txt"))
    print(f"Building index from {len(files)} files...")

    chunks   = []   # list of {"text": ..., "source": ...}
    tf       = []   # term frequency per chunk
    df       = defaultdict(int)  # doc frequency per term

    for filepath in files:
        text  = open(filepath, encoding="utf-8").read()
        title = os.path.basename(filepath).replace(".txt","").replace("_"," ")
        for chunk in _chunk_text(text):
            tokens = _tokenize(chunk)
            freq   = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            tf.append(dict(freq))
            chunks.append({"text": chunk, "source": title})
            for t in freq:
                df[t] += 1

    index = {"chunks": chunks, "tf": tf, "df": dict(df), "n": len(chunks)}
    with open(index_path, "wb") as f:
        pickle.dump(index, f)

    print(f"Indexed {len(chunks)} chunks.")
    return index

# ── BM25 retrieval ────────────────────────────────────────────────────────────
def _bm25_search(index: dict, query: str, top_k: int = TOP_K) -> list[dict]:
    n  = index["n"]
    df = index["df"]
    tf = index["tf"]
    k1, b, avgdl = 1.5, 0.75, 500

    query_terms = _tokenize(query)
    scores = []
    for i, freq in enumerate(tf):
        dl    = sum(freq.values())
        score = 0.0
        for term in query_terms:
            if term not in freq:
                continue
            idf  = math.log((n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
            tf_n = freq[term] * (k1 + 1) / (freq[term] + k1 * (1 - b + b * dl / avgdl))
            score += idf * tf_n
        scores.append((score, i))

    scores.sort(reverse=True)
    return [index["chunks"][i] for _, i in scores[:top_k] if _ > 0]

# ── Pipeline run ──────────────────────────────────────────────────────────────
def run(query: str, api_key: str, collection=None) -> dict:
    gemini = genai.Client(api_key=api_key)

    index = build_index()
    start = time.time()

    hits = _bm25_search(index, query)
    if not hits:
        context = "No relevant context found."
        sources = []
    else:
        context = "\n\n---\n\n".join(h["text"] for h in hits)
        sources  = list({h["source"] for h in hits})

    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {query}

Answer:"""

    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    latency = time.time() - start

    usage = response.usage_metadata
    prompt_tokens    = usage.prompt_token_count    or 0
    completion_tokens = usage.candidates_token_count or 0
    total_tokens     = usage.total_token_count      or 0

    return {
        "pipeline": "Basic RAG",
        "answer": response.text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": round(latency, 2),
        "cost_usd": round(total_tokens * 0.00000015, 6),
        "sources": sources,
        "chunks_retrieved": len(hits),
    }
