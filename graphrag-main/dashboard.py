"""GraphRAG Inference Hackathon — Comparison Dashboard"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

from config import GOOGLE_API_KEY
from pipelines import pipeline1_llm_only, pipeline2_basic_rag, pipeline3_graphrag

st.set_page_config(
    page_title="GraphRAG vs Basic RAG vs LLM-Only",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 GraphRAG Inference Hackathon")
st.markdown("**One query. Three pipelines. Side-by-side metrics.**")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    st.subheader("1. Build ChromaDB Index")
    st.caption("Run once before querying Pipeline 2.")
    if st.button("Build RAG Index", type="primary"):
        with st.spinner("Chunking & indexing dataset..."):
            try:
                collection = pipeline2_basic_rag.build_index(GOOGLE_API_KEY)
                st.success(f"Index built!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.subheader("2. Ingest into TigerGraph")
    st.caption("Run once before querying Pipeline 3.")
    max_articles = st.slider("Articles to ingest", 5, 30, 10)
    if st.button("Ingest into GraphRAG", type="primary"):
        with st.spinner("Extracting entities & loading graph..."):
            try:
                pipeline3_graphrag.ingest(GOOGLE_API_KEY, max_articles=max_articles)
                st.success("Graph ingested!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.caption("Dataset: 1M tokens | 164 Wikipedia AI/ML articles")

# ── Main Query Area ───────────────────────────────────────────────────────────
query = st.text_input(
    "Enter your question:",
    placeholder="e.g. How does backpropagation work in neural networks?",
    key="query_input"
)

run_col1, run_col2, run_col3 = st.columns(3)
run_p1 = run_col1.checkbox("Pipeline 1: LLM-Only", value=True)
run_p2 = run_col2.checkbox("Pipeline 2: Basic RAG", value=True)
run_p3 = run_col3.checkbox("Pipeline 3: GraphRAG", value=True)

if st.button("🚀 Run All Pipelines", type="primary", disabled=not query):
    results = {}

    if run_p1:
        with st.spinner("Running Pipeline 1: LLM-Only..."):
            results["p1"] = pipeline1_llm_only.run(query, GOOGLE_API_KEY)

    if run_p2:
        with st.spinner("Running Pipeline 2: Basic RAG..."):
            try:
                results["p2"] = pipeline2_basic_rag.run(query, GOOGLE_API_KEY)
            except Exception as e:
                results["p2"] = {"error": str(e), "pipeline": "Basic RAG"}

    if run_p3:
        with st.spinner("Running Pipeline 3: GraphRAG..."):
            try:
                results["p3"] = pipeline3_graphrag.run(query, GOOGLE_API_KEY)
            except Exception as e:
                results["p3"] = {"error": str(e), "pipeline": "GraphRAG"}

    st.divider()
    st.subheader("📊 Results")

    # ── Answers side by side ─────────────────────────────────────────────────
    cols = st.columns(len(results))
    for col, (key, res) in zip(cols, results.items()):
        with col:
            st.markdown(f"### {res['pipeline']}")
            if "error" in res:
                st.error(res["error"])
            else:
                st.markdown(res.get("answer", ""))

    st.divider()

    # ── Metrics Table ─────────────────────────────────────────────────────────
    st.subheader("📈 Metrics Comparison")
    metric_cols = st.columns(len(results))

    for col, (key, res) in zip(metric_cols, results.items()):
        with col:
            if "error" not in res:
                st.metric("Total Tokens", f"{res.get('total_tokens', 0):,}")
                st.metric("Latency", f"{res.get('latency_seconds', 0)}s")
                st.metric("Cost/Query", f"${res.get('cost_usd', 0):.6f}")
                if "chunks_retrieved" in res:
                    st.metric("Chunks Retrieved", res["chunks_retrieved"])
                if "context_facts" in res:
                    st.metric("Graph Facts Used", res["context_facts"])

    # ── Token reduction summary ───────────────────────────────────────────────
    if "p1" in results and "p3" in results:
        t1 = results["p1"].get("total_tokens", 1)
        t3 = results["p3"].get("total_tokens", 1)
        if t1 and t3:
            reduction = round((1 - t3 / t1) * 100, 1)
            st.divider()
            if reduction > 0:
                st.success(f"🎯 GraphRAG used **{reduction}% fewer tokens** than LLM-Only")
            else:
                st.info(f"GraphRAG used {abs(reduction)}% more tokens than LLM-Only for this query")

    if "p2" in results and "p3" in results:
        t2 = results["p2"].get("total_tokens", 1)
        t3 = results["p3"].get("total_tokens", 1)
        if t2 and t3:
            reduction = round((1 - t3 / t2) * 100, 1)
            if reduction > 0:
                st.success(f"🎯 GraphRAG used **{reduction}% fewer tokens** than Basic RAG")
