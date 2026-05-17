"""Pipeline 1: LLM-Only — direct Gemini call, no retrieval."""

import time
from google import genai
from google.genai import types

def run(query: str, api_key: str) -> dict:
    client = genai.Client(api_key=api_key)
    prompt = f"Answer the following question as accurately as possible:\n\n{query}"

    start = time.time()
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
        "pipeline": "LLM-Only",
        "answer": response.text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": round(latency, 2),
        "cost_usd": round(cost, 6),
    }
