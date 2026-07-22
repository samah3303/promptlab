"""LLM service — calls DeepSeek API and optional Ollama."""

import time
import json
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_PRICING,
    OLLAMA_BASE_URL,
    OLLAMA_ENABLED,
    DEFAULT_INPUT_PRICE,
    DEFAULT_OUTPUT_PRICE,
)

# ──────────────── DeepSeek ────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_deepseek(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Call DeepSeek chat-completions endpoint and return output + metrics."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(DEEPSEEK_BASE_URL, headers=headers, json=payload)
    latency_ms = (time.perf_counter() - start) * 1000

    if resp.status_code != 200:
        error_detail = resp.text[:500]
        raise RuntimeError(f"DeepSeek API error {resp.status_code}: {error_detail}")

    data = resp.json()
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cost = _estimate_cost(model, input_tokens, output_tokens)
    content = data["choices"][0]["message"]["content"]

    return {
        "model": model,
        "output": content,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_estimate": round(cost, 6),
    }


# ──────────────── Ollama ────────────────

async def list_ollama_models() -> list[str]:
    """Return list of available local Ollama models."""
    if not OLLAMA_ENABLED:
        return []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


async def call_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Call local Ollama model. Raises if Ollama is not available."""
    combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    payload = {
        "model": model,
        "prompt": combined,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
    latency_ms = (time.perf_counter() - start) * 1000

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    response_text = data.get("response", "")
    # Ollama doesn't give exact token counts in generate; estimate ~4 chars/token
    input_chars = len(system_prompt or "") + len(user_prompt or "")
    output_chars = len(response_text)
    input_tokens = max(1, input_chars // 4)
    output_tokens = max(1, output_chars // 4)
    cost = 0.0  # local models are free

    return {
        "model": f"ollama/{model}",
        "output": response_text,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_estimate": cost,
    }


# ──────────────── Multi-backend dispatch ────────────────

async def generate_multi(
    models: list[str],
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> list[dict[str, Any]]:
    """Send the same prompt to multiple LLM backends in parallel."""
    import asyncio

    async def _call_one(model: str) -> dict[str, Any]:
        try:
            if model.startswith("ollama/"):
                ollama_model = model.replace("ollama/", "")
                return await call_ollama(ollama_model, system_prompt, user_prompt, temperature, max_tokens)
            else:
                return await call_deepseek(model, system_prompt, user_prompt, temperature, max_tokens)
        except Exception as e:
            return {
                "model": model,
                "output": "",
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_estimate": 0,
                "error": str(e),
            }

    tasks = [_call_one(m) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            out.append({
                "model": models[i],
                "output": "",
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_estimate": 0,
                "error": str(r),
            })
        else:
            out.append(r)
    return out


# ──────────────── Helpers ────────────────

def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = DEEPSEEK_PRICING.get(model, {"input": DEFAULT_INPUT_PRICE, "output": DEFAULT_OUTPUT_PRICE})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def available_backends() -> list[dict[str, Any]]:
    """Return metadata about available LLM backends."""
    backends = []
    if DEEPSEEK_API_KEY:
        backends.append({
            "id": "deepseek-chat",
            "name": "DeepSeek Chat (V3)",
            "type": "cloud",
            "pricing": {"input": 0.14, "output": 0.28},
        })
        backends.append({
            "id": "deepseek-reasoner",
            "name": "DeepSeek Reasoner (R1)",
            "type": "cloud",
            "pricing": {"input": 0.55, "output": 2.19},
        })
    # Ollama availability checked at runtime
    backends.append({
        "id": "ollama",
        "name": "Ollama (Local)",
        "type": "local",
        "pricing": {"input": 0, "output": 0},
    })
    return backends
