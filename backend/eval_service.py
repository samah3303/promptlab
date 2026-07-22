"""A/B evaluation service — coordinates LLM calls and evaluation storage."""

from datetime import datetime, timezone
from typing import Any, Optional

from database import save_ab_evaluation, list_ab_evaluations, score_summary_for_prompt
from llm_service import call_deepseek
from models import ABEvalRequest


async def run_ab_evaluation(
    prompt_id_a: str,
    prompt_id_b: str,
    content_a: str,
    content_b: str,
    model: str = "deepseek-chat",
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Generate outputs for both prompts and return comparison data."""

    import asyncio

    async def _gen(content: str) -> dict:
        try:
            return await call_deepseek(model, system_prompt, content, temperature, max_tokens)
        except Exception as e:
            return {"output": "", "latency_ms": 0, "input_tokens": 0, "output_tokens": 0, "cost_estimate": 0, "error": str(e)}

    result_a, result_b = await asyncio.gather(_gen(content_a), _gen(content_b))

    return {
        "prompt_id_a": prompt_id_a,
        "prompt_id_b": prompt_id_b,
        "model": model,
        "output_a": result_a,
        "output_b": result_b,
        "content_a": content_a,
        "content_b": content_b,
    }


async def submit_evaluation(data: ABEvalRequest) -> dict[str, Any]:
    """Save A/B evaluation results."""
    return await save_ab_evaluation(
        prompt_id_a=data.prompt_id_a,
        prompt_id_b=data.prompt_id_b,
        model=data.model,
        content_a=data.content_a,
        content_b=data.content_b,
        output_a=data.output_a,
        output_b=data.output_b,
        ratings=data.ratings,
        notes=data.notes,
    )


async def get_evaluations(
    prompt_id: Optional[str] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    return await list_ab_evaluations(prompt_id=prompt_id, limit=limit)


async def get_score_summary(prompt_id: str) -> dict[str, Any]:
    return await score_summary_for_prompt(prompt_id)
