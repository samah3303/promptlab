"""PromptLab — FastAPI backend for Prompt Engineering Workbench."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ORIGINS, HOST, PORT
from database import init_db, close_pool
from database import (
    create_prompt, get_prompt, list_prompts, update_prompt, delete_prompt,
    get_prompt_versions, get_prompt_version, rollback_prompt,
    save_generation, list_history, cost_summary,
)
from eval_service import run_ab_evaluation, submit_evaluation, get_evaluations, get_score_summary
from llm_service import generate_multi, list_ollama_models, available_backends
from prompt_manager import (
    load_template_library, get_template, compute_diff,
    export_prompt_json, export_prompt_markdown, parse_import_json,
)
from models import (
    PromptCreate, PromptUpdate, PromptOut, PromptVersionOut, PromptDiffOut,
    LLMRequest, MultiLLMResponse, LLMResponse,
    ABEvalRequest, ABEvalOut, ScoreSummary,
    HistoryItem, CostSummary, TemplateMeta,
)

logger = logging.getLogger("promptlab")
logging.basicConfig(level=logging.INFO)


# ──────────────── App Lifecycle ────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    try:
        await init_db()
        logger.info("Database ready.")
    except Exception as e:
        logger.warning(f"DB init skipped (will retry on-demand): {e}")
    yield
    await close_pool()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="PromptLab API",
    description="Prompt Engineering Workbench — multi-LLM comparison, A/B evaluation, versioning",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────── Health ────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "PromptLab"}


# ──────────────── Backends ────────────────

@app.get("/api/backends")
async def get_backends():
    """List available LLM backends."""
    backends = available_backends()
    ollama_models = await list_ollama_models()
    for b in backends:
        if b["id"] == "ollama":
            b["models"] = ollama_models
            b["available"] = len(ollama_models) > 0
    return backends


# ──────────────── Templates ────────────────

@app.get("/api/templates", response_model=list[TemplateMeta])
async def list_templates():
    templates = load_template_library()
    return [
        TemplateMeta(
            filename=t["_filename"],
            name=t.get("name", "Untitled"),
            description=t.get("description", ""),
            category=t.get("category", "general"),
            tags=t.get("tags", []),
            variable_count=len(t.get("variables", [])),
        )
        for t in templates
    ]


@app.get("/api/templates/{filename}")
async def get_template_detail(filename: str):
    t = get_template(filename)
    if t is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return t


# ──────────────── Prompts CRUD ────────────────

@app.post("/api/prompts", response_model=PromptOut, status_code=201)
async def api_create_prompt(body: PromptCreate):
    try:
        prompt = await create_prompt(
            name=body.name,
            content=body.content,
            category=body.category,
            tags=body.tags,
            description=body.description,
        )
        return PromptOut(**prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prompts", response_model=list[PromptOut])
async def api_list_prompts(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
):
    prompts = await list_prompts(offset=offset, limit=limit, category=category)
    return [PromptOut(**p) for p in prompts]


@app.get("/api/prompts/{prompt_id}", response_model=PromptOut)
async def api_get_prompt(prompt_id: str):
    prompt = await get_prompt(prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return PromptOut(**prompt)


@app.put("/api/prompts/{prompt_id}", response_model=PromptOut)
async def api_update_prompt(prompt_id: str, body: PromptUpdate):
    try:
        prompt = await update_prompt(prompt_id, body.content, body.description)
        return PromptOut(**prompt)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/prompts/{prompt_id}", status_code=204)
async def api_delete_prompt(prompt_id: str):
    deleted = await delete_prompt(prompt_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prompt not found")


# ──────────────── Prompt Versions ────────────────

@app.get("/api/prompts/{prompt_id}/versions", response_model=list[PromptVersionOut])
async def api_get_versions(prompt_id: str):
    versions = await get_prompt_versions(prompt_id)
    if not versions:
        raise HTTPException(status_code=404, detail="Prompt not found or no versions")
    return [PromptVersionOut(**v) for v in versions]


@app.get("/api/prompts/{prompt_id}/versions/{version_number}", response_model=PromptVersionOut)
async def api_get_version(prompt_id: str, version_number: int):
    v = await get_prompt_version(prompt_id, version_number)
    if v is None:
        raise HTTPException(status_code=404, detail="Version not found")
    return PromptVersionOut(**v)


@app.get("/api/prompts/{prompt_id}/diff")
async def api_diff_versions(
    prompt_id: str,
    version_a: int = Query(...),
    version_b: int = Query(...),
):
    va = await get_prompt_version(prompt_id, version_a)
    vb = await get_prompt_version(prompt_id, version_b)
    if va is None or vb is None:
        raise HTTPException(status_code=404, detail="One or both versions not found")
    diff_lines = compute_diff(va["content"], vb["content"])
    return PromptDiffOut(version_a=version_a, version_b=version_b, diff_lines=diff_lines)


@app.post("/api/prompts/{prompt_id}/rollback/{version_number}", response_model=PromptOut)
async def api_rollback(prompt_id: str, version_number: int):
    try:
        prompt = await rollback_prompt(prompt_id, version_number)
        return PromptOut(**prompt)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ──────────────── LLM Generation ────────────────

@app.post("/api/generate", response_model=MultiLLMResponse)
async def api_generate(body: LLMRequest):
    results = await generate_multi(
        models=body.models,
        system_prompt=body.system_prompt,
        user_prompt=body.content,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )

    total_cost = sum(r.get("cost_estimate", 0) for r in results)
    total_latency = sum(r.get("latency_ms", 0) for r in results)

    # Save to history for each result
    for r in results:
        try:
            await save_generation(
                prompt_id=body.prompt_id,
                prompt_name="",
                model=r["model"],
                output=r["output"],
                latency_ms=r["latency_ms"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                cost_estimate=r["cost_estimate"],
                variables=body.variables or {},
            )
        except Exception:
            pass  # non-critical: history save failure shouldn't block response

    return MultiLLMResponse(
        prompt_id=body.prompt_id,
        prompt_content=body.content,
        results=[LLMResponse(**r) for r in results],
        total_cost=round(total_cost, 6),
        total_latency_ms=round(total_latency, 2),
    )


# ──────────────── A/B Evaluation ────────────────

@app.post("/api/eval/compare")
async def api_compare_ab(body: ABEvalRequest):
    """Run a live A/B comparison between two prompts."""
    result = await run_ab_evaluation(
        prompt_id_a=body.prompt_id_a,
        prompt_id_b=body.prompt_id_b,
        content_a=body.content_a,
        content_b=body.content_b,
        model=body.model,
    )
    return result


@app.post("/api/eval/submit", response_model=ABEvalOut, status_code=201)
async def api_submit_eval(body: ABEvalRequest):
    """Submit ratings for an A/B comparison."""
    saved = await submit_evaluation(body)
    return ABEvalOut(**saved)


@app.get("/api/eval/history", response_model=list[ABEvalOut])
async def api_list_evals(
    prompt_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
):
    evals = await get_evaluations(prompt_id=prompt_id, limit=limit)
    return [ABEvalOut(**e) for e in evals]


@app.get("/api/eval/scores/{prompt_id}", response_model=ScoreSummary)
async def api_score_summary(prompt_id: str):
    return ScoreSummary(**await get_score_summary(prompt_id))


# ──────────────── History / Analytics ────────────────

@app.get("/api/history", response_model=list[HistoryItem])
async def api_history(
    prompt_id: Optional[str] = None,
    model: Optional[str] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    history = await list_history(
        prompt_id=prompt_id, model=model, offset=offset, limit=limit,
    )
    return [HistoryItem(**h) for h in history]


@app.get("/api/analytics/cost", response_model=CostSummary)
async def api_cost_analytics():
    return CostSummary(**await cost_summary())


# ──────────────── Export / Import ────────────────

@app.get("/api/prompts/{prompt_id}/export")
async def api_export_prompt(prompt_id: str, format: str = Query("json", regex="^(json|markdown)$")):
    prompt = await get_prompt(prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    versions = await get_prompt_versions(prompt_id)
    if format == "markdown":
        return {"format": "markdown", "content": export_prompt_markdown(prompt, versions)}
    return {"format": "json", "content": export_prompt_json(prompt, versions)}


@app.post("/api/prompts/import", response_model=PromptOut, status_code=201)
async def api_import_prompt(body: dict):
    """Import a prompt from JSON export format."""
    if "content" not in body:
        raise HTTPException(status_code=400, detail="Missing 'content' field")
    try:
        parsed = parse_import_json(body["content"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    prompt = await create_prompt(
        name=parsed["name"],
        content=parsed["content"],
        category=parsed["category"],
        tags=parsed["tags"],
        description=parsed["description"],
    )
    return PromptOut(**prompt)


# ──────────────── Entry point ────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
