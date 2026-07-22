"""Neon PostgreSQL database layer — async using asyncpg / connection pool."""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

from config import NEON_HOST, NEON_DATABASE, NEON_USER, NEON_PASSWORD, DATABASE_URL


_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Return (and lazily create) the global connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=NEON_HOST,
            database=NEON_DATABASE,
            user=NEON_USER,
            password=NEON_PASSWORD,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ──────────────── Schema Init ────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prompts (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'general',
    tags        TEXT[] NOT NULL DEFAULT '{}',
    description TEXT NOT NULL DEFAULT '',
    current_version_number INTEGER NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id              TEXT PRIMARY KEY,
    prompt_id       TEXT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    version_number  INTEGER NOT NULL,
    content         TEXT NOT NULL,
    parent_version  INTEGER,
    variables_used  TEXT[] NOT NULL DEFAULT '{}',
    description     TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(prompt_id, version_number)
);

CREATE TABLE IF NOT EXISTS generation_history (
    id              TEXT PRIMARY KEY,
    prompt_id       TEXT REFERENCES prompts(id) ON DELETE SET NULL,
    prompt_name     TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL,
    output          TEXT NOT NULL,
    latency_ms      DOUBLE PRECISION NOT NULL DEFAULT 0,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_estimate   DOUBLE PRECISION NOT NULL DEFAULT 0,
    variables       JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ab_evaluations (
    id              TEXT PRIMARY KEY,
    prompt_id_a     TEXT NOT NULL,
    prompt_id_b     TEXT NOT NULL,
    model           TEXT NOT NULL,
    content_a       TEXT NOT NULL,
    content_b       TEXT NOT NULL,
    output_a        TEXT NOT NULL,
    output_b        TEXT NOT NULL,
    ratings         JSONB NOT NULL DEFAULT '{}',
    winner          TEXT,
    notes           TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_prompt_versions_prompt_id ON prompt_versions(prompt_id);
CREATE INDEX IF NOT EXISTS idx_generation_history_prompt_id ON generation_history(prompt_id);
CREATE INDEX IF NOT EXISTS idx_generation_history_created_at ON generation_history(created_at);
CREATE INDEX IF NOT EXISTS idx_ab_evaluations_prompt_a ON ab_evaluations(prompt_id_a);
CREATE INDEX IF NOT EXISTS idx_ab_evaluations_prompt_b ON ab_evaluations(prompt_id_b);
"""


async def init_db() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)


# ──────────────── Prompt CRUD ────────────────

async def create_prompt(
    name: str, content: str, category: str, tags: list[str], description: str
) -> dict[str, Any]:
    pool = await get_pool()
    prompt_id = str(uuid.uuid4())
    version_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """INSERT INTO prompts (id, name, category, tags, description,
                   current_version_number, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, 1, $6, $6)""",
                prompt_id, name, category, tags, description, now,
            )
            variables_used = _extract_variables(content)
            await conn.execute(
                """INSERT INTO prompt_versions (id, prompt_id, version_number,
                   content, parent_version, variables_used, description, created_at)
                   VALUES ($1, $2, 1, $3, NULL, $4, '', $5)""",
                version_id, prompt_id, content, variables_used, now,
            )
    return {
        "id": prompt_id,
        "name": name,
        "content": content,
        "category": category,
        "tags": tags,
        "description": description,
        "current_version_number": 1,
        "created_at": now,
        "updated_at": now,
    }


async def get_prompt(prompt_id: str) -> Optional[dict[str, Any]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM prompts WHERE id = $1", prompt_id,
        )
        if row is None:
            return None
        return _row_to_dict(row)


async def list_prompts(
    offset: int = 0, limit: int = 20, category: Optional[str] = None,
) -> list[dict[str, Any]]:
    pool = await get_pool()
    if category:
        rows = await pool.fetch(
            "SELECT * FROM prompts WHERE category = $1 ORDER BY updated_at DESC "
            "OFFSET $2 LIMIT $3",
            category, offset, limit,
        )
    else:
        rows = await pool.fetch(
            "SELECT * FROM prompts ORDER BY updated_at DESC OFFSET $1 LIMIT $2",
            offset, limit,
        )
    return [_row_to_dict(r) for r in rows]


async def update_prompt(prompt_id: str, content: str, description: str) -> dict[str, Any]:
    """Create a new version, bump version number, update prompt record."""
    pool = await get_pool()
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        async with conn.transaction():
            prompt = await conn.fetchrow(
                "SELECT * FROM prompts WHERE id = $1 FOR UPDATE", prompt_id,
            )
            if prompt is None:
                raise ValueError(f"Prompt {prompt_id} not found")

            new_version = prompt["current_version_number"] + 1
            version_id = str(uuid.uuid4())
            variables_used = _extract_variables(content)
            await conn.execute(
                """INSERT INTO prompt_versions (id, prompt_id, version_number,
                   content, parent_version, variables_used, description, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                version_id, prompt_id, new_version, content,
                prompt["current_version_number"], variables_used, description, now,
            )
            await conn.execute(
                """UPDATE prompts SET current_version_number = $1,
                   updated_at = $2 WHERE id = $3""",
                new_version, now, prompt_id,
            )
    return await get_prompt(prompt_id)


async def delete_prompt(prompt_id: str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM prompts WHERE id = $1", prompt_id,
        )
        return result == "DELETE 1"


async def get_prompt_versions(prompt_id: str) -> list[dict[str, Any]]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM prompt_versions WHERE prompt_id = $1 ORDER BY version_number DESC",
        prompt_id,
    )
    return [_row_to_dict(r) for r in rows]


async def get_prompt_version(prompt_id: str, version_number: int) -> Optional[dict[str, Any]]:
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM prompt_versions WHERE prompt_id = $1 AND version_number = $2",
        prompt_id, version_number,
    )
    return _row_to_dict(row) if row else None


async def rollback_prompt(prompt_id: str, version_number: int) -> dict[str, Any]:
    """Restore prompt content to a previous version (creates a new version)."""
    version = await get_prompt_version(prompt_id, version_number)
    if version is None:
        raise ValueError(f"Version {version_number} not found for prompt {prompt_id}")
    return await update_prompt(prompt_id, version["content"], f"Rollback to v{version_number}")


# ──────────────── Generation History ────────────────

async def save_generation(
    prompt_id: Optional[str],
    prompt_name: str,
    model: str,
    output: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    cost_estimate: float,
    variables: dict[str, str],
) -> dict[str, Any]:
    pool = await get_pool()
    gen_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO generation_history
               (id, prompt_id, prompt_name, model, output, latency_ms,
                input_tokens, output_tokens, cost_estimate, variables, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
            gen_id, prompt_id, prompt_name, model, output, latency_ms,
            input_tokens, output_tokens, cost_estimate,
            _to_jsonb(variables), now,
        )
    return {
        "id": gen_id, "prompt_id": prompt_id, "prompt_name": prompt_name,
        "model": model, "output": output, "latency_ms": latency_ms,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "cost_estimate": cost_estimate, "variables": variables, "created_at": now,
    }


async def list_history(
    prompt_id: Optional[str] = None,
    model: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
) -> list[dict[str, Any]]:
    pool = await get_pool()
    query = "SELECT * FROM generation_history WHERE 1=1"
    params: list[Any] = []
    idx = 1
    if prompt_id:
        query += f" AND prompt_id = ${idx}"; params.append(prompt_id); idx += 1
    if model:
        query += f" AND model = ${idx}"; params.append(model); idx += 1
    query += f" ORDER BY created_at DESC OFFSET ${idx} LIMIT ${idx + 1}"
    params.extend([offset, limit])
    rows = await pool.fetch(query, *params)
    return [_row_to_dict(r) for r in rows]


async def cost_summary() -> dict[str, Any]:
    pool = await get_pool()
    row = await pool.fetchrow("""
        SELECT
            COUNT(*) as total_generations,
            COALESCE(SUM(input_tokens), 0) as total_input_tokens,
            COALESCE(SUM(output_tokens), 0) as total_output_tokens,
            COALESCE(SUM(cost_estimate), 0) as total_cost,
            COALESCE(AVG(latency_ms), 0) as avg_latency_ms
        FROM generation_history
    """)
    by_model_rows = await pool.fetch("""
        SELECT model, COUNT(*) as count,
               COALESCE(SUM(input_tokens),0) as input_tokens,
               COALESCE(SUM(output_tokens),0) as output_tokens,
               COALESCE(SUM(cost_estimate),0) as total_cost
        FROM generation_history
        GROUP BY model
    """)
    by_model = {
        r["model"]: {
            "count": r["count"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "total_cost": float(r["total_cost"]),
        }
        for r in by_model_rows
    }
    return {
        "total_generations": row["total_generations"],
        "total_input_tokens": row["total_input_tokens"],
        "total_output_tokens": row["total_output_tokens"],
        "total_cost": float(row["total_cost"]),
        "avg_latency_ms": float(row["avg_latency_ms"]),
        "by_model": by_model,
    }


# ──────────────── A/B Evaluations ────────────────

async def save_ab_evaluation(
    prompt_id_a: str,
    prompt_id_b: str,
    model: str,
    content_a: str,
    content_b: str,
    output_a: str,
    output_b: str,
    ratings: dict,
    notes: str,
) -> dict[str, Any]:
    pool = await get_pool()
    eval_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    # Determine winner by comparing avg scores
    winner = None
    if ratings:
        def _avg(d: dict) -> float:
            return sum(d.values()) / len(d) if d else 0
        score_a = _avg(ratings.get("a", {}))
        score_b = _avg(ratings.get("b", {}))
        if score_a > score_b:
            winner = "a"
        elif score_b > score_a:
            winner = "b"

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO ab_evaluations
               (id, prompt_id_a, prompt_id_b, model, content_a, content_b,
                output_a, output_b, ratings, winner, notes, created_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)""",
            eval_id, prompt_id_a, prompt_id_b, model,
            content_a, content_b, output_a, output_b,
            _to_jsonb(ratings), winner, notes, now,
        )
    return {
        "id": eval_id, "prompt_id_a": prompt_id_a, "prompt_id_b": prompt_id_b,
        "model": model, "ratings": ratings, "winner": winner,
        "notes": notes, "created_at": now,
    }


async def list_ab_evaluations(
    prompt_id: Optional[str] = None, limit: int = 20,
) -> list[dict[str, Any]]:
    pool = await get_pool()
    if prompt_id:
        rows = await pool.fetch(
            """SELECT * FROM ab_evaluations
               WHERE prompt_id_a = $1 OR prompt_id_b = $1
               ORDER BY created_at DESC LIMIT $2""",
            prompt_id, limit,
        )
    else:
        rows = await pool.fetch(
            "SELECT * FROM ab_evaluations ORDER BY created_at DESC LIMIT $1", limit,
        )
    return [_row_to_dict(r) for r in rows]


async def score_summary_for_prompt(prompt_id: str) -> dict[str, Any]:
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT ratings, winner FROM ab_evaluations
           WHERE prompt_id_a = $1 OR prompt_id_b = $1""",
        prompt_id,
    )
    if not rows:
        return {
            "prompt_id": prompt_id, "avg_relevance": 0,
            "avg_accuracy": 0, "avg_creativity": 0, "total_evals": 0,
        }
    rel, acc, cre = [], [], []
    for r in rows:
        ratings = r["ratings"]
        if isinstance(ratings, str):
            import json
            ratings = json.loads(ratings)
        for side in ("a", "b"):
            metrics = ratings.get(side, {})
            if "relevance" in metrics:
                rel.append(metrics["relevance"])
            if "accuracy" in metrics:
                acc.append(metrics["accuracy"])
            if "creativity" in metrics:
                cre.append(metrics["creativity"])
    return {
        "prompt_id": prompt_id,
        "avg_relevance": sum(rel) / len(rel) if rel else 0,
        "avg_accuracy": sum(acc) / len(acc) if acc else 0,
        "avg_creativity": sum(cre) / len(cre) if cre else 0,
        "total_evals": len(rows),
    }


# ──────────────── Helpers ────────────────

def _extract_variables(content: str) -> list[str]:
    import re
    return list(dict.fromkeys(re.findall(r"\{\{(\w+)\}\}", content)))


def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def _to_jsonb(obj: Any) -> str:
    import json
    return json.dumps(obj, default=str)
