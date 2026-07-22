"""Pydantic models / schemas for PromptLab API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────── Prompt ────────────────

class PromptCreate(BaseModel):
    """Request body to create a new prompt."""
    name: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    category: str = Field(default="general", max_length=100)
    tags: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=500)


class PromptUpdate(BaseModel):
    """Request body to update a prompt (creates a new version)."""
    content: str = Field(..., min_length=1)
    description: str = Field(default="", max_length=500)


class PromptOut(BaseModel):
    id: str
    name: str
    content: str
    category: str
    tags: list[str]
    description: str
    current_version_number: int
    created_at: datetime
    updated_at: datetime


class PromptVersionOut(BaseModel):
    id: str
    prompt_id: str
    version_number: int
    content: str
    parent_version: Optional[int] = None
    variables_used: list[str] = Field(default_factory=list)
    description: str
    created_at: datetime


class PromptDiffOut(BaseModel):
    version_a: int
    version_b: int
    diff_lines: list[str]  # unified diff output lines


# ──────────────── LLM Generation ────────────────

class LLMRequest(BaseModel):
    """Request to generate output from one or more LLM backends."""
    prompt_id: Optional[str] = None
    content: str  # prompt text (with variables already substituted)
    models: list[str] = Field(
        default_factory=lambda: ["deepseek-chat"],
        description="LLM backends to query",
    )
    system_prompt: str = Field(
        default="You are a helpful assistant.",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="Variable values keyed by name (already substituted in content)",  # noqa: E501
    )


class LLMResponse(BaseModel):
    model: str
    output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_estimate: float
    error: Optional[str] = None


class MultiLLMResponse(BaseModel):
    prompt_id: Optional[str] = None
    prompt_content: str
    results: list[LLMResponse]
    total_cost: float
    total_latency_ms: float


# ──────────────── A/B Evaluation ────────────────

class ABEvalRequest(BaseModel):
    """Submit ratings for an A/B comparison."""
    prompt_id_a: str
    prompt_id_b: str
    model: str = "deepseek-chat"
    content_a: str
    content_b: str
    output_a: str
    output_b: str
    ratings: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Ratings keyed by label (a/b) -> metric -> score",
    )
    notes: str = Field(default="")


class ABEvalOut(BaseModel):
    id: str
    prompt_id_a: str
    prompt_id_b: str
    model: str
    ratings: dict
    winner: Optional[str] = None
    notes: str
    created_at: datetime


class ScoreSummary(BaseModel):
    prompt_id: str
    avg_relevance: float
    avg_accuracy: float
    avg_creativity: float
    total_evals: int


# ──────────────── History / Analytics ────────────────

class HistoryItem(BaseModel):
    id: str
    prompt_id: str
    prompt_name: str
    model: str
    output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_estimate: float
    variables: dict[str, str]
    created_at: datetime


class CostSummary(BaseModel):
    total_generations: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    avg_latency_ms: float
    by_model: dict[str, dict[str, float]]


class TemplateMeta(BaseModel):
    filename: str
    name: str
    description: str
    category: str
    tags: list[str]
    variable_count: int
