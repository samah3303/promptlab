"""Prompt manager — prompt template library, import/export, diff."""

import json
import difflib
from pathlib import Path
from typing import Optional

from config import PROMPTS_DIR


def load_template_library() -> list[dict]:
    """Load all JSON templates from the prompts/ directory."""
    templates = []
    if not PROMPTS_DIR.exists():
        return templates
    for fpath in sorted(PROMPTS_DIR.glob("*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_filename"] = fpath.name
                templates.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return templates


def get_template(filename: str) -> Optional[dict]:
    """Get a single template by filename."""
    fpath = PROMPTS_DIR / filename
    if not fpath.exists():
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_filename"] = fpath.name
            return data
    except (json.JSONDecodeError, OSError):
        return None


def compute_diff(text_a: str, text_b: str) -> list[str]:
    """Return unified diff between two texts."""
    a_lines = text_a.splitlines(keepends=True)
    b_lines = text_b.splitlines(keepends=True)
    diff = difflib.unified_diff(
        a_lines, b_lines,
        fromfile="version_a",
        tofile="version_b",
        lineterm="",
    )
    return list(diff)


def apply_diff(text: str, patch_lines: list[str]) -> str:
    """Apply a unified diff patch to text. Best-effort using difflib restore."""
    try:
        result = difflib.restore(patch_lines, 1)
        return "".join(result).rstrip("\n")
    except Exception:
        raise ValueError("Failed to apply diff patch")


def export_prompt_json(prompt: dict, versions: list[dict]) -> str:
    """Export a prompt and its version history as JSON string."""
    export_data = {
        "promptlab_export_version": "1.0",
        "prompt": {
            "name": prompt.get("name"),
            "category": prompt.get("category"),
            "tags": prompt.get("tags"),
            "description": prompt.get("description"),
        },
        "versions": [
            {
                "version_number": v["version_number"],
                "content": v["content"],
                "parent_version": v.get("parent_version"),
                "variables_used": v.get("variables_used", []),
                "description": v.get("description", ""),
                "created_at": v.get("created_at"),
            }
            for v in versions
        ],
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def export_prompt_markdown(prompt: dict, versions: list[dict]) -> str:
    """Export a prompt as Markdown string."""
    lines = [
        f"# {prompt.get('name', 'Untitled Prompt')}",
        "",
        f"**Category:** {prompt.get('category', 'N/A')}",
        f"**Tags:** {', '.join(prompt.get('tags', []))}",
        f"**Description:** {prompt.get('description', '')}",
        "",
        "---",
        "",
        "## Current Version (v{})".format(prompt.get("current_version_number", 1)),
        "",
    ]
    if versions:
        current = versions[0]
        lines.append("```")
        lines.append(current["content"])
        lines.append("```")
        lines.append("")
        if current.get("variables_used"):
            lines.append(f"**Variables:** {', '.join(current['variables_used'])}")
            lines.append("")
    return "\n".join(lines)


def parse_import_json(data: str) -> dict:
    """Parse a PromptLab JSON export, validate, and return structured data."""
    try:
        obj = json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object at top level")

    # Support both full export and simple prompt object
    if "promptlab_export_version" in obj:
        prompt_data = obj.get("prompt", {})
        versions = obj.get("versions", [])
    else:
        prompt_data = obj
        versions = [{"version_number": 1, "content": obj.get("prompt", ""), "variables_used": []}]

    return {
        "name": prompt_data.get("name", "Imported Prompt"),
        "category": prompt_data.get("category", "imported"),
        "tags": prompt_data.get("tags", []),
        "description": prompt_data.get("description", ""),
        "content": versions[-1]["content"] if versions else prompt_data.get("prompt", ""),
        "variables_used": versions[-1].get("variables_used", []) if versions else [],
    }


def extract_variables(content: str) -> list[str]:
    """Extract {{variable_name}} references from prompt content."""
    import re
    return list(dict.fromkeys(re.findall(r"\{\{(\w+)\}\}", content)))
