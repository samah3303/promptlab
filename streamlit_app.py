"""PromptLab — Streamlit Cloud deployment.
Merges the FastAPI frontend directly with backend modules (no HTTP calls).
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

import streamlit as st

# Ensure backend/ is importable
sys.path.insert(0, str(Path(__file__).parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

# ── Secrets / env fallback ───────────────────────────────────────────────
def _secret(key: str, default: str = "") -> str:
    """Try st.secrets first, then os.environ, then default."""
    import os
    try:
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)

# Inject config-level env vars before importing backend modules
os.environ.setdefault("NEON_HOST", _secret("NEON_HOST", "localhost"))
os.environ.setdefault("NEON_DATABASE", _secret("NEON_DATABASE", "promptlab"))
os.environ.setdefault("NEON_USER", _secret("NEON_USER", "user"))
os.environ.setdefault("NEON_PASSWORD", _secret("NEON_PASSWORD", "password"))
os.environ.setdefault("DATABASE_URL", _secret("DATABASE_URL", ""))
os.environ.setdefault("DEEPSEEK_API_KEY", _secret("DEEPSEEK_API_KEY", ""))
os.environ.setdefault("OLLAMA_BASE_URL", _secret("OLLAMA_BASE_URL", "http://localhost:11434"))

import os as _os  # noqa: E402

# ── Backend imports ──────────────────────────────────────────────────────
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_PRICING, OLLAMA_BASE_URL, OLLAMA_ENABLED,
    PROMPTS_DIR, PROJECT_ROOT,
)
from llm_service import (
    call_deepseek, call_ollama, generate_multi, available_backends, list_ollama_models,
)
from prompt_manager import (
    load_template_library, get_template, compute_diff,
    export_prompt_json, export_prompt_markdown, parse_import_json, extract_variables,
)
from eval_service import run_ab_evaluation, submit_evaluation, get_evaluations, get_score_summary
from models import ABEvalRequest

# ── Async helpers for database (asyncpg) ─────────────────────────────────
def _run_async(coro):
    """Run an async coroutine synchronously (Streamlit is sync)."""
    try:
        loop = asyncio.get_running_loop()
        # If there's already a running loop (unlikely in Streamlit but be safe)
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

_db_initialised = False

def _ensure_db():
    global _db_initialised
    if _db_initialised:
        return
    from database import get_pool, SCHEMA_SQL
    async def _init():
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
    _run_async(_init())
    _db_initialised = True

# ── Direct DB wrappers (replacing HTTP API calls) ────────────────────────
def db_list_prompts():
    _ensure_db()
    from database import list_prompts as _lp
    return _run_async(_lp())

def db_get_prompt(prompt_id: str):
    _ensure_db()
    from database import get_prompt as _gp
    return _run_async(_gp(prompt_id))

def db_create_prompt(name: str, content: str, category: str = "general", tags: list = None, description: str = ""):
    _ensure_db()
    from database import create_prompt as _cp
    return _run_async(_cp(name, content, category, tags or [], description))

def db_update_prompt(prompt_id: str, content: str, description: str = ""):
    _ensure_db()
    from database import update_prompt as _up
    return _run_async(_up(prompt_id, content, description))

def db_delete_prompt(prompt_id: str):
    _ensure_db()
    from database import delete_prompt as _dp
    return _run_async(_dp(prompt_id))

def db_get_versions(prompt_id: str):
    _ensure_db()
    from database import get_prompt_versions as _gpv
    return _run_async(_gpv(prompt_id))

def db_rollback_prompt(prompt_id: str, version_number: int):
    _ensure_db()
    from database import rollback_prompt as _rb
    return _run_async(_rb(prompt_id, version_number))

def db_get_version_diff(prompt_id: str, version_a: int, version_b: int):
    _ensure_db()
    from database import get_diff as _gd
    return _run_async(_gd(prompt_id, version_a, version_b))

def db_list_history(limit: int = 100, offset: int = 0, prompt_id: str = None, model: str = None):
    _ensure_db()
    from database import list_history as _lh
    return _run_async(_lh(limit=limit, offset=offset, prompt_id=prompt_id, model=model))

def db_get_cost_summary():
    _ensure_db()
    from database import cost_summary as _cs
    return _run_async(_cs())

def db_list_ab_evals(prompt_id: str = None, limit: int = 50):
    _ensure_db()
    from database import list_ab_evaluations as _lae
    return _run_async(_lae(prompt_id=prompt_id, limit=limit))

def db_score_summary(prompt_id: str):
    _ensure_db()
    from database import score_summary_for_prompt as _ss
    return _run_async(_ss(prompt_id))

def db_save_ab_eval(prompt_id_a, prompt_id_b, model, content_a, content_b, output_a, output_b, ratings, notes=""):
    _ensure_db()
    from database import save_ab_evaluation as _sae
    return _run_async(_sae(prompt_id_a, prompt_id_b, model, content_a, content_b, output_a, output_b, ratings, notes))

# ── Helpers ──────────────────────────────────────────────────────────────
def highlight_variables(content: str) -> str:
    return re.sub(
        r"(\{\{(\w+)\}\})",
        r'<span style="background:#1a73e8;color:#fff;padding:2px 6px;border-radius:4px;font-family:monospace">\1</span>',
        content,
    )

def fill_variables(content: str, vars_dict: dict) -> str:
    filled = content
    for k, v in vars_dict.items():
        filled = filled.replace("{{" + k + "}}", v)
    return filled

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PromptLab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 PromptLab")
    st.caption("Prompt Engineering Workbench")

    if DEEPSEEK_API_KEY:
        st.success("DeepSeek API: Configured")
    else:
        st.warning("DeepSeek API: Not configured — set DEEPSEEK_API_KEY in secrets")

    st.divider()
    tab = st.radio(
        "Navigation",
        ["📝 Editor", "⚖️ Compare (A/B)", "📚 Library", "📜 History", "📊 Analytics"],
        index=0,
    )

# ── Session state init ───────────────────────────────────────────────────
for key, default in [
    ("editor_content", ""), ("editor_name", ""), ("editor_prompt_id", None),
    ("editor_vars", {}), ("generate_results", None), ("ab_result", None),
    ("ab_ratings_submitted", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================================================
# TAB: Editor
# ==========================================================================
if tab == "📝 Editor":
    st.title("📝 Prompt Editor")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        prompts = db_list_prompts() or []
        prompt_options = {"Create New...": None}
        for p in prompts:
            prompt_options[f"{p['name']} ({p['category']})"] = p["id"]

        selected = st.selectbox("Select a prompt", list(prompt_options.keys()), key="editor_select")
        prompt_id = prompt_options.get(selected)

        if prompt_id and prompt_id != st.session_state.editor_prompt_id:
            detail = db_get_prompt(prompt_id)
            if detail:
                versions = db_get_versions(prompt_id) or []
                current = versions[0] if versions else {"content": detail.get("content", "")}
                st.session_state.editor_content = current["content"]
                st.session_state.editor_name = detail["name"]
                st.session_state.editor_prompt_id = prompt_id
                st.session_state.editor_vars = {}
                st.session_state.generate_results = None

        prompt_name = st.text_input("Prompt Name", value=st.session_state.editor_name, key="editor_name_input")
        category = st.selectbox("Category", [
            "general", "summarization", "code_generation", "creative_writing",
            "classification", "rag_qa", "chain_of_thought", "few_shot", "other"
        ], key="editor_category")
        tags_input = st.text_input("Tags (comma-separated)", key="editor_tags")

        vars_list = extract_variables(st.session_state.editor_content)
        if vars_list:
            st.markdown("**Variables detected:** " + " ".join([f"`{{{{{v}}}}}`" for v in vars_list]))

        st.markdown("### Prompt Content")
        new_content = st.text_area(
            "Edit your prompt", value=st.session_state.editor_content, height=250,
            key="editor_textarea",
            placeholder="Write your prompt here. Use {{variable_name}} for dynamic variables.",
        )

        if new_content:
            highlighted = highlight_variables(new_content)
            st.markdown(
                f'<div style="background:#1e1e1e;color:#d4d4d4;padding:16px;border-radius:8px;'
                f'font-family:monospace;white-space:pre-wrap;border:1px solid #333">{highlighted}</div>',
                unsafe_allow_html=True,
            )

        col_save, col_delete = st.columns(2)
        with col_save:
            if st.button("💾 Save Prompt", use_container_width=True):
                if not new_content.strip():
                    st.warning("Prompt content cannot be empty.")
                elif not prompt_name.strip():
                    st.warning("Prompt name cannot be empty.")
                else:
                    tags = [t.strip() for t in tags_input.split(",") if t.strip()]
                    if st.session_state.editor_prompt_id:
                        result = db_update_prompt(st.session_state.editor_prompt_id, new_content)
                        if result:
                            st.success(f"✅ Updated — now at version {result['current_version_number']}")
                            st.session_state.editor_prompt_id = result["id"]
                            st.session_state.editor_content = new_content
                            st.rerun()
                    else:
                        result = db_create_prompt(prompt_name, new_content, category, tags)
                        if result:
                            st.success(f"✅ Created prompt `{result['name']}`")
                            st.session_state.editor_prompt_id = result["id"]
                            st.session_state.editor_content = new_content
                            st.session_state.editor_name = result["name"]
                            st.rerun()

        with col_delete:
            if st.session_state.editor_prompt_id:
                if st.button("🗑️ Delete", use_container_width=True):
                    if db_delete_prompt(st.session_state.editor_prompt_id):
                        st.success("Deleted")
                        st.session_state.editor_prompt_id = None
                        st.session_state.editor_content = ""
                        st.session_state.editor_name = ""
                        st.rerun()

        # Version history
        if st.session_state.editor_prompt_id:
            st.divider()
            st.markdown("### 📋 Version History")
            versions = db_get_versions(st.session_state.editor_prompt_id) or []
            if versions:
                for v in versions[:10]:
                    col_v, col_r = st.columns([4, 1])
                    with col_v:
                        snippet = v["content"][:80].replace("\n", " ") + ("..." if len(v["content"]) > 80 else "")
                        st.caption(f"**v{v['version_number']}** — {v['created_at'][:16].replace('T',' ')} — {snippet}")
                    with col_r:
                        if st.button(f"↩️ Rollback", key=f"rb_{v['version_number']}"):
                            result = db_rollback_prompt(st.session_state.editor_prompt_id, v["version_number"])
                            if result:
                                st.success(f"Rolled back to v{v['version_number']}")
                                st.session_state.editor_content = v["content"]
                                st.rerun()

                st.markdown("#### Diff two versions")
                v_nums = [v["version_number"] for v in versions]
                va = st.selectbox("Version A", v_nums, key="diff_a")
                vb = st.selectbox("Version B", v_nums, key="diff_b")
                if st.button("Show Diff") and va != vb:
                    diff = db_get_version_diff(st.session_state.editor_prompt_id, va, vb)
                    if diff:
                        st.code("\n".join(diff["diff_lines"]) or "(no differences)", language="diff")

    with col_right:
        st.markdown("### 🚀 Generate")
        backends = available_backends()
        model_options = []
        for b in backends:
            if b["id"] == "ollama":
                ollama_models = _run_async(list_ollama_models())
                for m in ollama_models:
                    model_options.append(f"ollama/{m}")
            elif b["id"] in ("deepseek-chat", "deepseek-reasoner"):
                model_options.append(b["id"])
        if not model_options:
            model_options = ["deepseek-chat", "deepseek-reasoner"]

        selected_models = st.multiselect(
            "Models", model_options,
            default=["deepseek-chat"] if "deepseek-chat" in model_options else model_options[:1],
        )
        system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.", height=80)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        max_tokens = st.slider("Max Tokens", 64, 8192, 2048, 64)

        if vars_list:
            st.markdown("**Fill in variables:**")
            for var in vars_list:
                st.session_state.editor_vars[var] = st.text_input(
                    f"{{{{{var}}}}}", value=st.session_state.editor_vars.get(var, ""), key=f"var_{var}"
                )

        if st.button("⚡ Generate", type="primary", use_container_width=True):
            if not new_content.strip():
                st.warning("Enter a prompt first.")
            elif not selected_models:
                st.warning("Select at least one model.")
            else:
                filled = fill_variables(new_content, st.session_state.editor_vars)
                with st.spinner(f"Generating with {', '.join(selected_models)}..."):
                    results = _run_async(generate_multi(
                        selected_models, system_prompt, filled, temperature, max_tokens
                    ))
                total_cost = sum(r.get("cost_estimate", 0) for r in results)
                total_latency = sum(r.get("latency_ms", 0) for r in results)
                st.session_state.generate_results = {
                    "results": results, "total_cost": total_cost, "total_latency_ms": total_latency,
                    "prompt_id": st.session_state.editor_prompt_id, "prompt_content": filled,
                }
                # Save to history
                try:
                    from database import save_generation as _sg
                    for r in results:
                        _run_async(_sg(
                            prompt_id=st.session_state.editor_prompt_id or "",
                            prompt_name=prompt_name, model=r.get("model", ""),
                            output=r.get("output", ""), latency_ms=r.get("latency_ms", 0),
                            input_tokens=r.get("input_tokens", 0), output_tokens=r.get("output_tokens", 0),
                            cost_estimate=r.get("cost_estimate", 0), variables=st.session_state.editor_vars,
                        ))
                except Exception:
                    pass

        if st.session_state.generate_results:
            res = st.session_state.generate_results
            st.divider()
            st.markdown(f"### Results — Total cost: ${res['total_cost']:.6f}")
            for r in res["results"]:
                with st.expander(
                    f"**{r['model']}** — {r['latency_ms']:.0f}ms | "
                    f"{r['input_tokens']}+{r['output_tokens']} tokens | ${r['cost_estimate']:.6f}",
                    expanded=True,
                ):
                    if r.get("error"):
                        st.error(r["error"])
                    else:
                        st.markdown(r["output"])

# ==========================================================================
# TAB: Compare (A/B)
# ==========================================================================
elif tab == "⚖️ Compare (A/B)":
    st.title("⚖️ A/B Prompt Comparison")

    prompts = db_list_prompts() or []
    prompt_map = {f"{p['name']} (v{p['current_version_number']})": p for p in prompts}

    if len(prompt_map) < 2:
        st.warning("Create at least 2 prompts in the Editor first.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Prompt A")
            sel_a = st.selectbox("Select Prompt A", list(prompt_map.keys()), key="ab_sel_a")
            content_a = ""
            if sel_a:
                p_a = prompt_map[sel_a]
                versions_a = db_get_versions(p_a["id"]) or []
                content_a = versions_a[0]["content"] if versions_a else p_a.get("content", "")
                vars_a = extract_variables(content_a)
                for v in vars_a:
                    val = st.text_input(f"A: {{{{ {v} }}}}", key=f"ab_var_a_{v}")
                    content_a = content_a.replace("{{" + v + "}}", val)
                st.text_area("Content A", value=content_a, height=150, key="content_a", disabled=True)

        with col_b:
            st.markdown("### Prompt B")
            sel_b = st.selectbox("Select Prompt B", list(prompt_map.keys()), key="ab_sel_b")
            content_b = ""
            if sel_b:
                p_b = prompt_map[sel_b]
                versions_b = db_get_versions(p_b["id"]) or []
                content_b = versions_b[0]["content"] if versions_b else p_b.get("content", "")
                vars_b = extract_variables(content_b)
                for v in vars_b:
                    val = st.text_input(f"B: {{{{ {v} }}}}", key=f"ab_var_b_{v}")
                    content_b = content_b.replace("{{" + v + "}}", val)
                st.text_area("Content B", value=content_b, height=150, key="content_b", disabled=True)

        cloud_models = []
        for b in available_backends():
            if b["id"] in ("deepseek-chat", "deepseek-reasoner"):
                cloud_models.append(b["id"])
        if not cloud_models:
            cloud_models = ["deepseek-chat"]
        compare_model = st.selectbox("Model for comparison", cloud_models, key="ab_model")

        if st.button("⚡ Run A/B Comparison", type="primary", use_container_width=True):
            if sel_a and sel_b and sel_a != sel_b:
                with st.spinner(f"Generating with {compare_model}..."):
                    result = _run_async(run_ab_evaluation(
                        p_a["id"], p_b["id"], content_a, content_b,
                        model=compare_model,
                    ))
                if result:
                    st.session_state.ab_result = result
                    st.session_state.ab_ratings_submitted = False

        if st.session_state.ab_result:
            r = st.session_state.ab_result
            col_ar, col_br = st.columns(2)
            with col_ar:
                st.markdown("### Output A")
                st.markdown(r["output_a"].get("output", "Error"))
                st.caption(f"Latency: {r['output_a'].get('latency_ms', 0):.0f}ms")
            with col_br:
                st.markdown("### Output B")
                st.markdown(r["output_b"].get("output", "Error"))
                st.caption(f"Latency: {r['output_b'].get('latency_ms', 0):.0f}ms")

            st.divider()
            st.markdown("### 📊 Rate Outputs (1-5)")
            col_ra, col_rb = st.columns(2)
            with col_ra:
                st.markdown("**Score A**")
                rel_a = st.slider("Relevance", 1, 5, 3, key="rel_a")
                acc_a = st.slider("Accuracy", 1, 5, 3, key="acc_a")
                cre_a = st.slider("Creativity", 1, 5, 3, key="cre_a")
            with col_rb:
                st.markdown("**Score B**")
                rel_b = st.slider("Relevance", 1, 5, 3, key="rel_b")
                acc_b = st.slider("Accuracy", 1, 5, 3, key="acc_b")
                cre_b = st.slider("Creativity", 1, 5, 3, key="cre_b")

            notes = st.text_area("Notes (optional)", key="ab_notes")
            if st.button("💾 Submit Ratings", type="primary"):
                ratings = {
                    "a": {"relevance": rel_a, "accuracy": acc_a, "creativity": cre_a},
                    "b": {"relevance": rel_b, "accuracy": acc_b, "creativity": cre_b},
                }
                winner = "a" if (rel_a + acc_a + cre_a) >= (rel_b + acc_b + cre_b) else "b"
                try:
                    req = ABEvalRequest(
                        prompt_id_a=r["prompt_id_a"], prompt_id_b=r["prompt_id_b"],
                        model=compare_model, content_a=r["content_a"], content_b=r["content_b"],
                        output_a=r["output_a"]["output"], output_b=r["output_b"]["output"],
                        ratings=ratings, notes=notes,
                    )
                    _run_async(submit_evaluation(req))
                    st.success("✅ Ratings submitted!")
                    st.session_state.ab_ratings_submitted = True
                except Exception as e:
                    st.error(f"Failed to save: {e}")

# ==========================================================================
# TAB: Library
# ==========================================================================
elif tab == "📚 Library":
    st.title("📚 Prompt Template Library")
    templates = load_template_library()

    if not templates:
        st.info("No templates found in prompts/ directory. Add JSON template files to get started.")
    else:
        for tmpl in templates:
            name = tmpl.get("name", tmpl.get("_filename", "Unknown"))
            desc = tmpl.get("description", "")
            cat = tmpl.get("category", "general")
            tags = tmpl.get("tags", [])
            with st.expander(f"**{name}** — {cat}" + (f" — {desc}" if desc else "")):
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(f"⬇ Load into Editor", key=f"load_{tmpl['_filename']}"):
                        detail = get_template(tmpl["_filename"])
                        if detail:
                            prompt_text = detail.get("prompt", detail.get("content", ""))
                            st.session_state.editor_content = prompt_text
                            st.session_state.editor_name = detail.get("name", "")
                            st.session_state.editor_prompt_id = None
                            st.session_state.generate_results = None
                            st.success(f"Loaded `{detail.get('name', tmpl['_filename'])}` into Editor. Switch to Editor tab.")
                    if st.button(f"👁 Preview", key=f"prev_{tmpl['_filename']}"):
                        detail = get_template(tmpl["_filename"])
                        if detail:
                            prompt_text = detail.get("prompt", detail.get("content", ""))
                            highlighted = highlight_variables(prompt_text)
                            st.markdown(
                                f'<div style="background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:6px;'
                                f'font-family:monospace;white-space:pre-wrap;font-size:13px;border:1px solid #333">{highlighted}</div>',
                                unsafe_allow_html=True,
                            )
                            vars_found = detail.get("variables", extract_variables(prompt_text))
                            if vars_found:
                                st.caption(f"Variables: {', '.join(vars_found)}")
                with col2:
                    if tags:
                        st.caption("Tags: " + ", ".join(tags))

# ==========================================================================
# TAB: History
# ==========================================================================
elif tab == "📜 History":
    st.title("📜 Generation History")

    col_filters, col_data = st.columns([1, 3])
    with col_filters:
        prompts = db_list_prompts() or []
        prompt_filter = st.selectbox("Filter by prompt", ["All"] + [p["name"] for p in prompts], key="hist_filter")
        prompt_id_filter = None
        if prompt_filter != "All":
            for p in prompts:
                if p["name"] == prompt_filter:
                    prompt_id_filter = p["id"]
                    break
        model_filter = st.selectbox("Filter by model", ["All", "deepseek-chat", "deepseek-reasoner"], key="hist_model")

    params = {"limit": 100, "offset": 0}
    if prompt_id_filter:
        params["prompt_id"] = prompt_id_filter
    if model_filter != "All":
        params["model"] = model_filter

    history = db_list_history(**params) or []

    with col_data:
        if not history:
            st.info("No generation history yet. Generate some prompts first!")
        else:
            st.caption(f"Showing {len(history)} results")
            for h in history:
                with st.expander(
                    f"**{h.get('prompt_name', 'Prompt')}** → {h['model']} | "
                    f"{h['latency_ms']:.0f}ms | {h['input_tokens']}+{h['output_tokens']} tokens | "
                    f"${h['cost_estimate']:.6f} | {h['created_at'][:16].replace('T',' ')}"
                ):
                    st.markdown(h["output"][:2000])
                    if h.get("variables") and h["variables"] != {}:
                        st.caption(f"Variables: {json.dumps(h['variables'])}")

# ==========================================================================
# TAB: Analytics
# ==========================================================================
elif tab == "📊 Analytics":
    st.title("📊 Analytics & Cost Tracking")

    cost = db_get_cost_summary()
    if cost:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Generations", cost["total_generations"])
        with col2:
            st.metric("Total Tokens (In)", f"{cost['total_input_tokens']:,}")
        with col3:
            st.metric("Total Tokens (Out)", f"{cost['total_output_tokens']:,}")
        with col4:
            st.metric("Total Cost", f"${cost['total_cost']:.4f}")
        st.metric("Avg Latency", f"{cost['avg_latency_ms']:.1f} ms")

        st.divider()
        st.markdown("### Per-Model Breakdown")
        by_model = cost.get("by_model", {})
        if by_model:
            model_data = []
            for model_name, stats in by_model.items():
                model_data.append({
                    "Model": model_name, "Generations": stats["count"],
                    "Input Tokens": f"{stats['input_tokens']:,}",
                    "Output Tokens": f"{stats['output_tokens']:,}",
                    "Total Cost": f"${stats['total_cost']:.6f}",
                })
            st.dataframe(model_data, use_container_width=True)

        st.divider()
        st.markdown("### Evaluation Score Trends")
        evals = db_list_ab_evals(limit=50) or []
        if evals:
            scores_by_date = {}
            for e in evals:
                date_key = e["created_at"][:10]
                ratings = e.get("ratings", {})
                if isinstance(ratings, str):
                    try:
                        ratings = json.loads(ratings)
                    except Exception:
                        ratings = {}
                all_scores = []
                for side in ("a", "b"):
                    for metric, score in ratings.get(side, {}).items():
                        all_scores.append(score)
                if all_scores:
                    scores_by_date.setdefault(date_key, []).extend(all_scores)
            if scores_by_date:
                import pandas as pd
                chart_data = pd.DataFrame([
                    {"Date": d, "Avg Score": sum(s) / len(s)}
                    for d, s in sorted(scores_by_date.items())
                ])
                st.line_chart(chart_data, x="Date", y="Avg Score")
            else:
                st.info("No evaluation scores to chart yet.")
        else:
            st.info("No evaluations yet. Run A/B comparisons to see trends here.")

        st.divider()
        st.markdown("### Latency Over Time")
        history = db_list_history(limit=100) or []
        if history:
            import pandas as pd
            latency_data = pd.DataFrame([
                {"Time": h["created_at"][:16].replace("T", " "), "Latency (ms)": h["latency_ms"], "Model": h["model"]}
                for h in history
            ])
            st.scatter_chart(latency_data, x="Time", y="Latency (ms)", color="Model", size="Latency (ms)")
        else:
            st.info("No history data to chart.")

st.divider()
st.caption("🧪 PromptLab — Prompt Engineering Workbench | Streamlit Cloud")
