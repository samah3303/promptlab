"""PromptLab Frontend — Streamlit web interface for Prompt Engineering Workbench."""

import json
import re
import sys
import time
from pathlib import Path

import streamlit as st
import requests

# --- Config ---
API_BASE = "http://127.0.0.1:8000"
st.set_page_config(
    page_title="PromptLab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────── Helpers ────────────────

def api_get(path: str, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, data=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


def api_put(path: str, data=None):
    try:
        r = requests.put(f"{API_BASE}{path}", json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


def api_delete(path: str):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return False


def extract_variables(content: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\{\{(\w+)\}\}", content)))


def highlight_variables(content: str) -> str:
    """Wrap {{variables}} in HTML spans for visual highlighting."""
    return re.sub(
        r"(\{\{(\w+)\}\})",
        r'<span style="background:#1a73e8;color:#fff;padding:2px 6px;border-radius:4px;font-family:monospace">\1</span>',
        content,
    )


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ──────────────── Sidebar ────────────────

with st.sidebar:
    st.title("🧪 PromptLab")
    st.caption("Prompt Engineering Workbench")

    api_healthy = check_api_health()
    if api_healthy:
        st.success("Backend: Connected")
    else:
        st.error("Backend: Offline")
        st.info("Start the backend: `cd backend && python main.py`")

    st.divider()
    tab = st.radio(
        "Navigation",
        ["📝 Editor", "⚖️ Compare (A/B)", "📚 Library", "📜 History", "📊 Analytics"],
        index=0,
    )


# ──────────────── Tab: Editor ────────────────

if tab == "📝 Editor":
    st.title("📝 Prompt Editor")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # --- Load existing prompt or start fresh ---
        prompts = api_get("/api/prompts") or []
        prompt_options = {"Create New...": None}
        for p in prompts:
            prompt_options[f"{p['name']} ({p['category']})"] = p["id"]

        selected = st.selectbox("Select a prompt", list(prompt_options.keys()), key="editor_select")

        # --- Editor State ---
        if "editor_content" not in st.session_state:
            st.session_state.editor_content = ""
        if "editor_name" not in st.session_state:
            st.session_state.editor_name = ""
        if "editor_prompt_id" not in st.session_state:
            st.session_state.editor_prompt_id = None
        if "editor_vars" not in st.session_state:
            st.session_state.editor_vars = {}
        if "generate_results" not in st.session_state:
            st.session_state.generate_results = None

        prompt_id = prompt_options.get(selected)
        if prompt_id and prompt_id != st.session_state.editor_prompt_id:
            prompt_detail = api_get(f"/api/prompts/{prompt_id}")
            if prompt_detail:
                versions = api_get(f"/api/prompts/{prompt_id}/versions") or []
                current = versions[0] if versions else {"content": prompt_detail["content"]}
                st.session_state.editor_content = current["content"]
                st.session_state.editor_name = prompt_detail["name"]
                st.session_state.editor_prompt_id = prompt_id
                st.session_state.editor_vars = {}
                st.session_state.generate_results = None

        # Prompt metadata
        prompt_name = st.text_input("Prompt Name", value=st.session_state.editor_name, key="editor_name_input")
        category = st.selectbox("Category", ["general", "summarization", "code_generation", "creative_writing", "classification", "rag_qa", "chain_of_thought", "few_shot", "other"], key="editor_category")
        tags_input = st.text_input("Tags (comma-separated)", key="editor_tags")

        # Variable detection from content
        vars_list = extract_variables(st.session_state.editor_content)
        if vars_list:
            st.markdown("**Variables detected:** " + " ".join([f"`{{{{{v}}}}}`" for v in vars_list]))

        # Prompt content editor
        st.markdown("### Prompt Content")
        new_content = st.text_area(
            "Edit your prompt",
            value=st.session_state.editor_content,
            height=250,
            key="editor_textarea",
            placeholder="Write your prompt here. Use {{variable_name}} for dynamic variables.",
        )

        # Show preview with highlighted variables
        if new_content:
            highlighted = highlight_variables(new_content)
            st.markdown(f'<div style="background:#1e1e1e;color:#d4d4d4;padding:16px;border-radius:8px;font-family:monospace;white-space:pre-wrap;border:1px solid #333">{highlighted}</div>', unsafe_allow_html=True)

        # Save / Update button
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
                        result = api_put(
                            f"/api/prompts/{st.session_state.editor_prompt_id}",
                            {"content": new_content, "description": ""},
                        )
                        if result:
                            st.success(f"✅ Updated — now at version {result['current_version_number']}")
                            st.session_state.editor_prompt_id = result["id"]
                            st.session_state.editor_content = new_content
                            st.rerun()
                    else:
                        result = api_post("/api/prompts", {
                            "name": prompt_name,
                            "content": new_content,
                            "category": category,
                            "tags": tags,
                        })
                        if result:
                            st.success(f"✅ Created prompt `{result['name']}`")
                            st.session_state.editor_prompt_id = result["id"]
                            st.session_state.editor_content = new_content
                            st.session_state.editor_name = result["name"]
                            st.rerun()

        with col_delete:
            if st.session_state.editor_prompt_id:
                if st.button("🗑️ Delete", use_container_width=True):
                    if api_delete(f"/api/prompts/{st.session_state.editor_prompt_id}"):
                        st.success("Deleted")
                        st.session_state.editor_prompt_id = None
                        st.session_state.editor_content = ""
                        st.session_state.editor_name = ""
                        st.rerun()

        # Version history for current prompt
        if st.session_state.editor_prompt_id:
            st.divider()
            st.markdown("### 📋 Version History")
            versions = api_get(f"/api/prompts/{st.session_state.editor_prompt_id}/versions") or []
            if versions:
                for v in versions[:10]:
                    col_v, col_r = st.columns([4, 1])
                    with col_v:
                        version_snippet = v["content"][:80].replace("\n", " ") + ("..." if len(v["content"]) > 80 else "")
                        st.caption(f"**v{v['version_number']}** — {v['created_at'][:16].replace('T',' ')} — {version_snippet}")
                    with col_r:
                        if st.button(f"↩️ Rollback", key=f"rb_{v['version_number']}"):
                            result = api_post(f"/api/prompts/{st.session_state.editor_prompt_id}/rollback/{v['version_number']}")
                            if result:
                                st.success(f"Rolled back to v{v['version_number']}")
                                st.session_state.editor_content = v["content"]
                                st.rerun()

                # Diff viewer
                st.markdown("#### Diff two versions")
                v_nums = [v["version_number"] for v in versions]
                va = st.selectbox("Version A", v_nums, key="diff_a")
                vb = st.selectbox("Version B", v_nums, key="diff_b")
                if st.button("Show Diff") and va != vb:
                    diff = api_get(f"/api/prompts/{st.session_state.editor_prompt_id}/diff", {"version_a": va, "version_b": vb})
                    if diff:
                        diff_text = "\n".join(diff["diff_lines"])
                        st.code(diff_text if diff_text else "(no differences)", language="diff")

    with col_right:
        st.markdown("### 🚀 Generate")

        # Multi-LLM selection
        backends = api_get("/api/backends") or []
        model_options = []
        for b in backends:
            if b["id"] == "ollama":
                for m in b.get("models", []):
                    model_options.append(f"ollama/{m}")
            elif b["id"] in ("deepseek-chat", "deepseek-reasoner"):
                model_options.append(b["id"])

        if not model_options:
            model_options = ["deepseek-chat", "deepseek-reasoner"]  # fallback

        selected_models = st.multiselect(
            "Models",
            model_options,
            default=["deepseek-chat"] if "deepseek-chat" in model_options else model_options[:1],
        )

        system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.", height=80)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        max_tokens = st.slider("Max Tokens", 64, 8192, 2048, 64)

        # Variable inputs
        if vars_list:
            st.markdown("**Fill in variables:**")
            for var in vars_list:
                st.session_state.editor_vars[var] = st.text_input(
                    f"{{{{{var}}}}}",
                    value=st.session_state.editor_vars.get(var, ""),
                    key=f"var_{var}",
                )

        # Substitute variables before sending
        def fill_variables(content: str, vars_dict: dict) -> str:
            filled = content
            for k, v in vars_dict.items():
                filled = filled.replace("{{" + k + "}}", v)
            return filled

        if st.button("⚡ Generate", type="primary", use_container_width=True):
            if not new_content.strip():
                st.warning("Enter a prompt first.")
            elif not selected_models:
                st.warning("Select at least one model.")
            else:
                filled_content = fill_variables(new_content, st.session_state.editor_vars)
                with st.spinner(f"Generating with {', '.join(selected_models)}..."):
                    result = api_post("/api/generate", {
                        "prompt_id": st.session_state.editor_prompt_id,
                        "content": filled_content,
                        "models": selected_models,
                        "system_prompt": system_prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "variables": st.session_state.editor_vars,
                    })
                if result:
                    st.session_state.generate_results = result

        # Display results
        if st.session_state.generate_results:
            results = st.session_state.generate_results
            st.divider()
            st.markdown(f"### Results — Total cost: ${results['total_cost']:.6f}")

            for r in results["results"]:
                with st.expander(f"**{r['model']}** — {r['latency_ms']:.0f}ms | {r['input_tokens']}+{r['output_tokens']} tokens | ${r['cost_estimate']:.6f}", expanded=True):
                    if r.get("error"):
                        st.error(r["error"])
                    else:
                        st.markdown(r["output"])


# ──────────────── Tab: Compare (A/B) ────────────────

elif tab == "⚖️ Compare (A/B)":
    st.title("⚖️ A/B Prompt Comparison")

    prompts = api_get("/api/prompts") or []
    prompt_map = {f"{p['name']} (v{p['current_version_number']})": p for p in prompts}

    if len(prompt_map) < 2:
        st.warning("Create at least 2 prompts in the Editor first.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Prompt A")
            sel_a = st.selectbox("Select Prompt A", list(prompt_map.keys()), key="ab_sel_a")
            if sel_a:
                p_a = prompt_map[sel_a]
                versions_a = api_get(f"/api/prompts/{p_a['id']}/versions") or []
                content_a = versions_a[0]["content"] if versions_a else p_a["content"]
                # Fill variables
                vars_a = extract_variables(content_a)
                var_vals_a = {}
                for v in vars_a:
                    var_vals_a[v] = st.text_input(f"A: {{{{ {v} }}}}", key=f"ab_var_a_{v}")
                for k, val in var_vals_a.items():
                    content_a = content_a.replace("{{" + k + "}}", val)
                st.text_area("Content A", value=content_a, height=150, key="content_a", disabled=True)

        with col_b:
            st.markdown("### Prompt B")
            sel_b = st.selectbox("Select Prompt B", list(prompt_map.keys()), key="ab_sel_b")
            if sel_b:
                p_b = prompt_map[sel_b]
                versions_b = api_get(f"/api/prompts/{p_b['id']}/versions") or []
                content_b = versions_b[0]["content"] if versions_b else p_b["content"]
                vars_b = extract_variables(content_b)
                var_vals_b = {}
                for v in vars_b:
                    var_vals_b[v] = st.text_input(f"B: {{{{ {v} }}}}", key=f"ab_var_b_{v}")
                for k, val in var_vals_b.items():
                    content_b = content_b.replace("{{" + k + "}}", val)
                st.text_area("Content B", value=content_b, height=150, key="content_b", disabled=True)

        # Select model for comparison
        backends = api_get("/api/backends") or []
        cloud_models = []
        for b in backends:
            if b["id"] in ("deepseek-chat", "deepseek-reasoner"):
                cloud_models.append(b["id"])
        if not cloud_models:
            cloud_models = ["deepseek-chat"]

        compare_model = st.selectbox("Model for comparison", cloud_models, key="ab_model")

        # Run comparison button
        if "ab_result" not in st.session_state:
            st.session_state.ab_result = None
        if "ab_ratings_submitted" not in st.session_state:
            st.session_state.ab_ratings_submitted = False

        if st.button("⚡ Run A/B Comparison", type="primary", use_container_width=True):
            if sel_a and sel_b and sel_a != sel_b:
                with st.spinner(f"Generating with {compare_model}..."):
                    result = api_post("/api/eval/compare", {
                        "prompt_id_a": p_a["id"],
                        "prompt_id_b": p_b["id"],
                        "model": compare_model,
                        "content_a": content_a,
                        "content_b": content_b,
                        "output_a": "",
                        "output_b": "",
                    })
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

            # Rating widgets
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

            if st.button("💾 Submit Evaluation", type="primary"):
                submitted = api_post("/api/eval/submit", {
                    "prompt_id_a": p_a["id"],
                    "prompt_id_b": p_b["id"],
                    "model": compare_model,
                    "content_a": content_a,
                    "content_b": content_b,
                    "output_a": r["output_a"].get("output", ""),
                    "output_b": r["output_b"].get("output", ""),
                    "ratings": {
                        "a": {"relevance": rel_a, "accuracy": acc_a, "creativity": cre_a},
                        "b": {"relevance": rel_b, "accuracy": acc_b, "creativity": cre_b},
                    },
                    "notes": notes,
                })
                if submitted:
                    st.success(f"✅ Submitted! Winner: Prompt {submitted.get('winner', 'tie').upper()}")
                    st.session_state.ab_ratings_submitted = True

        # Past evaluations
        st.divider()
        st.markdown("### Past A/B Evaluations")
        evals = api_get("/api/eval/history", {"limit": 20}) or []
        for e in evals:
            prompt_a_name = next((p["name"] for p in prompts if p["id"] == e["prompt_id_a"]), e["prompt_id_a"][:8])
            prompt_b_name = next((p["name"] for p in prompts if p["id"] == e["prompt_id_b"]), e["prompt_id_b"][:8])
            winner = e.get("winner", "tie")
            st.markdown(
                f"**{prompt_a_name}** vs **{prompt_b_name}** — Winner: **{winner.upper()}** | "
                f"{e['created_at'][:16].replace('T',' ')} | {e['model']}"
            )


# ──────────────── Tab: Library ────────────────

elif tab == "📚 Library":
    st.title("📚 Prompt Library")

    templates = api_get("/api/templates") or []

    if not templates:
        st.info("No templates loaded. Add JSON files to the `prompts/` directory.")

    # Group by category
    categories = {}
    for t in templates:
        cat = t.get("category", "other")
        categories.setdefault(cat, []).append(t)

    for cat, items in sorted(categories.items()):
        st.markdown(f"### {cat.replace('_', ' ').title()}")
        cols = st.columns(2)
        for i, tmpl in enumerate(items):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**{tmpl['name']}**")
                    st.caption(tmpl.get("description", ""))
                    tag_html = " ".join([f"`{tag}`" for tag in tmpl.get("tags", [])])
                    st.markdown(tag_html)
                    if st.button(f"📝 Use This", key=f"use_{tmpl['filename']}"):
                        detail = api_get(f"/api/templates/{tmpl['filename']}")
                        if detail:
                            st.session_state.editor_content = detail["prompt"]
                            st.session_state.editor_name = detail["name"]
                            st.session_state.editor_prompt_id = None
                            st.session_state.generate_results = None
                            st.success(f"Loaded `{detail['name']}` into the Editor. Switch to Editor tab.")
                    # Preview
                    if st.button(f"👁 Preview", key=f"prev_{tmpl['filename']}"):
                        detail = api_get(f"/api/templates/{tmpl['filename']}")
                        if detail:
                            highlighted = highlight_variables(detail["prompt"])
                            st.markdown(f'<div style="background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:6px;font-family:monospace;white-space:pre-wrap;font-size:13px;border:1px solid #333">{highlighted}</div>', unsafe_allow_html=True)
                            if detail.get("variables"):
                                st.caption(f"Variables: {', '.join(detail['variables'])}")


# ──────────────── Tab: History ────────────────

elif tab == "📜 History":
    st.title("📜 Generation History")

    col_filters, col_data = st.columns([1, 3])

    with col_filters:
        prompts = api_get("/api/prompts") or []
        prompt_filter = st.selectbox(
            "Filter by prompt",
            ["All"] + [p["name"] for p in prompts],
            key="hist_filter",
        )
        prompt_id_filter = None
        if prompt_filter != "All":
            for p in prompts:
                if p["name"] == prompt_filter:
                    prompt_id_filter = p["id"]
                    break

        model_filter = st.selectbox(
            "Filter by model",
            ["All", "deepseek-chat", "deepseek-reasoner"],
            key="hist_model",
        )

    params = {"limit": 100, "offset": 0}
    if prompt_id_filter:
        params["prompt_id"] = prompt_id_filter
    if model_filter != "All":
        params["model"] = model_filter

    history = api_get("/api/history", params) or []

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


# ──────────────── Tab: Analytics ────────────────

elif tab == "📊 Analytics":
    st.title("📊 Analytics & Cost Tracking")

    cost = api_get("/api/analytics/cost")
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

        # Per-model breakdown
        st.divider()
        st.markdown("### Per-Model Breakdown")
        by_model = cost.get("by_model", {})
        if by_model:
            model_data = []
            for model_name, stats in by_model.items():
                model_data.append({
                    "Model": model_name,
                    "Generations": stats["count"],
                    "Input Tokens": f"{stats['input_tokens']:,}",
                    "Output Tokens": f"{stats['output_tokens']:,}",
                    "Total Cost": f"${stats['total_cost']:.6f}",
                })
            st.dataframe(model_data, use_container_width=True)

        # Score trends (if any evaluations)
        st.divider()
        st.markdown("### Evaluation Score Trends")
        evals = api_get("/api/eval/history", {"limit": 50}) or []
        if evals:
            scores_by_date = {}
            for e in evals:
                date_key = e["created_at"][:10]
                ratings = e.get("ratings", {})
                all_scores = []
                if isinstance(ratings, str):
                    try:
                        ratings = json.loads(ratings)
                    except Exception:
                        ratings = {}
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

        # Latency distribution
        st.divider()
        st.markdown("### Latency Over Time")
        history = api_get("/api/history", {"limit": 100}) or []
        if history:
            import pandas as pd
            latency_data = pd.DataFrame([
                {
                    "Time": h["created_at"][:16].replace("T", " "),
                    "Latency (ms)": h["latency_ms"],
                    "Model": h["model"],
                }
                for h in history
            ])
            st.scatter_chart(latency_data, x="Time", y="Latency (ms)", color="Model", size="Latency (ms)")
        else:
            st.info("No history data to chart.")


# ──────────────── Footer ────────────────

st.divider()
st.caption("🧪 PromptLab — Prompt Engineering Workbench | MIT License")
