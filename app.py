"""
app.py - Streamlit Frontend for RepoSage Codebase RAG

Interactive developer UI for:
  - Natural language querying of the codebase
  - Real-time token streaming via Server-Sent Events (SSE)
  - Interactive hyperparameter tuning (Top-k Seeds, Max Hops, Score Threshold)
  - Detailed inspection of Qdrant vector seeds and Neo4j graph relationships
"""
import json
import uuid
import requests
import streamlit as st
from typing import Dict, Any, List

# --------------------------------------------------------------------------- #
#  Page Configuration & Styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="RepoSage - Codebase Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://127.0.0.1:8000"

CUSTOM_CSS = """
<style>
    /* Sleek container styling */
    .stChatInput {
        padding-bottom: 1.5rem;
    }
    
    /* Metadata badge chips */
    .chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .chip-vector {
        background-color: #1e3a5f;
        color: #60a5fa;
        border: 1px solid #2563eb;
    }
    .chip-graph {
        background-color: #2e1065;
        color: #c084fc;
        border: 1px solid #7c3aed;
    }
    .chip-score {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
    }
    
    /* Code block refinements */
    pre code {
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.88rem !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Session State Initialization
# --------------------------------------------------------------------------- #
if "session_id" not in st.session_state:
    st.session_state.session_id = f"dev_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------------------------------- #
#  Sidebar: Controls & Hyperparameters
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🧠 RepoSage")
    st.caption("Graph-Augmented Codebase Assistant")

    # Backend Connection Status
    st.subheader("System Status")
    try:
        resp = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=2.0)
        if resp.status_code == 200:
            health_data = resp.json()
            st.success("🟢 API Server Online")
            with st.expander("Connected Engines", expanded=False):
                services = health_data.get("services", {})
                st.write(f"• **LLM**: `{services.get('llm_engine', 'N/A')}`")
                st.write(f"• **Embeddings**: `{services.get('embedding_model', 'N/A')}`")
                st.write(f"• **Vector DB**: `{services.get('vector_store', 'N/A')}`")
                st.write(f"• **Graph DB**: `{services.get('graph_database', 'N/A')}`")
        else:
            st.warning(f"🟡 Server returned status {resp.status_code}")
    except Exception:
        st.error("🔴 Backend Offline")
        st.info("Start API: `uv run uvicorn main:app --reload --port 8000`")

    st.divider()

    # Hyperparameters
    st.subheader("⚙️ Retrieval Hyperparameters")

    top_seeds = st.slider(
        "Top Vector Seeds (Qdrant)",
        min_value=1,
        max_value=8,
        value=2,
        help="Number of strict flat-scan primary code chunks retrieved from Qdrant.",
    )

    max_hops = st.slider(
        "Max Graph Hops (Neo4j)",
        min_value=1,
        max_value=3,
        value=2,
        help="Depth of structural traversal across [:CALLS], [:CONTAINS], and [:INHERITS_FROM].",
    )

    enable_score_threshold = st.checkbox("Filter by Score Threshold", value=False)
    score_threshold = None
    if enable_score_threshold:
        score_threshold = st.slider(
            "Min Cosine Similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
        )

    st.divider()

    # Session Management
    st.subheader("💬 Conversation Session")
    st.code(st.session_state.session_id, language="text")

    if st.button("🔄 New Chat / Reset Memory", use_container_width=True):
        try:
            requests.post(
                f"{BACKEND_URL}/api/v1/sessions/reset",
                json={"session_id": st.session_state.session_id},
                timeout=2.0,
            )
        except Exception:
            pass
        st.session_state.session_id = f"dev_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
#  Helper: Render Retrieved Context Box
# --------------------------------------------------------------------------- #
def render_retrieved_context(context_data: Dict[str, Any]):
    """Renders the expandable accordion with vector hits, graph links, and prompt context."""
    if not context_data:
        return

    seeds: List[Dict[str, Any]] = context_data.get("seeds", [])
    graph_relations: List[Dict[str, Any]] = context_data.get("graph_relations", [])
    raw_context: str = context_data.get("context", "")
    eff_query: str = context_data.get("effective_query", "")

    label = f"🔍 Retrieved Context ({len(seeds)} Seeds, {len(graph_relations)} Graph Relations)"
    with st.expander(label, expanded=False):
        if eff_query:
            st.caption(f"**Contextualized Query:** `{eff_query}`")

        tab_seeds, tab_graph, tab_prompt = st.tabs([
            f"🎯 Qdrant Seeds ({len(seeds)})",
            f"🕸️ Neo4j Relations ({len(graph_relations)})",
            "📄 Compiled Prompt Window",
        ])

        # Tab 1: Vector Hits
        with tab_seeds:
            if not seeds:
                st.info("No primary vector seeds retrieved.")
            else:
                for idx, seed in enumerate(seeds, 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(
                            f"**#{idx}. `{seed.get('name', 'Unknown')}`**  \n"
                            f"📁 `{seed.get('file_path', 'Unknown')}` "
                            f"(Lines {seed.get('start_line', '?')}-{seed.get('end_line', '?')})"
                        )
                    with col2:
                        score = seed.get("similarity_score", 0.0)
                        st.metric("Similarity", f"{score:.4f}")

                    snippet = seed.get("snippet", "").strip()
                    if snippet:
                        st.code(snippet, language="python")
                    st.divider()

        # Tab 2: Graph Expansion
        with tab_graph:
            if not graph_relations:
                st.info("No adjacent graph nodes expanded (Neo4j was offline or seed has no outgoing relations).")
            else:
                for idx, rel in enumerate(graph_relations, 1):
                    rels = " / ".join(rel.get("relationship_types", ["CONNECTED"]))
                    st.markdown(
                        f"**{idx}. `{rel.get('source_seed')}`** ──`[:{rels}]`──▶ **`{rel.get('target_name')}`** "
                        f"*(Hops: {rel.get('hops')}, Type: `{rel.get('target_type')}`)*  \n"
                        f"📁 `{rel.get('target_file')}`"
                    )
                    snippet = rel.get("snippet", "").strip()
                    if snippet:
                        with st.expander(f"View Code: {rel.get('target_name')}", expanded=False):
                            st.code(snippet, language="python")

        # Tab 3: Raw Context
        with tab_prompt:
            st.markdown("Raw compiled context provided to the LLM:")
            st.text_area(
                "Prompt Context Window",
                value=raw_context,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )


# --------------------------------------------------------------------------- #
#  Main Chat Interface
# --------------------------------------------------------------------------- #
st.header("RepoSage Codebase Assistant")
st.caption("Ask questions about code architecture, implementation details, dependencies, and APIs.")

# Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("context_data"):
            render_retrieved_context(msg["context_data"])

# User Query Input
user_query = st.chat_input("Ask a question about the codebase (e.g., 'How is Streamlit implemented in this app?')...")

if user_query:
    # Append and display user turn
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant Turn with Real-Time Streaming
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        context_placeholder = st.empty()

        status_placeholder.markdown("⏳ *Connecting to RepoSage RAG backend...*")

        payload = {
            "query": user_query,
            "session_id": st.session_state.session_id,
            "top_seeds": top_seeds,
            "max_hops": max_hops,
            "score_threshold": score_threshold,
        }

        full_answer = ""
        context_payload = None

        try:
            with requests.post(
                f"{BACKEND_URL}/api/v1/query",
                json=payload,
                stream=True,
                timeout=90.0,
            ) as response:
                if response.status_code != 200:
                    status_placeholder.empty()
                    error_msg = f"**Backend Error ({response.status_code}):** {response.text}"
                    answer_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    for line in response.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue

                        raw_json = line[6:].strip()
                        if not raw_json:
                            continue

                        try:
                            event = json.loads(raw_json)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type")

                        if event_type == "status":
                            stage_msg = event.get("message", "Processing...")
                            status_placeholder.markdown(f"⚙️ *{stage_msg}*")

                        elif event_type == "context":
                            context_payload = event
                            status_placeholder.markdown("⚡ *Generating answer via Gemini...*")

                        elif event_type == "token":
                            status_placeholder.empty()  # Clear status indicator once tokens arrive
                            token = event.get("token", "")
                            full_answer += token
                            answer_placeholder.markdown(full_answer + "▌")

                        elif event_type == "error":
                            status_placeholder.empty()
                            err = event.get("error", "Unknown error")
                            answer_placeholder.error(f"Error during generation: {err}")
                            full_answer += f"\n\n*[Error: {err}]*"

                        elif event_type == "done":
                            status_placeholder.empty()

                    # Final render without streaming cursor
                    answer_placeholder.markdown(full_answer)

                    # Render Retrieved Context Accordion
                    if context_payload:
                        with context_placeholder:
                            render_retrieved_context(context_payload)

                    # Persist assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "context_data": context_payload,
                    })

        except requests.exceptions.ConnectionError:
            status_placeholder.empty()
            err_text = (
                "❌ **Could not connect to the FastAPI backend.**  \n"
                f"Please ensure `uv run uvicorn main:app --reload --port 8000` is running at `{BACKEND_URL}`."
            )
            answer_placeholder.error(err_text)
            st.session_state.messages.append({"role": "assistant", "content": err_text})

        except Exception as exc:
            status_placeholder.empty()
            answer_placeholder.error(f"Request failed: {exc}")
            st.session_state.messages.append({"role": "assistant", "content": str(exc)})
