"""
prompt_templates.py - Prompt engineering for codebase RAG generation.

Provides system and user prompts to guide the LLM in generating accurate,
grounded, and citation-rich answers based on retrieved code context.
"""
from typing import List, Dict, Optional

CODEBASE_RAG_SYSTEM_PROMPT = """You are an expert AI software architect and codebase assistant.
Your task is to answer user questions about a codebase using ONLY the provided retrieval context (primary vector search matches and 1-2 hop structural graph expansion).

GUIDELINES FOR YOUR RESPONSE:
1. **Groundedness & Accuracy**:
   - Base your answer strictly on the provided code snippets and graph relationships.
   - If the provided context does not contain sufficient information to answer the question with certainty, state clearly what is known from the context and what information is missing. Do not invent non-existent functions, classes, or files.

2. **Precise Citations & Locations**:
   - Always cite exact file paths, line numbers (e.g. `path/to/file.py:L10-L45`), class names, and function names when explaining functionality.
   - Reference structural connections (e.g., "Function `foo` calls `bar` across files via `CALLS` relationship").

3. **Code Explanations & Examples**:
   - Include concise, syntax-highlighted code snippets from the context when illustrating key implementation details.
   - Explain the architectural flow (e.g., callers, callees, inheritance, and data flow).

4. **Tone & Formatting**:
   - Be clear, direct, and structured. Use headings, bullet points, and code blocks.
"""


def build_user_prompt(
    query: str,
    context: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Constructs the user message containing retrieved codebase context and the user query.
    """
    prompt_parts = []

    if history:
        prompt_parts.append("### PREVIOUS CONVERSATION CONTEXT:")
        for turn in history[-4:]:  # Keep last 2-4 turns
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            prompt_parts.append(f"{role}: {content}")
        prompt_parts.append("\n" + "=" * 60 + "\n")

    prompt_parts.append("### RETRIEVED CODEBASE CONTEXT (Vector Seeds + Graph Expansion):\n")
    prompt_parts.append(context.strip() if context else "No codebase snippets were retrieved.")
    prompt_parts.append("\n" + "=" * 60 + "\n")

    prompt_parts.append(f"### USER QUESTION:\n{query.strip()}\n")
    prompt_parts.append("Please provide a comprehensive, well-structured answer citing relevant files and functions.")

    return "\n".join(prompt_parts)
