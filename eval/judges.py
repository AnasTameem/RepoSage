"""
eval/judges.py - Centralized evaluation judge and embeddings provider.

Configures explicit LLM judges for both DeepEval and Ragas using Google Gemini
from config.py (avoiding OpenAI key demands and ensuring reproducible evaluation).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    EVAL_EMBEDDING_MODEL,
    GEMINI_API_KEY,
    JUDGE_GEMINI_API_KEY,
    JUDGE_MODEL,
)


def get_deepeval_judge(
    model: str | None = None,
    api_key: str | None = None,
):
    """
    Returns an explicit GeminiModel instance for DeepEval metric assertions.
    Passing an explicit judge prevents DeepEval from defaulting to OpenAI.
    """
    from deepeval.models import GeminiModel

    target_key = api_key or JUDGE_GEMINI_API_KEY or GEMINI_API_KEY or os.getenv("JUDGE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    target_model = model or JUDGE_MODEL or os.getenv("JUDGE_MODEL", "gemini-2.5-flash")

    if not target_key:
        raise ValueError("Neither JUDGE_GEMINI_API_KEY nor GEMINI_API_KEY is set. An API key is required to initialize the DeepEval judge.")

    return GeminiModel(
        model=target_model,
        api_key=target_key,
    )


def get_ragas_judge(
    model: str | None = None,
    api_key: str | None = None,
):
    """
    Returns an explicit LangChain ChatGoogleGenerativeAI wrapper for Ragas.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLMWrapper

    target_key = api_key or JUDGE_GEMINI_API_KEY or GEMINI_API_KEY or os.getenv("JUDGE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    target_model = model or JUDGE_MODEL or os.getenv("JUDGE_MODEL", "gemini-2.5-flash")

    if not target_key:
        raise ValueError("Neither JUDGE_GEMINI_API_KEY nor GEMINI_API_KEY is set. An API key is required to initialize the Ragas judge.")

    chat = ChatGoogleGenerativeAI(
        model=target_model,
        google_api_key=target_key,
    )
    return LangchainLLMWrapper(chat)


def get_ragas_embeddings(
    model: str | None = None,
    api_key: str | None = None,
):
    """
    Returns explicit GoogleGenerativeAIEmbeddings for Ragas metrics that require embeddings
    (such as AnswerRelevancy), avoiding any implicit fallback to OpenAI.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    target_key = api_key or JUDGE_GEMINI_API_KEY or GEMINI_API_KEY or os.getenv("JUDGE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    target_model = model or EVAL_EMBEDDING_MODEL or os.getenv("EVAL_EMBEDDING_MODEL", "models/text-embedding-004")

    if not target_key:
        raise ValueError("Neither JUDGE_GEMINI_API_KEY nor GEMINI_API_KEY is set. An API key is required to initialize Ragas embeddings.")

    emb = GoogleGenerativeAIEmbeddings(
        model=target_model,
        google_api_key=target_key,
    )
    return LangchainEmbeddingsWrapper(emb)

