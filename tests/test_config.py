"""Tests for rag.config — verify all config values are loaded and typed correctly."""

from rag.config import (
    ADD_BATCH_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_BATCH,
    EMBED_MODEL,
    KNOWLEDGE_DIR,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_URL,
    OVERFETCH_FACTOR,
    RERANKER_BATCH_SIZE,
    RERANKER_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
    WORKERS,
)


def test_paths_are_pathlib():
    from pathlib import Path
    assert isinstance(KNOWLEDGE_DIR, Path)
    assert isinstance(CHROMA_DIR, Path)


def test_string_configs_not_empty():
    assert isinstance(COLLECTION_NAME, str) and COLLECTION_NAME
    assert isinstance(EMBED_MODEL, str) and EMBED_MODEL
    assert isinstance(RERANKER_MODEL, str) and RERANKER_MODEL
    assert isinstance(LLM_MODEL, str) and LLM_MODEL
    assert isinstance(LLM_PROVIDER, str) and LLM_PROVIDER
    assert isinstance(OLLAMA_URL, str) and OLLAMA_URL


def test_int_configs_positive():
    for name, val in [
        ("CHUNK_SIZE", CHUNK_SIZE),
        ("CHUNK_OVERLAP", CHUNK_OVERLAP),
        ("TOP_K", TOP_K),
        ("OVERFETCH_FACTOR", OVERFETCH_FACTOR),
        ("RERANKER_BATCH_SIZE", RERANKER_BATCH_SIZE),
        ("EMBED_BATCH", EMBED_BATCH),
        ("ADD_BATCH_SIZE", ADD_BATCH_SIZE),
        ("WORKERS", WORKERS),
    ]:
        assert isinstance(val, int), f"{name} should be int, got {type(val)}"
        assert val > 0, f"{name} should be > 0, got {val}"


def test_chunk_overlap_less_than_size():
    assert CHUNK_OVERLAP < CHUNK_SIZE, "overlap must be < chunk size"


def test_system_prompt_has_rules():
    assert "STRICT RULES" in SYSTEM_PROMPT
    assert "NEVER" in SYSTEM_PROMPT


def test_llm_provider_known():
    assert LLM_PROVIDER in ("ollama", "openai"), f"Unknown provider: {LLM_PROVIDER}"
