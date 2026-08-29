"""
FastAPI dependency injection.

Provides singleton instances of:
- QdrantVectorStore (configured from environment variables)
- EmbeddingProvider (production RealEmbeddingProvider via Sentence Transformers)
- SearchAgent (constructed from above singletons)
- LLMProvider (pluggable LLM provider for grounded answer synthesis)
- AnswerSynthesizer (orchestrates citation-aware answer generation)

The EmbeddingProvider protocol remains the unified abstraction throughout the
ingestion and retrieval pipeline.
"""

from __future__ import annotations

import logging
import os
import time

try:
    from dotenv import load_dotenv
    _dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(_dotenv_path, override=False)
except ImportError:
    pass

from agents.search_agent import SearchAgent
from backend.embedding_provider import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL_NAME,
    RealEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from backend.llm import AnswerSynthesizer, LLMProvider, create_llm_provider
from ingestion.embedding_generator import EmbeddingProvider
from ingestion.qdrant_config import QdrantConfig
from ingestion.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)

# ── Configuration from environment ───────────────────────────────────────────

COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "omnibrain_documents")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", str(DEFAULT_DIMENSION)))

# LLM Configuration
LLM_PROVIDER_NAME: str = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL: str | None = os.getenv("LLM_MODEL")

# Track application start time for uptime reporting
APP_START_TIME: float = time.time()


# ── Singleton Builders ───────────────────────────────────────────────────────

def _build_qdrant_store() -> QdrantVectorStore:
    """Construct QdrantVectorStore from environment configuration."""
    config = QdrantConfig.from_env()
    logger.info("Qdrant config: url=%s collection=%s", config.url, config.default_collection)
    return QdrantVectorStore(config=config)


def _build_embedding_provider() -> EmbeddingProvider:
    """Construct the production RealEmbeddingProvider singleton."""
    provider = SentenceTransformerEmbeddingProvider(
        model_name=EMBEDDING_MODEL,
        dimension=EMBEDDING_DIMENSION,
        normalize_embeddings=True,
    )
    logger.info(
        "Embedding provider initialized: RealEmbeddingProvider(model='%s', dimension=%d)",
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSION,
    )
    return provider


def _build_search_agent(
    embedding_provider: EmbeddingProvider,
    store: QdrantVectorStore,
) -> SearchAgent:
    """Construct the SearchAgent using the injected singletons."""
    agent = SearchAgent(
        embedding_provider=embedding_provider,
        store=store,
        collection_name=COLLECTION_NAME,
        top_k=10,
        min_score=0.0,
        max_results=10,
        agent_name="OmnibrainSearchAgent",
        expected_dimension=EMBEDDING_DIMENSION,
    )
    logger.info("SearchAgent constructed with collection=%s", COLLECTION_NAME)
    return agent


def _build_llm_provider() -> LLMProvider | None:
    """Construct the production LLMProvider singleton."""
    try:
        provider = create_llm_provider(
            provider_type=LLM_PROVIDER_NAME,
            model_name=LLM_MODEL,
        )
        logger.info("LLMProvider constructed successfully (type=%s)", LLM_PROVIDER_NAME)
        return provider
    except Exception as exc:
        logger.warning("Could not initialize LLMProvider: %s", exc)
        return None


def _build_answer_synthesizer(llm_provider: LLMProvider | None) -> AnswerSynthesizer:
    """Construct the AnswerSynthesizer with the injected LLM provider."""
    return AnswerSynthesizer(llm_provider=llm_provider)


# Module-level singletons (constructed once at import / first access time)
_qdrant_store: QdrantVectorStore | None = None
_embedding_provider: EmbeddingProvider | None = None
_search_agent: SearchAgent | None = None
_llm_provider: LLMProvider | None = None
_answer_synthesizer: AnswerSynthesizer | None = None


def get_qdrant_store() -> QdrantVectorStore:
    """FastAPI dependency: return the shared QdrantVectorStore instance."""
    global _qdrant_store
    if _qdrant_store is None:
        _qdrant_store = _build_qdrant_store()
    return _qdrant_store


def get_embedding_provider() -> EmbeddingProvider:
    """FastAPI dependency: return the shared EmbeddingProvider instance."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = _build_embedding_provider()
    return _embedding_provider


def get_search_agent() -> SearchAgent:
    """FastAPI dependency: return the shared SearchAgent instance."""
    global _search_agent
    if _search_agent is None:
        _search_agent = _build_search_agent(
            get_embedding_provider(),
            get_qdrant_store(),
        )
    return _search_agent


def get_llm_provider() -> LLMProvider | None:
    """FastAPI dependency: return the shared LLMProvider instance (or None if not configured)."""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = _build_llm_provider()
    return _llm_provider


def get_answer_synthesizer() -> AnswerSynthesizer:
    """FastAPI dependency: return the shared AnswerSynthesizer instance."""
    global _answer_synthesizer
    if _answer_synthesizer is None:
        _answer_synthesizer = _build_answer_synthesizer(get_llm_provider())
    return _answer_synthesizer
