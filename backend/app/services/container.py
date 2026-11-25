"""Application container: builds long-lived services once and hot-reloads indexes."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict

from app.core.config import Settings
from app.core.exceptions import LLMUnavailableError
from app.core.logging import log_event
from app.generation.answer_generator import GroundedAnswerGenerator
from app.generation.llm_client import GroqLLMClient, LLMClient
from app.graph.state import GraphDeps
from app.graph.workflow import build_workflow
from app.ingestion.ingestion_service import IngestionService
from app.memory.session_store import SessionRepository, SqliteSessionRepository
from app.schemas.retrieval import Candidate, EvidencePackage, RetrievalRequest
from app.services.manual_profile import ManualProfile
from app.services.manual_repository import ManualRepository

logger = logging.getLogger(__name__)


class UnavailableRetrieval:
    """Used when the manual has not been indexed: every READY request ends in an honest refusal."""

    def build_request(self, state, message, extraction) -> RetrievalRequest:  # type: ignore[no-untyped-def]
        from app.retrieval.pipeline import build_request

        return build_request(state, message, extraction)

    def applicability_filter(self, request: RetrievalRequest) -> dict:
        return {}

    def dense(self, request: RetrievalRequest, flt: dict) -> list[Candidate]:
        return []

    def bm25(self, request: RetrievalRequest, flt: dict) -> list[Candidate]:
        return []

    def fuse(self, dense: list[Candidate], bm25: list[Candidate]) -> list[Candidate]:
        return []

    def rerank(self, request: RetrievalRequest, candidates: list[Candidate]) -> list[Candidate]:
        return []

    def expand(self, request: RetrievalRequest, candidates: list[Candidate]) -> EvidencePackage:
        return EvidencePackage(request=request)

    def visuals_and_tables(self, package: EvidencePackage) -> EvidencePackage:
        return package

    def assess(self, package: EvidencePackage) -> EvidencePackage:
        package.sufficient = False
        package.insufficiency_reason = "the manual index has not been built yet (run ingestion)"
        return package


class Container:
    def __init__(self, settings: Settings, *, llm: LLMClient | None = None, session_repo: SessionRepository | None = None,
                 load_indexes: bool = True) -> None:  # fmt: skip
        self.settings = settings
        self.sessions = session_repo or SqliteSessionRepository(
            settings.database_url, settings.session_message_limit, settings.session_expiry_days
        )
        self.llm = llm if llm is not None else self._make_llm()
        self.ingestion = IngestionService(settings)
        self.session_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._reload_lock = threading.Lock()
        self.manual: ManualRepository = ManualRepository([], [], [])
        self.profile = ManualProfile({})
        self.retrieval = UnavailableRetrieval()
        self.index_ready = False
        if load_indexes:
            self.reload_indexes()
        self.workflow = self._build_workflow()

    def _make_llm(self) -> LLMClient | None:
        try:
            return GroqLLMClient(self.settings)
        except LLMUnavailableError:
            log_event(logger, "LLM not configured; answers fall back to verbatim manual excerpts", logging.WARNING)
            return None

    def _build_workflow(self):  # type: ignore[no-untyped-def]
        title = self.document_title
        generator = GroundedAnswerGenerator(self.settings, self.llm, self.sessions, title)
        deps = GraphDeps(settings=self.settings, repo=self.sessions, profile=self.profile, retrieval=self.retrieval,
                         generator=generator, llm=self.llm)  # fmt: skip
        return build_workflow(deps)

    @property
    def document_title(self) -> str:
        any_chunk = next(iter(self.manual.chunks.values()), None)
        return any_chunk.document_title if any_chunk else "Workshop Manual"

    def reload_indexes(self) -> None:
        """(Re)load profile, chunks, vector store, BM25 and reranker. Safe to call after ingestion."""
        with self._reload_lock:
            s = self.settings
            from app.services.manual_profile import load_profile

            load_profile.cache_clear()
            self.profile = load_profile(str(s.manual_profile_path))
            self.manual = ManualRepository.load(s.processed_dir)
            bm25_path = s.indexes_dir / "bm25_corpus.jsonl"
            if not self.manual.ready or not bm25_path.exists() or not self.profile.available:
                self.index_ready = False
                self.retrieval = UnavailableRetrieval()
                log_event(logger, "manual index not ready; retrieval disabled", logging.WARNING)
                return
            from app.embeddings.embedding_service import SentenceTransformerEmbedder
            from app.retrieval.bm25_search import BM25Index
            from app.retrieval.pipeline import HybridRetrievalPipeline, RetrievalConfig
            from app.retrieval.reranker import CrossEncoderReranker, NoopReranker
            from app.vectorstore.chroma_store import ChromaStore

            store = ChromaStore(s.chroma_persist_directory, s.chroma_collection_name,
                                SentenceTransformerEmbedder(s.embedding_model, s.embedding_batch_size))  # fmt: skip
            reranker = CrossEncoderReranker(s.reranker_model) if s.reranker_enabled else NoopReranker()
            self.retrieval = HybridRetrievalPipeline(RetrievalConfig.from_settings(s), self.manual, store,
                                                     BM25Index.load(bm25_path), reranker)  # fmt: skip
            self.index_ready = store.count() > 0
            log_event(logger, "indexes loaded", vectors=store.count(), **self.manual.stats())
        if hasattr(self, "workflow"):
            self.workflow = self._build_workflow()

    def warmup(self) -> None:
        """Load embedding/reranker models so the first request is not slow."""
        r = self.retrieval
        if self.index_ready and hasattr(r, "store") and r.store is not None:
            r.store.embedder.embed(["warmup"])
            r.reranker.score("warmup", ["warmup"])
