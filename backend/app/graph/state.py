"""LangGraph state and injected dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from app.conversation.context_extractor import Extraction
from app.conversation.state import PendingQuestion, ReadinessDecision, SessionState
from app.core.config import Settings
from app.generation.llm_client import LLMClient
from app.memory.session_store import SessionRepository
from app.schemas.retrieval import Candidate, Citation, EvidencePackage, GeneratedAnswer, RetrievalRequest
from app.services.manual_profile import ManualProfile


class RetrievalPipeline(Protocol):
    """Stage-wise retrieval API. Every method must only be reachable after READY."""

    def build_request(self, state: SessionState, message: str, extraction: Extraction) -> RetrievalRequest: ...
    def applicability_filter(self, request: RetrievalRequest) -> dict: ...
    def dense(self, request: RetrievalRequest, flt: dict) -> list[Candidate]: ...
    def bm25(self, request: RetrievalRequest, flt: dict) -> list[Candidate]: ...
    def fuse(self, dense: list[Candidate], bm25: list[Candidate]) -> list[Candidate]: ...
    def rerank(self, request: RetrievalRequest, candidates: list[Candidate]) -> list[Candidate]: ...
    def expand(self, request: RetrievalRequest, candidates: list[Candidate]) -> EvidencePackage: ...
    def visuals_and_tables(self, package: EvidencePackage) -> EvidencePackage: ...
    def assess(self, package: EvidencePackage) -> EvidencePackage: ...


class AnswerGenerator(Protocol):
    def build_prompt(self, state: SessionState, message: str, package: EvidencePackage) -> list[dict]: ...
    def generate(self, messages: list[dict]) -> str: ...
    def validate(self, text: str, package: EvidencePackage) -> GeneratedAnswer: ...
    def citations(self, answer: GeneratedAnswer, package: EvidencePackage) -> list[Citation]: ...
    def fallback_answer(self, package: EvidencePackage) -> GeneratedAnswer: ...


@dataclass
class GraphDeps:
    settings: Settings
    repo: SessionRepository
    profile: ManualProfile
    retrieval: RetrievalPipeline | None
    generator: AnswerGenerator | None
    llm: LLMClient | None = None  # used for context extraction


class GraphState(TypedDict, total=False):
    session_id: str | None
    message: str
    session: SessionState
    history: list[dict]
    extraction: Extraction
    events: list[str]
    decision: ReadinessDecision
    question: PendingQuestion | None
    request: RetrievalRequest
    filter: dict
    dense: list[Candidate]
    bm25: list[Candidate]
    candidates: list[Candidate]
    package: EvidencePackage
    prompt: list[dict]
    raw_answer: str
    answer: GeneratedAnswer
    citations: list[Citation]
    assistant_message: str
    interaction_mode: str
    trace: list[str]
    error: str | None
    response: dict[str, Any]
