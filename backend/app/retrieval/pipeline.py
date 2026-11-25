"""Hybrid retrieval pipeline (implements the graph's RetrievalPipeline protocol).

query -> applicability filter -> dense (Chroma, multi-window) + BM25 -> weighted RRF
-> cross-encoder rerank (+ task-type / follow-up / applicability boosts) -> procedure
expansion -> original figures & tables -> evidence assessment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.conversation.context_extractor import Extraction, manual_terms_for
from app.conversation.state import SessionState
from app.core.config import Settings
from app.retrieval.applicability_filter import VehicleFilter
from app.retrieval.bm25_search import BM25Index
from app.retrieval.context_expansion import ExpansionConfig, detect_variant_ambiguity, expand
from app.retrieval.fusion import rrf_fuse
from app.retrieval.reranker import NoopReranker, Reranker
from app.retrieval.visual_retrieval import retrieve_images, retrieve_tables
from app.schemas.retrieval import Candidate, EvidencePackage, RetrievalRequest
from app.services.manual_repository import ManualRepository
from app.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

STEP_REF_RE = re.compile(r"\bstep\s*(\d{1,3})\b", re.I)
IDENT_RE = re.compile(
    r"\b(?:[A-Z]{1,3}\d{3,}[A-Z0-9-]*|\d{3}-[A-Z]?\d{3}[A-Z0-9-]*|[BCPU][0-9A-F]{4}(?::[0-9A-F]{2})?)\b"
)
TASK_QUERY_TERMS = {
    "replace": "removal installation",
    "removal": "removal",
    "installation": "installation",
    "diagnose": "diagnosis testing symptom pinpoint test",
    "specification": "specifications torque",
    "inspect": "inspection check",
    "adjust": "adjustment",
    "service": "draining filling bleeding",
    "overview": "description operation",
    "repair": "repair",
    "disassembly": "disassembly assembly",
}
TASK_PHRASES = {
    "replace": "removal and installation",
    "removal": "removal",
    "installation": "installation",
    "diagnose": "diagnosis and testing",
    "specification": "specifications",
    "inspect": "inspection",
    "adjust": "adjustment",
    "service": "draining and filling",
    "overview": "description and operation",
    "repair": "repair",
    "disassembly": "disassembly and assembly",
}
PREFERRED_TYPES = {
    "replace": ["removal", "installation", "procedure"],
    "removal": ["removal", "procedure"],
    "installation": ["installation", "procedure"],
    "diagnose": ["diagnostic", "troubleshooting"],
    "specification": ["torque_table", "specification", "table"],
    "inspect": ["inspection", "procedure", "diagnostic"],
    "adjust": ["adjustment", "procedure"],
    "service": ["procedure"],
    "overview": ["overview"],
}
TORQUE_Q_RE = re.compile(r"\b(torque|tighten|nm|lb-?ft|lb-?in)\b", re.I)


@dataclass
class RetrievalConfig:
    top_k: int = 5
    dense_candidates: int = 30
    bm25_candidates: int = 30
    rrf_k: int = 60
    w_dense: float = 1.0
    w_bm25: float = 1.0
    rerank_candidates: int = 20
    min_rerank_score: float = 0.2  # sigmoid(cross-encoder logit); calibrated in docs/evaluation_report.md
    min_dense_score: float = 0.45  # relevance floor when no cross-encoder is configured
    strictness: str = "strict"
    max_images: int = 6
    expansion: ExpansionConfig | None = None

    @classmethod
    def from_settings(cls, s: Settings) -> RetrievalConfig:
        return cls(
            top_k=s.retrieval_top_k,
            dense_candidates=s.dense_candidates,
            bm25_candidates=s.bm25_candidates,
            rrf_k=s.rrf_k,
            w_dense=s.rrf_dense_weight,
            w_bm25=s.rrf_bm25_weight,
            rerank_candidates=s.rerank_candidates,
            strictness=s.applicability_strictness,
            max_images=s.max_images_per_answer,
            expansion=ExpansionConfig(s.retrieval_top_k, s.max_expanded_chunks, s.evidence_token_budget),
        )


def build_request(state: SessionState, message: str, extraction: Extraction | None) -> RetrievalRequest:
    """Structured query from confirmed context + user wording. Synonyms are added, facts never altered.

    Vehicle variants (engine/transmission/body) are *filters*, not query terms: adding "5.0L" to the query
    would bias ranking toward engine-section content for components that are not engine-specific."""
    comp = state.value("component")
    task = state.value("task_type")
    terms = list(extraction.manual_terms) if extraction and extraction.manual_terms else []
    if comp:
        terms += [t for t in manual_terms_for(comp) if t not in terms]
    is_followup = bool(extraction and extraction.is_followup and state.active_procedure)
    focus = state.active_procedure if is_followup else None
    steps = [int(s) for s in STEP_REF_RE.findall(message)]
    dtcs = state.value("dtcs") or []
    engine = state.value("engine")
    sem_parts = [message]
    if focus:
        sem_parts.insert(0, " ".join(filter(None, [focus.title, focus.subsection])))
    if comp and comp.lower() not in message.lower():
        sem_parts.append(comp)
    sem_parts += terms
    if task and not focus:
        sem_parts.append(TASK_QUERY_TERMS.get(task, ""))
    kw_parts = [message, *terms, *dtcs, *IDENT_RE.findall(message)]
    if focus and focus.title:
        kw_parts.append(focus.title)
    if comp:
        kw_parts.append(comp)
    if task:
        kw_parts.append(TASK_QUERY_TERMS.get(task, ""))
    preferred = PREFERRED_TYPES.get(task or "", [])
    if TORQUE_Q_RE.search(message):
        preferred = ["torque_table", "specification", *preferred]
        if focus:
            preferred = ["installation", *preferred]
    return RetrievalRequest(
        query_text=" ".join(p for p in sem_parts if p).strip(),
        keyword_query=" ".join(p for p in kw_parts if p).strip(),
        original_message=message,
        model_year=state.value("model_year"),
        engine=engine,
        transmission=state.value("transmission"),
        body_style=state.value("body_style"),
        task_type=task,
        component=comp,
        manual_terms=terms,
        dtcs=dtcs,
        focus_parent_id=focus.parent_id if focus else None,
        focus_steps=steps,
        preferred_chunk_types=preferred,
    )


class HybridRetrievalPipeline:
    def __init__(self, cfg: RetrievalConfig, repo: ManualRepository, store: ChromaStore | None, bm25: BM25Index,
                 reranker: Reranker | None = None) -> None:  # fmt: skip
        self.cfg = cfg
        self.repo = repo
        self.store = store
        self.bm25_index = bm25
        self.reranker = reranker or NoopReranker()
        self._hash = {cid: c.content_hash for cid, c in repo.chunks.items()}

    # -- protocol ---------------------------------------------------------------
    def build_request(self, state: SessionState, message: str, extraction: Extraction) -> RetrievalRequest:
        return build_request(state, message, extraction)

    def applicability_filter(self, request: RetrievalRequest) -> dict:
        flt = VehicleFilter.from_request(request, self.cfg.strictness)
        return {"vehicle": flt, "where": flt.chroma_where()}

    def dense(self, request: RetrievalRequest, flt: dict) -> list[Candidate]:
        if self.store is None:
            return []
        hits = self.store.search(request.query_text, self.cfg.dense_candidates, where=flt["where"])
        vf: VehicleFilter = flt["vehicle"]
        out = []
        for h in hits:
            ch = self.repo.chunk(h.chunk_id)
            if ch is None or not vf.allows(ch):
                continue
            out.append(Candidate(chunk_id=h.chunk_id, dense_score=round(h.score, 4), dense_rank=len(out) + 1))
        return out

    def bm25(self, request: RetrievalRequest, flt: dict) -> list[Candidate]:
        vf: VehicleFilter = flt["vehicle"]

        def allow(cid: str) -> bool:
            ch = self.repo.chunk(cid)
            return ch is not None and vf.allows(ch)

        hits = self.bm25_index.search(request.keyword_query, self.cfg.bm25_candidates, allow=allow)
        return [Candidate(chunk_id=h.chunk_id, bm25_score=round(h.score, 3), bm25_rank=i + 1)
                for i, h in enumerate(hits)]  # fmt: skip

    def fuse(self, dense: list[Candidate], bm25: list[Candidate]) -> list[Candidate]:
        return rrf_fuse(dense, bm25, self.cfg.rrf_k, self.cfg.w_dense, self.cfg.w_bm25, self._hash)

    @staticmethod
    def rerank_queries(request: RetrievalRequest) -> list[str]:
        """Cross-encoders score far better with manual terminology: rewrite user wording with synonyms
        and add a structured '<manual term> <task> <engine>' query. Facts are never changed."""
        msg = request.original_message
        term = request.manual_terms[0] if request.manual_terms else request.component
        if request.component and term and request.component != term:
            msg = re.sub(re.escape(request.component), term, msg, flags=re.I)
        if request.focus_parent_id and request.query_text:
            msg = f"{request.query_text.split(request.original_message)[0].strip()}: {msg}".strip(": ")
        structured = " ".join(filter(None, [term, TASK_PHRASES.get(request.task_type or "", "")]))
        return [q for q in dict.fromkeys([msg, structured]) if q]

    def rerank(self, request: RetrievalRequest, candidates: list[Candidate]) -> list[Candidate]:
        pool = candidates[: self.cfg.rerank_candidates]
        passages = []
        for c in pool:
            ch = self.repo.chunk(c.chunk_id)
            passages.append(f"{ch.context_header}\n{ch.content}"[:2000] if ch else "")
        active = self.reranker.name != "none"
        if active:
            per_query = [self.reranker.score(q, passages) for q in self.rerank_queries(request)]
            scores = [max(vals) for vals in zip(*per_query, strict=True)] if per_query else [0.0] * len(pool)
        else:
            scores = [0.0] * len(pool)
        max_rrf = max((c.rrf_score for c in pool), default=1.0) or 1.0
        for c, s in zip(pool, scores, strict=True):
            ch = self.repo.chunk(c.chunk_id)
            boost = 0.0
            if ch is not None:
                if ch.chunk_type in request.preferred_chunk_types:
                    boost += 0.1 * (1 - request.preferred_chunk_types.index(ch.chunk_type) * 0.2)
                if request.focus_parent_id and ch.parent_id == request.focus_parent_id:
                    boost += 0.3
                if (
                    request.focus_steps
                    and ch.step_range
                    and any(ch.step_range[0] <= st <= ch.step_range[1] for st in request.focus_steps)
                ):
                    boost += 0.2
            c.rerank_raw = round(s, 4) if active else None
            c.rerank_score = round((s if active else c.rrf_score / max_rrf) + boost, 4)
        return sorted(pool, key=lambda c: -(c.rerank_score or 0.0)) + candidates[self.cfg.rerank_candidates :]

    def expand(self, request: RetrievalRequest, candidates: list[Candidate]) -> EvidencePackage:
        flt = VehicleFilter.from_request(request, self.cfg.strictness)
        cfg = self.cfg.expansion or ExpansionConfig(self.cfg.top_k)
        items = expand(request, candidates, self.repo, flt, cfg)
        pkg = EvidencePackage(request=request, items=items, candidates=candidates[: self.cfg.top_k * 2])
        pkg.variant_ambiguity = detect_variant_ambiguity(candidates[: self.cfg.top_k], self.repo, flt)
        pkg.token_estimate = sum(self.repo.chunk(i.chunk_id).token_count for i in items)  # type: ignore[union-attr]
        return pkg

    def visuals_and_tables(self, package: EvidencePackage) -> EvidencePackage:
        package.images = retrieve_images(package.items, self.repo, self.cfg.max_images)
        package.tables = retrieve_tables(package.items, self.repo)
        return package

    def assess(self, package: EvidencePackage) -> EvidencePackage:
        if not package.items:
            package.sufficient, package.insufficiency_reason = False, "no applicable manual content matched"
            return package
        top = package.candidates[0] if package.candidates else None
        if (
            top is not None
            and top.rerank_raw is None
            and not top.bm25_rank
            and (top.dense_score or 0) < self.cfg.min_dense_score
        ):
            package.sufficient = False
            package.insufficiency_reason = "no manual passage matched the request closely enough"
            return package
        if top is not None and top.rerank_raw is not None and top.rerank_raw < self.cfg.min_rerank_score:
            package.sufficient = False
            package.insufficiency_reason = "the best-matching manual passages were not relevant enough"
            return package
        # the main procedure differs only by a variant the user has not specified -> ask, don't blend
        main_titles = {i.procedure_title for i in package.items[:2]}
        if (
            package.variant_ambiguity
            and len(set(package.variant_ambiguity) & main_titles)
            and not package.request.focus_parent_id
        ):
            package.sufficient = False
            package.insufficiency_reason = "multiple manual variants match"
            return package
        package.variant_ambiguity = []
        package.sufficient = True
        return package
