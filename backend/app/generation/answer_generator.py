"""Grounded answer generation (implements the graph's AnswerGenerator protocol)."""

from __future__ import annotations

import re

from app.conversation.question_planner import describe_context
from app.conversation.state import SessionState
from app.core.config import Settings
from app.generation import response_validator as rv
from app.generation.llm_client import LLMClient
from app.generation.prompts import build_messages
from app.memory.session_store import SessionRepository
from app.schemas.retrieval import Citation, EvidenceItem, EvidencePackage, GeneratedAnswer


def estimate_tokens(messages: list[dict]) -> int:
    """Conservative estimate for the provider tokenizer (technical text ~3.2 chars/token)."""
    return int(sum(len(m["content"]) for m in messages) / 3.2) + 16 * len(messages)


FALLBACK_STOP = {"what", "which", "the", "and", "for", "how", "should", "they", "have", "does", "with", "this",
                 "that", "there", "need", "torque", "spec", "specs", "specification"}  # fmt: skip


def _qwords(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", text.lower())} - FALLBACK_STOP


def fallback_items(package: EvidencePackage, limit: int = 4) -> list[EvidenceItem]:
    """Excerpts most related to the question: main procedure first, then by word overlap, in manual order."""
    if not package.items:
        return []
    q = _qwords(package.request.original_message)
    main = package.items[0].parent_id

    def score(i: EvidenceItem) -> float:
        words = _qwords(f"{i.procedure_title} {i.subsection} {i.content}")
        return len(q & words) + (2 if i.parent_id == main else 0)

    ranked = sorted(package.items, key=score, reverse=True)[:limit]
    return [i for i in package.items if i in ranked]


def focus_table_rows(content: str, question: str, max_lines: int = 14) -> str:
    """Long spec tables: keep header rows and the rows that mention the question's words."""
    lines = content.splitlines()
    if len(lines) <= max_lines or sum(" | " in ln for ln in lines) < len(lines) / 2:
        return content
    q = _qwords(question)
    keep = [
        ln for ln in lines
        if " | " not in ln or ln.startswith("#") or re.search(r"\b(Description|Item)\b \|", ln) or q & _qwords(ln)
    ]  # fmt: skip
    return "\n".join(keep) if len(keep) < len(lines) else content


class GroundedAnswerGenerator:
    def __init__(self, settings: Settings, llm: LLMClient | None, repo: SessionRepository, document_title: str) -> None:
        self.s = settings
        self.llm = llm
        self.repo = repo
        self.document_title = document_title

    def build_prompt(self, state: SessionState, message: str, package: EvidencePackage) -> list[dict]:
        """Build the grounded prompt, trimming lowest-priority evidence so prompt + output fit the provider's
        per-request token cap. Trimmed items are removed from the package so citations stay consistent."""
        history = self.repo.recent_messages(state.session_id, 6)[:-1]  # exclude the current user turn
        summary = describe_context(state)
        cap = int(self.s.llm_tpm_limit * 0.92) - self.s.llm_max_output_tokens
        msgs = build_messages(self.document_title, summary, message, history, package)
        main = package.items[0].parent_id if package.items else None
        while estimate_tokens(msgs) > cap and len(package.items) > 1:
            others = [i for i in package.items if i.parent_id != main]
            victim = others[-1] if others else package.items[-1]
            package.items.remove(victim)
            package.images = [im for im in package.images if im.source_id != victim.source_id]
            package.tables = [t for t in package.tables if t.source_id != victim.source_id]
            msgs = build_messages(self.document_title, summary, message, history, package)
        return msgs

    def generate(self, messages: list[dict]) -> str:
        if self.llm is None:
            from app.core.exceptions import LLMUnavailableError

            raise LLMUnavailableError("No language model configured.")
        return self.llm.complete(messages, model=self.s.llm_model, reasoning_effort=self.s.llm_reasoning_effort)

    def validate(self, text: str, package: EvidencePackage) -> GeneratedAnswer:
        ans = rv.validate(text, package)
        ans.model = self.s.llm_model
        return ans

    def citations(self, answer: GeneratedAnswer, package: EvidencePackage) -> list[Citation]:
        return rv.build_citations(answer, package, self.document_title)

    def fallback_answer(self, package: EvidencePackage) -> GeneratedAnswer:
        """Verbatim manual excerpts — used when the model is unavailable or its answer failed validation."""
        parts = [
            "I couldn't produce a verified summary, so here are the exact manual excerpts for your vehicle. "
            "Follow them in order and check the cited pages."
        ]
        ids = []
        for item in fallback_items(package):
            head = f"**[{item.source_id}] {item.procedure_title or ''} — {item.subsection or 'procedure'}** "
            head += f"(PDF p. {item.page_start}{'-' + str(item.page_end) if item.page_end != item.page_start else ''})"
            body = focus_table_rows(item.content, package.request.original_message)
            parts.append(f"{head}\n_Applicability: {item.applicability_note}_\n\n{body}")
            ids.append(item.source_id)
        return GeneratedAnswer(text="\n\n".join(parts), cited_source_ids=ids, grounded=True,
                               validation_warnings=["verbatim excerpt fallback"])  # fmt: skip
