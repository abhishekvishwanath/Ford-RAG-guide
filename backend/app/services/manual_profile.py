"""The manual profile: discovered model years, covered variants and the procedure catalogue.

The readiness gate consults this *metadata* (procedure titles, sections and their
applicability) to decide which vehicle fields discriminate between procedures for the
requested component. It never reads manual body text and is not a retrieval step.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

VARIANT_TERM_RE = re.compile(r"\d\.\dl|\dv|gt500|coupe|convertible|shelby")
WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+l)?")
STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "the",
        "of",
        "for",
        "to",
        "with",
        "in",
        "on",
        "at",
        "my",
        "i",
        "need",
        "want",
        "replace",
        "remove",
        "install",
        "change",
        "how",
        "do",
        "is",
        "it",
        "front",
        "rear",
        "lh",
        "rh",
        "left",
        "right",
        "new",
        "old",
        "car",
        "vehicle",
        "mustang",
        "ford",
        "please",
        "help",
    ]
)
TASK_DOC_TYPES = {
    "replace": ("REMOVAL AND INSTALLATION", "IN-VEHICLE REPAIR", "REMOVAL", "INSTALLATION"),
    "removal": ("REMOVAL AND INSTALLATION", "IN-VEHICLE REPAIR", "REMOVAL"),
    "installation": ("REMOVAL AND INSTALLATION", "IN-VEHICLE REPAIR", "INSTALLATION"),
    "repair": ("IN-VEHICLE REPAIR", "REMOVAL AND INSTALLATION", "GENERAL PROCEDURES"),
    "disassembly": ("DISASSEMBLY", "ASSEMBLY", "DISASSEMBLY AND ASSEMBLY"),
    "diagnose": ("DIAGNOSIS AND TESTING",),
    "specification": ("SPECIFICATIONS",),
    "overview": ("DESCRIPTION AND OPERATION",),
    "adjust": ("GENERAL PROCEDURES",),
    "inspect": ("GENERAL PROCEDURES", "DIAGNOSIS AND TESTING"),
    "service": ("GENERAL PROCEDURES", "REMOVAL AND INSTALLATION"),
}


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def terms(text: str) -> set[str]:
    return {_stem(w) for w in WORD_RE.findall(text.lower()) if w not in STOP and len(w) > 1}


@dataclass
class CatalogueMatch:
    parent_id: str
    title: str
    section: str
    doc_type: str
    score: float
    engines: list[str] = field(default_factory=list)
    transmissions: list[str] = field(default_factory=list)
    body_styles: list[str] = field(default_factory=list)
    equipment_conditions: list[str] = field(default_factory=list)


class ManualProfile:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.model_years: list[int] = data.get("model_years", [])
        self.model_name: str | None = data.get("model_name")
        cov = data.get("covered", {})
        self.engines: list[str] = cov.get("engines", [])
        self.transmissions: list[str] = cov.get("transmissions", [])
        self.body_styles: list[str] = cov.get("body_styles", [])
        self.procedures: list[dict] = data.get("procedures", [])
        for p in self.procedures:
            p["_title_terms"] = terms(p.get("title") or "")
            p["_section_terms"] = terms(" ".join(filter(None, [p.get("section_name"), *p.get("subsections", [])[:6]])))

    @classmethod
    def load(cls, path: Path) -> ManualProfile:
        return cls(json.loads(path.read_text()))

    @property
    def available(self) -> bool:
        return bool(self.model_years)

    def covers_year(self, year: int) -> bool:
        return year in self.model_years

    def match_component(self, text: str, task_type: str | None = None, limit: int = 12) -> list[CatalogueMatch]:
        q = terms(text)
        if not q or all(VARIANT_TERM_RE.fullmatch(t) for t in q):
            return []  # "5.0L" alone names a variant, not a component
        wanted = TASK_DOC_TYPES.get(task_type or "", ())
        out: list[CatalogueMatch] = []
        for p in self.procedures:
            tt, st = p["_title_terms"], p["_section_terms"]
            core = {t for t in tt if not VARIANT_TERM_RE.fullmatch(t)} or tt
            title_hit = len(q & tt) / len(q)  # how much of the query the title explains
            title_cov = len(q & core) / len(core) if core else 0.0  # how much of the title was mentioned
            section_hit = len(q & st) / len(q)
            score = max(title_hit, 0.6 * section_hit, 0.95 * title_cov if title_cov == 1.0 else 0.6 * title_cov)
            if title_hit and len(q & core) == len(core):
                score += 0.2  # every (non-variant) title word was mentioned
            if score < 0.5:
                continue
            if wanted and (p.get("doc_type") or "").upper() in wanted:
                score += 0.1
            out.append(
                CatalogueMatch(
                    parent_id=p["parent_id"],
                    title=p.get("title") or "",
                    section=" ".join(filter(None, [p.get("section_code"), p.get("section_name")])),
                    doc_type=p.get("doc_type") or "",
                    score=round(score, 3),
                    engines=p.get("engines", []),
                    transmissions=[t for t in p.get("transmissions", []) if t in ("automatic", "manual")],
                    body_styles=p.get("body_styles", []),
                    equipment_conditions=p.get("equipment_conditions", []),
                )
            )
        out.sort(key=lambda m: -m.score)
        if out:
            top = out[0].score
            out = [m for m in out if m.score >= top - 0.25]
        return out[:limit]


@lru_cache
def load_profile(path: str) -> ManualProfile:
    p = Path(path)
    if not p.exists():
        return ManualProfile({})
    return ManualProfile.load(p)
