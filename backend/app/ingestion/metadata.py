"""Applicability metadata extraction from document *structure* (section, title, qualifier lines).

Body text (e.g. "remove the 5.0L engine cover") is deliberately not used: a procedure's
applicability is established by how the manual scopes it, not by passing mentions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.schemas.chunk import Applicability

VOCAB_PATH = PROJECT_ROOT / "config/applicability_vocabulary.json"


@dataclass(frozen=True)
class Vocabulary:
    engine_re: re.Pattern
    variants: dict[str, dict]
    transmission_terms: dict[str, list[str]]
    transmission_codes: list[str]
    body_terms: dict[str, list[str]]
    raw: dict


@lru_cache
def load_vocabulary(path: Path = VOCAB_PATH) -> Vocabulary:
    raw = json.loads(path.read_text())
    return Vocabulary(
        engine_re=re.compile(raw["engine_pattern"], re.I),
        variants=raw["variants"],
        transmission_terms=raw["transmission_terms"],
        transmission_codes=raw["transmission_codes"],
        body_terms=raw["body_terms"],
        raw=raw,
    )


def _has_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, re.I) is not None


def parse_phrase(text: str, vocab: Vocabulary | None = None) -> Applicability:
    """Normalise one applicability phrase (no source attribution)."""
    vocab = vocab or load_vocabulary()
    app = Applicability()
    if re.match(r"^\s*vehicles?\s+without\b", text, re.I):
        app.exclusions.append(text.strip())
        return app
    app.engines = sorted({f"{m.group(1)}L" for m in vocab.engine_re.finditer(text)})
    for name, spec in vocab.variants.items():
        if any(_has_term(text, t) for t in spec["terms"]):
            app.variants.append(name)
            for e in spec.get("implies", {}).get("engines", []):
                if e not in app.engines:
                    app.engines.append(e)
    trans = {kind for kind, terms in vocab.transmission_terms.items() if any(_has_term(text, t) for t in terms)}
    codes = [c for c in vocab.transmission_codes if _has_term(text, c)]
    app.transmissions = sorted(trans | set(codes))
    app.body_styles = sorted(b for b, terms in vocab.body_terms.items() if any(_has_term(text, t) for t in terms))
    return app


def merge(*apps: Applicability) -> Applicability:
    out = Applicability()
    for a in apps:
        for field in (
            "model_years",
            "engines",
            "transmissions",
            "body_styles",
            "variants",
            "equipment_conditions",
            "exclusions",
            "sources",
            "raw_qualifiers",
        ):
            cur = getattr(out, field)
            for v in getattr(a, field):
                if v not in cur:
                    cur.append(v)
    return out


def from_source(text: str | None, source: str) -> Applicability:
    if not text:
        return Applicability()
    app = parse_phrase(text)
    if app.engines or app.transmissions or app.body_styles or app.variants or app.exclusions:
        app.sources.append(source)
    return app


def from_qualifier(text: str) -> Applicability:
    """A qualifier line ('Vehicles with 6 piston calipers'). Unnormalised conditions are kept verbatim."""
    app = parse_phrase(text)
    app.raw_qualifiers.append(text)
    app.sources.append("qualifier")
    is_all = re.match(r"^\s*all vehicles\s*$", text, re.I) is not None
    normalised = app.engines or app.transmissions or app.body_styles or app.variants or app.exclusions
    if not normalised and not is_all:
        app.equipment_conditions.append(text.strip())
    return app


def procedure_applicability(
    model_years: list[int], toc_path: list[str], section_name: str | None, title: str | None, doc_type: str | None
) -> Applicability:
    doc_level = Applicability(model_years=list(model_years), sources=["document"] if model_years else [])
    section_text = " | ".join([*toc_path[:-1], section_name or ""])
    return merge(
        doc_level,
        from_source(section_text, "section"),
        from_source(" | ".join(filter(None, [title, doc_type])), "title"),
    )


def flatten_for_vector_store(app: Applicability) -> dict[str, bool | str]:
    """Chroma metadata must be scalar: boolean flags per variant + 'any' flags for non-specific chunks."""
    meta: dict[str, bool | str] = {
        "engine_any": not app.engines,
        "trans_any": not app.transmissions,
        "body_any": not app.body_styles,
        "model_years": ",".join(str(y) for y in app.model_years),
        "equipment_conditions": " ; ".join(app.equipment_conditions)[:500],
        "exclusions": " ; ".join(app.exclusions)[:500],
    }
    for e in app.engines:
        meta[f"eng_{flag_key(e)}"] = True
    for t in app.transmissions:
        meta[f"trans_{flag_key(t)}"] = True
    for b in app.body_styles:
        meta[f"body_{flag_key(b)}"] = True
    return meta


def flag_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
