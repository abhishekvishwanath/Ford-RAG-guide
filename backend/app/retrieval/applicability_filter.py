"""Vehicle applicability filtering.

* Explicitly incompatible variants (a 3.7L-only chunk for a 5.0L vehicle) are excluded.
* Chunks that are not variant-specific are kept and labelled "general".
* A chunk whose variant metadata is unknown for a dimension the vehicle specifies is kept
  but labelled "uncertain" (lenient mode) — it is never presented as confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.metadata import flag_key
from app.schemas.chunk import Chunk
from app.schemas.retrieval import RetrievalRequest


@dataclass(frozen=True)
class VehicleFilter:
    model_year: int | None
    engine: str | None
    transmission: str | None
    body_style: str | None
    strict: bool = True

    @classmethod
    def from_request(cls, req: RetrievalRequest, strictness: str = "strict") -> VehicleFilter:
        return cls(req.model_year, req.engine, req.transmission, req.body_style, strictness == "strict")

    def chroma_where(self) -> dict | None:
        clauses: list[dict] = []
        if self.engine:
            clauses.append({"$or": [{"engine_any": True}, {f"eng_{flag_key(self.engine)}": True}]})
        if self.transmission:
            clauses.append({"$or": [{"trans_any": True}, {f"trans_{flag_key(self.transmission)}": True}]})
        if self.body_style:
            clauses.append({"$or": [{"body_any": True}, {f"body_{flag_key(self.body_style)}": True}]})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def classify(self, chunk: Chunk) -> str:
        """matches | general | incompatible"""
        a = chunk.applicability
        if self.model_year and a.model_years and self.model_year not in a.model_years:
            return "incompatible"
        specific = False
        for want, have in (
            (self.engine, a.engines),
            (self.transmission, a.transmissions),
            (self.body_style, a.body_styles),
        ):
            if not have:
                continue
            specific = specific or bool(want)
            if want and want not in have:
                return "incompatible"
        return "matches" if specific else "general"

    def allows(self, chunk: Chunk) -> bool:
        return self.classify(chunk) != "incompatible"

    def note(self, chunk: Chunk) -> str:
        a = chunk.applicability
        parts = []
        if a.model_years:
            parts.append("MY " + "/".join(map(str, a.model_years)))
        if a.engines:
            parts.append("engines: " + ", ".join(a.engines))
        if a.transmissions:
            parts.append("transmission: " + ", ".join(a.transmissions))
        if a.body_styles:
            parts.append("body: " + ", ".join(a.body_styles))
        if a.equipment_conditions:
            parts.append("conditions in text: " + "; ".join(a.equipment_conditions[:3]))
        if a.exclusions:
            parts.append("excludes: " + "; ".join(a.exclusions[:2]))
        status = self.classify(chunk)
        label = {
            "matches": "matches your vehicle",
            "general": "not variant-specific",
            "incompatible": "NOT for your vehicle",
        }[status]
        return f"{label} ({'; '.join(parts) or 'no variant qualifiers'})"
