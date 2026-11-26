"""Post-generation validation: citations, numeric specs, identifiers, safety statements.

* Citation markers must refer to evidence blocks; unknown markers are removed.
* Every value-with-unit and every part/tool/DTC identifier in the answer must occur in the
  evidence (or the user's question). Unsupported values make the answer ungrounded -> the
  caller falls back to verbatim manual excerpts.
* WARNING/CAUTION statements from cited evidence that the answer dropped are re-inserted
  verbatim at the top (safety statements are never lost to summarisation).
"""

from __future__ import annotations

import re

from app.schemas.retrieval import Citation, EvidencePackage, GeneratedAnswer

CITE_RE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
NORMALISE = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u00a0": " ", "\u202f": " ", "\u2009": " "})
VALUE_RE = re.compile(
    r"(?<![\w.-])(\d+(?:\.\d+)?)\s?(Nm|N·m|N-m|lb-ft|lb-in|mm|cm|in\.?(?=\s|\)|,|$)|L\b|ml|qt|quarts?|psi|kPa|bar|V\b|volts?|"
    r"ohms?|Ω|°C|°F|rpm|%|seconds?|minutes?|amps?|A\b)",
    re.I,
)
IDENT_RE = re.compile(
    r"\b(?:[A-Z]{1,2}\d{5,}[A-Z0-9-]*|\d{3}-[A-Z]{0,2}\d{3}[A-Z0-9-]*|[BCPU][0-9A-F]{4}(?::[0-9A-F]{2})?)\b"
)
SAFETY_RE = re.compile(r"((?:WARNING|CAUTION|DANGER):[^\n]+)")
UNIT_NORMAL = {"n·m": "nm", "n-m": "nm", "in.": "in", "quarts": "qt", "quart": "qt", "volts": "v", "volt": "v",
               "ohms": "ohm", "ω": "ohm", "seconds": "second", "minutes": "minute", "amps": "amp"}  # fmt: skip


def _norm_unit(u: str) -> str:
    u = u.lower()
    return UNIT_NORMAL.get(u, u)


RANGE_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s?(?=[A-Za-z°%Ω])")


def _expand_ranges(text: str) -> str:
    """'1.25 - 1.35 mm' -> '1.25 mm 1.35 mm' so each end of a range is a checkable value."""

    def rep(m: re.Match) -> str:
        unit = VALUE_RE.match("0 " + text[m.end() : m.end() + 8].lstrip())
        if not unit:
            return m.group(0)
        return f"{m.group(1)} {unit.group(2)} {m.group(2)} "

    return RANGE_RE.sub(rep, text)


def _values(text: str) -> set[tuple[str, str]]:
    return {(float_str(n), _norm_unit(u)) for n, u in VALUE_RE.findall(_expand_ranges(text))}


def float_str(n: str) -> str:
    f = float(n)
    return str(int(f)) if f.is_integer() else str(f)


UNIT_HEADER_RE = re.compile(r"^(Nm|N·m|lb-ft|lb-in|mm|in|L|qt|psi|kPa|bar|V|ohms?|°C|°F|rpm|%)$", re.I)
NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?\s*$")


def _table_values(text: str) -> set[tuple[str, str]]:
    """Spec tables carry units in the header row ("Description | Nm | lb-ft") and bare numbers in cells."""
    out: set[tuple[str, str]] = set()
    units: list[str | None] | None = None
    for line in text.splitlines():
        if " | " not in line:
            units = None
            continue
        cells = [re.sub(r"\s*-\s*", "-", c.strip()) for c in line.split(" | ")]
        header_units = [c if UNIT_HEADER_RE.match(c) else None for c in cells]
        if any(header_units):
            units = header_units
            continue
        if units is None or len(cells) != len(units):
            continue
        for cell, unit in zip(cells, units, strict=True):
            m = NUM_RE.match(cell)
            if unit and m:
                for n in filter(None, m.groups()):
                    out.add((float_str(n), _norm_unit(unit)))
    return out


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


VEHICLE_CLAIM_RE = re.compile(
    r"\b(mustang|your vehicle|this vehicle|the vehicle|\d\.\d\s?l|gt500|gt|coupe|convertible)\b.{0,60}\b"
    r"(uses|has|have|is fitted with|is equipped with|are equipped with|comes with|come with|equipped with)\b"
    r"|\b(uses|has|is fitted with|is equipped with|comes with)\b.{0,40}\b(mustang|your vehicle|\d\.\d\s?l)\b",
    re.I,
)


def strip_uncited_vehicle_claims(text: str) -> tuple[str, int]:
    """Remove sentences asserting what equipment the vehicle has unless they carry a citation."""
    removed = 0
    out_lines = []
    for line in text.split("\n"):
        parts = re.split(r"(?<=[.!?])\s+", line)
        kept = []
        for sent in parts:
            if VEHICLE_CLAIM_RE.search(sent) and not CITE_RE.search(sent):
                removed += 1
                continue
            kept.append(sent)
        new_line = " ".join(kept)
        if line.strip() and not new_line.strip("*_ ").strip():
            continue  # line consisted only of the removed claim (e.g. an italic note)
        out_lines.append(new_line if kept else "")
    return "\n".join(out_lines).strip(), removed


WRAP_RE = re.compile(r"(?<=\d)-\s+(?=\d)")  # "1-800-959- 3673": identifier wrapped at a line end


def evidence_text(pkg: EvidencePackage) -> str:
    """Evidence the answer may quote: content, figure callouts and the procedure/section titles."""
    parts = [
        "\n".join(filter(None, [i.procedure_title, i.section, i.subsection, i.content, i.figure_ocr]))
        for i in pkg.items
    ]
    return WRAP_RE.sub("-", "\n".join(parts).translate(NORMALISE))


def validate(text: str, pkg: EvidencePackage) -> GeneratedAnswer:
    text = WRAP_RE.sub("-", text.translate(NORMALISE))
    valid_ids = {i.source_id for i in pkg.items}
    warnings: list[str] = []
    cited: list[str] = []

    def fix_marker(m: re.Match) -> str:
        ids = [s.strip() for s in re.split(r"[,;]", m.group(1))]
        good = [i for i in ids if i in valid_ids]
        bad = [i for i in ids if i not in valid_ids]
        if bad:
            warnings.append(f"removed unknown citation(s) {', '.join(bad)}")
        cited.extend(i for i in good if i not in cited)
        return f"[{', '.join(good)}]" if good else ""

    # bare references such as "(Section 303-03A, S2)" -> "[S2]" when S2 is a real evidence id
    text = re.sub(
        r"(?<![\w\[])(S\d+)(?![\w\]])", lambda m: f"[{m.group(1)}]" if m.group(1) in valid_ids else m.group(1), text
    )
    text = CITE_RE.sub(fix_marker, text).strip()
    text, removed = strip_uncited_vehicle_claims(text)
    if removed:
        warnings.append(f"removed {removed} uncited claim(s) about the vehicle's equipment")
    ev = evidence_text(pkg)
    req = pkg.request
    vehicle_facts = " ".join(str(v) for v in (req.engine, req.model_year) if v)  # confirmed context, not specs
    allowed_values = _values(ev) | _table_values(ev) | _values(req.original_message) | _values(vehicle_facts)
    unsupported = sorted(
        {
            m.group(0).strip()
            for m in VALUE_RE.finditer(_expand_ranges(text))
            if (float_str(m.group(1)), _norm_unit(m.group(2))) not in allowed_values
        }
    )
    ev_idents = set(IDENT_RE.findall(ev)) | set(IDENT_RE.findall(pkg.request.original_message))
    unsupported += sorted({i for i in IDENT_RE.findall(text) if i not in ev_idents})
    if unsupported:
        warnings.append("values or identifiers not found in the manual evidence: " + ", ".join(unsupported))
    has_technical = bool(_values(text) or IDENT_RE.search(text) or re.search(r"^\s*\d+\.", text, re.M))
    uncited_technical = has_technical and not cited
    if uncited_technical:
        warnings.append("technical content without citations")

    # safety statements from cited (or all, if none cited) evidence must survive
    sources = [i for i in pkg.items if i.source_id in cited] or pkg.items
    missing = []
    answer_words = _words(text)
    for item in sources:
        for stmt in SAFETY_RE.findall(item.content):
            w = _words(stmt)
            if w and len(w & answer_words) / len(w) < 0.6 and stmt not in missing:
                missing.append(f"{stmt.strip()} [{item.source_id}]")
    if missing:
        warnings.append(f"re-inserted {len(missing)} manual safety statement(s) omitted by the model")
        text = "**Manual warnings (verbatim):**\n" + "\n".join(f"- {m}" for m in missing) + "\n\n" + text
        for item in sources:
            if item.source_id not in cited:
                cited.append(item.source_id)

    grounded = not unsupported and not uncited_technical
    return GeneratedAnswer(text=text, cited_source_ids=cited, validation_warnings=warnings,
                           unsupported_values=unsupported, grounded=grounded)  # fmt: skip


def build_citations(answer: GeneratedAnswer, pkg: EvidencePackage, document_title: str) -> list[Citation]:
    by_id = {i.source_id: i for i in pkg.items}
    ids = answer.cited_source_ids or [i.source_id for i in pkg.items]
    out = []
    for sid in ids:
        i = by_id.get(sid)
        if i is None:
            continue
        pdf_pages = str(i.page_start) if i.page_start == i.page_end else f"{i.page_start}-{i.page_end}"
        printed = i.printed_page_start
        if i.printed_page_end and i.printed_page_end != i.printed_page_start:
            printed = f"{i.printed_page_start} – {i.printed_page_end}"
        out.append(
            Citation(
                source_id=sid,
                chunk_id=i.chunk_id,
                document_title=document_title,
                section=i.section,
                procedure_title=i.procedure_title,
                procedure_id=i.parent_id.removeprefix("proc_"),
                subsection=i.subsection,
                pdf_pages=pdf_pages,
                printed_pages=printed,
                figure_ids=i.figure_ids,
                table_ids=i.table_ids,
            )
        )
    return out
