"""Resolve draft evaluation cases against the ingested manual and verify them.

A draft case names the expected procedure by its catalogue title + section. Resolution
turns that into source procedure ids and page ranges, and *verifies* that the stated key
fact actually occurs in that procedure's text — so no expectation is invented.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services.manual_repository import ManualRepository


def resolve_cases(draft_path: Path, repo: ManualRepository) -> tuple[list[dict], list[str]]:
    draft = json.loads(draft_path.read_text())["cases"]
    resolved, problems = [], []
    for case in draft:
        matches = []
        for p in repo.procedures.values():
            if (p.section_code or "") != case["section"]:
                continue
            title = p.title or ""
            ok = title == case["title"] if "title" in case else title.startswith(case["title_prefix"])
            if ok and case.get("doc_type") and (p.doc_type or "") != case["doc_type"]:
                ok = False
            if ok:
                matches.append(p)
        if not matches:
            problems.append(
                f"{case['id']}: no procedure titled {case.get('title') or case.get('title_prefix')!r} in {case['section']}"
            )
            continue
        fact = case["fact"].lower()
        verified = []
        for p in matches:
            text = "\n".join(
                (repo.chunks[c].content + "\n" + repo.chunks[c].context_header).lower() for c in p.chunk_ids
            )
            if fact in text:
                verified.append(p)
        if not verified:
            problems.append(f"{case['id']}: key fact {case['fact']!r} not found in {[m.procedure_id for m in matches]}")
            continue
        resolved.append(
            {
                **{k: v for k, v in case.items() if k not in ("title", "title_prefix", "doc_type")},
                "expected_title": verified[0].title,
                "expected_parent_ids": [p.parent_id for p in verified],
                "expected_source_ids": [p.source_procedure_id for p in verified],
                "expected_pages": [[p.page_start, p.page_end] for p in verified],
                "expected_has_figures": any(
                    any(f.image_type == "illustration" for f in repo.figures_for_chunk(repo.chunks[c]))
                    for p in verified
                    for c in p.chunk_ids
                ),
            }
        )
    return resolved, problems
