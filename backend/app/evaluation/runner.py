"""Evaluation suites: retrieval (structure-based + ablations), chunking baselines, context gate, faithfulness."""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.conversation.context_extractor import Extraction, manual_terms_for
from app.conversation.state import SessionState, Slot, SlotStatus
from app.core.config import Settings
from app.embeddings.embedding_service import SentenceTransformerEmbedder
from app.embeddings.tokenizer import get_token_counter
from app.evaluation import metrics as M
from app.evaluation.baselines import fixed_size_chunks, page_texts, recursive_chunks
from app.ingestion import metadata as md
from app.ingestion.pdf_parser import PageParseRunner, ParseOptions
from app.retrieval.applicability_filter import VehicleFilter
from app.retrieval.bm25_search import BM25Index
from app.retrieval.pipeline import HybridRetrievalPipeline, RetrievalConfig, build_request
from app.services.manual_repository import ManualRepository
from app.vectorstore.chroma_store import ChromaStore

K = 5


def case_state(case: dict, year: int = 2014) -> SessionState:
    st = SessionState(session_id=f"eval_{case['id']}")
    slots = {"model_year": year, "task_type": case.get("task"), "component": case.get("component"),
             "dtcs": case.get("dtcs"), **case.get("veh", {})}  # fmt: skip
    for k, v in slots.items():
        if v:
            st.set_slot(Slot(name=k, value=v, status=SlotStatus.CONFIRMED, confidence=1.0))
    return st


def case_request(case: dict):  # type: ignore[no-untyped-def]
    comp = case.get("component")
    ex = Extraction(manual_terms=manual_terms_for(comp) if comp else [])
    return build_request(case_state(case), case["q"], ex)


# ----------------------------------------------------------------------------- structure-based (system C)
def run_structure(pipe: HybridRetrievalPipeline, repo: ManualRepository, cases: list[dict], variant: str) -> dict:
    per_case = []
    for case in cases:
        req = case_request(case)
        flt = pipe.applicability_filter(req)
        if variant == "no_filter":
            flt = {"vehicle": VehicleFilter(None, None, None, None), "where": None}
        t0 = time.perf_counter()
        dense = pipe.dense(req, flt) if variant != "bm25_only" else []
        bm25 = pipe.bm25(req, flt) if variant != "dense_only" else []
        fused = pipe.fuse(dense, bm25)
        ranked = pipe.rerank(req, fused) if variant in ("full", "no_filter") else fused
        ms = (time.perf_counter() - t0) * 1000
        top = [repo.chunk(c.chunk_id) for c in ranked[:K]]
        rel = [c is not None and c.parent_id in case["expected_parent_ids"] for c in top]
        vf = VehicleFilter.from_request(req)
        violations = sum(1 for c in top if c is not None and vf.classify(c) == "incompatible")
        row = {
            "id": case["id"],
            "recall@5": M.recall_at_k(rel, K),
            "precision@5": M.precision_at_k(rel, K),
            "mrr": M.reciprocal_rank(
                [
                    c is not None and c.parent_id in case["expected_parent_ids"]
                    for c in (repo.chunk(x.chunk_id) for x in ranked[:20])
                ]
            ),  # fmt: skip
            "ndcg@5": M.ndcg_at_k(rel, K),
            "applicability_violations@5": violations,
            "top1": top[0].procedure_title if top and top[0] else None,
            "ms": round(ms),
        }
        if variant == "full":
            pkg = pipe.assess(pipe.visuals_and_tables(pipe.expand(req, ranked)))
            exp_chunks = {cid for pid in case["expected_parent_ids"] for cid in repo.procedures[pid].chunk_ids}
            got = {i.chunk_id for i in pkg.items}
            row["procedure_completeness"] = round(len(exp_chunks & got) / len(exp_chunks), 3) if exp_chunks else 0.0
            first = pkg.items[0] if pkg.items else None
            row["citation_page_correct"] = bool(
                first and any(a <= first.page_start <= b for a, b in case["expected_pages"])
            )
            row["sufficient"] = pkg.sufficient
            if case.get("expected_has_figures"):
                own = [im for im in pkg.images if repo.figures[im.figure_id].procedure_id in
                       {p.removeprefix("proc_") for p in case["expected_parent_ids"]}]  # fmt: skip
                row["figure_precision"] = round(len(own) / len(pkg.images), 3) if pkg.images else 0.0
                row["figure_recall"] = 1.0 if own else 0.0
        per_case.append(row)
    return summarise(per_case)


def summarise(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float, bool)) and k not in ("ms",)] if rows else []
    summary = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r]
        summary[k] = M.mean(vals)
    summary["median_ms"] = sorted(r["ms"] for r in rows)[len(rows) // 2] if rows else 0
    summary["n"] = len(rows)
    for opt in ("figure_precision", "figure_recall"):
        vals = [r[opt] for r in rows if opt in r]
        if vals:
            summary[opt] = M.mean(vals)
            summary[f"{opt}_n"] = len(vals)
    return {"summary": summary, "cases": rows}


# ----------------------------------------------------------------------------- baselines (A, B)
def procedure_page_map(repo: ManualRepository, year: int = 2014) -> dict[int, list]:
    out: dict[int, list] = {}
    for p in repo.procedures.values():
        app = md.procedure_applicability([year], p.toc_path, p.section_name, p.title, p.doc_type)
        for pg in range(p.page_start, p.page_end + 1):
            out.setdefault(pg, []).append((p.parent_id, app))
    return out


def run_baseline(name: str, chunks: list, s: Settings, cases: list[dict], repo: ManualRepository, reranker) -> dict:  # type: ignore[no-untyped-def]
    counter = get_token_counter(s.embedding_model)
    store = ChromaStore(
        s.indexes_dir / "eval_chroma", f"baseline_{name}", SentenceTransformerEmbedder(s.embedding_model)
    )
    if store.count() == 0:
        store.upsert_chunks(chunks, counter)
    brepo = ManualRepository(chunks, [], [])
    pipe = HybridRetrievalPipeline(RetrievalConfig(top_k=K), brepo, store, BM25Index.from_chunks(chunks), reranker)
    pmap = procedure_page_map(repo)
    rows = []
    for case in cases:
        req = case_request(case)
        flt = {"vehicle": VehicleFilter(None, None, None, None), "where": None}  # no metadata to filter on
        t0 = time.perf_counter()
        ranked = pipe.rerank(req, pipe.fuse(pipe.dense(req, flt), pipe.bm25(req, flt)))
        ms = (time.perf_counter() - t0) * 1000

        def relevant(cid: str, pages: list = case["expected_pages"]) -> bool:  # noqa: B008
            c = brepo.chunk(cid)
            return c is not None and any(a <= c.page_end and c.page_start <= b for a, b in pages)

        rel = [relevant(c.chunk_id) for c in ranked[:K]]
        vf = VehicleFilter.from_request(req)
        violations = 0
        for c in ranked[:K]:
            ch = brepo.chunk(c.chunk_id)
            apps = [a for pg in range(ch.page_start, ch.page_end + 1) for _, a in pmap.get(pg, [])] if ch else []
            if any(a.engines and vf.engine and vf.engine not in a.engines for a in apps):
                violations += 1
        rows.append({
            "id": case["id"], "recall@5": M.recall_at_k(rel, K), "precision@5": M.precision_at_k(rel, K),
            "mrr": M.reciprocal_rank([relevant(c.chunk_id) for c in ranked[:20]]), "ndcg@5": M.ndcg_at_k(rel, K),
            "applicability_violations@5": violations, "ms": round(ms),
        })  # fmt: skip
    out = summarise(rows)
    tokens = [c.token_count for c in chunks]
    out["chunk_stats"] = {
        "chunks": len(chunks),
        "median_tokens": sorted(tokens)[len(tokens) // 2],
        "max_tokens": max(tokens),
    }
    return out


def build_baselines(s: Settings) -> dict[str, list]:
    runner = PageParseRunner(s.manual_pdf_path, s.processed_dir, s.images_dir, ParseOptions())
    pages = page_texts(runner.iter_pages())
    counter = get_token_counter(s.embedding_model)
    return {"A_fixed_size": fixed_size_chunks(pages, counter), "B_recursive": recursive_chunks(pages, counter)}


# ----------------------------------------------------------------------------- context gate
def run_context(cases: list[dict], workflow_factory) -> dict:  # type: ignore[no-untyped-def]
    rows = []
    for conv in cases:
        wf, recorder = workflow_factory()
        sid = None
        turn_rows = []
        premature = 0
        for t in conv["turns"]:
            before = len(recorder.calls)
            r = wf.invoke({"session_id": sid, "message": t["user"], "trace": []})["response"]
            sid = r["session_id"]
            status = r["readiness"]["status"]
            retrieved = len(recorder.calls) > before
            if retrieved and status != "READY":
                premature += 1
            asked = (r.get("pending_question") or {}).get("slots", [])
            turn_rows.append({"user": t["user"], "expected": t["expect"], "got": status, "asked": asked,
                              "ok": status == t["expect"], "retrieved": retrieved})  # fmt: skip
        final = r
        exp_slots = conv.get("final_slots", {})
        got_slots = {k: v["value"] for k, v in {**final["vehicle_context"], **final["repair_context"]}.items()}
        slot_hits = sum(1 for k, v in exp_slots.items() if str(got_slots.get(k, "")).lower() == str(v).lower())
        necessary = set(conv.get("necessary_slots", []))
        questions = [s for tr in turn_rows for s in tr["asked"]]
        unnecessary = [s for s in questions if necessary and s not in necessary]
        rows.append({
            "id": conv["id"], "desc": conv["desc"], "turns": turn_rows,
            "decision_accuracy": M.mean([1.0 if tr["ok"] else 0.0 for tr in turn_rows]),
            "slot_accuracy": round(slot_hits / len(exp_slots), 3) if exp_slots else 1.0,
            "clarification_turns": sum(1 for tr in turn_rows if tr["got"] in ("NEEDS_INFORMATION", "NEEDS_CONFIRMATION")),
            "unnecessary_questions": len(unnecessary), "questions": len(questions), "premature_retrievals": premature,
        })  # fmt: skip
    summary = {
        "conversations": len(rows),
        "turns": sum(len(r["turns"]) for r in rows),
        "decision_accuracy": M.mean([tr_ok for r in rows for tr_ok in [1.0 if t["ok"] else 0.0 for t in r["turns"]]]),
        "slot_accuracy": M.mean([r["slot_accuracy"] for r in rows]),
        "mean_clarification_turns": M.mean([r["clarification_turns"] for r in rows]),
        "unnecessary_question_rate": round(
            sum(r["unnecessary_questions"] for r in rows) / max(1, sum(r["questions"] for r in rows)), 4
        ),
        "premature_retrievals": sum(r["premature_retrievals"] for r in rows),
        "hard_gate_enforced": all(r["premature_retrievals"] == 0 for r in rows),
    }
    return {"summary": summary, "conversations": rows}


# ----------------------------------------------------------------------------- faithfulness proxy
def run_faithfulness(pipe, generator, cases: list[dict], sleep_s: float) -> dict:  # type: ignore[no-untyped-def]
    rows = []
    for i, case in enumerate(cases):
        if i:
            time.sleep(sleep_s)  # stay under the provider's tokens-per-minute cap
        req = case_request(case)
        flt = pipe.applicability_filter(req)
        ranked = pipe.rerank(req, pipe.fuse(pipe.dense(req, flt), pipe.bm25(req, flt)))
        pkg = pipe.assess(pipe.visuals_and_tables(pipe.expand(req, ranked)))
        st = case_state(case)
        msgs = generator.build_prompt(st, case["q"], pkg)
        t0 = time.perf_counter()
        try:
            raw = generator.generate(msgs)
            err = None
        except Exception as exc:  # recorded, not hidden
            raw, err = "", type(exc).__name__
        ans = generator.validate(raw, pkg) if raw else None
        rows.append({
            "id": case["id"], "error": err, "seconds": round(time.perf_counter() - t0, 1),
            "grounded": bool(ans and ans.grounded), "unsupported_values": ans.unsupported_values if ans else [],
            "citations": len(ans.cited_source_ids) if ans else 0,
            "safety_reinserted": any("re-inserted" in w for w in ans.validation_warnings) if ans else False,
            "warnings": ans.validation_warnings if ans else [],
        })  # fmt: skip
    ok = [r for r in rows if not r["error"]]
    return {
        "summary": {
            "n": len(rows),
            "llm_errors": len(rows) - len(ok),
            "grounded_rate": M.mean([1.0 if r["grounded"] else 0.0 for r in ok]),
            "answers_with_unsupported_values": sum(1 for r in ok if r["unsupported_values"]),
            "mean_citations": M.mean([r["citations"] for r in ok]),
            "safety_reinsertions": sum(1 for r in ok if r["safety_reinserted"]),
        },
        "cases": rows,
    }


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str))
