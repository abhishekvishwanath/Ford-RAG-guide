from __future__ import annotations

import pytest

from app.conversation.context_extractor import Extraction
from app.conversation.state import ActiveProcedure, SessionState, Slot, SlotStatus
from app.retrieval.applicability_filter import VehicleFilter
from app.retrieval.bm25_search import BM25Index
from app.retrieval.context_expansion import base_title, detect_variant_ambiguity
from app.retrieval.fusion import rrf_fuse
from app.retrieval.pipeline import HybridRetrievalPipeline, RetrievalConfig, build_request
from app.retrieval.reranker import NoopReranker
from app.schemas.chunk import FigureRecord, ProcedureRecord
from app.schemas.retrieval import Candidate, RetrievalRequest
from app.services.manual_repository import ManualRepository
from app.vectorstore.chroma_store import ChromaStore
from tests.fakes import HashEmbedder, make_chunk
from tests.unit.test_indexing import WordCounter


def test_rrf_math_and_dedup():
    dense = [Candidate(chunk_id="a", dense_rank=1), Candidate(chunk_id="b", dense_rank=2)]
    bm25 = [Candidate(chunk_id="b", bm25_rank=1), Candidate(chunk_id="c", bm25_rank=2)]
    fused = rrf_fuse(dense, bm25, k=60)
    assert [c.chunk_id for c in fused] == ["b", "a", "c"]
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    deduped = rrf_fuse(dense, bm25, k=60, content_hash={"a": "h", "b": "h", "c": "x"})
    assert [c.chunk_id for c in deduped] == ["b", "c"]


def test_vehicle_filter_classify_and_where():
    f = VehicleFilter(2099, "5.0L", "automatic", None)
    assert f.classify(make_chunk("x", "t", engines=["3.7L"])) == "incompatible"
    assert f.classify(make_chunk("y", "t", engines=["5.0L"])) == "matches"
    assert VehicleFilter(2014, None, None, None).classify(make_chunk("z", "t")) == "incompatible"  # wrong year
    f = VehicleFilter(2014, "5.0L", "automatic", None)
    gen = make_chunk("g", "t")
    gen.applicability.model_years = [2014]
    assert f.classify(gen) == "general"
    where = f.chroma_where()
    assert where == {"$and": [{"$or": [{"engine_any": True}, {"eng_5_0l": True}]},
                              {"$or": [{"trans_any": True}, {"trans_automatic": True}]}]}  # fmt: skip
    assert VehicleFilter(None, None, None, None).chroma_where() is None


def _corpus():
    chunks = [
        make_chunk("cp50_0", "Special Tool(s) Remover 303-123. WARNING: Hot coolant can cause burns.", title="Coolant Pump — 5.0L",
                   parent="proc_CP50", seq=0, engines=["5.0L"]),
        make_chunk("cp50_1", "Removal 1. Drain the cooling system. 2. Remove the 4 bolts and the coolant pump.",
                   title="Coolant Pump — 5.0L", parent="proc_CP50", seq=1, engines=["5.0L"], chunk_type="removal",
                   subsection="Removal", steps=(1, 2), figure_ids=["fig1"]),
        make_chunk("cp50_2", "Installation 1. Install the coolant pump and the 4 bolts. Tighten to 25 Nm (18 lb-ft).",
                   title="Coolant Pump — 5.0L", parent="proc_CP50", seq=2, engines=["5.0L"], chunk_type="installation",
                   subsection="Installation", steps=(1, 1)),
        make_chunk("cp37_0", "Removal 1. Remove the coolant pump pulley. 2. Remove the 3.7L coolant pump bolts.",
                   title="Coolant Pump — 3.7L", parent="proc_CP37", seq=0, engines=["3.7L"], chunk_type="removal"),
        make_chunk("door_0", "Removal 1. Remove the door trim panel screws.", title="Door Trim Panel", parent="proc_DTP"),
    ]  # fmt: skip
    for c in chunks:
        c.applicability.model_years = [2014]
    procs = [
        ProcedureRecord(parent_id=p, procedure_id=p[5:], source_procedure_id=p[5:], title=t, section_code="303-03A",
                        section_name="Engine Cooling", doc_type="REMOVAL AND INSTALLATION", toc_path=[], page_start=1,
                        page_end=3, printed_total=3, revision_date=None,
                        chunk_ids=[c.chunk_id for c in chunks if c.parent_id == p], applicability=chunks[0].applicability)
        for p, t in [("proc_CP50", "Coolant Pump — 5.0L"), ("proc_CP37", "Coolant Pump — 3.7L"), ("proc_DTP", "Door Trim Panel")]
    ]  # fmt: skip
    fig = FigureRecord(figure_id="fig1", asset_id="img_0123456789abcdef0123", document_id="doc", source_hash="h",
                       pdf_page_index=10, page_number=11, printed_page="Page 2 of 3", bbox=(0, 0, 10, 10),
                       image_type="illustration", extraction_method="embedded_image", procedure_id="CP50",
                       source_procedure_id="CP50", procedure_title="Coolant Pump — 5.0L", section_code="303-03A",
                       related_chunk_ids=["cp50_1"], nearest_step=2, subsection="Removal", association_method="in_step",
                       association_confidence=0.9, figure_numbers=["N0000001"], width_px=600, height_px=400,
                       asset_relpath="01/img.png")  # fmt: skip
    return chunks, procs, [fig]


@pytest.fixture
def pipeline(tmp_path):
    chunks, procs, figs = _corpus()
    repo = ManualRepository(chunks, procs, figs)
    store = ChromaStore(tmp_path / "chroma", "testcoll", HashEmbedder())
    store.upsert_chunks(chunks, WordCounter())
    return HybridRetrievalPipeline(RetrievalConfig(top_k=5), repo, store, BM25Index.from_chunks(chunks), NoopReranker())


def _state(**slots) -> SessionState:
    s = SessionState(session_id="x")
    for k, v in slots.items():
        s.set_slot(Slot(name=k, value=v, status=SlotStatus.CONFIRMED, confidence=1.0))
    return s


def run(pipeline, req):
    flt = pipeline.applicability_filter(req)
    cands = pipeline.rerank(req, pipeline.fuse(pipeline.dense(req, flt), pipeline.bm25(req, flt)))
    return pipeline.assess(pipeline.visuals_and_tables(pipeline.expand(req, cands)))


def test_build_request_adds_synonyms_and_preserves_facts():
    st = _state(model_year=2014, engine="5.0L", component="water pump", task_type="replace")
    req = build_request(st, "I need to replace the water pump", Extraction())
    assert "coolant pump" in req.manual_terms and "coolant pump" in req.keyword_query
    assert req.engine == "5.0L" and req.model_year == 2014 and req.task_type == "replace"
    assert req.preferred_chunk_types[:2] == ["removal", "installation"]


def test_build_request_followup_focus_and_steps():
    st = _state(model_year=2014, engine="5.0L", component="water pump", task_type="replace")
    st.active_procedure = ActiveProcedure(parent_id="proc_CP50", procedure_id="CP50", title="Coolant Pump — 5.0L")
    req = build_request(st, "what is step 2 and the torque?", Extraction(is_followup=True))
    assert req.focus_parent_id == "proc_CP50" and req.focus_steps == [2]
    assert req.preferred_chunk_types[0] == "installation"


def test_pipeline_filters_incompatible_and_expands_procedure(pipeline):
    st = _state(model_year=2014, engine="5.0L", component="water pump", task_type="replace")
    pkg = run(pipeline, build_request(st, "replace the water pump", Extraction()))
    ids = [i.chunk_id for i in pkg.items]
    assert "cp37_0" not in ids  # 3.7L excluded for a 5.0L vehicle
    assert ids[:3] == ["cp50_0", "cp50_1", "cp50_2"]  # whole procedure, in manual order
    assert pkg.sufficient
    assert pkg.items[0].source_id == "S1" and "matches your vehicle" in pkg.items[0].applicability_note
    assert [im.figure_id for im in pkg.images] == ["fig1"]
    assert pkg.images[0].url == "/api/manual-assets/img_0123456789abcdef0123" and pkg.images[0].nearest_step == 2


def test_pipeline_irrelevant_query_is_insufficient(pipeline):

    req = RetrievalRequest(query_text="zzz qqq", keyword_query="zzz qqq", original_message="zzz qqq", model_year=2014)
    pkg = run(pipeline, req)
    assert not pkg.sufficient


def test_variant_ambiguity_detected_when_variant_unknown():
    chunks, procs, figs = _corpus()
    repo = ManualRepository(chunks, procs, figs)
    cands = [Candidate(chunk_id="cp50_1"), Candidate(chunk_id="cp37_0")]
    amb = detect_variant_ambiguity(cands, repo, VehicleFilter(2014, None, None, None))
    assert set(amb) == {"Coolant Pump — 5.0L", "Coolant Pump — 3.7L"}
    assert base_title("Quarter Trim Panel — Convertible") == "quarter trim panel"
