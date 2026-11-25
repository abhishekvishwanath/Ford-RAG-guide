from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from app.embeddings.tokenizer import TokenCounter
from app.ingestion.index_stage import plan_index_changes, run_index_stage
from app.retrieval.bm25_search import BM25Index, tokenize
from app.vectorstore.chroma_store import ChromaStore
from tests.fakes import HashEmbedder, make_chunk


class WordCounter(TokenCounter):
    def __init__(self) -> None:
        self._tok = None


def test_tokenize_preserves_identifiers():
    toks = tokenize("Tighten W710141 to 103 Nm (76 lb-ft); DTC B11E8:16; tool 303-D087 on 5.0L")
    for t in ["w710141", "lb-ft", "b11e8:16", "303-d087", "5.0l", "103", "nm"]:
        assert t in toks
    assert "b11e8" in toks and "d087" in toks  # sub-parts indexed too
    assert "the" not in tokenize("the bolt")


def test_bm25_ranks_exact_identifier_first_and_respects_filter():
    chunks = [
        make_chunk("c1", "Install the water pump. Tighten bolts to 25 Nm."),
        make_chunk("c2", "DTC B11E8:16 ABS power supply circuit voltage below threshold."),
        make_chunk("c3", "Remove the thermostat housing."),
    ]
    idx = BM25Index.from_chunks(chunks)
    assert idx.search("B11E8:16", 3)[0].chunk_id == "c2"
    assert [h.chunk_id for h in idx.search("water pump", 3, allow=lambda c: c != "c1")] == []


def test_bm25_save_load_roundtrip(tmp_path):
    chunks = [make_chunk("c1", "water pump removal"), make_chunk("c2", "brake caliper"), make_chunk("c3", "door trim")]
    toks = [tokenize(c.content) for c in chunks]
    BM25Index([c.chunk_id for c in chunks], toks).save(tmp_path / "bm25.jsonl", toks)
    idx = BM25Index.load(tmp_path / "bm25.jsonl")
    assert idx.search("caliper", 2)[0].chunk_id == "c2"


def test_chroma_window_aggregation_and_filters(tmp_path):
    store = ChromaStore(tmp_path / "chroma", "test", HashEmbedder())
    long = " ".join(["filler"] * 400) + " water pump housing gasket"
    chunks = [
        make_chunk("c1", long, engines=["5.0L"]),
        make_chunk("c2", "water pump removal 3.7L", engines=["3.7L"]),
        make_chunk("c3", "door trim panel"),
    ]
    n = store.upsert_chunks(chunks, WordCounter())
    assert n > 3  # the long chunk produced several windows
    hits = store.search("water pump gasket", 5)
    assert len({h.chunk_id for h in hits}) == len(hits)  # aggregated: one hit per chunk
    where = {"$or": [{"engine_any": True}, {"eng_5_0l": True}]}
    ids = {h.chunk_id for h in store.search("water pump", 5, where=where)}
    assert "c2" not in ids and "c1" in ids


def test_plan_index_changes():
    a, b = make_chunk("a", "x"), make_chunk("b", "y")
    upsert, stale, changed = plan_index_changes([a, b], {"a": a.content_hash, "b": "old", "z": "gone"})
    assert [c.chunk_id for c in upsert] == ["b"] and stale == ["z"] and changed == ["b"]


@pytest.fixture
def service(tmp_path):
    s = Settings(
        processed_dir=tmp_path / "processed",
        indexes_dir=tmp_path / "indexes",
        chroma_persist_directory=tmp_path / "chroma",
        chroma_collection_name="testcoll",
    )
    s.processed_dir.mkdir()

    class Svc:
        pass

    svc = Svc()
    svc.s = s
    return svc


def _write(service, chunks):
    with (service.s.processed_dir / "chunks.jsonl").open("w") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")


def test_incremental_indexing_duplicate_protection_and_stale_cleanup(service):
    emb = HashEmbedder()
    _write(service, [make_chunk("a", "water pump"), make_chunk("b", "brake caliper")])
    r1 = run_index_stage(service, emb)
    assert r1["chunks_embedded"] == 2 and r1["vector_count"] == 2
    # re-run with identical chunks: nothing re-embedded, no duplicates
    r2 = run_index_stage(service, emb)
    assert r2["chunks_embedded"] == 0 and r2["vector_count"] == 2
    # change one, remove one, add one
    _write(service, [make_chunk("a", "water pump revised"), make_chunk("c", "thermostat")])
    r3 = run_index_stage(service, emb)
    assert r3["chunks_embedded"] == 2 and r3["stale_deleted"] == 1 and r3["vector_count"] == 2
    manifest = json.loads((service.s.indexes_dir / "index_manifest.json").read_text())
    assert set(manifest["chunks"]) == {"a", "c"}


def test_embedding_model_change_forces_rebuild(service):
    _write(service, [make_chunk("a", "water pump")])
    run_index_stage(service, HashEmbedder(model_name="m1"))
    r = run_index_stage(service, HashEmbedder(model_name="m2"))
    assert r["full_rebuild"] is True and r["chunks_embedded"] == 1 and r["vector_count"] == 1
