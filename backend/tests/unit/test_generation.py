from __future__ import annotations

from app.generation import response_validator as rv
from app.generation.prompts import build_messages
from app.schemas.retrieval import EvidenceItem, EvidencePackage, RetrievalRequest


def pkg() -> EvidencePackage:
    req = RetrievalRequest(query_text="q", keyword_query="q", original_message="torque for the pump bolts?")
    items = [
        EvidenceItem(source_id="S1", chunk_id="c1", parent_id="proc_CP50", procedure_title="Coolant Pump — 5.0L",
                     section="303-03A Engine Cooling", doc_type="REMOVAL AND INSTALLATION", subsection="Installation",
                     page_start=1661, page_end=1662, printed_page_start="Page 3 of 3", printed_page_end="Page 3 of 3",
                     applicability_note="matches your vehicle (engines: 5.0L)",
                     content="WARNING: Never remove the pressure relief cap while the engine is hot.\n"
                             "1. Install the coolant pump and the 4 bolts.\n   • Tighten to 25 Nm (18 lb-ft).",
                     figure_ids=["f1"]),
    ]  # fmt: skip
    return EvidencePackage(request=req, items=items, sufficient=True)


def test_valid_answer_passes_and_keeps_citations():
    text = ("WARNING: Never remove the pressure relief cap while the engine is hot. [S1]\n"
            "1. Install the coolant pump and the 4 bolts. Tighten to 25 Nm (18 lb-ft). [S1]")  # fmt: skip
    a = rv.validate(text, pkg())
    assert a.grounded and a.cited_source_ids == ["S1"] and not a.unsupported_values


def test_unknown_citations_removed_and_fabricated_value_flagged():
    a = rv.validate("Tighten to 30 Nm (22 lb-ft). [S1][S9]", pkg())
    assert "removed unknown citation(s) S9" in a.validation_warnings[0]
    assert "30 Nm" in a.unsupported_values and not a.grounded


def test_fabricated_tool_number_flagged():
    a = rv.validate("Use special tool 303-999 to remove the pump. [S1]", pkg())
    assert "303-999" in a.unsupported_values and not a.grounded


def test_uncited_technical_content_is_ungrounded():
    a = rv.validate("1. Install the coolant pump. Tighten to 25 Nm (18 lb-ft).", pkg())
    assert not a.grounded and "technical content without citations" in a.validation_warnings


def test_dropped_warning_is_reinserted_verbatim():
    a = rv.validate("1. Install the coolant pump and the 4 bolts. Tighten to 25 Nm (18 lb-ft). [S1]", pkg())
    assert a.text.startswith("**Manual warnings (verbatim):**")
    assert "WARNING: Never remove the pressure relief cap while the engine is hot." in a.text
    assert a.grounded


def test_citations_built_from_metadata_not_model_text():
    p = pkg()
    a = rv.validate("Tighten to 25 Nm (18 lb-ft). [S1]", p)
    cites = rv.build_citations(a, p, "Mustang 2014 Workshop Manual")
    c = cites[0]
    assert (c.source_id, c.pdf_pages, c.printed_pages, c.procedure_id) == ("S1", "1661-1662", "Page 3 of 3", "CP50")
    assert c.figure_ids == ["f1"] and c.section == "303-03A Engine Cooling"


def test_prompt_contains_rules_evidence_and_applicability():
    msgs = build_messages("Mustang 2014 Workshop Manual", "2014 Mustang, 5.0L", "torque?", [], pkg())
    assert "ONLY from the numbered manual evidence" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "[S1] Section: 303-03A Engine Cooling" in user and "Applicability: matches your vehicle" in user
    assert "PDF p. 1661-1662 (Page 3 of 3)" in user


def test_prompt_trimmed_to_provider_token_cap(tmp_path):
    from app.conversation.state import SessionState
    from app.core.config import Settings
    from app.generation.answer_generator import GroundedAnswerGenerator, estimate_tokens
    from app.memory.session_store import SqliteSessionRepository

    p = pkg()
    extra = p.items[0].model_copy(update={"source_id": "S2", "chunk_id": "c2", "parent_id": "proc_OTHER",
                                          "content": "filler text " * 3000})  # fmt: skip
    p.items.append(extra)
    s = Settings(llm_tpm_limit=4000, llm_max_output_tokens=1000)
    repo = SqliteSessionRepository(f"sqlite:///{tmp_path / 's.db'}")
    gen = GroundedAnswerGenerator(s, None, repo, "Manual")
    msgs = gen.build_prompt(SessionState(session_id="x"), "q", p)
    assert estimate_tokens(msgs) <= 4000 * 0.92 - 1000
    assert [i.source_id for i in p.items] == ["S1"]  # the other procedure was dropped, main kept


def test_values_from_spec_table_columns_are_supported():
    p = pkg()
    p.items[0].content = ("Description | Nm | lb- ft | lb- in\nBrake caliper bolt — 4 and 6 piston calipers | 115 | 85 | —\n"
                          "Brake tube fittings | 17 | — | 150")  # fmt: skip
    a = rv.validate("Caliper bolts: 115 Nm (85 lb-ft); tube fittings 17 Nm (150 lb-in). [S1]", p)
    assert a.grounded and not a.unsupported_values
    bad = rv.validate("Caliper bolts: 120 Nm (89 lb-ft). [S1]", p)
    assert not bad.grounded and set(bad.unsupported_values) == {"120 Nm", "89 lb-ft"}


def test_uncited_vehicle_equipment_claim_is_removed():
    p = pkg()
    text = ("Caliper bolt: 25 Nm (18 lb-ft). [S1]\n"
            "*Note: The 2014 Mustang 5.0 L uses 4-piston front calipers; therefore 25 Nm applies.*")  # fmt: skip
    a = rv.validate(text, p)
    assert "4-piston" not in a.text and "25 Nm (18 lb-ft). [S1]" in a.text
    assert any("uncited claim" in w for w in a.validation_warnings)


def test_wrapped_identifiers_and_vehicle_facts_are_not_false_positives():
    p = pkg()
    p.request.engine = "5.0L"
    p.items[0].content += "\nFor information call: 1-800-959- 3673."
    a = rv.validate("For your 2014 Mustang 5.0 L: call 1-800-959-3673. Tighten to 25 Nm (18 lb-ft). [S1]", p)
    assert a.grounded and not a.unsupported_values, a.unsupported_values
