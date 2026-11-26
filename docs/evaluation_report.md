# Evaluation Report

All numbers below were measured on the ingested 2014 Mustang Workshop Manual (5,007 pages,
942 procedures, 4,039 chunks, 9,847 vectors). Raw per-case results are in `docs/evaluation/*.json`;
reproduce with `scripts/evaluate_rag.py`.

## 1. Datasets

| Set | Size | How it was built |
|---|---|---|
| Retrieval cases | 48 queries | Draft cases name the expected procedure by catalogue title + section and a **key fact**. `evaluate_rag.py resolve` maps each to source procedure ids/pages and verifies that the key fact actually occurs in that procedure's text. Nothing is accepted on trust. |
| Conversation cases | 16 conversations, 29 turns | Scripted technician dialogues: missing year/engine, trim-name inference, conflicts, out-of-scope vehicles, hazards, follow-ups. Each turn has an expected gate decision and expected slots. |
| Faithfulness subset | 8 queries | Every 6th retrieval case, answered live by Groq `gpt-oss-120b`. |

Relevance is **procedure-level**: a retrieved chunk is relevant when it belongs to an expected procedure.

## 2. Retrieval (structure-based system, K = 5)

| Variant | Recall@5 | P@5 | MRR | nDCG@5 | Applicability violations@5 | Median ms |
|---|---|---|---|---|---|---|
| **Full** (hybrid + filter + rerank fusion) | **0.958** | 0.288 | **0.788** | **0.823** | **0.00** | 323 |
| No rerank | 0.958 | 0.288 | 0.778 | 0.812 | 0.00 | 46 |
| Dense only | 0.958 | 0.242 | 0.759 | 0.803 | 0.00 | 45 |
| BM25 only | 0.813 | 0.250 | 0.663 | 0.679 | 0.00 | 3 |
| No applicability filter | 0.938 | 0.279 | 0.701 | 0.755 | 1.02 | 302 |

Additional full-system metrics:

| Metric | Value |
|---|---|
| Procedure completeness (expected procedure's chunks present after expansion) | 0.97 (was 0.92) |
| Citation page correct (top evidence page inside expected range) | 0.75 (was 0.65) |
| Evidence judged sufficient | 0.98 |
| Figure precision / recall (n = 37 cases whose procedure has diagrams) | 0.84 / 0.84 (was 0.73) |

**Reading the numbers**
- The applicability filter matters most for ranking: without it, about one chunk in every top 5 belongs to
  an incompatible variant (e.g. a 3.7L procedure in a 5.0L job), and MRR drops by 0.09.
- BM25 alone misses paraphrased queries; dense alone ranks well but is less precise; the fusion wins.
- **P@5 is structurally low**: most questions have exactly one relevant procedure. When it occupies one or two
  of the five slots, P@5 can't go above 0.2–0.4. Recall, MRR and completeness are the meaningful measures here.
- "Was" values are from before the technician stress test (§8), whose fixes chose the main procedure better.
- Citation page correctness (0.75) is strict. It counts a citation as wrong when the top chunk is from the
  right procedure but begins on a different page of a multi-page procedure.

### Reranker strategy experiment

The cross-encoder (`ms-marco-MiniLM-L-6-v2`) by itself made ranking **worse** on this domain. Strategies were
tuned on odd-numbered cases and checked on held-out even-numbered cases (Recall@5 / MRR):

| Strategy | Odd (tune) | Even (held-out) | All |
|---|---|---|---|
| Hybrid RRF only | 0.958 / 0.809 | 0.958 / 0.747 | 0.958 / 0.778 |
| Cross-encoder, max of two queries | 0.875 / 0.629 | 0.958 / 0.764 | 0.917 / 0.697 |
| Cross-encoder, mean of two queries | 0.917 / 0.700 | 1.000 / 0.692 | 0.958 / 0.696 |
| Cross-encoder, user message only | 0.917 / 0.639 | 0.958 / 0.707 | 0.938 / 0.673 |
| Cross-encoder, structured query only | 0.875 / 0.657 | 0.875 / 0.685 | 0.875 / 0.671 |
| **RRF(hybrid, rerank-mean, k = 10)** (chosen) | 0.917 / 0.826 | 1.000 / 0.750 | 0.958 / 0.788 |
| + task-type boost 0.01 | 0.958 / 0.774 | 1.000 / 0.750 | 0.979 / 0.762 |
| + task-type boost 0.04 | 0.958 / 0.778 | 1.000 / 0.733 | 0.979 / 0.755 |

Fusing the cross-encoder ranks with the hybrid order gives a small, consistent MRR gain on both halves, so it
was kept. A task-type boost (e.g. favouring "Removal and Installation" pages) improves recall slightly but
lowers MRR, so it is off.

## 3. Chunking comparison

All three strategies use the same parsed text (watermark removed), the same embedder, BM25, RRF and reranker.
Only the chunking changes. Baselines have no procedure structure, so their relevance is judged **leniently**:
a baseline chunk counts as relevant when its page range overlaps an expected procedure's pages. Applicability
violations for baselines are computed by mapping their pages back to procedure applicability.

| Strategy | Chunks | Median tokens | Recall@5 | P@5 | MRR | nDCG@5 | Violations@5 |
|---|---|---|---|---|---|---|---|
| A. Fixed-size (~400-token target, no overlap, ignores structure) | 2,861 | 426 | 0.917 | 0.313 | 0.671 | 0.721 | 1.31 |
| B. Recursive (paragraph → line → sentence, 400 tok, 50 overlap) | 3,597 | 388 | 0.833 | 0.300 | 0.625 | 0.649 | 1.33 |
| **C. Structure-based hierarchical** | 4,039 | 315 | **0.958** | 0.288 | **0.788** | **0.823** | **0.00** |

The structure-based chunker does better on every ranking measure. Even with lenient page-overlap
relevance, the baselines fall behind. More importantly, the baselines can't be filtered by
engine/transmission/body: about 1.3 chunks in every top 5 came from a variant the vehicle doesn't have. Their
slightly higher P@5 comes from the lenient page-overlap relevance rule, not from better retrieval.

## 4. Conversational readiness gate

| Metric | Value |
|---|---|
| Gate decision accuracy (29 turns) | 1.00 |
| Slot extraction accuracy | 1.00 |
| Mean clarification turns before READY | 0.63 |
| Unnecessary question rate | 0.00 |
| **Premature retrievals** (retrieval called before READY) | **0** |

The suite runs the real LangGraph workflow with a recording retrieval stub, so "0 premature retrievals"
is measured, not assumed. The unit tests (`test_graph_gate.py`) assert the same for every non-READY path.
Extraction used the deterministic extractor (LLM extraction disabled) to keep the suite reproducible.

## 5. Generation faithfulness (proxy)

Eight live answers from `gpt-oss-120b`, spaced 50 s apart because of the Groq on-demand limit of 8,000 tokens per minute:

| Metric | Value |
|---|---|
| Answers produced | 7 / 8 (1 Groq call failed → the system served the verbatim-excerpt fallback) |
| Passed validator (every value / part no. / tool no. found in cited evidence) | 7 / 7 |
| Answers with unsupported values | 0 |
| Mean citations per answer | 3.0 |
| Safety warnings re-inserted by validator | 0 |

An earlier run, before the validator fixes, scored 0.625. All three failures were validator false positives:
`5.0 L` vs `5.0L`, the vehicle's own engine, and a phone number wrapped across lines. They were fixed by
Unicode/wrap normalisation and by allowing known vehicle facts.

**Limitation:** this is a *proxy*. It checks that every number and identifier in the answer appears in the cited
evidence and that citations resolve. It does **not** judge semantic faithfulness, such as a step paraphrased
wrongly or two steps in the wrong order. A human or LLM-judge review of answers is the recommended next step.

## 6. Bugs found by evaluation (and fixed)

- Mixed-qualifier chunks were tagged as one engine only, which would have hidden shared procedures from other
  engines. Now chunk applicability is "general" unless the whole chunk is variant-specific.
- The index manifest ignored metadata-only changes. It now hashes content + metadata (`index_hash`).
- Including the engine in the search query skewed ranking. Variants are now filters only.
- Raw cross-encoder reranking hurt MRR. Replaced with rank fusion (see §2).
- Validator false positives (spaced citations, Unicode hyphens, units in spec-table headers, wrapped identifiers).

## 7. Limitations

- 48 retrieval cases and 16 conversations are enough to compare designs. They're too few to make precise
  population claims; the differences of ±0.02 in the tables are within noise.
- The cases were written by the builder against the manual's catalogue. Real technician phrasing may differ.
- Figure linkage is 0.84 precision/recall. Some diagrams sit on a page next to the steps they illustrate, not on the same page.
- The Groq on-demand tier (8k TPM) forces prompt trimming and spacing between calls. A higher tier would allow
  larger evidence budgets and concurrent users.
- Faithfulness is checked lexically, not semantically (see §5).

## 8. Technician stress test (10 live conversations)

`scripts/run_technician_scenarios.py` runs ten hard, technician-style conversations through the live API.
They cover trim-name inference, caliper variants, follow-ups, a hazard, an out-of-scope year, a spec
lookup and a contradictory engine. Answers were graded against facts pulled from the manual.

| # | Scenario | First run | After fixes |
|---|---|---|---|
| T01 | GT spark plugs: torque, then gap | Gap answer (1.25–1.35 mm) rejected by the validator | ✅ Both answered. The 14 vs 15 Nm torque is flagged because the manual lists both (ignition vs engine section). |
| T02 | Front pads on a GT | Blended the 2-piston procedure. Irrelevant coupe/convertible question. | ✅ Asks 2/4/6-piston, answers the 4-piston procedure. No body-style question. |
| T03 | 5.0 clutch disc + pressure plate | Diagnosis pages chosen as main procedure. Wrong diagrams. | ✅ Clutch Disc and Pressure Plate — 5.0L is the main procedure, with its own diagrams. |
| T04 | GT500 accessory belt + routing | ✅ | ✅ |
| T05 | V6 automatic fluid level | Level-check procedure not retrieved | ✅ Retrieved. Model invented an 80 °C figure → rejected → focused excerpts. |
| T06 | Fuel smell + puddle, then fuel pump module | "Safe now" turned the job into *battery / diagnose*. A false **fire** escalation followed. | ✅ Stop-work first, then the fuel pump module procedure with its fuel warnings |
| T07 | 2011 (out of scope) → corrected to 2014 | Front hub diagram shown for a rear axle seal | ✅ Rear Wheel Bearing and Axle Shaft Seal and its own diagrams |
| T08 | 5.0 timing chains → tools → "camshaft sprocket bolts" | First answer rejected ("4V" read as 4 volts). Sprocket follow-up lost the procedure. | ✅ Answers and tools. The follow-up stays on Timing Drive Components. The fallback model (quota) answered the torque question without citations → excerpts. |
| T09 | Front camber: method + spec | Spec missing from evidence | ✅ Procedure plus −0.75° ± 0.75° from the section spec table |
| T10 | "2014 GT with the 3.7 V6" | Contradiction not noticed | ✅ Asks which engine. The answer resolves it in one turn. |

**Fixes made from these findings**
- Variant detection covers caliper type (`— 2/4/6 Piston`), side (LH/RH), axle (Front/Rear) and transmission
  suffixes. Options come from the whole catalogue, filtered by vehicle and by "front"/"rear" in the question.
  The reply ("4 piston", "driver side", "second") is matched to an option and becomes the focus procedure.
  A variant already named in the question is used directly instead of being asked back.
- Main-procedure choice scores the top hits by rank, by how well the doc type fits the task (e.g. Removal and
  Installation for a replacement) and by each title word the technician used. Previously it was simply the top hit.
- A spec/torque/gap question adds the best-matching row block of the section's SPECIFICATIONS table.
- Short replies ("yes", "coupe") reuse the technician's original wording of the job for search and reranking.
- Safety: an acknowledgement never changes the job. The extraction LLM cannot re-open an acknowledged hazard
  (it re-read the history). "Fuel smell/puddle" was added to the rule-based hazards.
- A trim that contradicts a stated engine ("GT" + "3.7") is a conflict. Choosing one value resolves it.
- Validator: each end of a range ("1.25 - 1.35 mm") is a checkable value. Titles and section names count as
  evidence. The diagram fallback uses the cited procedure.
- Follow-ups: the active procedure keeps its vocabulary, so "camshaft sprocket bolts" stays on Timing Drive
  Components. Manual-term synonyms were added (cam sprocket/phaser → VCT, timing chain → Timing Drive Components, ATF).
- Readiness ignores look-alike catalogue matches ("Stay Pad" for "pads") when deciding which variant to ask about.
- Fallback excerpts are ranked by relevance to the question, and long spec tables are cut to the rows asked about.
- Groq: the on-demand tier has a **200,000 tokens/day** cap on `gpt-oss-120b`, which a day of testing used up.
  The client now falls back to `LLM_FALLBACK_MODEL` (`gpt-oss-20b`, which has its own quota) on rate limits.
  The same validator applies.

**Remaining issues seen**
- The smaller fallback model sometimes omits citations, so the answer is rejected and the technician gets
  focused manual excerpts instead.
- Labels can be slightly off even when values are right. T02 called the 115 Nm 2-piston *anchor-plate* row
  applicable to a 4-piston caliper; the 4/6-piston caliper bolt is also 115 Nm. Values are checked, wording is not.
- Retrieval benchmark after the fixes: Recall@5 / MRR unchanged (0.958 / 0.788). Completeness 0.92 → 0.97,
  citation page 0.65 → 0.75, figure P/R 0.73 → 0.84. Gate benchmark unchanged (1.0, 0 premature retrievals).
