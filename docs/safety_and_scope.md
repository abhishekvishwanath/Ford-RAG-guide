# Safety and Scope

## Scope

* **Source of truth:** the supplied *2014* Ford Mustang Workshop Manual only. Other model years,
  other vehicles and engines/transmissions the manual does not cover are answered with an explicit
  out-of-scope message. Supported years/variants are read from the ingested manual profile.
* **No general knowledge for technical content.** Steps, values, tools, warnings and DTC meanings must
  come from retrieved manual evidence. If evidence is missing the assistant says what it could not find.
* The assistant provides **manual information retrieval**, not an inspection or diagnosis, and never
  states that a vehicle is safe to drive.

## Before retrieval

* The readiness gate must be `READY`; retrieval nodes refuse to run otherwise (tested).
* Variant fields that change the procedure are asked for; trim-name inferences (e.g. "GT" → 5.0L) are
  confirmed before use; conflicting statements are resolved first.
* **Immediate hazards** (fire/smoke, fuel leak, brake failure, unstable vehicle support, electrical
  hazards, injury) trigger a short stop-work message first; the conversation resumes only after the user
  confirms the situation is safe. Detection is rule-based (the LLM can only *add* a flag when its text
  matches the same hazard vocabulary), so it errs toward escalation.

## In answers

* WARNING, CAUTION, NOTICE and NOTE statements are kept with — and before — the steps they govern
  (structure-level reordering at ingestion, prompt rule, and a validator that re-inserts any dropped
  WARNING/CAUTION verbatim).
* Special tools and prerequisites are quoted as the manual states them.
* Variant-specific steps for other configurations are omitted and labelled as omitted.
* Values with units, part/tool numbers and DTCs must appear in the evidence; otherwise the model answer
  is rejected and verbatim manual excerpts are shown instead.
* Uncited claims about which equipment the vehicle has are removed.
* Diagram callout text obtained by OCR is labelled as such; the original diagram is always shown as
  the authority. No image is generated or altered.

## Privacy and security

* VINs are not required; VIN/e-mail patterns are masked in logs. The source PDF's personal e-mail
  watermark is stripped during parsing and never indexed.
* Secrets only in `.env` (git-ignored); admin endpoints require `X-Admin-Key`; images are served by
  validated asset id, never by path; request size limits, rate limiting and session isolation.

## Known limitations relevant to safety

* Table reconstruction is heuristic; a small number of un-ruled tables fall back to text.
* Rule-based hazard detection can miss unusual phrasings (the LLM can add, but not create, a hazard).
* Answer faithfulness beyond numbers/identifiers (e.g. paraphrase drift in step wording) is not
  automatically verified; the UI always shows citations and page references for checking.
