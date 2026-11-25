# Source Manual Assessment

Inspection performed 2026-11-25 with PyMuPDF over all 5,007 pages
(`scripts/inspect_manual.py` reproduces these numbers). Values below are measured
from the file, not assumed.

## 1. File facts

| Property | Value |
|---|---|
| File | `data/raw/319-Ford-Mustang-Workshop-Service-Manual-_-2004-2014.pdf` |
| Size | 205.9 MB (205,916,729 bytes) |
| Pages | 5,007 (US Letter, 612 × 792 pt) |
| PDF version | 1.6, produced by Acrobat Distiller 10.1.9 from PScript5 |
| Created / modified | 2014-04-25 / 2015-02-26 |
| Encryption | Owner-password only (Standard V4 R4 AES-128). Opens and extracts **without** a user password. |
| Text layer | Present on every content page. **Not a scanned document.** |
| Bookmarks (TOC) | 1,140 entries, depth 1–5 (L1: 5, L2: 37, L3: 145, L4: 433, L5: 520) |

## 2. What the manual actually covers

**The filename says "2004-2014" but the content is the 2014 model year only.**
Every page footer reads `2014 Mustang Workshop Manual` (5,934 occurrences, no other year),
and all 940 procedure revision dates are in 2013.

- **Model year:** 2014 (S197 platform, final year). No 2004–2013 content was found.
- **Market:** US English ("SEB~us~en" in the source path). Market is not stated elsewhere.
- **Engines (by mention count):** 5.0L (365), 3.7L (345), 5.8L supercharged GT500 (272).
  Stray mentions of 5.4L (8), 3.3L, 3.8L, 4.6L etc. are cross-references/legacy notes, not covered engines.
- **Transmissions:** 6R80 6-speed automatic (sections 307-xx), MT82 Getrag 6-speed manual (308-03A).
  TR6060 (GT500 manual) is mentioned 3× but has no dedicated section — **coverage of TR6060 is uncertain.**
- **Body styles:** Coupe and Convertible (body sections contain Coupe/Convertible-specific procedures).
- **Variant naming:** GT500 / Shelby, GT, V6, Boss 302 (1 mention).
- **Decision (user-confirmed):** the assistant is strictly scoped to MY2014. Other years → `OUT_OF_SCOPE`.
  Supported years are read from the ingested manual profile, not hardcoded.

## 3. Organisation

The PDF is a browser print-out of Ford's online workshop manual. Each procedure is a separate
HTML document ("SEB" file), printed with its own pagination.

- **Hierarchy (bookmarks):** `1. General Information`, `2. Chassis`, `3. Powertrain`, `4. Electrical`,
  `5. Body and Paint` → groups (e.g. `03 Engine`) → sections (`303-01B Engine - 5.0L`) →
  document types (`Removal and Installation`) → procedures (`Lower Arm`).
  Note: bookmark depth is not uniform (Body sections sit one level higher), and a few bookmarks
  point to wrong pages (`09 Exhaust System` → 674, `15 Audio` → 2807). The parser must tolerate this.
- **Procedure documents:** 941 distinct `SEBxxxxx.HTM` IDs, present in every page footer →
  stable procedure identifier.
- **Document types (first page header):** REMOVAL AND INSTALLATION 386, GENERAL PROCEDURES 146,
  DIAGNOSIS AND TESTING 99, DESCRIPTION AND OPERATION 99, IN-VEHICLE REPAIR 77, SPECIFICATIONS 64,
  DISASSEMBLY/ASSEMBLY (various) ~30.
- **Sections:** 83 distinct `SECTION nnn-nn[X]: Name` identifiers (e.g. `204-01`, `303-01B`, `307-01`).

## 4. Page layout conventions (measured on representative pages)

| Element | Signature |
|---|---|
| Procedure title | Arial-Bold 12 pt, first line of "Page 1 of N" |
| Header block | Arial-Italic 9 pt: `SECTION 204-01: Front Suspension` / `2014 Mustang Workshop Manual` / doc type |
| Revision | Arial 8 pt `Procedure revision date: MM/DD/YYYY` |
| Sub-headings | Arial-Bold 10 pt (`Special Tool(s)`, `Removal`, `Installation`) |
| Applicability qualifiers | Arial-Bold 10 pt lines: `All vehicles`, `Vehicles with 5.8L (4V) engine`, `Vehicles with automatic transmission`, `Vehicles with 6 piston calipers`, … |
| Steps | Number span `7.` followed by step text |
| Admonitions | Arial-BoldItalic `WARNING:` (290 pages), `CAUTION:`, `NOTICE:` (738), `NOTE:` (1,751) |
| Tables | Printed HTML tables (Item / Part Number / Description; torque tables; pinpoint tests). Drawn as vector lines; `find_tables()` does not detect them reliably → layout-based row reconstruction needed. |
| Footer (boilerplate) | Times 12 pt: `Page X of N`, `2014 Mustang Workshop Manual`, print date `2014-04-23/24`, `file://C:\TSO\…SEBxxxxx.HTM` |
| Watermark | A personal e-mail address in 4 pt text on ~1,000 pages. **Must be stripped** from all chunks (privacy). |
| Banner images | 3 images per page (600×100 px at top/bottom). Decorative → filtered. |

**Page numbering:** there is no global printed page number. The printed reference is
procedure-local `Page X of N`. Citations therefore use: section + procedure (SEB id) +
`Page X of N` + PDF page index.

## 5. Visual content

- 30,616 image placements in total; 15,021 are banners (3/page).
- 5,421 "content-sized" images (> 80 pt each side) on 2,781 pages.
- Special-tool icons: ~99×99 px images next to tool numbers.
- Exploded views / location views (e.g. 650×650 px) carry **callout numbers, torque values
  (e.g. `103 Nm (76 lb-ft)`) and figure IDs (`N0133215`) inside the raster**. These values are
  *not* in the text layer. Figure OCR (Tesseract 5) is used to make them searchable, but the
  original image remains the authority and OCR text is labelled as such.
- Item/Part/Description legends for each exploded view are in the text layer directly below the figure.

## 6. Diagnostic content

- 549 distinct DTC-shaped codes (B/C/P/U + 4 hex chars).
- 593 pages mention Pinpoint Tests; symptom charts in DIAGNOSIS AND TESTING documents.

## 7. Low-text pages

482 pages have < 40 characters of body text after boilerplate removal. Sampled pages are the
blank tail pages of procedures (only banners), or pages that contain only a figure.
No scanned text pages were found in sampling; OCR routing remains enabled for safety and is
reported per page.

## 8. Uncertainties

- Whether TR6060 (GT500 manual transmission) procedures exist beyond passing mentions.
- Market applicability beyond the "us/en" source path.
- A small number of bookmark targets are wrong; the parser uses the SEB footer as the primary
  procedure boundary and bookmarks only for hierarchy labels.
- Some procedure applicability is only implied by the section title (e.g. `303-01A Engine - 3.7L`);
  this is recorded as `section`-derived applicability, distinct from explicit in-text qualifiers.

## 9. Parsing implications

1. Group pages by SEB id → procedure documents (941). This is a stronger boundary than bookmarks.
2. Use font signatures for heading / qualifier / admonition / step detection (no hardcoded headings).
3. Strip boilerplate footer, banners, and watermark.
4. Reconstruct tables from span geometry.
5. Extract content images, link them to the nearest preceding step / heading, OCR them.
6. OCR route pages without a text layer (expected near-zero).
7. Embedding model max sequence is 256 tokens → embed chunks as multiple windows.
