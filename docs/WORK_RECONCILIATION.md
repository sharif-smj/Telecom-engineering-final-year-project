# Work Reconciliation

This file reconciles scattered notes with active deliverables, so future edits stay focused.

## Canonical Sources by Purpose
- `ProjectReport.md`: primary manuscript for chapters, objectives, methodology, and references.
- `DAE_AMC_Use_Cases.md`: scenario narrative for demos and presentation storytelling.
- `Uganda-Mobile-Network-Noise-and-Mitigation.md`: broad Uganda telecom context and policy background.
- `reports/LowSNR_AMC_Denoising_Research.md`: deep technical literature synthesis (2020-2025).
- `reports/LowSNR_Service_Resilience.md`: policy/enforcement framing tied to UCC constraints.
- `reports/LowSNR_Consolidated_Claims.md`: normalized working sheet for chapter and slide drafting.

## Supporting (Non-Canonical) Notes
- `reports/LowSNR_notes.txt`, `reports/LowSNR_Research_Notes.txt`, `reports/Spectrum_Congestion_Notes.txt`: draft/compiled note dumps used as source extraction.
- `reports/UCC_Market_Report_June2023.txt`: OCR/text extraction from report PDF.
- Rule: do not cite these directly in final writing without verifying against original PDF/web source.

## Mapping Notes to Deliverables
- Report chapters (`ProjectReport.md`):
  - Chapter 1 problem framing: `Uganda-Mobile-Network-Noise-and-Mitigation.md`, `LowSNR_Service_Resilience.md`
  - Chapter 2 literature and metrics: `LowSNR_AMC_Denoising_Research.md`, `LowSNR_Research_Notes.txt`
  - Chapter 3 methodology and validation framing: `LowSNR_AMC_Denoising_Research.md`, `DAE_AMC_Use_Cases.md`
- Simulation stories (`simulations/*.html`): scenario cues from `DAE_AMC_Use_Cases.md`.
- Presentations (`presentation-v2/`, `deliverables/presentations/`): distilled claims from chapter-ready report content.

## Immediate Reconciliation Tasks
1. Consolidate numeric claims (for example, 71.76%, -90 dBm, QoS thresholds) into one verified citation block in `ProjectReport.md`.
2. Promote reusable facts from `reports/*.txt` into one cleaned markdown source, then treat the `.txt` files as archive-only.
3. Keep simulation claims aligned with the latest chapter text before exporting new slides.
4. Store all new exports in `deliverables/` to avoid root-level clutter.

## Status Snapshot (2026-03-05)
- Structure cleanup started: root artifacts moved into `deliverables/` and `assets/maps/`.
- Consolidated drafting sheet created at `reports/LowSNR_Consolidated_Claims.md`.
- Citation normalization against primary sources is still pending.
