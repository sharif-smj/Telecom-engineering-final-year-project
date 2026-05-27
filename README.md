# GSM Signal Denoising and AMC for Rural Uganda

This repository contains the final-year project workspace for **DAE-enhanced Automatic Modulation Classification (AMC)** in Uganda telecom scenarios. It combines research notes, report writing, interactive simulations, and presentation artifacts.

## Repository Map
- `ProjectReport.md`: primary working report/proposal manuscript.
- `simulations/`: interactive HTML demos (signal denoising, spectrum guardian, coverage, classifier, end-to-end pipeline).
- `presentation-v2/`: current presentation deck (`presentation_v2.html`, `style.css`, `script.js`, `assets/`).
- `presentation_v1/`: earlier deck snapshot.
- `reports/`: technical literature summaries and supporting source files.
- `reports/AI_Enhanced_Rural_GSM_Connectivity.pdf`: AI-enhanced rural GSM connectivity reference/report.
- `reports/LowSNR_Consolidated_Claims.md`: reconciled working claims sheet for writing/slides.
- `references/`: captured reference pages and citation images.
- `NCC/`: related NCC 2026 strategy, UCC proposal material, source captures, supervisor feedback, and proposal QA artifacts.
- `deliverables/`: exported submission assets:
  - `deliverables/papers/`
  - `deliverables/presentations/`
  - `deliverables/proposals/`
- `assets/maps/`: map files used across materials.
- `docs/WORK_RECONCILIATION.md`: canonical mapping of notes to deliverables and next actions.

## Run Locally
This project is static HTML/CSS/JS (no package install required).

```powershell
python -m http.server 8000
```

Then open:
- `http://localhost:8000/simulations/index.html`
- `http://localhost:8000/presentation-v2/presentation_v2.html`
- `http://localhost:8000/deliverables/presentations/presentation_panel_jan2026.html`

## Working Rules
- Keep report/proposal edits in `ProjectReport.md`.
- Keep simulation logic inside `simulations/`; avoid embedding large scripts in unrelated files.
- Put new exports (`.pptx`, `.pdf`, `.docx`) under `deliverables/` instead of root.
- Keep evidence and citations traceable to `references/` or `reports/`.
- Use focused commits by workstream (`report`, `simulations`, `presentations`, `references`).

## Current Focus
The structure cleanup has started; next priority is consolidating overlapping Low-SNR notes into one citation-ready summary and syncing those facts into `ProjectReport.md`.
