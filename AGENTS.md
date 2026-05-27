# Agent Operating Rules

## Project Overview
- Treat this repository as the active final-year project workspace for GSM signal denoising and automatic modulation classification in the Uganda telecom context.
- Support four active workstreams: browser-based simulations in `simulations/`, manuscript and project narrative files at the repository root and in `reports/`, source evidence in `references/`, `references/images/`, `assets/maps/`, and `docs/`, and NCC/UCC strategy material in `NCC/`.
- Keep technical claims, terminology, and scenario language aligned across `ProjectReport.md`, `DAE_AMC_Use_Cases.md`, `reports/*.md`, `docs/WORK_RECONCILIATION.md`, and any edited simulation page.
- Treat audience comprehension as a project constraint. Assume some supervisors, reviewers, or professors may understand classical telecom blocks better than machine-learning terminology.
- Preserve the current working tree reality. Do not assume older cleanup plans were completed.

## Tech Stack
- Use static HTML, CSS, and vanilla JavaScript for interactive pages.
- Expect most simulation pages to be single-file HTML documents with inline `<style>` and `<script>` blocks.
- Treat `simulations/uganda_svg_data.js` as the only shared local JavaScript data asset currently present.
- Expect existing remote browser dependencies only where already used: Google Fonts plus `three.js`, `OrbitControls.js`, and `SVGLoader.js` loaded from CDNs.
- Do not introduce frameworks, package managers, bundlers, transpilers, or a build system without explicit approval.

## Current Repository Layout
- Treat these directories as present and active: `simulations/`, `reports/`, `references/`, `references/images/`, `assets/maps/`, `docs/`, and `NCC/`.
- Treat `NCC/` as the in-repository home for NCC 2026 strategy, sources, paper drafts, UCC proposal submissions, supervisor feedback, and proposal QA artifacts.
- Treat these root files as active source material: `ProjectReport.md`, `DAE_AMC_Use_Cases.md`, `README.md`, and the root-level office/PDF artifacts currently checked in.
- Treat `.gemini-clipboard/` and `.playwright-mcp/` as tool state. Do not edit them unless the task explicitly targets those directories.
- Keep new implementation and literature evidence files with `references/` or `reports/` instead of scattering them across the root.
- Keep new NCC/UCC strategy, proposal, feedback, and source-capture files under the matching `NCC/` subfolder.
- Keep new reusable geographic assets in `assets/maps/`.

## Build And Run Commands
- Serve the repository from the root with `python -m http.server 8000`.
- Open the simulation portal with `start http://localhost:8000/simulations/index.html`.
- Open individual simulation pages directly when validating specific changes.
- Inspect the tracked workspace with `rg --files simulations reports references docs assets NCC`.
- Check scope before and after edits with `git status --short` and `git diff --name-only`.
- Treat the repository as static. Do not assume `npm`, `pip install`, or other setup steps are required for normal page work.

## Testing And Verification
- Assume no automated test suite exists.
- Run manual browser QA for every changed page.
- Validate every edited UI at desktop and mobile widths, at minimum `1280px` and `375px`.
- Check for console errors, missing assets, broken internal navigation, and animation regressions.
- For simulation pages with controls, exercise at least two distinct scenario or slider settings and confirm displayed metrics update.
- For `simulations/uganda_terrain_map.html`, verify that CDN-hosted `three.js` dependencies load and that rotation, hover, and layer toggles still work.
- For research and manuscript edits, verify every changed claim against a primary source or a clearly identified repository source file.
- For model or experiment work, log dataset/source, preprocessing, impairment model, SNR range, random seed, model architecture, baseline AMC metrics, DAE-AMC metrics, reconstruction error, macro F1-score, confusion matrices, model size, inference latency, and failure cases.

## Coding And Content Rules
- Use 4-space indentation in HTML, CSS, and JavaScript.
- Keep semicolons in JavaScript.
- Use `camelCase` for JavaScript identifiers.
- Use `kebab-case` for CSS classes and IDs.
- Use lowercase filenames with underscores for new simulation pages.
- Define new colors and theme tokens in `:root` before using them.
- Keep simulation-specific logic inside the relevant file unless a local shared asset already exists and reuse is clearly justified.
- Prefer editing Markdown source files over regenerating `.pdf`, `.pptx`, or `.docx` artifacts.
- Do not rename or relocate root-level artifacts without approval.
- Keep citations traceable to a repository source file under `references/`, `reports/`, `NCC/sources/`, or another explicitly cited primary document.
- Treat `reports/*.txt` files as extraction notes. Do not use them as final citation authority without verifying the original source.

## Presentation Context
- When assisting with presentation-facing or supervisor-facing content, account for this project context: the user has reported that some professors understand telecom blocks and filters more readily than machine-learning abstractions.
- When assisting with explanatory writing, visuals, or demos, treat confusion between “algorithm” and “device” as an expected audience risk.
- When the task involves presentation support, prefer framing that connects the denoising autoencoder to familiar DSP ideas such as denoising, signal recovery, preprocessing, or adaptive filtering.
- When the task involves presentation support, prefer showing the system as a signal chain such as `received noisy signal -> denoising block -> cleaned signal -> modulation classifier -> decision`.
- When the task involves presentation support, prefer before/after signal evidence over abstract ML-first explanations.
- When the task involves presentation support, remember that the user wants simulations and demonstrations to paint a strong picture before the final implementation reveal.
- When the task involves presentation support, prefer visuals and demos that show generated or captured signals being corrupted, denoised, and then measured or classified.

## Current FYP Direction From NCC/UCC Work
- Keep the active final-year project technical core unchanged: GSM-family signal denoising plus automatic modulation classification for weak/noisy edge-of-coverage signal interpretation in rural Uganda.
- Treat `NCC/` as related supervisor-facing, administrative, proposal, and publication strategy material inside this repository, not as proof that implementation results already exist.
- Treat `/Users/sharif/telecom/final-year-project/NCC/ucc-submission/` as supervisor-facing UCC Research Support proposal material. Do not treat that proposal as the implementation source of truth for this repository.
- Use the broader UCC proposal for context, motivation, diagrams, and supervisor alignment only after checking its source map at `/Users/sharif/telecom/final-year-project/NCC/ucc-submission/source-map-and-compliance-checklist.md`.
- Frame the defensible technical claim as improved weak-signal recoverability or improved interpretation of noisy I/Q samples under controlled low-SNR or interference conditions.
- Do not claim physical tower-range extension, live call/USSD/SMS/Mobile Money improvement, completed field deployment, or measured SNR gain without verified experiment artifacts.
- Treat supervisor concept-note numbers such as 3-5 dB gain, 4.2 dB gain, 52% to 84% decode-rate improvement, 30% usable-radius increase, sub-1.2 MB model size, sub-40 ms latency, Wakiso pre-test outcomes, district rollout outcomes, and 25% failed-session reduction as expected targets or future-pilot language only until verified.
- Keep basic GSM/2G "kabiriti" phones central to the user problem. Treat Android phones and SDR equipment as research and measurement tools, not as a requirement for rural users.

## Prototype Direction
- Use Mode 2 as the current prototype default: Android on-device inference using exported ONNX DAE and AMC models.
- Train models off-device in Python/PyTorch, export to ONNX, then load the ONNX models into the Android workflow for inference and visualization.
- Use replay mode as the guaranteed demonstration path: the Android app loads prepared fixed-length two-channel I/Q windows and runs preprocessing, denoising, classification, metrics, and logging locally.
- Use receive-only SDR capture through USB OTG as a feasible extension when hardware and Android compatibility allow it. Keep SDR hardware options open: RTL-SDR Blog V4, ADALM-Pluto, HackRF One, and LimeSDR Mini.
- Treat microcontrollers as optional support components for control, triggering, power, or accessory handling, not as the main GSM I/Q receiver.
- Show the prototype as a signal-processing and measurement workflow: `I/Q sample source -> Android preprocessing -> ONNX DAE -> ONNX AMC -> visualization, latency, confidence, and observation log`.
- Do not describe the Android prototype as a live cellular modem modification, booster, or direct improvement to the phone's built-in calls, SMS, USSD, or Mobile Money sessions.
- When supervisors ask for "data collection tools," clarify whether they mean research instruments or physical equipment. For physical tools, include Android measurement phones, basic GSM/2G phones, receive-only SDR dongles, antennas, RF cables/adapters/filters/attenuators, USB OTG adapters, laptop/workstation, power banks, SIM cards/airtime/data, and field observation materials.

## Security And Data Rules
- Never add secrets, credentials, tokens, or personal data.
- Never add new remote scripts, fonts, or analytics without approval.
- Never change existing CDN dependency URLs or versions without approval.
- Never present unverified numeric, regulatory, or literature claims as final facts.
- Preserve provenance for captured reference pages and screenshots.
- For any future RF or field activity, keep the workflow receive-only unless separate authorization is obtained.
- Do not record, decode, store, or summarize private subscriber communication content.
- Treat WhatsApp supervisor messages and UCC proposal drafts as operational context, not final academic citations.

## Operating Boundaries
- Edit only the files required for the task.
- Preserve intentional constraints already present in repository guidance unless the current working tree proves they are stale.
- Ask for approval before performing broad directory moves, restoring deleted files, deleting checked-in artifacts, or regenerating binary exports.
- Do not “clean up” the root by moving files unless the user explicitly asks for that restructure.
- Do not assume missing paths should be recreated. Verify the current file map first.
- Keep documentation and rule updates explicit when repository drift is discovered.

## Known Danger Zones
- Treat these paths as intentionally removed from the current working tree even though other files still reference them: `presentation-v2/`, `presentation_v1/`, `deliverables/`, `simulations/spectrum_guardian.html`, `simulations/pipeline_visualizer.html`, and `simulations/coverage_map_3d.html`.
- Never restore or recreate those intentionally removed paths unless the user explicitly asks for that work.
- Re-check navigation in `simulations/index.html` and `simulations/uganda_terrain_map.html` after any edit because both currently link to missing pages.
- Treat `README.md` and `docs/WORK_RECONCILIATION.md` as partially stale where they describe deleted presentation and deliverable paths.
- Treat root-level `.pdf`, `.pptx`, and `.docx` files as active checked-in artifacts. Do not delete or relocate them without approval.
- Treat `reports/LowSNR_Consolidated_Claims.md` as a working synthesis file, not a substitute for primary-source verification.
