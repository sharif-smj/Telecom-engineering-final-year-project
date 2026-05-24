# FYP Slide Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a submission-ready final-year project presentation deck today, supported by real preliminary controlled simulation artifacts rather than fabricated completed-project results.

**Architecture:** Generate a small reproducible NumPy/Pillow simulation that creates synthetic GSM-family/proxy I/Q samples, applies controlled low-SNR impairment, applies a reproducible denoising front end, and exports metrics/figures. Build a fresh editable PowerPoint deck with the bundled artifact-tool presentation runtime, using the simulation artifacts as proof objects and the existing project report/deck as content sources.

**Tech Stack:** Bundled Python at `/Users/sharif/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`, NumPy, Pandas, Pillow, bundled Node.js, `@oai/artifact-tool`, existing Markdown/PPTX source material in `/Users/sharif/telecom/final-year-project`.

---

## File Structure

- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/run_preliminary_dae_amc.py`
  - Responsibility: deterministic controlled simulation, metrics generation, and figure export.
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/results.csv`
  - Responsibility: machine-readable preliminary metrics by SNR level and pipeline.
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/confusion_baseline.csv`
  - Responsibility: confusion counts for the noisy-signal baseline classifier.
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/confusion_denoised.csv`
  - Responsibility: confusion counts for the denoising-assisted classifier.
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/*.png`
  - Responsibility: slide-ready proof figures: noisy/denoised waveform, constellation comparison, accuracy chart, confusion matrix image.
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/README.md`
  - Responsibility: provenance note explaining that artifacts are preliminary controlled simulation outputs.
- Create: `/Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/slides/slide-*.mjs`
  - Responsibility: editable artifact-tool slide modules.
- Create: `/Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output/fyp-final-project-presentation-2026-05-24.pptx`
  - Responsibility: final editable PowerPoint deck for submission.
- Create: `/Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output/fyp-final-project-presentation-2026-05-24.pdf`
  - Responsibility: print/share fallback if PDF conversion is available.

## Task 1: Generate Preliminary Simulation Artifacts

**Files:**
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/run_preliminary_dae_amc.py`
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/results.csv`
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/confusion_baseline.csv`
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/confusion_denoised.csv`
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/signal_comparison.png`
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/constellation_comparison.png`
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/accuracy_by_snr.png`
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/confusion_denoised.png`
- Create: `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/README.md`

- [ ] **Step 1: Create the simulation script**

Write `/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/run_preliminary_dae_amc.py` with these implementation requirements:

```python
SEED = 20260524
CLASSES = ["GMSK-like", "QPSK", "8PSK", "16QAM"]
SNR_LEVELS = [-8, -4, 0, 4]
WINDOW = 256
SAMPLES_PER_CLASS_PER_SNR = 120
```

The script must:

- Generate deterministic synthetic complex I/Q windows for the four classes.
- Add AWGN plus a narrowband sinusoidal interference term.
- Apply a simple reproducible denoising stage using FFT-domain spectral shrinkage and normalization.
- Extract features from raw noisy and denoised windows: amplitude mean, amplitude standard deviation, phase-difference mean, phase-difference standard deviation, real/imag variance, kurtosis-like fourth moment, and peak-to-average ratio.
- Split train/test deterministically using alternating sample indices.
- Train a nearest-centroid classifier for the noisy baseline and another for the denoised pipeline.
- Export accuracy by SNR and pipeline to `results.csv`.
- Export baseline and denoised confusion matrices to CSV.
- Render all figures using Pillow only, not Matplotlib, so the script works with the bundled runtime.

- [ ] **Step 2: Run the simulation**

Run:

```bash
/Users/sharif/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  /Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/run_preliminary_dae_amc.py
```

Expected:

- Script exits with code `0`.
- The output folder contains `results.csv`, two confusion CSVs, four PNGs, and `README.md`.
- Terminal output includes an overall baseline accuracy and denoising-assisted accuracy.

- [ ] **Step 3: Verify artifact integrity**

Run:

```bash
find /Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24 \
  -maxdepth 1 -type f -print | sort

/Users/sharif/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
from pathlib import Path
import pandas as pd
base = Path("/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24")
results = pd.read_csv(base / "results.csv")
required = {"snr_db", "pipeline", "accuracy", "macro_f1", "samples"}
assert required.issubset(results.columns), results.columns
assert set(results["pipeline"]) == {"Noisy baseline", "Denoising-assisted"}
assert results["accuracy"].between(0, 1).all()
for name in ["signal_comparison.png", "constellation_comparison.png", "accuracy_by_snr.png", "confusion_denoised.png"]:
    assert (base / name).stat().st_size > 1000, name
print(results)
PY
```

Expected:

- All required files exist.
- Accuracy values are numeric and between `0` and `1`.
- PNG files are non-empty.

- [ ] **Step 4: Commit the simulation artifacts**

Run:

```bash
cd /Users/sharif/telecom/final-year-project
git add reports/preliminary_simulation_2026_05_24
git commit -m "Add preliminary DAE-AMC simulation artifacts"
```

Expected:

- A commit is created for the simulation evidence only.
- Existing unstaged rule-file changes remain unstaged unless already committed separately by the user.

## Task 2: Build the Submission Deck

**Files:**
- Create: `/Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/profile-plan.txt`
- Create: `/Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/source-notes.txt`
- Create: `/Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/slides/slide-01.mjs` through `slide-13.mjs`
- Create: `/Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output/fyp-final-project-presentation-2026-05-24.pptx`

- [ ] **Step 1: Create presentation workspace**

Run:

```bash
mkdir -p \
  /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/slides \
  /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/preview \
  /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/layout \
  /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/assets \
  /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output
```

Expected:

- The presentation workspace exists under `outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation`.

- [ ] **Step 2: Write source notes and profile plan**

Create `profile-plan.txt` with:

```text
task mode: create
primary deck-profile: engineering-platform
secondary profile gates: academic assessment deck, final-year project defense
required proof objects: DAE-AMC signal chain, controlled simulation setup, accuracy-by-SNR chart, noisy-vs-denoised signal figure, confusion matrix, scope/limitations note
source requirements: ProjectReport.md, existing root PPTX slide inventory, reports/preliminary_simulation_2026_05_24 outputs
brand constraints: no invented UCC/Kyambogo logos; use clean academic typography and telecom signal visuals
QA gates: every required project-presentation particular is represented; results are labelled preliminary controlled simulation; no field deployment or live-service improvement claim
known missing inputs: final completed model results and field validation are not available today
```

Create `source-notes.txt` with:

```text
Primary content sources:
- /Users/sharif/telecom/final-year-project/ProjectReport.md
- /Users/sharif/telecom/final-year-project/GSM Signal Denoising and Modulation Classification for Rural Uganda.pptx
- /Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/README.md
- /Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/results.csv
- /Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24/*.png

Evidence rule:
The deck uses preliminary controlled simulation artifacts for result slides. It does not claim completed field deployment, live call/USSD/SMS improvement, or physical tower-range extension.
```

- [ ] **Step 3: Create 13 slide modules**

Create editable artifact-tool slide modules with this slide spine:

1. `slide-01.mjs`: Title, students, supervisor, department.
2. `slide-02.mjs`: Background: rural GSM service and weak edge-of-coverage problem.
3. `slide-03.mjs`: Problem statement.
4. `slide-04.mjs`: Main objective and specific objectives.
5. `slide-05.mjs`: Justification, significance, and scope.
6. `slide-06.mjs`: Literature review and technical gap.
7. `slide-07.mjs`: Proposed DAE-AMC methodology signal chain.
8. `slide-08.mjs`: Preliminary controlled simulation setup.
9. `slide-09.mjs`: Preliminary result: noisy versus denoised signal and constellation.
10. `slide-10.mjs`: Preliminary result: baseline versus denoising-assisted accuracy by SNR.
11. `slide-11.mjs`: Discussion of preliminary results and limitations.
12. `slide-12.mjs`: Conclusion and recommendations.
13. `slide-13.mjs`: References and artifact provenance.

Each slide module must:

- Add exactly one slide.
- Use 16:9 slide size through the builder.
- Use a consistent dark-blue/teal academic engineering theme.
- Keep body text large enough for projection.
- Avoid unsupported completed-results wording.

- [ ] **Step 4: Build and render the deck**

Run:

```bash
NODE_PATH=/Users/sharif/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
/Users/sharif/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  /Users/sharif/.codex/plugins/cache/openai-primary-runtime/presentations/26.521.10419/skills/presentations/scripts/build_artifact_deck.mjs \
  --slides-dir /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/slides \
  --out /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output/fyp-final-project-presentation-2026-05-24.pptx \
  --preview-dir /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/preview \
  --layout-dir /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/layout \
  --contact-sheet /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output/contact-sheet.png \
  --slide-count 13
```

Expected:

- PPTX exists and has 13 slides.
- Preview PNGs exist for all 13 slides.
- `contact-sheet.png` exists.

- [ ] **Step 5: Commit the generated deck workspace**

Run:

```bash
cd /Users/sharif/telecom/final-year-project
git add outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation
git commit -m "Create final project presentation deck"
```

Expected:

- A commit is created for the deck output.

## Task 3: Export PDF And Run Submission QA

**Files:**
- Create if possible: `/Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output/fyp-final-project-presentation-2026-05-24.pdf`
- Read: `/Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output/contact-sheet.png`

- [ ] **Step 1: Convert PPTX to PDF if LibreOffice is available**

Run:

```bash
if command -v soffice >/dev/null 2>&1; then
  soffice --headless --convert-to pdf \
    --outdir /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output \
    /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output/fyp-final-project-presentation-2026-05-24.pptx
else
  echo "soffice not available; PDF export skipped"
fi
```

Expected:

- PDF is created if LibreOffice exists.
- If not available, the skipped export is disclosed in the final response.

- [ ] **Step 2: Scan deck text for unsafe claims**

Run:

```bash
/Users/sharif/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
from zipfile import ZipFile
from pathlib import Path
from xml.etree import ElementTree as ET
import re

pptx = Path("/Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output/fyp-final-project-presentation-2026-05-24.pptx")
ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
text = []
with ZipFile(pptx) as z:
    for name in z.namelist():
        if re.match(r"ppt/slides/slide\\d+\\.xml$", name):
            root = ET.fromstring(z.read(name))
            text.extend(t.text or "" for t in root.findall(".//a:t", ns))
joined = "\\n".join(text).lower()
blocked = [
    "field deployment completed",
    "improved live calls",
    "improved ussd",
    "improved mobile money",
    "extended tower range",
    "we achieved",
    "we demonstrated in the field",
]
hits = [phrase for phrase in blocked if phrase in joined]
assert not hits, hits
for required in ["problem statement", "objectives", "methodology", "preliminary", "conclusion", "references"]:
    assert required in joined, required
print("Deck text QA passed")
PY
```

Expected:

- Command prints `Deck text QA passed`.

- [ ] **Step 3: Inspect contact sheet**

Open or view:

```bash
open /Users/sharif/telecom/final-year-project/outputs/fyp-slide-recovery-2026-05-24/presentations/fyp-final-project-presentation/output/contact-sheet.png
```

Expected:

- The contact sheet shows 13 readable slides.
- Result slides contain actual generated figures.
- The deck is visually consistent and not a wall of text.

- [ ] **Step 4: Final git/status check**

Run:

```bash
cd /Users/sharif/telecom/final-year-project
git status --short --branch
```

Expected:

- Only known unrelated rule-file changes and `.superpowers/` scratch remain, unless separately committed.
- Simulation and deck commits are present.
