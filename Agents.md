# Repository Guidelines

## Project Structure & Module Organization
- `simulations/`: interactive DAE-AMC demos (`index.html`, signal visualizers, coverage maps, classifier pages) plus shared data like `uganda_svg_data.js`.
- `presentation-v2/`: primary slide deck (`presentation_v2.html`) with supporting `script.js`, `style.css`, and `assets/`.
- `presentation_v1/`: legacy single-file presentation archive.
- `references/`: saved source pages and `references/images/` snapshots used for citations/figures.
- `reports/`: research notes (`.md`, `.txt`) and supporting PDFs.
- `deliverables/`: exported submission assets (`papers/`, `presentations/`, `proposals/`).
- `assets/maps/`: reusable map files and geographic visuals.
- `docs/`: process docs such as `WORK_RECONCILIATION.md`.
- Repository root: keep only core control documents (for example `ProjectReport.md`, `README.md`, `AGENTS.md`).

## Build, Test, and Development Commands
- `python -m http.server 8000`: serve the repository locally (recommended for relative assets/scripts).
- `start http://localhost:8000/simulations/index.html`: open the simulation portal.
- `start http://localhost:8000/presentation-v2/presentation_v2.html`: open the main presentation.
- `start http://localhost:8000/deliverables/presentations/presentation_panel_jan2026.html`: open panel deck export.
- `rg --files simulations presentation-v2 references reports docs`: quickly inspect tracked files before commits.
- `git status` and `git diff --name-only`: verify scope before opening a PR.

## Coding Style & Naming Conventions
- Use 4-space indentation in HTML, CSS, and JavaScript; keep semicolons in JS.
- Keep implementations framework-free (vanilla HTML/CSS/JS) unless a change explicitly introduces tooling.
- File names: lowercase with underscores for simulation/presentation pages (for example, `signal_denoising_visualizer.html`).
- JavaScript: `camelCase` for variables/functions; CSS classes/IDs: `kebab-case`.
- Define palette and theme tokens in `:root` variables before adding new colors.

## Testing Guidelines
- No automated test suite is currently configured; run manual UI checks for every change.
- Validate each edited page in browser at mobile and desktop widths (at least 375px and 1280px).
- Confirm there are no console errors and all navigation controls, links, and animations work.
- For simulation updates, verify slider/input interactions and metric displays with at least two scenario settings.

## Commit & Pull Request Guidelines
- Prefer concise, artifact-first commit subjects in imperative style (for example, `simulations: refine coverage map legend`).
- Keep commits focused (do not mix report writing, simulation logic, and presentation styling in one commit).
- PRs should include:
  - clear summary of what changed and why,
  - list of affected paths,
  - screenshots/GIFs for UI updates,
  - linked issue/task when available.
- For updated claims or figures, include the supporting source file path under `references/` or `reports/` and reflect resolved conflicts in `docs/WORK_RECONCILIATION.md`.
