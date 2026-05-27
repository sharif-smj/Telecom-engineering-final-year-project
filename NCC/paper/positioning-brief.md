# NCC Paper Positioning Brief

Status date: April 23, 2026

Purpose:

- Keep the eventual NCC paper strategically aligned before drafting starts.
- Translate the analysis files into a compact writing brief.
- Prevent drift back into a generic final-year-project narrative.

Non-draft rule:

- This file is a positioning control sheet.
- Do not turn these bullets into polished paper prose here.

## Working identity

- Paper identity: infrastructure-first GSM denoising paper
- Primary promise: recover more usable signal intelligence from weak and noisy GSM conditions
- Practical outcome: improve effective service reach interpretation at the edge of rural coverage
- Technical engine: denoising autoencoder plus modulation-recognition pipeline
- Country context: Uganda
- Likely users: UCC, UCUSAF, operators, and rural connectivity planners

## Submission target

- Primary track: Track 1 - Digital Infrastructure, Connectivity and Future Networks
- Backup track: Track 4 - Artificial Intelligence, Data Science and Trustworthy Technology
- Optional policy bridge: Track 8 - Policy, Regulation, Cybersecurity and Innovation Ecosystems

## What the paper must feel like

- Locally grounded
- Implementation-aware
- Useful to both academics and sector practitioners
- Honest about limits
- Strong on deployment logic
- Written for mixed reviewers, not ML specialists only

## The story we are selling

- Uganda has weak and noisy service zones where usable GSM signal interpretation degrades.
- Extending physical infrastructure is costly and slow.
- A denoising-first software pipeline may recover more actionable signal structure from low-SNR conditions.
- That makes coverage troubleshooting and underserved-area planning more evidence-driven.

## Claims to emphasize

- improves effective or usable service reach
- improves edge-of-coverage signal interpretation
- recovers more useful information from weak and noisy GSM signals
- supports troubleshooting, interference assessment, and rural coverage decisions
- extracts more value from existing network assets before expensive physical expansion

## Claims to avoid

- physically extends radio range
- increases tower coverage footprint
- guarantees live network QoS improvement
- proves nationwide field performance
- solves rural connectivity on its own
- uses `on-device` or `without new towers` as if already proven

Evidence basis:

- [analysis/project-alignment.md](/Users/sharif/telecom/final-year-project/NCC/analysis/project-alignment.md)
- [analysis/supervisor-concept-note-claim-ledger.md](/Users/sharif/telecom/final-year-project/NCC/analysis/supervisor-concept-note-claim-ledger.md)

## Reviewer hooks to keep visible

- Infrastructure hook: software-first improvement to cell-edge utility
- Innovation hook: practical and market-relevant value from existing assets
- Regulator hook: supports service evidence, interference awareness, and underserved-area decisions
- Academic hook: denoising is tied to a concrete communications problem
- Operator hook: lower-cost diagnostic and planning value

## Evidence that must appear in the paper

- Uganda-side problem statement
- reason low-SNR and noisy coverage matter in practice
- method block showing denoising and classification stages
- measured improvement beyond raw classification accuracy alone
- practical deployment path
- limitations and next-step validation path
- claim-ledger signoff for any practical concept-note number

## Evidence layers

- `Layer A: NCC evidence core`
  - official UCC / NCC context
  - literature benchmarks
  - project-generated baseline-versus-hybrid results
  - any newly verified practical metrics
- `Layer B: future deployment pathway`
  - Android + SDR kit concept
  - field pre-test idea
  - district pilot path
  - UCC-aligned scale story

Writing rule:

- The paper's results section can only draw from `Layer A`.
- `Layer B` belongs in motivation, discussion, impact, or future pilot language.

Evidence basis:

- [analysis/practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md)
- [analysis/supervisor-concept-note-claim-ledger.md](/Users/sharif/telecom/final-year-project/NCC/analysis/supervisor-concept-note-claim-ledger.md)

## Tone rules for the eventual draft

- Open with the service problem before the model
- Mention Uganda in the title or first two sentences
- Use plain-English problem framing before acronyms
- Translate metrics into operational meaning
- End with deployment logic, not abstract future-work filler
- If we mention the concept note's applied path, label it as next-step deployment logic rather than present-tense proof

## References to consult while drafting

- [analysis/project-alignment.md](/Users/sharif/telecom/final-year-project/NCC/analysis/project-alignment.md)
- [analysis/ncc-landscape.md](/Users/sharif/telecom/final-year-project/NCC/analysis/ncc-landscape.md)
- [analysis/winner-patterns.md](/Users/sharif/telecom/final-year-project/NCC/analysis/winner-patterns.md)
- [analysis/ucc-priority-map.md](/Users/sharif/telecom/final-year-project/NCC/analysis/ucc-priority-map.md)
- [analysis/supervisor-concept-note-claim-ledger.md](/Users/sharif/telecom/final-year-project/NCC/analysis/supervisor-concept-note-claim-ledger.md)
- [analysis/practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md)
- [sources/manifest.csv](/Users/sharif/telecom/final-year-project/NCC/sources/manifest.csv)
- [ProjectReport.md](/Users/sharif/telecom/final-year-project/ProjectReport.md)
