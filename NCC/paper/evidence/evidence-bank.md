# Evidence Bank

Status date: April 23, 2026

Purpose:

- Track the claims we want to make and the evidence each claim needs.
- Separate already-sourced external context from project-generated evidence still to be produced.
- Keep the eventual paper honest, specific, and review-ready.

Legend:

- `ready`: public source already archived or current project material already exists
- `partial`: some support exists but needs sharper extraction or confirmation
- `missing`: must be generated, measured, or written before submission

## External context evidence

| Claim or section need | Evidence type | Current status | Best current source IDs | Notes |
| --- | --- | --- | --- | --- |
| Uganda still faces access and usage gaps | official sector context | ready | OFF-08 | Use for rural and underserved framing |
| UCC is pushing market-driven innovation | official strategic framing | ready | OFF-05 | Good for aligning to 2026 theme |
| Affordability and investment constraints matter | official economic context | ready | OFF-09 | Useful when arguing software-first value |
| Compliance and signal-integrity issues remain relevant | official regulatory context | ready | OFF-11 | Supports interference and signal-quality motivation |
| Satellite and future-network readiness are active policy themes | official infrastructure context | ready | OFF-12 | Useful for future-network language without overclaiming |
| UCC values proposal quality, translation to practice, and collaboration | official research framework | ready | OFF-06, OFF-07, OFF-19 | Use for grant-winning alignment logic |

## NCC format and winner evidence

| Claim or section need | Evidence type | Current status | Best current source IDs | Notes |
| --- | --- | --- | --- | --- |
| 2026 submission categories, dates, and tracks | official conference metadata | ready | OFF-01, OFF-02 | Deadline and acceptance timing are already confirmed |
| Template expectations appear conventional and somewhat stale | official template context | ready | OFF-03, OFF-04 | Confirm final formatting before submission |
| 2025 winners were practical, solution-first, and deployment-facing | official winner evidence | ready | WIN-01, WIN-02 | Important for title and abstract posture |
| At least one full winner paper is available for structure analysis | full accessible paper | ready | WIN-05 | FareFlow is our clearest format benchmark |
| Best-paper network-infrastructure framing example | winner-adjacent article | ready | WIN-06 | Useful for network performance outcome language |
| Older NCC framing around inclusion and digital transformation | historical artifact | ready | WIN-07, WIN-08 | Helpful for continuity and long-run conference identity |

## Project-generated evidence still needed

| Claim or section need | Evidence type | Current status | Existing reference | What we still need |
| --- | --- | --- | --- | --- |
| What exact GSM signal classes or conditions we target | project definition | partial | [ProjectReport.md](/Users/sharif/telecom/final-year-project/ProjectReport.md) | Lock the signal/problem scope for the paper |
| Denoising pipeline description | method description | partial | Project repo and v1 report | Clean block diagram and concise method narrative |
| Dataset or signal-generation setup | experiment context | partial | Project repo and v1 report | Final paper-ready description and parameter table |
| Baseline comparison | experiment result | missing | project experiments | Need explicit baseline models or no-denoising comparison |
| Improvement metric beyond accuracy | experiment result | missing | project experiments | Prefer usable-SNR shift, recovery rate, precision-recall, or error-rate trend |
| Practical implication for operator or regulator workflows | deployment interpretation | partial | analysis files | Need one crisp workflow example |
| Limitations and non-claims | discussion evidence | partial | analysis files | Tie to lack of live deployment and field validation |

## Supervisor concept note evidence screen

| Claim or section need | Evidence type | Current status | Best current source IDs | Notes |
| --- | --- | --- | --- | --- |
| Practical pilot framing and software-defined coverage narrative | user-supplied strategic input | ready for motivation only | SUP-01 | Good for future pilot and impact framing, not automatic evidence |
| Exact lab metrics from the concept note | candidate empirical evidence | missing | SUP-01 | Needs artifact verification before NCC use |
| Device deployability metrics from the concept note | candidate empirical evidence | missing | SUP-01, ProjectReport.md:380 | Only usable if quantization and latency artifacts exist |
| Exact pilot and field outcomes from the concept note | projected pilot target | missing | SUP-01 | Keep out of abstract/results/conclusion |
| UCC-side alignment for underserved access and practical innovation | official external context | ready | OFF-05, OFF-07, OFF-08, OFF-18 | Safe to use as high-level policy alignment |
| GSMA-side alignment language | external alignment candidate | partial | SUP-01 | Needs official GSMA source before public factual use |

See:

- [analysis/supervisor-concept-note-claim-ledger.md](/Users/sharif/telecom/final-year-project/NCC/analysis/supervisor-concept-note-claim-ledger.md)
- [analysis/practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md)

## Must-extract items from the current project before drafting

- Exact model names, inputs, outputs, and training setup
- Exact test conditions and low-SNR range covered
- Best-performing configuration and baseline comparison
- Any plots or tables that can be translated into service-level meaning
- Anything that demonstrates robustness rather than one-off peak accuracy
- Any artifact that could verify a supervisor-note lab or device claim

## Evidence priorities for a winning submission

1. A strong Uganda problem statement backed by official UCC material
2. A method description that sounds practical and reproducible
3. A result that translates into network-service meaning
4. A realistic deployment story for UCC or operator use
5. A careful limitations section that improves trust

## Reference files

- [sources/manifest.csv](/Users/sharif/telecom/final-year-project/NCC/sources/manifest.csv)
- [analysis/ncc-landscape.md](/Users/sharif/telecom/final-year-project/NCC/analysis/ncc-landscape.md)
- [analysis/winner-patterns.md](/Users/sharif/telecom/final-year-project/NCC/analysis/winner-patterns.md)
- [analysis/project-alignment.md](/Users/sharif/telecom/final-year-project/NCC/analysis/project-alignment.md)
- [analysis/supervisor-concept-note-claim-ledger.md](/Users/sharif/telecom/final-year-project/NCC/analysis/supervisor-concept-note-claim-ledger.md)
- [analysis/practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md)
