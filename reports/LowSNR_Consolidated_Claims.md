# Low-SNR Consolidated Claims (Working Sheet)

This file reconciles repeated claims across `reports/*.md` and `reports/*.txt` into one writing-ready sheet.
Use it as an internal drafting source, then verify each claim against its original paper/report before final submission.

## Core Operational Constraints (Uganda Context)
- Rural/edge deployments are frequently near the `-90 dBm` minimum service threshold used in UCUSAF framing.
- Low-SNR and interference environments reduce reliability of conventional AMC and increase misclassification risk.
- UCC QoS enforcement context includes rapid restoration expectations (commonly cited as `95%` of reported faults within `24 hours` in notes).

Source notes:
- `reports/LowSNR_Service_Resilience.md`
- `Uganda-Mobile-Network-Noise-and-Mitigation.md`

## AMC and Denoising Performance Benchmarks (From Literature Notes)
- MoE-AMC average accuracy across SNR sweep: `71.76%` (as cited in internal notes).
- Robust CNN (2023) reportedly reaches `86.1%` at `-2 dB` and higher at moderate SNR.
- DenoMAE-style denoising notes report strong gains in extreme low-SNR regimes (including sub-0 dB operation).

Source notes:
- `reports/LowSNR_AMC_Denoising_Research.md`
- `reports/LowSNR_Research_Notes.txt`

## Documented Service Impact Signals
- Interference is repeatedly linked to degraded QoS and dropped/blocked calls in Uganda-focused notes.
- Internal notes cite historical QoS snapshots where failures were heavily interference-driven in specific operators/areas.
- Enforcement examples include illegal repeaters/unauthorized links affecting licensed services.

Source notes:
- `reports/LowSNR_notes.txt`
- `reports/Spectrum_Congestion_Notes.txt`
- `reports/LowSNR_Service_Resilience.md`

## Data Gaps to Keep Explicit in Writing
- Limited recent public field measurements for rural Uganda SNR/BER are repeatedly noted.
- Several figures are currently secondary-note extracts and must be validated against primary publications before final thesis submission.

## Recommended Writing Workflow
1. Draft report text from this consolidated sheet.
2. Replace each claim with a verified citation from the original paper/report.
3. Update `ProjectReport.md` and presentation text in the same commit to keep claims synchronized.
