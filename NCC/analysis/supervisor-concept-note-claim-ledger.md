# Supervisor Concept Note Claim Ledger

Status date: April 23, 2026

Purpose:

- Convert the supervisor concept note into an evidence-safe drafting control sheet for the NCC paper.
- Force every practical claim into one of four statuses before it can enter the paper.
- Keep the NCC paper aligned with the winner-style format already captured in [winner-patterns.md](/Users/sharif/telecom/final-year-project/NCC/analysis/winner-patterns.md). Source IDs: `WIN-01`, `WIN-05`, `SUP-01`.

Updated interpretation:

- After the April 23, 2026 chat clarification, the concept-note numbers should be treated as `proposal-level expected results / practical ideology`, not as established project evidence. Source basis: `SUP-02`.

Status meanings:

- `verified empirical`: supported by experiment artifacts or reproducible project outputs
- `externally cited`: supported by official or reputable external sources
- `projected pilot target`: acceptable only in future-work, pilot, or deployment-pathway language
- `excluded from NCC paper`: do not use in the NCC draft unless evidence arrives

## Current rule

Any numeric claim from the concept note that is not yet backed by an artifact or primary source is barred from the NCC paper's:

- title
- abstract
- empirical results section
- conclusion

This rule follows the current project state, where the report defines metrics and logging plans but does not yet contain recorded experiment outputs. [ProjectReport.md](/Users/sharif/telecom/final-year-project/ProjectReport.md:373) [ProjectReport.md](/Users/sharif/telecom/final-year-project/ProjectReport.md:380) [weekly_log_timeline.md](/Users/sharif/telecom/final-year-project/reports/weekly_log_timeline.md:17)

Proposal-level exception:

- For a separate concept note, budget note, or proposal, expected results may be stated if they are clearly labeled as `expected`, `anticipated`, `target`, or `proposed pilot outcome`, following the supervisor clarification. Source basis: `SUP-02`.
- That exception does not promote those numbers into NCC paper evidence.

## Claim ledger

| Claim or claim family | Current status | Current support in pack | NCC usage rule | Next evidence needed | Source basis |
| --- | --- | --- | --- | --- | --- |
| Rural Uganda still depends on GSM or other legacy mobile services for essential communication | externally cited at high level | High-level dependency is consistent with the current project framing and Uganda access-gap material, but the exact `11M` figure is not backed in the current source pack | Use only the high-level dependence claim; do not use `11M` without a source | Official or reputable source for the exact numeric estimate | `SUP-01`; `OFF-08`; [ProjectReport.md](/Users/sharif/telecom/final-year-project/ProjectReport.md:94) |
| `30-40%` USSD/SMS failure in districts more than `8 km` from towers due to low SNR | excluded from NCC paper | No supporting UCC artifact or project dataset was found in the workspace | Do not use in paper prose or slides as fact | Official UCC report, field log, or supervisor dataset note | `SUP-01`; [practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md) |
| New towers cost `$150k-$250k` each and require grid power | excluded from NCC paper as exact numeric claim | The current pack supports only the general point that infrastructure expansion is costly and constrained, not this specific range | Use only general cost/slow-rollout language unless a primary or reputable cost source is provided | Tower cost source or budget sheet | `SUP-01`; `OFF-09`; `OFF-08` |
| Lightweight AI pipeline can run on low-cost Android phones with a `$25` SDR dongle | projected pilot target | No device-build artifact or BOM in the workspace | Use only in future pilot/deployment pathway sections | Hardware BOM, prototype photos, or device run logs | `SUP-01`; [practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md) |
| `3-5 dB` SNR gain on weak GSM signals | projected pilot target / expected result | The concept note states the target, and the supervisor later clarified that such figures may be used as expected results at proposal level | Use only as an expected result in proposal-style sections; do not use as a measured NCC result until backed | Result table, script output, or notebook export | `SUP-01`; `SUP-02`; [practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md) |
| `30%` increase in usable cell radius with zero operator changes | excluded from NCC paper as a measured claim | No derivation or field validation found in the workspace | Use only as a future pilot hypothesis, not a present result | Propagation-based derivation or field validation | `SUP-01`; [analysis/project-alignment.md](/Users/sharif/telecom/final-year-project/NCC/analysis/project-alignment.md) |
| Lab evaluation on captured Ugandan GSM bands exists | projected proposal expectation unless independently verified | The concept note presents this as evidence-to-date, but the supervisor later clarified that the imported figures were proposal-level and adapted as ideology | Keep out of results until artifacts arrive | Capture metadata, acquisition note, or result tables | `SUP-01`; `SUP-02`; [practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md) |
| `4.2 dB` average SNR improvement | projected proposal expectation unless independently verified | No supporting artifact found, and the supervisor clarified that results were being used as anticipated proposal material | Use only as an expected-result placeholder in proposal-style documents if clearly labeled; keep out of NCC abstract/results until verified | Result table or reproducible output | `SUP-01`; `SUP-02`; [practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md) |
| Burst decode rate improves from `52%` to `84%` at `-102 dBm` | projected proposal expectation unless independently verified | No supporting artifact found, and the supervisor clarified that results were being used as anticipated proposal material | Use only as an expected-result placeholder in proposal-style documents if clearly labeled; keep out of NCC abstract/results until verified | Decode logs, evaluation script output, or table | `SUP-01`; `SUP-02`; [practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md) |
| Quantized models remain below `1.2 MB` total | projected proposal expectation unless independently verified | The report plans computational metrics, but no measured quantization output exists yet; the supervisor clarified that such figures may be expectation-level | Optional proposal expectation only; NCC device-feasibility subsection requires verification | Quantized model file size and build notes | `SUP-01`; `SUP-02`; [ProjectReport.md](/Users/sharif/telecom/final-year-project/ProjectReport.md:380) |
| Inference latency remains below `40 ms` on Tecno Spark phones | projected proposal expectation unless independently verified | The report plans latency evaluation, but no device benchmark exists in the workspace; the supervisor clarified that such figures may be expectation-level | Optional proposal expectation only; NCC device-feasibility subsection requires verification | Device benchmark logs and phone model details | `SUP-01`; `SUP-02`; [ProjectReport.md](/Users/sharif/telecom/final-year-project/ProjectReport.md:380) |
| Wakiso pre-test may improve call setup from `9/20` to `16/20` | projected pilot target / expected result | The concept note itself labels this as an expected pre-test outcome, and the supervisor later confirmed such items are acceptable as expected results in a proposal | Use only in future pilot or expected-results language; not in NCC results | Field protocol and measured pre-test record | `SUP-01`; `SUP-02` |
| Three-district pilot in Iganga, Kiryandongo, and Kabale | projected pilot target | District names are plausible and locally relevant, but the pilot has no backing artifacts in the pack | Use only in future-work or proposal narrative | Pilot protocol, partner confirmations, and site criteria | `SUP-01`; `OFF-11` |
| `20` kits at `$95` per unit | projected pilot target | No budget sheet or bill of materials found | Exclude from NCC paper; retain for concept note only | Budget or BOM | `SUP-01` |
| Target `25%` reduction in failed sessions | projected pilot target / expected result | No field protocol or baseline data found, but this fits the supervisor's allowed expected-results framing for proposal use | Future pilot target only; not current NCC evidence | Pilot design and baseline metrics | `SUP-01`; `SUP-02` |
| Open-source APK, bilingual training manual, and technical report | projected pilot outputs | No APK, app source, or manual is present in the current workspace | Future-work output list only | Implementation roadmap and deliverables | `SUP-01` |
| Local assembly on William Street and `2 hour` training | excluded from NCC paper as unsupported specifics | No supporting logistics source or pilot ops note found | Do not use as fact in the paper | Operations plan or partner note | `SUP-01` |
| Equip `1000` VHTs for `<$100k` versus `$200k` for one new tower | excluded from NCC paper as unsupported financial comparison | No validated cost sheet or primary comparison source found | Do not use in paper | Budget model and tower cost source | `SUP-01` |
| Aligns with UCC Universal Access goals | externally cited at high level | UCC research and underserved-area priorities are well supported in the source pack | Safe to use as high-level alignment language | None, if kept high-level and not over-specific | `SUP-01`; `OFF-05`; `OFF-07`; `OFF-18` |
| Aligns with GSMA's rural connectivity focus | projected external alignment pending source | No official GSMA source is archived in the current pack; keep as proposal motivation only for now | Mention only as concept-note motivation until an official GSMA source is added | Official GSMA program/source | `SUP-01`; `SUP-02` |
| `$18,000` funding request and co-funding arrangement | excluded from NCC paper | Funding ask belongs to proposal collateral, not the conference paper | Keep out of the NCC paper | Proposal budget documents if needed for a separate concept note | `SUP-01` |
| Team history, SIM800/ESP32 deployments, and live demo readiness | excluded from NCC paper unless needed in proposal collateral | No supporting deployment dossier is in the current pack | Keep out of paper; reserve for proposal or demo notes | Team bios and deployment references | `SUP-01` |

## Drafting consequences

- The safest NCC title and abstract should stay with `effective service reach` and `cell-edge GSM signal recovery`, not `on-device` or `without new towers`, unless the device and deployment claims are verified. Source basis: `SUP-01`; [analysis/project-alignment.md](/Users/sharif/telecom/final-year-project/NCC/analysis/project-alignment.md).
- The concept note is still valuable because it strengthens the `problem significance`, `deployment logic`, `expected results`, and `future pilot pathway` sections in proposal-style writing. Source basis: `SUP-01`; `SUP-02`; `OFF-05`; `OFF-07`.
- The claim pattern also matches the winner analysis: practical value is good, but only if the implementation story is tied to real evidence rather than pitch-only numbers. Source basis: `WIN-01`; `WIN-05`.

## Sources

- [SUP-01: supervisor concept note](/Users/sharif/telecom/final-year-project/NCC/sources/background/supervisor-concept-note-2026-04-23.md)
- [OFF-05: UCC launches NCC 2026](/Users/sharif/telecom/final-year-project/NCC/sources/official/ucc-launches-ncc-2026.html)
- [OFF-07: UCC Research Support and Collaboration Framework 2022-2026](/Users/sharif/telecom/final-year-project/NCC/sources/official/ucc-research-support-collaboration-framework-2022-2026.pdf)
- [OFF-08: UCC access and usage gaps report news](/Users/sharif/telecom/final-year-project/NCC/sources/official/ucc-access-usage-gaps.html)
- [OFF-09: UCC tax policy and telecom growth news](/Users/sharif/telecom/final-year-project/NCC/sources/official/ucc-tax-policy-telecom-growth.html)
- [OFF-18: James Beronda profile / UCUSAF lens](/Users/sharif/telecom/final-year-project/NCC/sources/official/ucc-james-beronda-profile.html)
- [WIN-01: NCC 2025 winners](/Users/sharif/telecom/final-year-project/NCC/sources/winners/ncc-2025-winners.html)
- [WIN-05: FareFlow paper](/Users/sharif/telecom/final-year-project/NCC/sources/winners/fareflow-paper.pdf)
- [ProjectReport.md](/Users/sharif/telecom/final-year-project/ProjectReport.md:373)
- [weekly_log_timeline.md](/Users/sharif/telecom/final-year-project/reports/weekly_log_timeline.md:17)
