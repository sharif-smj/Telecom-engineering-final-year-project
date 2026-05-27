# Practical Claim Verification

Status date: April 23, 2026

Purpose:

- Document what has actually been verified in the workspace for the supervisor concept note.
- Create a citation-backed boundary between `practical ambition` and `NCC-usable evidence`.

Updated interpretation:

- The supervisor later clarified that the shared practical numbers were being used as `proposal-level expected results / ideology`, not as confirmed project evidence. Source basis: `SUP-02`.

Verification scope:

- `/Users/sharif/telecom/final-year-project`
- `/Users/sharif/telecom/final-year-project/NCC`

Verification rule:

- A practical claim counts as verified only if the workspace contains a directly inspectable artifact such as a result table, notebook output, benchmark log, quantized model file, capture metadata, or traceable external primary source. Source basis: `SUP-01`.
- Expected results are allowed in a proposal if labeled carefully, but they still do not count as verified paper evidence. Source basis: `SUP-02`.

## What the workspace does support

### 1. The project intends to measure deployability and robustness

The current proposal explicitly plans:

- experiment logging and checkpoints
- denoising metrics such as MSE and PSNR
- classification metrics such as accuracy, macro F1, and confusion matrices
- computational metrics such as parameter counts and inference latency

These are valid NCC directions, but they are still method commitments in the current report rather than recorded results. [ProjectReport.md](/Users/sharif/telecom/final-year-project/ProjectReport.md:373) [ProjectReport.md](/Users/sharif/telecom/final-year-project/ProjectReport.md:380)

### 2. The weekly schedule shows that major experiments were still planned, not archived

The project's own weekly timeline schedules:

- denoising model development for `Apr 27 - May 3, 2026`
- baseline and integrated pipeline work for `May 4 - May 10, 2026`
- evaluation and comparative testing for `May 11 - May 17, 2026`

That timeline strongly suggests the practical metrics in the supervisor note are not yet represented as archived experiment artifacts in the repository as of this implementation pass. [weekly_log_timeline.md](/Users/sharif/telecom/final-year-project/reports/weekly_log_timeline.md:17)

### 3. The visual demo metrics are illustrative, not paper-grade measurements

The signal denoising visualizer contains:

- hard-coded display values such as `+12.4 dB`, `14.2%`, `0.3%`, and `98.7%`
- an explicit simulated denoising step with `const improvement = 0.85`

Those values are useful for demos but cannot be cited as empirical NCC results. [signal_denoising_visualizer.html](/Users/sharif/telecom/final-year-project/simulations/signal_denoising_visualizer.html:754) [signal_denoising_visualizer.html](/Users/sharif/telecom/final-year-project/simulations/signal_denoising_visualizer.html:872)

## What the workspace does not currently verify

No inspectable workspace artifact was found for the following supervisor-note claims:

- `4.2 dB` average SNR improvement
- burst decode rate from `52%` to `84%` at `-102 dBm`
- captured Ugandan GSM-band lab evaluation
- quantized model size below `1.2 MB`
- latency below `40 ms` on Tecno Spark phones
- Wakiso pre-test expectation from `9/20` to `16/20`
- district pilot deployment economics, kit counts, or assembly claims

Current status for all of the above:

- not verified in workspace
- not safe for NCC abstract/results/conclusion
- acceptable only as future pilot, concept-note, or clearly labeled expected-results material until evidence arrives

See the full claim-by-claim treatment in [supervisor-concept-note-claim-ledger.md](/Users/sharif/telecom/final-year-project/NCC/analysis/supervisor-concept-note-claim-ledger.md). Source basis: `SUP-01`.

## Safe NCC usage after this verification pass

### Safe now

- Uganda access-gap and underserved-area significance
- low-SNR GSM edge conditions as the motivating technical problem
- denoising-plus-classification as the proposed method
- deployment relevance for UCC, UCUSAF, and operators
- a future applied path that may include on-device kits or a pilot
- carefully labeled expected results in a proposal-style document

These are all consistent with the existing NCC research pack and UCC priorities. Source basis: `OFF-05`; `OFF-07`; `OFF-08`; [analysis/project-alignment.md](/Users/sharif/telecom/final-year-project/NCC/analysis/project-alignment.md).

### Unsafe until verified

- exact device-performance numbers
- exact laboratory improvement numbers
- exact pre-test field numbers
- exact coverage-radius expansion percentages
- exact cost comparisons against new towers
- exact rural-user population counts from the concept note unless separately sourced

For NCC specifically:

- all of the items above remain unsafe for the title, abstract, empirical results, and conclusion until independently verified

## Recommended next artifacts to request or generate

1. A result table or notebook export for the `4.2 dB` and `52% to 84%` claims
2. A device benchmark note for model size and latency
3. A short metadata note for the claimed Ugandan GSM captures
4. A one-page field pre-test protocol if the Wakiso work is real and imminent
5. A primary source for any tower-cost or GSMA-alignment claim intended for public use

## Sources

- [SUP-01: supervisor concept note](/Users/sharif/telecom/final-year-project/NCC/sources/background/supervisor-concept-note-2026-04-23.md)
- [SUP-02: Dr. Ephrance chat clarification](/Users/sharif/telecom/final-year-project/NCC/sources/background/dr-ephrance-chat-clarification-2026-04-23.md)
- [ProjectReport.md](/Users/sharif/telecom/final-year-project/ProjectReport.md:373)
- [weekly_log_timeline.md](/Users/sharif/telecom/final-year-project/reports/weekly_log_timeline.md:17)
- [signal_denoising_visualizer.html](/Users/sharif/telecom/final-year-project/simulations/signal_denoising_visualizer.html:754)
- [OFF-05: UCC launches NCC 2026](/Users/sharif/telecom/final-year-project/NCC/sources/official/ucc-launches-ncc-2026.html)
- [OFF-07: UCC Research Support and Collaboration Framework 2022-2026](/Users/sharif/telecom/final-year-project/NCC/sources/official/ucc-research-support-collaboration-framework-2022-2026.pdf)
- [OFF-08: UCC access and usage gaps report news](/Users/sharif/telecom/final-year-project/NCC/sources/official/ucc-access-usage-gaps.html)
