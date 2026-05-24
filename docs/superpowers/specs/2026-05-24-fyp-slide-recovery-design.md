# FYP Slide Recovery Design

Date: 2026-05-24

## Situation

The immediate missing submission item is the PowerPoint slide deck for the final-year project presentation. The project coordinator message requires PowerPoint slides and lists the expected presentation particulars: project title, introduction/background, problem statement, objectives, justification, significance, scope, literature review, methodology, results, discussion of results, conclusion and recommendations, and references.

The project is not yet complete. Submitting a deck that implies completed field validation, completed deployment, or final measured model performance would create a defense risk. Submitting a purely proposal-level deck would also be weak for assessment scheduling.

## Chosen Strategy

Create a submission-ready final-year presentation deck using a defensible "preliminary controlled simulation" results layer.

The deck should present the work as an active final-year implementation that has progressed beyond pure proposal stage:

- Problem, objectives, literature gap, and methodology are established from the current project report and research notes.
- The technical pipeline is the DAE-AMC signal chain: noisy GSM-family I/Q sample, denoising stage, cleaned signal representation, AMC classifier, and evaluation metrics.
- Results slides use only artifacts that can be generated locally today through controlled simulation.
- Preliminary results are labelled as controlled simulation or preliminary validation, not final field results.
- Remaining work is framed as model refinement, broader evaluation, and final reporting before presentation day.

## Preliminary Results Scope

Generate real local artifacts for the slide deck:

- Synthetic GSM-family I/Q windows for a small modulation set.
- Controlled low-SNR noise and simple interference impairment.
- A baseline classifier on noisy features.
- A denoising-assisted classifier using a simple reproducible denoising front end.
- Summary metrics such as accuracy by SNR band, confusion matrix, and before/after signal plots.

If time or tooling prevents a full classifier run, the fallback is still real simulation evidence:

- Noisy versus denoised waveform and constellation plots.
- Reconstruction/noise-reduction indicators from the controlled simulation.
- A clearly marked evaluation plan table showing the final metrics to be completed.

## Deck Shape

Use the existing PowerPoint as source material where it is useful, but produce a cleaner submission copy rather than relying on the current 25-slide draft unchanged.

Recommended deck size: 12 to 15 slides.

Required coverage:

- Title and team details.
- Background and rural GSM service problem.
- Problem statement.
- Main and specific objectives.
- Justification, significance, and project scope.
- Literature review and identified gap.
- Methodology and DAE-AMC architecture.
- Preliminary controlled simulation setup.
- Preliminary controlled simulation results.
- Discussion of results and limitations.
- Conclusion and recommendations.
- References.

## Evidence Rules

Allowed:

- "Preliminary controlled simulation results."
- "Synthetic GSM-family I/Q samples."
- "Controlled low-SNR impairment."
- "Denoising-assisted pipeline showed improvement in this limited simulation."
- "Final validation remains ongoing."

Not allowed:

- Claims that field deployment has been completed.
- Claims that live calls, SMS, USSD, or Mobile Money sessions have improved.
- Claims that tower range has physically increased.
- Claims that Android/SDR prototype testing has been completed unless an artifact exists.
- Unverified concept-note metrics such as exact dB gains, radius gains, or latency claims.

## Deliverables

Primary:

- A cleaned final-year project PowerPoint deck suitable for immediate submission.
- A PDF export of the same deck for printing or backup submission.

Supporting:

- A small preliminary results folder containing the simulation script, generated figures, and results table.
- A short README or notes file explaining how the preliminary results were generated.

## Implementation Priority

1. Generate the preliminary simulation artifacts.
2. Select and revise the deck structure.
3. Insert the strongest figures and metrics.
4. Export PPTX and PDF.
5. Verify that the deck contains the required presentation particulars and does not overclaim completed work.

## Open Risks

- The project has not yet produced final model results, so the deck must avoid language that suggests the final system has been fully implemented.
- The existing PowerPoint contains some older strong claims that may need trimming or rewording.
- The department may expect a completed-project tone, so the deck must sound progress-backed and confident while keeping the results technically defensible.
