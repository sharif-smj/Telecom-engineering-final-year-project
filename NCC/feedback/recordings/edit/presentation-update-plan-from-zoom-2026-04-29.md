# Presentation Update Plan From Zoom Feedback - 2026-04-29

Source video: `/Users/sharif/telecom/final-year-project/NCC/feedback/recordings/GMT20260429-164419_Recording_1576x950.mp4`

Transcript: `/Users/sharif/telecom/final-year-project/NCC/feedback/recordings/edit/transcripts/GMT20260429-164419_Recording.transcript.md`

Likely current deck to update: `/Users/sharif/telecom/final-year-project/GSM Signal Denoising and Modulation Classification for Rural Uganda.pptx`

Note: transcript is automated. A few words are misheard, for example "DSM" should usually be read as GSM, and "Jambo Golden" should be read as Kyambogo.

## Core Direction

The supervisors are not asking for a bigger wall of text. They are asking for a clearer, more fundable, more demonstrable story.

The updated presentation should show:

1. Exactly how the work helps UCC priority areas.
2. A practical prototype path, not modelling only.
3. A simple system architecture diagram that explains the whole project at a glance.
4. Evidence and metrics we will use to prove denoising value.
5. Rural focus areas selected from UCC/UCUSAF reports, not random districts.
6. A realistic six-month budget and timeline in UGX.
7. A cleaner, faster presentation style with less text.

## Feedback Themes With Timestamps

### 1. Make UCC Alignment Obvious

Feedback:
- We must show which UCC areas the project is augmenting or helping.
- The project is cross-cutting across access/user experience, digital innovation/emerging technologies, and next-generation telecom infrastructure/spectrum management.

Evidence:
- [00:00-00:34] "show which areas ... helping the UCC" and the three UCC priority areas are named.
- [14:43-15:09] assessors check whether the work matches what the institution requires.

Presentation update:
- Add an early slide titled `Why This Matters to UCC`.
- Use a three-column fit map:
  - Access, Affordability, and User Experience: weak rural GSM service, SMS/USSD/Mobile Money reliability.
  - Digital Innovation and Emerging Technologies: AI-assisted denoising and low-cost edge prototype.
  - Next-Generation Telecommunications Infrastructure and Spectrum Management: weak-signal interpretation and interference diagnosis.

### 2. Prototype Must Be Visible

Feedback:
- Do not stop at models.
- The prototype can be simple, but it must prove model viability.
- Use a simple mobile application, GSM handset, SDR/device, microprocessor, or edge-processing setup to show before/after denoising.
- Show graphs of noisy signal versus denoised signal and SNR improvement.

Evidence:
- [05:42-09:18] Dr. Ephrance explains the need for a prototype, mobile/device deployment, SNR proof, and a simple interface showing before/after denoising.
- [36:11-36:17] prototyping equipment should be visible under tools/equipment.

Presentation update:
- Replace the current `Scope` slide that says the project is entirely offline.
- Add `Prototype Pathway` slide:
  - Input: weak GSM-family signal / I/Q samples.
  - Processing: denoising model + classifier.
  - Output: graph dashboard showing noisy signal, denoised signal, classification, SNR-related improvement.
  - Device path: Android + low-cost SDR or other edge device for supervised demonstration.
- Keep wording proposal-safe: "planned prototype", "field-support concept", "expected demonstration", not "already deployed".

### 3. Add One Strong Architecture Diagram

Feedback:
- The supervisors repeatedly said diagrams matter more than too much explanation.
- The architecture should show the full system, including data source, denoising, classification, prototype/end-user path, and UCC value.

Evidence:
- [13:53-14:15] "system diagrams" and "end user applications" are raised.
- [14:10-14:15] "At least we should have an architecture of some sort to show the design of the system."
- [16:22-17:58] Dr. Dickson and Dr. Ephrance agree one comprehensive architecture diagram can reduce over-explanation.

Presentation update:
- Add one central slide titled `Proposed System Architecture`.
- Diagram flow:
  - Rural weak-signal context
  - Signal capture / public I/Q dataset / controlled impairment generation
  - Noise and interference modelling
  - Denoising autoencoder
  - AMC classifier
  - Before/after metrics
  - Android-SDR / edge prototype interface
  - UCC outputs: weak-service evidence, interference follow-up, coverage planning support

### 4. Equations Should Support the Contribution, Not Look Raw

Feedback:
- Equations should be properly rendered using Word equation formatting.
- Do not write "Equation (1):" as part of the equation body in the proposal.
- Existing equations are acceptable if they support methodology/results, but the contribution should show how pieces are combined or improved.

Evidence:
- [09:19-10:37] questions about whether equations will be used for graphs/results and instruction to use equation editor.
- [11:21-13:46] equations should be numbered properly and the project contribution can be shown by combining methods into a fuller final equation or pipeline.

Presentation update:
- Do not show many formulas in the main deck.
- Use one slide or appendix titled `Technical Model`.
- Show only:
  - received signal model
  - DAE reconstruction objective
  - classification decision
  - final combined pipeline idea
- Use visual equation formatting, not raw text.
- Tie equations directly to plots: reconstruction error, SNR-related improvement, accuracy/F1, confusion matrix.

### 5. Keep It Short, Clear, and Error-Free

Feedback:
- Assessors have limited time.
- They look at the basics: what the project does, fit with institution priorities, organization, and budget.
- Literature and methodology should be concise.

Evidence:
- [14:16-15:48] Dr. Dickson says reviewers do not have time, the work must be simple, clear, error-free, and not over-explained.

Presentation update:
- Reduce the current 25-slide FYP deck to a sharper 12-14 slide UCC/NCC-style deck.
- Move detailed literature, references, detailed budget, and extra equations to appendix.
- Each slide should answer one reviewer question.

### 6. Budget Needs Realism and UCC Units

Feedback:
- Budget must match the total figure.
- The work schedule should be Gantt-like and tied to activities.
- Use the units in the call, which are UGX.
- Be specific about costs and avoid vague lines like office supplies, gifts, or 12-month costs when the project is six months.
- Participant support should be framed as data/airtime for respondents or participants, not "gifts".

Evidence:
- [03:10-05:21] budget table and Gantt-style schedule are reviewed.
- [21:12-22:02] REC/UNCST cost ranges discussed.
- [23:06-25:59] remove irrelevant induction/gift wording; use participants/respondents and data/airtime where needed.
- [31:40-35:16] office supplies and communication costs must be realistic and specific.
- [38:16-39:09] budget total and currency units must match the UCC call.

Presentation update:
- Replace old `1,750,000 UGX` budget slide.
- Add `Budget Summary` slide with only major categories and total UGX.
- Keep detailed line-item budget in appendix if needed.
- Add one sentence: "All amounts are presented in UGX to match the UCC call."

### 7. Choose Field Areas From UCC/UCUSAF Evidence

Feedback:
- We need to identify actual rural areas/focus areas from UCC or UCUSAF reports.
- The travel/field plan must say where we are going, why those places matter, and how those sites connect to UCC priorities.
- Consider ranges from near base station to edge-of-coverage conditions.

Evidence:
- [28:02-31:32] the team discusses identifying base stations, UCC deployment areas, and testing across distance/range conditions.
- [42:37-43:57] supervisors ask to mention rural areas from UCC reports and use them as focus areas.

Presentation update:
- Add `Proposed Focus Sites` slide.
- It should include 2-3 candidate rural districts or site categories from UCC/UCUSAF evidence.
- For each site, show why it was chosen:
  - underserved/weak service
  - UCC/UCUSAF relevance
  - field validation practicality
  - distance/range testing potential
- If specific districts are not yet verified, label them as "to be selected from UCC/UCUSAF reports" rather than inventing them.

### 8. Engage Derrick and Update Team Slide

Feedback:
- Derrick feels left out and should be engaged in Zoom calls and project work.
- The title page should have strong hierarchy and not make the title look small.

Evidence:
- [44:55-45:26] Derrick should be actively included.
- [45:29-45:42] title must not be small; UCC heading should be prominent.

Presentation update:
- Update title slide:
  - Big institutional heading.
  - Clear project title.
  - Student researchers: Ssemujju Sharif Abdukarim and Kisige Tom Derrick.
  - Supervisors: Dr. Dickson Mugerwa and Dr. Ephrance Eunice Namugenyi.
- Add a lightweight team/contribution slide if needed:
  - Sharif: model/pipeline, documentation, presentation.
  - Derrick: prototype setup, signal testing, evaluation support.
  - Supervisors: RF, AI/ML, proposal quality, field/prototype guidance.

## Recommended New Deck Structure

Target: 12-14 main slides plus appendix.

1. `Title`
   - Big UCC/NCC heading, project title, team, supervisors.

2. `One-Sentence Thesis`
   - "We propose software-assisted GSM signal denoising to improve interpretation of weak edge-of-coverage rural signals before expensive infrastructure decisions are made."

3. `Problem: Covered Does Not Always Mean Usable`
   - Rural weak signal, basic phones, SMS/USSD/Mobile Money, noisy edge-of-coverage conditions.

4. `Why This Matters to UCC`
   - Map to the three UCC priority areas.

5. `Proposed Focus Sites`
   - Candidate UCC/UCUSAF-backed rural areas or selection logic.

6. `What We Are Building`
   - DAE-AMC pipeline in plain language.

7. `Proposed System Architecture`
   - The main architecture diagram.

8. `Prototype Pathway`
   - Android/SDR or simple edge/mobile interface showing noisy vs denoised signal and metrics.

9. `Evaluation Plan`
   - SNR-related improvement, reconstruction error, accuracy, macro F1, confusion matrix, model size, latency.

10. `Technical Model`
   - One clean visual equation/pipeline slide, not raw equation text.

11. `Implementation Plan`
   - Six-month Gantt from May to November 2026.

12. `Budget Summary`
   - UGX total and major categories only.

13. `Expected Outputs for UCC`
   - Technical brief, prototype workflow, evidence for weak-service interpretation, future pilot path.

14. `Ask / Next Step`
   - Approval to proceed, feedback needed, submission milestone.

Appendix:
- Detailed literature.
- Full equations.
- Detailed budget.
- References.

## Slide-Level Changes to Current Deck

- Slide 1: Rebuild title slide with UCC heading, bigger hierarchy, Dr. Ephrance added, Derrick visible.
- Slide 2: Replace long outline with a simple story map or remove entirely.
- Slides 3-4: Compress background into one local problem slide.
- Slide 5: Rewrite problem statement to include UCC/UCUSAF-backed focus areas once verified.
- Slides 6-7: Keep aim/objectives but add prototype and field-support objective.
- Slides 8-9: Convert justification into UCC priority fit map.
- Slide 10: Replace "offline experimentation only" with two-layer scope: offline evidence core + prototype/field-support path.
- Slides 11-15: Compress literature into 1-2 slides and move details to appendix.
- Slides 16-19: Replace text-heavy methodology with architecture diagram, prototype workflow, and evaluation metrics.
- Slide 20: Replace old UGX 1,750,000 budget with current UCC proposal budget summary.
- Slide 21: Update timeline to May-November 2026, six months.
- Slides 22-24: Move references to appendix or use smaller source footers.
- Slide 25: Replace generic thank-you with clear next-step/feedback ask.

## Immediate Research Needed Before Slide Rebuild

1. Find UCC/UCUSAF report sections that identify underserved/rural focus areas, planned sites, or base-station deployment areas.
2. Confirm whether the presentation target is NCC, UCC supervisor review, or final-year defense, because the title slide and budget emphasis will differ.
3. Confirm the final UCC budget total to present in the deck.
4. Confirm Derrick's role so the team slide does not feel decorative.
5. Decide whether the prototype path is Android + SDR, GSM handset-based demo, or "edge-device field-support kit" as the umbrella language.

## Recommended Priority Order

1. Fix title/team/supervisor slide.
2. Add UCC priority fit and focus-sites slide.
3. Add architecture diagram.
4. Add prototype pathway slide.
5. Replace old budget and timeline.
6. Compress literature/methodology text.
7. Clean equations into appendix/technical slide.
8. Final pass for speed, clarity, and consistency.
