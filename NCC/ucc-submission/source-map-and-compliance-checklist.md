# Research Proposal Source Map and Review Checklist

This file is for internal control only. It supports the research-proposal review path and is not the NCC conference submission source map. Do not upload it unless provenance or screening notes are requested.

## Active Research Proposal Package

- Proposal Markdown: `/Users/sharif/telecom/final-year-project/NCC/ucc-submission/ucc-research-support-proposal.md`
- Proposal Word document: `/Users/sharif/telecom/final-year-project/NCC/ucc-submission/ucc-research-support-proposal.docx`

## Working Administrative Route

- Current purpose: research proposal for Dr. Dickson Mugerwa and Dr. Ephrance Eunice Namugenyi review.
- NCC tailoring is paused for this deliverable. The proposal remains practical, interdisciplinary, and UCC/UCUSAF-aware because that is the direction currently needed for supervision and final-year-project progress.
- Selected route: Lot 3 - Inter-University Research Collaboration, Interdisciplinary.
- Budget route: UGX 66,600,000.
- Reason: This preserves the higher practical ambition while staying below the UGX 100,000,000 ceiling for Lot 3 interdisciplinary projects.
- Critical pre-upload confirmation: Makerere University CoCIS must be accepted as the collaborating institution, or the proposal should be downgraded to Lot 1 with a UGX 20,000,000 cap.

## Review Format

- Microsoft Word document exists.
- Times New Roman is used throughout the generated Word document.
- Font size is set to 12 pt for body content.
- Single line spacing is used, with 6 pt paragraph spacing for readability.
- A4 page size is configured with 0.85 inch margins.
- The title page uses a no-border information table, stronger heading hierarchy, a clean top divider rule, and a signature block for easier scanning.
- Body paragraphs are justified.
- Main section headings are centered, uppercase, black, and kept with following text. The cut version now starts only major blocks on separate pages so the document stays within the 20-page limit.
- Equations and inline variables use formatted math runs with Greek symbols, subscripts, superscripts, hats, and summation where required.
- Visible equation labels such as `Equation (1):` have been removed from the generated Word body; equations are centered and numbered at the right as `(1)`, `(2)`, and so on.
- Figure 1 is included after Research Design as a generated PNG system pipeline diagram, not a table. It shows the practical denoising and testing workflow from signal source through Android on-device inference and exportable observation logs. Reporting deliverables are separated from the processing pipeline. The asset is stored at `/Users/sharif/telecom/final-year-project/NCC/ucc-submission/assets/denoising-assisted-gsm-architecture.png`.
- Figure 2 is included after Automatic Modulation Classification as a generated PNG model architecture diagram. It shows the noisy I/Q input, preprocessing, denoising autoencoder, denoised output, automatic modulation classifier, raw-input baseline path, and evaluation outputs. The asset is stored at `/Users/sharif/telecom/final-year-project/NCC/ucc-submission/assets/dae-amc-model-architecture.png`.
- Figure 3 is included in the Implementation Plan section as a generated PNG Gantt chart. It replaces the old dense schedule table and shows the May-November 2026 timeline with monthly headers, week numbers, and overlapping phase bars. The image is now a clean embedded chart without the earlier poster-style outer frame, title box, top description box, or footer note. The asset is stored at `/Users/sharif/telecom/final-year-project/NCC/ucc-submission/assets/project-timeline-gantt.png`.
- Extracted text count is 4,175 words.
- Rendered page count is 15 pages after the latest cut-down pass, which is below the 20-page maximum raised in WhatsApp feedback.
- Title is under 20 words.
- Project Summary is 260 words, below the 300-word screening target.
- Problem Statement is 193 words after reduction.
- Justification and Direct Benefit to UCC is 242 words after reduction.
- Literature Review is 271 words after reduction.
- Budget is in UGX only.
- No tuition, salary, stipend, remuneration, or infrastructure line appears in the budget.
- Budget total is UGX 66,600,000 across seven functional cost blocks tied to approvals, prototype hardware, laboratory analysis, field validation logistics, coordination, dissemination, and contingency.
- Full page-image rendering was completed with LibreOffice/Poppler. QA outputs are in `/Users/sharif/telecom/final-year-project/NCC/ucc-submission/rendered-qa/`.

## Required Sections

- Title Page: included.
- Project Summary: included.
- Introduction and Background: included.
- Problem Statement: included.
- Research Objectives: included.
- Research Questions or Hypotheses: included as Research Questions.
- Justification of the Study: included.
- Methodology: included.
- Implementation Plan: included.
- Contribution to Cross-Cutting Issues: included.
- Literature Review using APA-style citations: included.
- Budget and Budget Justification: included.
- Curriculum Vitae of the Research Team: included.

## Evidence and Citation Control

- UCC call requirements and priority areas: Uganda Communications Commission (2026a).
- UCC access and usage gap framing: Uganda Communications Commission (2025).
- UCUSAF mandate, strategic themes, and underserved-area programmes: Uganda Communications Commission (2026b, 2026c, 2026d).
- UCUSAF field-site selection logic, including coverage mapping and MNO-report evidence: Uganda Communications Commission (2024a).
- Local interference context: Uganda Communications Commission (2021, 2024b).
- Public I/Q dataset basis: DeepSig (2018), RF Signal Data (2025).
- Propagation and attenuation support: International Telecommunication Union (2005, 2021).
- Denoising and modulation-classification literature: Abd-Elaziz et al. (2023), An and Lee (2023), Gao et al. (2026), Zhang et al. (2023).
- Latest reference check: UCC, UCUSAF, ITU, DeepSig, Kaggle, and PMC URLs returned live HTTP responses. MDPI article pages returned bot-blocking/403 responses from the script but were retained as direct article pages. The incorrect IEEE DOI was replaced with `10.1109/ACCESS.2023.3238995`, and the IEEE article metadata was corrected to pages 7860-7872.

## Zoom Feedback Applied

- Strengthened UCC/UCUSAF fit in the Project Summary, Introduction, Problem Statement, Justification, and Proposed Focus Areas while keeping the document in research-proposal form.
- Added criteria-based field-site logic instead of inventing districts.
- Replaced the earlier table-style architecture figure with a generated system pipeline image after Research Design, removing `UCC outputs` as a pipeline step.
- Added a second generated model architecture figure showing the DAE-AMC signal path and baseline comparison.
- Replaced the implementation-plan table with a generated Gantt chart image matching the requested visual reference style.
- Made the prototype pathway more concrete as a receive-only Android-SDR or edge-device prototype workflow.
- Made Kisige Tom Derrick's prototype setup and field-support validation role visible in the team responsibilities.
- Rebuilt the budget into seven transcript-informed UGX cost blocks while keeping the total at UGX 66,600,000. The revision makes REC/UNCST-style approvals, prototype hardware, shipping/spares, six-month communication, field logistics, participant data/airtime, system-analysis support, and UCC-facing dissemination more explicit while removing broad unsupported cost assumptions.
- Regenerated the canonical Word document from the canonical Markdown source.

## Latest WhatsApp Cut Feedback Applied

- Latest root ZIP preserved at `/Users/sharif/telecom/final-year-project/NCC/feedback/original-zips/WhatsApp Chat - UCC paper 2026-05-05.zip`.
- Latest extracted feedback record stored at `/Users/sharif/telecom/final-year-project/NCC/feedback/latest-whatsapp-ucc-paper-2026-05-05/`.
- Dr. Ephrance's 20-page concern was addressed by relaxing the every-section-page-break rule and cutting dense prose.
- Dr. Dickson's problem-statement and literature reduction request was addressed.
- Marked justification/UCC-benefit material was compressed.
- Broad mechanical phrasing was replaced with more direct engineering language while keeping formal team voice.
- Reference metadata was checked and corrected where needed.

## NCC Status

- NCC tailoring is paused for this deliverable.
- Existing NCC analysis, paper skeletons, and conference-facing materials remain preserved elsewhere under `/Users/sharif/telecom/final-year-project/NCC/`.
- This file and the canonical Word proposal should not be treated as the NCC extended abstract, NCC full paper, or NCC submission-ready manuscript.

## Screening Notes

- The proposal does not claim the model has already been built or tested.
- The proposal does not present expected SNR gain as measured evidence.
- The proposal keeps the Android-SDR setup as engineering test equipment, not a requirement for rural users.
- Basic phones, SMS, USSD, voice, and Mobile Money remain visible in the problem framing.
- The active review risk is readability and alignment with the doctors' practical direction; this version is optimized for that.
- The main later administrative risk is eligibility proof for Lot 3. The second institution should be confirmed before any public portal submission.
- This version is now compressed for the stated page cap while preserving the practical supervisor-facing proposal structure.
- Current active Word document count in `NCC/ucc-submission` is one: `/Users/sharif/telecom/final-year-project/NCC/ucc-submission/ucc-research-support-proposal.docx`.
