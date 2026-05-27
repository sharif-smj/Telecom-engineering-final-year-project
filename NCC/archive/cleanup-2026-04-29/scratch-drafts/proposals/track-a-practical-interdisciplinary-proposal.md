# Improving Rural GSM Access in Uganda Through AI-Assisted Signal Denoising

## Title Page

**Proposal type:** Strategic interdisciplinary practical proposal for GSMA/UCC/UCUSAF-style rural-connectivity funding  
**Funding request:** `$18,000`  
**Duration:** `8 months`  
**Lead applicants:** `Ssemujju Sharif Abdukarim`, `Kisige Tom Derrick`  
**Department:** Department of Electrical and Electronics Engineering, Kyambogo University  
**Supervisory leads:** `Dr. Dickson Mugerwa`, `Dr. Ephrance`

**Working problem fit:** Rural Uganda still contains weak and noisy GSM service zones where essential communication services degrade, yet physical tower expansion remains slower, costlier, and harder to justify in underserved areas than software-first service improvement measures `[OFF-05, OFF-08, OFF-09, OFF-12]`.

**Evidence note:** All quantitative performance claims in this document are clearly marked as either published external evidence or proposal-level expected outcomes. No unverified supervisor-note metric is presented as a completed project result `[SUP-01, SUP-02]`.

## Executive Summary

Uganda's communications policy direction now centers on an inclusive digital economy, practical innovation, and better value from infrastructure already in the field `[OFF-05, OFF-06, OFF-07, OFF-08]`. At the same time, UCC's public communications continue to highlight access gaps, affordability constraints, and persistent service challenges in underserved and hard-to-reach areas `[OFF-08, OFF-09, OFF-12]`. In those settings, many households still depend on basic `2G`-capable phones for voice, SMS, USSD, and Mobile Money, but usable service quality degrades when weak signals are further affected by noise and interference `[OFF-08, OFF-11, V1-01]`.

This proposal addresses that gap through an interdisciplinary project that combines communications engineering, signal processing, artificial intelligence, low-cost prototyping, and rural service-delivery thinking. The core innovation is an AI-assisted GSM signal-denoising and modulation-recognition pipeline designed to recover more usable signal structure from weak and noisy conditions before classification or downstream interpretation takes place `[LIT-01, LIT-02, LIT-03, LIT-04, V1-01]`. Rather than treating rural users as future smartphone-only beneficiaries, the proposal keeps basic phone users at the center and treats Android + SDR kits as optional field-support tools for capture, testing, and prototype-assisted recovery experiments `[SUP-01, SUP-02]`.

The requested `$18,000` will fund prototype hardware, field validation logistics, data collection, documentation, and tightly scoped implementation support for an eight-month pilot pathway. The intended outcome is a practical software-defined coverage-support concept that can help UCC, UCUSAF, operators, and rural-service stakeholders test whether denoising-driven weak-signal recovery can improve the effective service reach of existing GSM infrastructure before resorting to more expensive physical expansion `[OFF-05, OFF-07, OFF-09, OFF-18, SUP-01]`.

## Background and Problem

UCC's current strategic and public research language makes two priorities unmistakable. First, Uganda wants communications research that is market-aware, locally grounded, and capable of moving from knowledge to practical impact `[OFF-05, OFF-07]`. Second, the sector still faces a combined access-and-usage problem in which infrastructure gaps, affordability pressures, and uneven service quality limit meaningful participation in the digital economy, especially outside major urban centers `[OFF-08, OFF-09, OFF-12]`.

This matters because rural communication is not only a broadband issue. In many communities, the critical user interaction is still a weak `2G` voice call, a delayed SMS, a failed USSD session, or a Mobile Money transaction attempted from the edge of coverage on a basic handset `[OFF-08, V1-01]`. When already marginal GSM signals are further affected by interference, repeater misuse, non-compliant transmissions, weather-related attenuation, or general low-SNR conditions, service can become unreliable at exactly the point where low-cost communication matters most `[OFF-11, LIT-05, LIT-06, V1-01]`.

UCC's own enforcement and monitoring activity shows that signal integrity and harmful interference are not abstract concerns. Public action against illegal and non-compliant broadcasters demonstrates that interference, harmonics, and poorly controlled emissions can damage service quality and create wider public-risk implications `[OFF-11]`. At the same time, affordability and capital-allocation pressures make it unrealistic to assume that every weak-service problem will be solved quickly through new towers alone `[OFF-09, OFF-10, OFF-12]`.

The practical problem, then, is not simply "coverage versus no coverage." It is the presence of noisy, low-confidence, weak-signal conditions in which users, operators, and regulators lose usable communication performance and usable signal evidence at the same time. That is the gap this proposal targets `[OFF-08, OFF-11, V1-01]`.

## Innovation and Interdisciplinary Contribution

The proposal is explicitly interdisciplinary because the problem crosses technical, economic, and service-delivery boundaries.

| Discipline area | Contribution in this proposal | Why the combination matters |
| --- | --- | --- |
| Telecommunications engineering | GSM signal behavior, weak-signal conditions, interference context, and rural access framing | Keeps the work tied to real mobile-service constraints rather than generic ML benchmarks |
| Signal processing and AI/ML | Denoising autoencoders, low-SNR feature recovery, and modulation recognition | Provides the technical engine for recovering usable signal structure `[LIT-01, LIT-02, LIT-03, LIT-04]` |
| Computing and prototyping | Low-cost device workflows, SDR-assisted capture, lightweight deployment experiments | Makes the concept testable, demonstrable, and adaptable to field conditions `[SUP-01, SUP-02]` |
| Rural service delivery and public-interest communications | Focus on voice, SMS, USSD, Mobile Money, health, and agricultural messaging | Keeps the proposal aligned with inclusion, usability, and real community value `[OFF-05, OFF-08, OFF-18]` |

The innovation is not only "a new model." The innovation is a software-defined service-support approach that tries to recover more usable GSM intelligence from weak and noisy environments before expensive physical expansion decisions are made. In other words, the technical model is the mechanism, but the real proposition is lower-cost improvement to effective service reach, troubleshooting, and underserved-area decision support `[OFF-05, OFF-07, OFF-09, WIN-01, WIN-05]`.

## Objectives

### Main objective

To develop and pilot an interdisciplinary, AI-assisted GSM signal-denoising approach that improves weak-signal interpretation and supports more reliable rural communication access in Uganda `[V1-01, OFF-05, OFF-08]`.

### Specific objectives

1. To model the weak-signal and interference conditions that are most relevant to rural GSM service in Uganda, including low-SNR reception and selected locally relevant degradation patterns `[OFF-11, LIT-05, LIT-06, V1-01]`.
2. To design a denoising-plus-modulation-recognition pipeline that can recover useful signal structure from noisy I/Q streams before downstream interpretation `[LIT-01, LIT-02, LIT-03, LIT-04]`.
3. To build a low-cost prototype and field-support workflow that can be used for controlled evaluation and limited pilot validation without assuming changes to operator infrastructure `[SUP-01, SUP-02]`.
4. To assess how the approach could support more reliable edge-of-coverage communication and better evidence for UCC, UCUSAF, and operator decision-making in underserved areas `[OFF-05, OFF-07, OFF-18]`.

## Expected Outcomes and Impact

### Proposal integrity note

The outcomes below combine two evidence classes:

- published strategic and technical context from official and literature sources
- clearly labeled proposal-level expected results and pilot targets inspired by the supervisor concept note

Where a number is not yet backed by a project artifact, it is expressed as an `expected`, `anticipated`, or `target` outcome rather than a completed result `[SUP-01, SUP-02]`.

### Expected technical outcomes

- A documented GSM denoising and modulation-recognition pipeline tailored to weak-signal rural-service conditions in Uganda `[V1-01, LIT-01, LIT-02, LIT-03, LIT-04]`.
- A benchmarked comparison between denoised and non-denoised signal-processing paths under controlled low-SNR conditions `[V1-01]`.
- A low-cost prototype workflow that can support lab testing, demonstration, and limited field validation using basic-phone service scenarios plus optional Android + SDR support tools `[SUP-01, SUP-02]`.

### Expected performance targets

The following figures are proposal targets, not completed empirical findings from the current project:

- an anticipated `3-5 dB` improvement in usable weak-signal interpretation under controlled denoising tests `[SUP-01, SUP-02]`
- materially improved burst or session recovery under low-SNR conditions compared with a no-denoising baseline `[SUP-01, SUP-02, LIT-01, LIT-03]`
- lightweight models and workflows that can support practical field demonstrations after prototype maturation `[SUP-01, SUP-02]`

### Expected public-value impact

- Better support for weak-service troubleshooting and underserved-area evidence gathering `[OFF-05, OFF-07, OFF-18]`
- A software-first complement to physical coverage expansion in settings where tower growth is slower, costlier, or harder to prioritize immediately `[OFF-09, OFF-10, OFF-12]`
- A pathway toward stronger continuity for rural SMS, USSD, Mobile Money, and other basic communications services that still matter deeply for health, agriculture, and local economic participation `[OFF-08, OFF-18]`

## Method and Implementation Approach

### Work package 1: Problem modeling and signal preparation

The first stage will consolidate the technical problem definition using public benchmark datasets, documented interference patterns, and project-side GSM modulation targets. The project will prioritize `GMSK`, `GFSK`, and `QPSK`-oriented scenarios, since these align with the preserved v1 research framing and provide a manageable first scope for development `[V1-01]`. Where lawful capture conditions and permissions are available, the team may add small-scale local signal snapshots for qualitative sanity checks; otherwise the core evaluation will remain anchored in public datasets and Uganda-motivated noise models `[SUP-02, V1-01]`.

### Work package 2: AI-assisted denoising and classification

The technical core is a denoising autoencoder followed by a modulation-recognition stage. This design is well aligned with recent literature showing that denoising frontends can improve low-SNR modulation recognition and recover useful structure before classification `[LIT-01, LIT-03, LIT-04]`. The classifier component will be benchmarked against a no-denoising baseline so that any gains are interpretable as service-relevant improvement rather than raw model complexity alone `[LIT-02, V1-01]`.

### Work package 3: Prototype pathway

The proposal includes a practical prototyping layer because the project is meant to be deployment-aware, not paper-only. Basic `2G` handsets remain the real communication endpoint in the beneficiary setting. The optional Android + SDR concept is therefore treated as a field-support tool for capture, assisted interpretation, and controlled demonstration rather than a requirement for every rural user `[SUP-01, SUP-02]`.

### Work package 4: Practical evaluation

Evaluation will be structured around both technical and service-facing metrics. Technical metrics will include denoising quality, classification accuracy, F1-score, confusion matrices, and low-SNR performance curves `[V1-01, LIT-02, LIT-03]`. Service-facing interpretation will examine whether denoising improves the practical recoverability of weak GSM conditions enough to justify further pilot work aimed at reducing failed sessions and improving effective service reach `[OFF-05, OFF-08, SUP-01, SUP-02]`.

## Pilot / Deployment Pathway

This proposal is designed to move beyond lab-only framing while staying honest about what is and is not yet proven.

### Proposed pilot shape

- `Phase 1`: controlled development and benchmarking in the lab
- `Phase 2`: limited field-support testing around known weak-signal environments
- `Phase 3`: small pilot validation in selected districts such as `Iganga`, `Kiryandongo`, and `Kabale`, subject to partner readiness and sponsor approval `[SUP-01]`

### Proposed users and stakeholders

- rural residents who still depend on basic `2G` communication flows
- Village Health Teams and farmer-cooperative coordinators using SMS and USSD-dependent workflows `[SUP-01]`
- UCC / UCUSAF teams interested in underserved-area service evidence `[OFF-07, OFF-18]`
- operator engineering or planning teams assessing edge-of-coverage performance `[OFF-05, OFF-08]`

### Proposed pilot measures

- SMS and USSD completion success under weak-signal conditions
- time-to-delivery for critical low-bandwidth messages
- user-reported ability to communicate at known weak-signal sites
- field comparison between baseline conditions and denoising-assisted prototype conditions `[SUP-01, SUP-02]`

### Proposed outputs

- a demonstrable low-cost prototype workflow
- operator/regulator-facing technical brief
- training materials for pilot use in accessible English and, where appropriate, Luganda
- a decision memo on whether a larger coverage-support pilot is justified `[SUP-01, SUP-02]`

## Budget and Justification

### Requested budget

| Budget line | Amount (USD) | Justification |
| --- | ---: | --- |
| Prototype field kits and RF accessories | 4,100 | Covers a practical stock of low-cost RF capture and prototype materials, including a proposed set of Android + SDR support kits, antennas, OTG adapters, power accessories, and replacement parts for controlled pilot work `[SUP-01]` |
| Bench equipment and calibration materials | 2,200 | Supports lab-side testing, measurement repeatability, calibration accessories, shielding, and bench consumables needed for clean signal experiments |
| Field travel, transport, and site logistics | 4,900 | Covers site visits, local transport, lodging, meals reimbursement, and structured weak-signal verification work across pilot locations |
| Data collection, storage, compute, and connectivity | 1,850 | Supports cloud or local compute needs, storage, internet bundles, backups, and secure project data handling |
| Prototype integration, documentation, and training materials | 2,250 | Funds prototype packaging, user guides, field manuals, translation-ready training material, and sponsor-facing reporting outputs |
| Technical implementation support | 2,700 | Provides tightly limited engineering support for prototype integration, testing discipline, and field-readiness tasks under student and supervisor direction |
| **Total requested** | **18,000** |  |

### Co-funding and in-kind support

Kyambogo University is expected to contribute in-kind through student time, supervision, basic lab access, and use of the existing software-development environment. Additional collaboration with Makerere-linked expertise or a rural implementation partner can be added later without weakening the student-led ownership of the work `[SUP-01, SUP-02]`.

### Budget logic

This budget is intentionally larger than a small undergraduate lab budget because the proposal is not only for offline modeling. It funds the interdisciplinary bridge between algorithm development, prototype packaging, limited field validation, and practical sponsor-facing outputs. That scale is consistent with the proposal's ambition to test a software-defined rural-access idea rather than deliver another desktop-only academic exercise `[OFF-05, OFF-07, WIN-01, WIN-05]`.

## Team and Supervision

### Student leads

**Ssemujju Sharif Abdukarim** is an undergraduate Electrical and Electronics Engineering student at Kyambogo University and a lead proposer on the GSM denoising project. His role in this proposal is to coordinate the technical narrative, signal-processing workflow, and overall project integration `[V1-01]`.

**Kisige Tom Derrick** is an undergraduate Electrical and Electronics Engineering student at Kyambogo University and co-proposer on the project. His role is to support model development, prototype testing, documentation, and field coordination within the student team `[V1-01]`.

### Supervisory leads

**Dr. Dickson Mugerwa** will provide academic supervision, engineering guidance, and proposal-quality oversight within the university research environment `[V1-01]`.

**Dr. Ephrance** will provide applied concept guidance, deployment-pathway support, and practical framing for the interdisciplinary funding route `[SUP-01, SUP-02]`.

### Collaboration posture

The project is student-led but intentionally collaborative. It is built to sit at the intersection of engineering, AI, low-cost prototyping, and rural service utility so that academic work can be translated into a fundable and testable public-value innovation `[OFF-05, OFF-07, WIN-01]`.

## References

- `OFF-05`. Uganda Communications Commission. "UCC launches 10th National Conference on Communications to drive market-driven innovation." 20 Apr 2026. <https://www.ucc.co.ug/ucc-launches-10th-national-conference-on-communications-to-drive-market-driven-innovation/>
- `OFF-06`. Uganda Communications Commission. "Call for Research Proposals for UCC Research Support to Academia." Apr 2026 PDF. <https://www.ucc.co.ug/wp-content/uploads/2026/04/Call-for-Research-Proposals-April-2026.pdf>
- `OFF-07`. Uganda Communications Commission. "UCC Research Support and Collaboration Framework 2022-2026." 31 Oct 2023. <https://www.ucc.co.ug/wp-content/uploads/2023/10/UCC-Research-Support-and-Collaboration-Framework-2022-2026.pdf>
- `OFF-08`. Uganda Communications Commission. "New Communications Sector Report Highlights Efforts to Bridge Access and Usage Gaps." 15 Dec 2025. <https://www.ucc.co.ug/new-communications-sector-report-highlights-efforts-to-bridge-access-and-usage-gaps/>
- `OFF-09`. Uganda Communications Commission. "New study digs into how tax policy shapes telecom sector growth." 11 Dec 2025. <https://www.ucc.co.ug/new-study-digs-into-how-tax-policy-shapes-telecom-sector-growth/>
- `OFF-10`. Uganda Communications Commission. "Study validates energy use and environmental impact of Uganda's telecom sector." 9 Feb 2026. <https://www.ucc.co.ug/study-validates-energy-use-and-environmental-impact-of-ugandas-telecom-sector/>
- `OFF-11`. Uganda Communications Commission. "UCC cracks down on illegal and non-compliant broadcasters." 25 Sep 2024. <https://www.ucc.co.ug/ucc-cracks-down-on-illegal-and-non-compliant-broadcasters/>
- `OFF-12`. Uganda Communications Commission. "UCC Leaders Engage Experts on Satellite Communications and Regulatory Preparedness." 13 Apr 2026. <https://www.ucc.co.ug/ucc-leaders-engage-experts-on-satellite-communications-and-regulatory-preparedness/>
- `OFF-18`. Uganda Communications Commission. "James Beronda profile." Accessed 22 Apr 2026. <https://www.ucc.co.ug/staff-member/james-beronda/>
- `SUP-01`. Supervisor Practical Concept Note for GSMA/UCC. Archived locally at [supervisor-concept-note-2026-04-23.md](/Users/sharif/telecom/final-year-project/NCC/sources/background/supervisor-concept-note-2026-04-23.md).
- `SUP-02`. Dr. Ephrance chat clarification. Archived locally at [dr-ephrance-chat-clarification-2026-04-23.md](/Users/sharif/telecom/final-year-project/NCC/sources/background/dr-ephrance-chat-clarification-2026-04-23.md).
- `V1-01`. Ssemujju Sharif Abdukarim and Kisige Tom Derrick. "GSM Signal Denoising and Modulation Classification for Rural Uganda." Preserved working draft at [GSM-Signal-Denoising-and-Modulation-Classification-for-rural-Uganda.md](/Users/sharif/telecom/final-year-project/GSM-Signal-Denoising-and-Modulation-Classification-for-rural-Uganda.md).
- `WIN-01`. NCC 2025 conference recap and winners. <https://ncc.co.ug/blog/harnessing-digital-innovation-to-power-sustainable-local-solutions-for-ugandas-development-goals-10/harnessing-digital-innovation-to-power-sustainable-local-solutions-for-ugandas-development-goals-7>
- `WIN-05`. "FareFlow: An IoT and Cloud-Based Smart Bus Fare Collection System for Sustainable Urban Transport." Accessible paper PDF archived locally at [fareflow-paper.pdf](/Users/sharif/telecom/final-year-project/NCC/sources/winners/fareflow-paper.pdf).
- `LIT-01`. Zhang, X., et al. "Dual Residual Denoising Autoencoder with Channel Attention Mechanism for Modulation of Signals." *Sensors*, 2023. <https://pmc.ncbi.nlm.nih.gov/articles/PMC9861137/>
- `LIT-02`. Abd-Elaziz, O. F., El-Ghandour, A. M., and Ismail, F. H. "Deep Learning-Based Automatic Modulation Classification Using Robust CNN Architecture for Cognitive Radio Networks." *Sensors*, 2023. doi:10.3390/s23239467
- `LIT-03`. Gao, X., Xu, X., Li, D., Liu, X., Yang, J., and Zhai, D. "Enhancing Noise Robustness in Few-Shot Automatic Modulation Classification via Complex-Valued Autoencoders." *Electronics*, 2026. <https://www.mdpi.com/2079-9292/15/3/674>
- `LIT-04`. An, H., and Lee, B.-M. "Robust Automatic Modulation Classification in Low Signal-to-Noise Ratio." *IEEE Access*, 2023. doi:10.1109/ACCESS.2023.3321108
- `LIT-05`. ITU-R. *Recommendation ITU-R P.530-18: Propagation data and prediction methods required for the design of terrestrial line-of-sight systems*. 2021. <https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.530-18-202109-I!!PDF-E.pdf>
- `LIT-06`. ITU-R. *Recommendation ITU-R P.838-3: Specific attenuation model for rain for use in prediction methods*. 2005. <https://www.itu.int/rec/R-REC-P.838-3-200503-P/en>
