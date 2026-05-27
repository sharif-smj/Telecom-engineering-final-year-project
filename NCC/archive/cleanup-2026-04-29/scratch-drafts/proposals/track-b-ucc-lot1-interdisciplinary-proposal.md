# AI-Assisted GSM Signal Denoising for Improved Access and Service Quality in Rural Uganda

## 1. Title Page

**Proposed funding route:** UCC Research Support to Academia  
**Lot:** `Lot 1 - Undergraduate Research Support`  
**Category:** `Interdisciplinary`  
**Primary theme fit:** `Access, Affordability, and User Experience of Communication Services`  
**Secondary theme fit:** `Digital Innovation, Entrepreneurship, and Emerging Technologies`; `Next-Generation Telecommunications Infrastructure and Spectrum Management` `[OFF-06, OFF-07]`

**Applicants**

- `Ssemujju Sharif Abdukarim`, Department of Electrical and Electronics Engineering, Kyambogo University
- `Kisige Tom Derrick`, Department of Electrical and Electronics Engineering, Kyambogo University

**Supervisory leads**

- `Dr. Dickson Mugerwa`
- `Dr. Ephrance`

**Project duration:** `6 months`  
**Budget request:** `UGX 20,000,000`

**Drafting note:** This draft already follows the UCC-required section order. On export to Word, it should be formatted in `Times New Roman 12`, `single spacing`, and kept within `20 pages` `[OFF-06, OFF-07]`.

## 2. Project Summary

Uganda's communications sector is growing, but UCC's recent public reporting still highlights access and usage gaps, affordability constraints, and hard-to-reach areas where service quality remains uneven `[OFF-08, OFF-09, OFF-12]`. In many of those areas, basic `2G`-capable phones remain important for voice calls, SMS, USSD, and Mobile Money, yet weak or noisy GSM conditions reduce the usability of those services `[OFF-08, V1-01]`.

This project proposes an interdisciplinary undergraduate study that combines telecommunications engineering, signal processing, artificial intelligence, and low-cost prototype thinking to investigate whether AI-assisted GSM signal denoising can improve weak-signal interpretation in rural Uganda. The study will design and test a denoising-autoencoder-plus-modulation-classification pipeline on controlled low-SNR conditions and Uganda-motivated interference scenarios, then evaluate its usefulness for rural communication access and underserved-area service analysis `[LIT-01, LIT-02, LIT-03, LIT-04, V1-01]`.

The project is honest about its present stage. It does not claim that the model has already been fully built and tested. Instead, it proposes a structured six-month research pathway with expected technical and practical outcomes. Any figures adapted from the supervisor concept note are treated as anticipated outcomes or pilot targets rather than completed project evidence `[SUP-01, SUP-02]`.

## 3. Introduction and Background

UCC's current strategic direction emphasizes an inclusive digital economy, stronger research translation, and communications services that are accessible, affordable, and secure `[OFF-05, OFF-06, OFF-07]`. Public reporting from late 2025 and early 2026 shows that while subscriptions and connectivity efforts have grown, meaningful access is still constrained by affordability, relevant usage, and uneven service conditions in underserved areas `[OFF-08, OFF-09, OFF-12]`.

This context matters for rural Uganda because communications need do not begin with high-end smartphones or broadband-intensive applications. In many communities, essential interactions still happen through lower-bandwidth mobile services such as voice, SMS, USSD, and Mobile Money on basic handsets. When GSM signals become weak, noisy, or interference-heavy, these essential services become less reliable `[OFF-08, V1-01]`.

UCC's enforcement against illegal and non-compliant broadcasters also shows that signal integrity remains a live regulatory concern. Harmful emissions and interference can complicate monitoring, degrade service, and make weak-signal environments even harder to interpret `[OFF-11]`. At the same time, expanding infrastructure in every underserved location is constrained by investment, affordability, and broader network-economics realities `[OFF-09, OFF-10, OFF-12]`.

The proposed study therefore asks whether a software-first signal-processing approach can improve weak-signal interpretation before costly infrastructure interventions are pursued. Specifically, it explores whether denoising a noisy GSM signal before modulation classification can recover more useful information from edge-of-coverage conditions relevant to rural Uganda `[V1-01, LIT-01, LIT-03, LIT-04]`.

## 4. Problem Statement

Rural and underserved areas in Uganda still experience weak and noisy mobile-service conditions that reduce the usability of basic communication services and complicate signal interpretation for operators and regulators `[OFF-08, OFF-11, OFF-12]`. Existing monitoring and coverage-assessment practices can indicate degraded service, but they do not necessarily recover usable signal structure once a GSM transmission is buried in noise or interference `[V1-01]`.

This creates a practical gap. Users experience failed or unreliable service at the edge of coverage, while institutions responsible for improving service quality have limited low-cost tools for extracting more useful information from those same weak-signal conditions. There is therefore a need to investigate whether AI-assisted signal denoising can improve GSM weak-signal interpretation and thereby support more reliable communication access and underserved-area decision-making in rural Uganda `[OFF-05, OFF-07, OFF-18, LIT-01, LIT-02]`.

## 5. Research Objectives

### Main objective

To design and evaluate an AI-assisted GSM signal-denoising pipeline for improving weak-signal interpretation and service quality analysis in rural Uganda `[V1-01]`.

### Specific objectives

1. To characterize weak-signal and interference conditions relevant to rural GSM service in Uganda using public datasets and Uganda-motivated signal-degradation models `[OFF-11, LIT-05, LIT-06, V1-01]`.
2. To develop a denoising autoencoder and a modulation-classification stage for GSM-oriented signals under low-SNR conditions `[LIT-01, LIT-02, LIT-03, LIT-04]`.
3. To compare the denoising-assisted pipeline against a non-denoising baseline using accuracy, macro F1-score, confusion matrices, and denoising-quality metrics `[V1-01, LIT-02, LIT-03]`.
4. To interpret the results in relation to rural voice, SMS, USSD, and Mobile Money service continuity, as well as possible regulator or operator use cases `[OFF-05, OFF-08, OFF-18]`.

## 6. Research Questions or Hypotheses

This proposal uses research questions.

1. How do weak-signal and interference-heavy GSM conditions affect signal interpretability in rural-communication scenarios relevant to Uganda `[OFF-08, OFF-11, V1-01]`?
2. Can a denoising autoencoder improve the recoverability of useful GSM signal features before modulation classification under low-SNR conditions `[LIT-01, LIT-03, LIT-04]`?
3. Does a denoising-assisted classification pipeline outperform a no-denoising baseline on selected GSM-family signal classes `[LIT-02, V1-01]`?
4. How can the resulting pipeline be interpreted as a low-cost support tool for rural service-quality analysis, underserved-area planning, or future prototype deployment `[OFF-05, OFF-07, OFF-18, SUP-02]`?

## 7. Justification of the Study

The study is justified on technical, policy, and public-value grounds.

First, it addresses a locally relevant communications problem. UCC's own recent reporting shows that the national challenge is not only whether connectivity exists, but whether it is usable, affordable, and meaningful in underserved settings `[OFF-08, OFF-09]`. A project that focuses on recovering more usable signal information from weak-service environments is therefore directly relevant to current sector priorities `[OFF-05, OFF-07]`.

Second, the study is interdisciplinary in a way that matches the nature of the problem. The project combines wireless communications, signal processing, machine learning, low-cost prototype design, and rural service-use thinking. That combination is necessary because weak-service conditions are simultaneously a network problem, an inference problem, and a public-service problem `[OFF-06, OFF-07, V1-01]`.

Third, the study is appropriate for undergraduate support because it is research-driven, scope-controlled, and capable of producing tangible outputs within six months. It does not depend on a nationwide deployment or a full commercial product. Instead, it aims to produce a reproducible pipeline, comparative evaluation results, and a grounded recommendation on whether a larger pilot pathway is justified `[OFF-06, OFF-07, SUP-02]`.

Finally, the study has clear potential value to UCC, UCUSAF, and operators because it explores a software-first way of extracting more value from existing infrastructure before costly physical expansion decisions are made `[OFF-05, OFF-09, OFF-18]`.

## 8. Methodology

### 8.1 Research design

The project will use an experimental research design anchored in reproducible data preparation, model development, benchmarking, and interpretation of results for rural Uganda communication scenarios `[V1-01]`. The work will be conducted primarily as an offline study using public datasets and controlled signal-degradation models, with limited exploratory field-support activity only if time, permissions, and equipment allow `[SUP-02]`.

### 8.2 Data sources and signal scope

The study will prioritize public datasets already identified in the preserved v1 project materials, particularly `RadioML 2018.01A` and the Kaggle `RF Signal Data` collection, because they provide a realistic starting point for controlled low-SNR experiments `[V1-01]`. The initial scope will focus on `GMSK`, `GFSK`, and `QPSK`-oriented conditions so that the study remains technically coherent and manageable within the undergraduate timeline `[V1-01]`.

### 8.3 Signal degradation and Uganda relevance

The signal-preparation stage will model low-SNR and interference-heavy conditions relevant to Uganda's rural-service problem. These conditions include general low-power reception, selected interference patterns motivated by UCC's signal-integrity context, and rain-related attenuation concepts from standard propagation literature `[OFF-11, LIT-05, LIT-06, V1-01]`. The goal is not to claim exact field replication, but to create a defensible Uganda-motivated evaluation environment for the proposed denoising approach `[SUP-02]`.

### 8.4 System design

The proposed system contains two main stages:

1. A denoising autoencoder that receives noisy I/Q signal inputs and attempts to recover cleaner signal structure `[LIT-01, LIT-03, LIT-04]`
2. A modulation-classification stage that operates on the denoised output and predicts the target signal class `[LIT-02, V1-01]`

This two-stage design is chosen because recent literature shows that denoising frontends can materially improve low-SNR modulation recognition by restoring useful features before classification begins `[LIT-01, LIT-03, LIT-04]`.

### 8.5 Baseline and evaluation

The denoising-assisted system will be compared with a baseline classifier operating directly on noisy signals. Evaluation metrics will include:

- classification accuracy
- macro F1-score
- confusion matrices
- denoising quality measures such as reconstruction error
- low-SNR trend analysis across selected test conditions `[V1-01, LIT-02, LIT-03]`

If feasible within the project timeline, the study will also record practical metrics such as model size and inference latency, but these will only be reported as measured results if actual artifacts are generated during the project `[SUP-02, V1-01]`.

### 8.6 Interpretation and practical meaning

The results will be interpreted beyond algorithmic accuracy alone. The analysis will ask whether denoising improves the recoverability of useful signal information in a way that could matter for rural voice, SMS, USSD, or Mobile Money continuity, as well as for regulator or operator troubleshooting in underserved areas `[OFF-05, OFF-08, OFF-18, WIN-05]`.

### 8.7 Ethical and regulatory considerations

The study will avoid handling subscriber content or personal communications data. The focus is on signal characteristics, controlled datasets, and system-performance analysis rather than user-message interception. Any future exploratory field captures will follow lawful, permission-based, and privacy-conscious procedures under supervisor guidance `[OFF-07, OFF-11, SUP-02]`.

## 9. Implementation Plan

| Month | Activity | Expected output |
| --- | --- | --- |
| 1 | Literature refinement, dataset preparation, and problem formalization | Clean research scope, dataset notes, and experiment plan |
| 2 | Signal preprocessing and Uganda-motivated low-SNR / interference modeling | Prepared training and evaluation inputs |
| 3 | Denoising autoencoder development and initial training | Initial denoising model and reconstruction outputs |
| 4 | Baseline AMC and denoising-assisted classification integration | Comparable baseline and hybrid pipelines |
| 5 | Evaluation, result interpretation, and limited practical-feasibility checks | Comparative plots, tables, and discussion notes |
| 6 | Final analysis, report writing, dissemination materials, and submission preparation | Final report, presentation material, and recommendation memo |

### Anticipated outputs

- one reproducible GSM denoising and classification workflow
- one comparative evaluation between denoising-assisted and non-denoising baselines
- one undergraduate research report and dissemination package
- one recommendation on the feasibility of a future practical pilot pathway `[SUP-02, OFF-07]`

### Anticipated outcomes

This study may produce performance improvements in controlled low-SNR settings. Where proposal-level targets from the supervisor concept note are mentioned in later review conversations, they should be treated as expected or anticipated outcomes rather than completed evidence until the project itself generates the relevant artifacts `[SUP-01, SUP-02]`.

## 10. Contribution to Cross-Cutting Issues

### Inclusion and equitable access

The project focuses on service conditions that matter to underserved communities, including users who still depend on basic `2G` phones rather than newer smartphones. This keeps the research grounded in equitable access rather than only high-end device scenarios `[OFF-08, OFF-18]`.

### Affordability

By examining a software-first approach to weak-signal improvement, the project speaks directly to affordability and cost-efficiency pressures in the communications sector. The value proposition is to extract more usefulness from existing conditions before assuming immediate infrastructure expansion `[OFF-09, OFF-10]`.

### Sustainability

The study supports a more resource-conscious approach to communications improvement by exploring whether better signal interpretation can complement, and in some cases defer, more energy- and capital-intensive interventions `[OFF-10, OFF-12]`.

### Capacity building

The project builds interdisciplinary undergraduate capacity in telecommunications, AI, signal processing, and applied research translation. This aligns with UCC's research-support emphasis on developing locally relevant academic capability `[OFF-06, OFF-07]`.

### Ethics and public interest

The study is designed around non-content signal analysis, lawful experimentation, and public-value communications outcomes. It avoids framing that would normalize invasive monitoring or unsupported deployment claims `[OFF-07, SUP-02]`.

## 11. Literature Review

Recent research shows that deep-learning-based automatic modulation classification performs better than many classical feature-engineering approaches, especially in low-SNR conditions where robust feature extraction becomes difficult `[LIT-02, LIT-03, LIT-04]`. This literature shift matters because the rural Uganda problem is fundamentally one of weak, noisy, and difficult-to-interpret signals rather than ideal clean-signal classification `[V1-01]`.

Among the most relevant studies are denoising-centered approaches. Zhang et al. showed that a dual residual denoising autoencoder can improve modulation recognition under noisy conditions, while Gao et al. and An and Lee demonstrated that denoising-oriented models can increase low-SNR robustness and recover discriminative signal structure before classification `[LIT-01, LIT-03, LIT-04]`. These studies support the core architectural decision in this proposal.

However, most of the available literature is still framed around general benchmark datasets and global low-SNR challenges rather than Uganda-specific communication conditions or regulator-facing rural-service interpretation. The preserved v1 project material already identifies a gap around booster-related interference, non-compliant transmissions, rain-related attenuation, and weak GSM service in rural Uganda `[V1-01]`. This proposal narrows that gap into a feasible undergraduate study.

The study is also shaped by UCC's current policy context. Recent official materials emphasize inclusive digital participation, underserved access, affordability, research translation, and communications-sector usefulness `[OFF-05, OFF-06, OFF-07, OFF-08, OFF-09]`. That makes a rural-service-focused denoising study more strategically relevant than a generic machine-learning benchmark project.

The literature and policy context together therefore justify an interdisciplinary project that treats denoising not as an isolated AI exercise, but as a practical communications-support tool for weak-signal rural service environments `[LIT-01, LIT-02, OFF-05, OFF-08]`.

## 12. Budget and Budget Justification

### Budget summary

| Budget item | Amount (UGX) | Justification |
| --- | ---: | --- |
| RF acquisition and test hardware | 5,800,000 | SDR-class capture tools, antennas, adapters, cables, test devices, and replacement accessories needed for controlled GSM signal experiments |
| Prototype design, development, testing, and calibration | 3,000,000 | Calibration materials, bench consumables, shielding aids, and prototype assembly needs within allowable project testing costs |
| Data collection, local travel, and field verification | 4,900,000 | Transport, lodging, and meals reimbursement for limited weak-signal verification visits and research coordination |
| Compute, storage, internet, and data handling | 2,500,000 | Cloud or local compute support, internet bundles, storage media, backups, and structured experiment logging |
| Software, documentation, copying, and dissemination | 1,500,000 | Software services, printing, binding, poster preparation, report submission, and dissemination materials |
| Research operations reserve within allowable categories | 2,300,000 | Controlled reserve for replacement accessories, extra calibration runs, and small cost fluctuations within approved research activities |
| **Total** | **20,000,000** |  |

### Budget justification

The budget is designed to stay within the `UGX 20,000,000` interdisciplinary cap while still giving the study enough practical depth to produce credible research outputs `[OFF-06, OFF-07]`. The largest share is assigned to RF and testing hardware because this project depends on signal-oriented experimentation rather than desk review alone. Travel and field-verification costs are included because the study must stay tied to real rural-service conditions, even if full-scale deployment is outside scope. The budget does not include salaries, stipends, tuition, or general institutional infrastructure costs, in line with the public call's funding logic `[OFF-06, OFF-07]`.

## 13. Curriculum Vitae of the Research Team

### Ssemujju Sharif Abdukarim

Ssemujju Sharif Abdukarim is an undergraduate student in the Department of Electrical and Electronics Engineering at Kyambogo University. His current research interest is the use of machine learning and signal processing to improve communications performance in underserved environments. In this project, he will coordinate the technical direction, signal-processing workflow, experiment integration, and reporting `[V1-01]`.

### Kisige Tom Derrick

Kisige Tom Derrick is an undergraduate student in the Department of Electrical and Electronics Engineering at Kyambogo University. His interests include embedded systems, applied communications engineering, and implementation-focused experimentation. In this project, he will support dataset preparation, prototype testing, evaluation, and documentation `[V1-01]`.

### Supervisory support

**Dr. Dickson Mugerwa** will provide academic supervision, engineering guidance, and project-quality oversight within the university environment `[V1-01]`.

**Dr. Ephrance** will provide co-supervision focused on practical concept development, interdisciplinary framing, and deployment-oriented guidance `[SUP-01, SUP-02]`.

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
- `LIT-01`. Zhang, X., et al. "Dual Residual Denoising Autoencoder with Channel Attention Mechanism for Modulation of Signals." *Sensors*, 2023. <https://pmc.ncbi.nlm.nih.gov/articles/PMC9861137/>
- `LIT-02`. Abd-Elaziz, O. F., El-Ghandour, A. M., and Ismail, F. H. "Deep Learning-Based Automatic Modulation Classification Using Robust CNN Architecture for Cognitive Radio Networks." *Sensors*, 2023. doi:10.3390/s23239467
- `LIT-03`. Gao, X., Xu, X., Li, D., Liu, X., Yang, J., and Zhai, D. "Enhancing Noise Robustness in Few-Shot Automatic Modulation Classification via Complex-Valued Autoencoders." *Electronics*, 2026. <https://www.mdpi.com/2079-9292/15/3/674>
- `LIT-04`. An, H., and Lee, B.-M. "Robust Automatic Modulation Classification in Low Signal-to-Noise Ratio." *IEEE Access*, 2023. doi:10.1109/ACCESS.2023.3321108
- `LIT-05`. ITU-R. *Recommendation ITU-R P.530-18: Propagation data and prediction methods required for the design of terrestrial line-of-sight systems*. 2021. <https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.530-18-202109-I!!PDF-E.pdf>
- `LIT-06`. ITU-R. *Recommendation ITU-R P.838-3: Specific attenuation model for rain for use in prediction methods*. 2005. <https://www.itu.int/rec/R-REC-P.838-3-200503-P/en>
- `WIN-05`. "FareFlow: An IoT and Cloud-Based Smart Bus Fare Collection System for Sustainable Urban Transport." Accessible paper PDF archived locally at [fareflow-paper.pdf](/Users/sharif/telecom/final-year-project/NCC/sources/winners/fareflow-paper.pdf).
