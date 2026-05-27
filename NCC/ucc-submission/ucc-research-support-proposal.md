# Signal Denoising for Improved Edge-of-Coverage GSM Service in Rural Uganda

## 1. Title Page

**Proposal type:** Uganda Communications Commission Research Support to Academia

**Research study title:** Signal Denoising for Improved Edge-of-Coverage GSM Service in Rural Uganda

**Funding category:** Lot 3 - Inter-University Research Collaboration, Interdisciplinary

**Priority research areas:** Access, Affordability, and User Experience of Communication Services; Digital Innovation, Entrepreneurship, and Emerging Technologies; Next-Generation Telecommunications Infrastructure and Spectrum Management

**Principal Investigator:** Dr. Dickson Mugerwa, Department of Electrical and Electronics Engineering, Kyambogo University

**Co-Principal Investigator:** Dr. Ephrance Eunice Namugenyi, Department of Electrical and Electronics Engineering, Kyambogo University / Makerere University research collaboration

**Student researchers / research assistants:**  
Ssemujju Sharif Abdukarim, Department of Electrical and Electronics Engineering, Kyambogo University  
Kisige Tom Derrick, Department of Electrical and Electronics Engineering, Kyambogo University

**Lead institution:** Kyambogo University, Kampala, Uganda

**Collaborating institution:** Makerere University, College of Computing and Information Sciences, Kampala, Uganda

**Proposed period of performance:** May 2026 to November 2026

**Total budget requested:** UGX 66,600,000

**Date prepared:** 3 May 2026

**Addressed to:**  
The Executive Director  
Uganda Communications Commission  
UCC House, Plot 42-44 Spring Road, Bugolobi  
P.O. Box 7376, Kampala, Uganda

**Signature of Principal Investigator:** _______________________________

**Signature of Co-Principal Investigator:** ____________________________

## 2. Project Summary

Rural mobile service in Uganda is often judged by whether network coverage exists, yet many users experience a more practical problem: the signal is present but too weak, noisy, or unstable to support dependable voice calls, SMS, USSD, and Mobile Money sessions. UCC's current Research Support to Academia call prioritizes access, user experience, digital innovation, emerging technologies, and next-generation telecommunications infrastructure and spectrum management (Uganda Communications Commission, 2026a). This study responds to those priorities by proposing a denoising-assisted GSM signal processing approach for interpreting weak edge-of-coverage signals in rural and underserved areas.

The project will design and evaluate a pipeline that combines a denoising autoencoder with automatic modulation classification. Noisy I/Q signal samples will first be reconstructed by the denoising model, then passed to a classifier so that modulation decisions are made from a cleaner signal representation. The proposed approach will be compared with a baseline classifier operating directly on noisy signals. Evaluation will use public I/Q datasets, controlled low-SNR impairments, reconstruction error, SNR-related measures, classification accuracy, macro F1-score, confusion matrices, model size, and inference latency.

The study is interdisciplinary, combining telecommunications engineering, RF signal processing, machine learning, low-cost prototyping, and rural service-quality analysis. The expected outputs include a documented experimental workflow, a denoising-assisted classification prototype, an Android on-device inference prototype using ONNX Runtime, and a technical report showing how such a tool could support UCC and UCUSAF in weak-service interpretation, interference follow-up, underserved-area planning, and evidence-based rural connectivity decisions. The work will not make tower-range, deployment, or live-service claims unless validated by evidence.

## 3. Introduction

Uganda's communications sector has expanded, but good service is still uneven in rural and underserved locations. UCC's Research Support to Academia call prioritizes access, user experience, digital innovation, emerging technologies, and next-generation telecommunications infrastructure and spectrum management (Uganda Communications Commission, 2026a). This proposal responds by studying weak GSM-family service at the edge of coverage.

For many users, the services that matter most are still voice calls, SMS, USSD, and Mobile Money, often on basic 2G-capable phones. A place may therefore appear covered while the actual service remains unreliable because the received signal is weak, noisy, or affected by interference. UCC has also warned about interference from network repeaters and boosters and has acted against illegal or non-compliant broadcasters (Uganda Communications Commission, 2021, 2024b).

New towers, optimization, power improvements, and enforcement remain necessary. This project does not replace that work. It investigates whether GSM-family signal denoising can help recover useful signal structure from noisy I/Q samples and produce clearer evidence for weak-service interpretation, rural planning, and later receive-only field validation.

## 4. Problem Statement

Rural and underserved users may see a GSM signal yet still fail to complete calls, SMS, USSD, or Mobile Money sessions because the signal is too weak, noisy, or unstable. UCC identifies access and user experience as priority research areas, while UCUSAF focuses on unserved and underserved communities that are difficult to serve commercially (Uganda Communications Commission, 2026a, 2026b).

The engineering gap is weak-service interpretation. At the edge of coverage, a service problem may be caused by low received power, interference, fading, environmental loss, or a combination of these factors. UCC's notices on repeaters, boosters, and non-compliant transmissions show that signal quality is also a spectrum and compliance concern, not only a coverage-map issue (Uganda Communications Commission, 2021, 2024b).

This study asks how a denoising-assisted GSM signal processing pipeline can improve recovery and interpretation of weak GSM-family I/Q samples under controlled edge-of-coverage conditions. The project will test a denoising autoencoder connected to an automatic modulation classifier, compare it with a no-denoising baseline, and package the workflow for a receive-only Android-SDR or replay-based prototype. It will not claim live service improvement or tower-range impact unless future evidence proves it.

## 5. Aim and Objectives

### 5.1 Aim

The aim of this project is to design and evaluate a denoising-assisted GSM-family signal processing pipeline for improved interpretation of weak, noisy, edge-of-coverage signals in rural Uganda.

### 5.2 Specific Objectives

1. To characterize weak-signal and interference-heavy GSM service conditions relevant to rural Uganda using UCC materials, literature, and controlled signal impairment models.

2. To design a denoising autoencoder that reconstructs cleaner I/Q signal representations from noisy GSM-family samples.

3. To integrate the denoising autoencoder with an automatic modulation classification stage for selected GSM-family or GSM-adjacent modulation classes.

4. To compare the denoising-assisted pipeline with a baseline classifier operating directly on noisy signals.

5. To develop a low-cost Android on-device inference prototype that can load or receive I/Q windows, run exported ONNX denoising and classification models, and display before-and-after signal outputs for practical review.

6. To interpret the results in terms of UCC value, rural service reliability, interference follow-up, coverage planning, and future pilot feasibility.

## 6. Research Questions

1. What weak-signal, noise, and interference conditions are most relevant to GSM-family service interpretation in rural Uganda?

2. Can a denoising autoencoder recover useful signal structure from noisy GSM-family I/Q samples under controlled low-SNR conditions?

3. Does a denoising-assisted modulation classification pipeline perform better than a no-denoising baseline under selected weak-signal conditions?

4. Which evaluation metrics best translate the technical result into practical meaning for UCC, operators, and rural service troubleshooting?

5. How can the denoising pipeline be packaged as a low-cost Android on-device inference prototype without requiring rural users to own smartphones or suggesting that the app directly improves live calls?

6. What ethical, regulatory, and technical limitations must be addressed before any field validation is attempted?

## 7. Justification and Direct Benefit to UCC

The study is useful because it deals with a real service problem in a technical way. UCC's call asks for research that improves communication access, supports underserved communities, and contributes to policy, regulation, and innovation (Uganda Communications Commission, 2026a). This proposal fits that direction by focusing on weak GSM-family service, where a user may be connected in theory but still unable to rely on basic services.

For UCC, the main benefit is better evidence for weak-service analysis. If the denoising-assisted pipeline improves recoverability under controlled low-SNR conditions, the result can help explain whether a marginal service problem looks more like coverage weakness, interference, distortion, or poor signal quality. That kind of evidence is useful before deciding on optimization, additional measurements, enforcement follow-up, or infrastructure planning.

For UCUSAF, the work supports underserved-area planning without pretending to replace towers or operators. UCUSAF already works on access infrastructure and devices for underserved communities (Uganda Communications Commission, 2026b, 2026c, 2026d). This project can add a reproducible weak-signal testing workflow and a short UCC-facing technical brief. Any field-support sites will be selected only from official UCC/UCUSAF materials or approved institutional evidence, not guessed district names.

The project also builds local capacity. It combines RF engineering, machine learning, Android/SDR prototyping, ethical measurement, and practical rural service analysis. The student researchers will support a lecturer-led Lot 3 collaboration while producing tools and documentation that can be reviewed beyond the university.

## 8. Literature Review

Weak mobile service is not only a question of whether a location appears on a coverage map. UCC's call places access, affordability, user experience, digital innovation, and spectrum management among its priority areas, which makes weak basic-service reliability a relevant research problem (Uganda Communications Commission, 2026a). UCC's public notices on boosters, repeaters, and non-compliant broadcasters also show that interference and signal integrity remain practical sector concerns (Uganda Communications Commission, 2021, 2024b).

Propagation and attenuation are well-established causes of radio-link degradation. ITU-R recommendations provide standard models for terrestrial propagation and attenuation prediction, and they support the use of controlled impairment models in a laboratory setting (International Telecommunication Union, 2005, 2021). In this study, such models will be used only to create test conditions for denoising and classification, not to claim full replication of every rural channel.

Automatic modulation classification identifies the modulation type of a received signal and is useful in spectrum monitoring, radio-environment awareness, and interference diagnosis. Deep learning methods can learn directly from I/Q samples, and CNN-based AMC work has shown useful results in cognitive-radio settings (Abd-Elaziz et al., 2023). Public I/Q datasets such as RadioML 2018.01A also provide a starting point for repeatable experiments (DeepSig, 2018).

Denoising autoencoders reconstruct cleaner versions of corrupted inputs. Recent studies have applied residual autoencoders, attention mechanisms, and complex-valued autoencoders to improve modulation-signal interpretation under noisy conditions (An & Lee, 2023; Gao et al., 2026; Zhang et al., 2023). The gap this project addresses is local framing: these methods are rarely connected to Uganda's weak GSM-family service problem, basic-phone use, UCC/UCUSAF planning needs, and low-cost receive-only field-support workflows.

## 9. Methodology

### 9.1 Research Design

The study will use an experimental research design. The core work will be conducted offline using public I/Q datasets, generated GSM-family test signals where necessary, and controlled signal impairments. This design is appropriate because the first task is to test whether denoising improves weak-signal recoverability before any field-support validation is attempted.

Two processing chains will be compared. The baseline chain will classify noisy I/Q samples directly. The proposed chain will first reconstruct the noisy I/Q sample using a denoising autoencoder and then classify the denoised output using the same classifier family. This keeps the comparison fair because the main difference is the denoising front end.

[ARCHITECTURE_FIGURE]

Figure 1 shows the full workflow. The core evidence comes from offline validation, while the Android-SDR path gives a receive-only prototype for importing or capturing short I/Q windows, running exported ONNX models, and displaying denoising, classification, confidence, and latency outputs.

### 9.2 Data Sources and Signal Scope

The first stage will use public I/Q datasets. RadioML 2018.01A provides labelled modulation samples with structured SNR information (DeepSig, 2018). RF Signal Data will be considered as a secondary SDR-style source (RF Signal Data, 2025). If a GSM-specific class is not sufficiently represented, limited GSM-family test signals will be generated for controlled experiments. The initial scope will focus on selected GSM-family or adjacent classes such as GMSK, GFSK, and QPSK where available.

### 9.2.1 Field-Site Selection Logic

Field-support validation will proceed only after approval. Candidate sites will be selected from official UCC/UCUSAF materials, access-infrastructure priorities, MNO-reported or UCC-recognized weak-service evidence, safety, and receive-only feasibility. UCUSAF access-infrastructure work already uses coverage mapping and MNO-report evidence, so this proposal will not invent districts without documentary support (Uganda Communications Commission, 2024a).

### 9.3 Signal Impairment Modelling

The received baseband I/Q signal will be modelled as:

Equation (1): r[n] = a[n]s[n] + i_wb[n] + i_nb[n] + w[n]

where r[n] is the received noisy signal, s[n] is the clean reference signal, a[n] is a time-varying attenuation factor, i_wb[n] is a wideband interference component, i_nb[n] is a narrowband interference component, and w[n] is additive noise.

Additive white Gaussian noise will be used to represent thermal noise and low-SNR edge-of-coverage conditions:

Equation (2): w[n] ~ CN(0, sigma_w^2)

For a target SNR, noise power will be selected using:

Equation (3): SNR_dB = 10 log10(P_s / P_w)

where P_s is the average signal power and P_w is the average noise power. This allows the experiment to compare performance across selected low-SNR ranges.

Narrowband interference will be modelled as:

Equation (4): i_nb[n] = A_i cos(2 pi f_i n / F_s + phi)

where A_i is the interference amplitude, f_i is the interference frequency, F_s is the sampling frequency, and phi is the phase. This approximates the effect of a concentrated interfering tone or poorly filtered transmitter component.

Wideband disturbance will be modelled using filtered random noise or a multi-tone disturbance spread across a wider part of the sampled band. This is included because UCC has warned about improperly used repeaters or boosters that can disturb legitimate network service (Uganda Communications Commission, 2021).

Time-varying attenuation will be represented by:

Equation (5): a[n] = 10^(-L[n]/20)

where L[n] is a loss term informed by propagation and attenuation concepts from ITU-R recommendations (International Telecommunication Union, 2005, 2021). The purpose is not to reproduce every rural channel condition, but to create controlled degradation scenarios that can test whether denoising preserves useful GSM-family signal structure.

### 9.4 Denoising Model

The denoising model will be a one-dimensional autoencoder designed for I/Q signal windows. The encoder will compress a corrupted input r into a latent representation z, and the decoder will reconstruct an estimate of the cleaner signal:

Equation (6): z = E_theta(r), x_hat = D_theta(z)

The model will be trained to minimize reconstruction error:

Equation (7): L_DAE = (1/N) sum from n=1 to N |s[n] - x_hat[n]|^2

where s[n] is the reference signal and x_hat[n] is the reconstructed signal. Reconstruction quality will be inspected numerically and visually, but the main test will be whether the denoised output improves downstream classification.

### 9.5 Automatic Modulation Classification

The classifier will operate on either raw noisy inputs or denoised outputs. For the proposed chain:

Equation (8): y_hat = argmax g_phi(x_hat)

where g_phi is the classifier and y_hat is the predicted modulation class. A compact CNN or similar lightweight sequence classifier will be used depending on dataset structure and available computing resources.

[MODEL_ARCHITECTURE_FIGURE]

Figure 2 shows how the proposed model chain differs from the baseline. The DAE reconstructs a denoised I/Q window before AMC classification, while the dashed path keeps a raw-noisy-input classifier for fair comparison.

### 9.6 Experimental Procedure

The experiment will: select and document datasets and SNR ranges; segment and normalize I/Q windows; create controlled noisy versions using noise, interference, and attenuation models; train the baseline classifier; train the denoising autoencoder; evaluate the denoising-assisted classifier across selected SNR ranges; measure model size and latency; and prepare the Android-SDR workflow for supervised validation if approved.

## 10. Data Collection Methods and Tools

Three tools will be used. The experiment log will record dataset source, modulation classes, SNR range, preprocessing, impairment settings, model settings, random seed, date, and output files. The results sheet will capture reconstruction error, SNR-related measures, accuracy, macro F1-score, confusion matrices, low-SNR trends, model size, latency, and failure cases. The field checklist will be used only after approval and will record site category, basic-phone service observations, receive-only Android-SDR setup, antenna placement, approval notes, and safety notes without recording private communication content.

## 11. Data Analysis Plan

Data analysis will compare denoising quality, classification quality, low-SNR behaviour, and practical feasibility. Reconstruction error and SNR-related measures will show whether useful signal structure is being recovered. Accuracy, macro F1-score, and confusion matrices will show whether denoising improves classification compared with the no-denoising baseline. Macro F1-score is included because it is more informative than accuracy when classes are not equally easy to classify.

Equation (9): Precision = TP / (TP + FP)

Equation (10): Recall = TP / (TP + FN)

Equation (11): F1 = 2(Precision x Recall) / (Precision + Recall)

Equation (12): Macro F1 = average of F1 across all classes

Low-SNR results will be examined separately so that any gain is not hidden inside an overall average. Model size, inference latency, and prototype suitability will also be reported. Positive offline results will be described only as improved weak-signal recoverability under controlled conditions, not as proof of live service improvement or tower-range impact.

## 12. Ethical and Regulatory Considerations

The core study uses public datasets, generated signals, and controlled impairments. It will not record, decode, or store private subscriber communications, and it will not transmit or interfere with licensed networks. If field-support validation is approved, the required institutional ethical and regulatory approvals will be obtained before field activity. Any Android-SDR kit will be receive-only unless separate authorization is granted, and the project will not present denoising as a substitute for licensed operators, infrastructure planning, spectrum regulation, or network optimization.

## 13. Implementation Plan and Gantt-Style Work Schedule

The project will run from May to November 2026 and will be completed before 30 November 2026, in line with the UCC call requirement (Uganda Communications Commission, 2026a).

[GANTT_TIMELINE]

Figure 3 presents the overlapping work schedule from approvals and data preparation through model development, prototype workflow, validation, reporting, and dissemination.

## 14. Contribution to Cross-Cutting Issues

The project supports inclusion by focusing on basic services used by many groups, including women, youth, farmers, small traders, community health workers, and households that rely on basic handsets. Any approved field observation will seek balanced service-use perspectives where human feedback is collected. It also supports responsible innovation by using receive-only measurement, avoiding private-content collection, separating expected outcomes from measured results, and treating AI as a support tool for evidence, not as a replacement for licensed infrastructure or regulation.

## 15. Expected Outputs and Deliverables

Expected outputs are: a documented weak-signal preprocessing workflow; controlled impairment scripts; a denoising autoencoder; a baseline AMC model; a denoising-assisted AMC pipeline; result tables and plots; an Android-SDR receive-only prototype workflow; a UCC-facing technical brief; the final report; presentation materials; and a conference-style manuscript.

## 16. Sustainability and Prototype Pathway

The project moves beyond modelling through a receive-only Android-SDR or edge-device workflow. The interface will show noisy input, denoised output, SNR-related or reconstruction indicators, classification output, and inference latency. It will not modify a user's basic phone, improve live calls directly, or act as a booster.

The pathway is: offline validation first; approved field-support observation second; and only then a small pilot based on service scenarios such as SMS, USSD, and call setup if the evidence justifies it. Sustainability will depend on whether the model is lightweight, reproducible, and useful to real planning decisions, so model size and latency will be reported alongside accuracy.

## 17. Budget and Budget Justification

The project requests UGX 66,600,000 under Lot 3 Inter-University Research Collaboration, Interdisciplinary. The request is below the UGX 100,000,000 interdisciplinary ceiling in the UCC call and avoids tuition, salaries, stipends, remuneration, and infrastructure construction. Costs are grouped by approvals, prototype hardware, laboratory work, field observation, coordination, dissemination, and contingency.

| Budget category | Amount (UGX) | Justification |
| --- | ---: | --- |
| Study approvals and protocol setup | 6,100,000 | Covers institutional ethics or REC preparation, UNCST or equivalent research clearance where required, protocol documentation, consent materials, and a focused inception meeting for scope, roles, methodology, work plan, and approval responsibilities. |
| Tools, equipment, and prototype hardware | 22,800,000 | Provides the tangible engineering platform requested in the feedback: receive-only SDR kits, antennas, OTG adapters, RF cables, filters, attenuators, connectors, Android measurement phones, power and protection accessories, prototype enclosures, mounting, safe packaging, shipping, spares, and price variation for imported RF parts. |
| Laboratory experimentation and model/system analysis | 10,700,000 | Supports repeatable bench experiments, controlled impairment testing, shielding, calibration aids, signal-source accessories, storage, backups, experiment repository preparation, compute/model optimization support, ONNX export testing, and visualization of denoising and classification results. |
| Field observation and validation logistics | 12,400,000 | Supports supervised transport, local movement, lodging, meals, site coordination, field safety, and criteria-based weak-service observation at approved sites selected using UCC/UCUSAF underserved-area and service-gap logic. It also includes limited respondent data and airtime for consented observations where human feedback is approved. |
| Communication, coordination, and stakeholder review | 4,600,000 | Covers six-month team communication data bundles, inter-university technical review meetings, PI and Co-PI coordination, and a UCC/UCUSAF-facing stakeholder validation session for feedback on the prototype workflow and technical report. |
| Reporting, dissemination, and handover | 5,000,000 | Covers final technical report production, printing and binding, UCC technical brief preparation, presentation materials, repository packaging, code/results handover documentation, and conference-style manuscript formatting. |
| Contingency and controlled price variation | 5,000,000 | Provides a controlled reserve for unavoidable changes in approval costs, RF hardware prices, transport, field logistics, or dissemination items, subject to supervisor and institutional approval. |
| **Total** | **66,600,000** |  |

## 18. Curriculum Vitae of the Research Team

### 18.1 Dr. Dickson Mugerwa

**Proposed role:** Principal Investigator  
**Institution:** Kyambogo University  
**Department:** Department of Electrical and Electronics Engineering  
**Relevant qualifications:** PhD in Radio and Communication Engineering; MSc in Radio Science, Electronics and Communication; BSc in Information Technology and Computing; Diploma in Computer Science.  
**Relevant experience:** Senior ICT technical academic staff at Kyambogo University, embedded systems consultant, doctoral researcher, and communications/IoT researcher.  
**Selected research areas:** Radio and communication engineering, IoT LoRa networks, wireless sensor networks, embedded systems, multi-hop communication, and distributed communication protocols.  
**Project responsibilities:** Lead the research grant, supervise technical quality, guide RF experimentation, coordinate Kyambogo University participation, approve deliverables, and ensure the final report meets UCC expectations.

### 18.2 Dr. Ephrance Eunice Namugenyi

**Proposed role:** Co-Principal Investigator  
**Institutional link:** Kyambogo University / Makerere University research collaboration  
**Academic profile:** Lecturer and researcher with experience in Data Communications, Software Engineering, Communication Networks, wireless communications, IoT systems, edge intelligence, and adaptive network architectures.  
**Education:** PhD in Software Engineering, Makerere University, 2021-2026; MSc in Data Communications and Software Engineering, Makerere University, 2015-2019; BSc in Telecommunications Engineering, Makerere University, 2008-2012.  
**Relevant appointment:** Lecturer, Electrical and Electronics Engineering, Kyambogo University, 2013-present.  
**Relevant skills:** Python, C, C++, machine learning for network optimization, NS-3, MATLAB, IoT, LPWAN, Wi-Fi, GSM/LTE, and edge computing.  
**Project responsibilities:** Lead computing and AI/ML framing, advise on practical prototype pathway, support inter-university collaboration, review model feasibility, and guide translation of the work into an applied communications-sector output.

### 18.3 Ssemujju Sharif Abdukarim

**Proposed role:** Student researcher / research assistant  
**Institution:** Kyambogo University  
**Department:** Department of Electrical and Electronics Engineering  
**Current level:** Undergraduate engineering student  
**Research focus:** Telecommunications engineering, weak-signal GSM service, RF signal processing, and machine learning-assisted signal interpretation.  
**Relevant skills:** Digital communication systems, signal processing fundamentals, Python/MATLAB-style technical computing, RF measurement concepts, machine learning experimentation, technical writing, and research documentation.  
**Project responsibilities:** Support literature review, dataset preparation, denoising model development, experiment logging, result interpretation, documentation, prototype review, and final reporting.

### 18.4 Kisige Tom Derrick

**Proposed role:** Student researcher / research assistant  
**Institution:** Kyambogo University  
**Department:** Department of Electrical and Electronics Engineering  
**Current level:** Undergraduate engineering student  
**Research focus:** Telecommunications systems, signal quality analysis, low-cost RF prototyping, and practical rural connectivity support.  
**Relevant skills:** Communication networks, electronics prototyping, RF equipment setup, dataset organization, model evaluation support, documentation, and field observation planning.  
**Project responsibilities:** Coordinate prototype setup support, Android-SDR or edge-kit assembly planning, field-support validation preparation, signal-testing logs, baseline classifier testing, result verification, and dissemination preparation.

## 19. References

Abd-Elaziz, O. F., Abdalla, M., & Elsayed, R. A. (2023). Deep learning-based automatic modulation classification using robust CNN architecture for cognitive radio networks. *Sensors, 23*(23), Article 9467. https://www.mdpi.com/1424-8220/23/23/9467

An, T. T., & Lee, B. M. (2023). Robust automatic modulation classification in low signal-to-noise ratio. *IEEE Access, 11*, 7860-7872. https://doi.org/10.1109/ACCESS.2023.3238995

DeepSig. (2018). *RadioML 2018.01A dataset*. https://www.deepsig.ai/datasets/

Gao, M., Zhang, B., Wang, L., Tang, X., & Huan, H. (2026). Enhancing noise robustness in few-shot automatic modulation classification via complex-valued autoencoders. *Electronics, 15*(3), Article 674. https://www.mdpi.com/2079-9292/15/3/674

International Telecommunication Union. (2005). *Recommendation ITU-R P.838-3: Specific attenuation model for rain for use in prediction methods*. https://www.itu.int/rec/R-REC-P.838-3-200503-P/en

International Telecommunication Union. (2021). *Recommendation ITU-R P.530-18: Propagation data and prediction methods required for the design of terrestrial line-of-sight systems*. https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.530-18-202109-I!!PDF-E.pdf

RF Signal Data. (2025). *RF signal data* [Data set]. Kaggle. https://www.kaggle.com/datasets/suraj520/rf-signal-data

Uganda Communications Commission. (2021, July 26). *Public notice: Signal interference arising out of usage of network repeaters - "boosters"*. https://uccinfoblog.com/2021/07/26/public-notice-signal-interference-arising-out-of-usage-of-network-repeaters-boosters/

Uganda Communications Commission. (2024a, March 5). *UCC/UCUSAF affirmative action telecom infrastructure program launched in Agago*. https://www.ucc.co.ug/wp-content/uploads/2024/03/UCC-UCUSAF-TELECOM-INFRASTRUCTURE-LAUNCHED-IN-AGAGO-06.FEB_.2024.pdf

Uganda Communications Commission. (2024b, September 25). *UCC cracks down on illegal and non-compliant broadcasters*. https://www.ucc.co.ug/ucc-cracks-down-on-illegal-and-non-compliant-broadcasters/

Uganda Communications Commission. (2025, December 15). *New communications sector report highlights efforts to bridge access and usage gaps*. https://www.ucc.co.ug/new-communications-sector-report-highlights-efforts-to-bridge-access-and-usage-gaps/

Uganda Communications Commission. (2026a). *Call for proposals for the Uganda Communications Commission (UCC) research support to academia*. https://www.ucc.co.ug/wp-content/uploads/2026/04/Call-for-Research-Proposals-April-2026.pdf

Uganda Communications Commission. (2026b). *Uganda Communications Universal Service and Access Fund (UCUSAF)*. https://www.ucc.co.ug/ucusaf/

Uganda Communications Commission. (2026c). *UCUSAF Access Infrastructure Program*. https://www.ucc.co.ug/ucusaf/access-infrastructure-program/

Uganda Communications Commission. (2026d). *UCUSAF Devices for Underserved Communities Program*. https://www.ucc.co.ug/ucusaf/devices-for-underserved-communities-program/

Zhang, X., et al. (2023). Dual residual denoising autoencoder with channel attention mechanism for modulation of signals. *Sensors, 23*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9861137/

## 20. Appendices: Data Collection Tools

### Appendix A: Experiment Log Template

Fields to be completed for every experiment run: experiment ID; date; researcher; dataset name; dataset version or URL; selected modulation classes; sample length; sampling rate where available; preprocessing method; normalization method; SNR range; impairment model; impairment parameters; model architecture; training epochs; optimizer; loss function; random seed; hardware used; output file path; technical review notes.

### Appendix B: Results Extraction Sheet Template

Fields to be completed after each evaluation run: experiment ID; dataset; signal class; SNR level; baseline accuracy; denoising-assisted accuracy; baseline macro F1; denoising-assisted macro F1; reconstruction error; estimated SNR-related improvement; confusion matrix file path; model size; inference latency; observed failure cases; interpretation notes.

### Appendix C: Field Observation Checklist

Fields to be completed only after approval for field-support validation: site code; district or location category; date and time; weather; terrain description; nearby visible tower or obstruction notes; handset type used for service observation; basic service observed, such as SMS, USSD, or call setup; Android-SDR kit ID; antenna position; receive-only confirmation; no private content collected; approval record; safety notes; field remarks.
