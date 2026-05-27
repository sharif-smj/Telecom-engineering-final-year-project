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

**Date prepared:** 29 April 2026

**Addressed to:**  
The Executive Director  
Uganda Communications Commission  
UCC House, Plot 42-44 Spring Road, Bugolobi  
P.O. Box 7376, Kampala, Uganda

**Signature of Principal Investigator:** _______________________________

**Signature of Co-Principal Investigator:** ____________________________

## 2. Project Summary

Rural mobile service in Uganda is often judged by whether network coverage exists, yet many users experience a more practical problem: the signal is present but too weak, noisy, or unstable to support dependable voice calls, SMS, USSD, and Mobile Money sessions. This study proposes a denoising-assisted GSM signal processing approach for improving the interpretation of weak edge-of-coverage signals in rural and underserved areas.

The project will design and evaluate a pipeline that combines a denoising autoencoder with automatic modulation classification. Noisy I/Q signal samples will first be reconstructed by the denoising model, then passed to a classifier so that modulation decisions are made from a cleaner signal representation. The proposed approach will be compared with a baseline classifier operating directly on noisy signals. Evaluation will use public I/Q datasets, controlled low-SNR impairments, reconstruction error, SNR-related measures, classification accuracy, macro F1-score, confusion matrices, model size, and inference latency.

The study is interdisciplinary, combining telecommunications engineering, RF signal processing, machine learning, low-cost prototyping, and rural service-quality analysis. The expected outputs include a documented experimental workflow, a denoising-assisted classification prototype, a low-cost Android-SDR field-support concept, and a technical report showing how such a tool could support UCC in weak-service interpretation, interference follow-up, and evidence-based rural connectivity planning. The work will not claim physical tower range extension unless validated by evidence.

## 3. Introduction

Uganda's communications sector has grown rapidly, but meaningful service quality remains uneven in rural and underserved locations. The Uganda Communications Commission (UCC) Research Support to Academia call emphasizes research that improves access to communication services, supports underserved communities, strengthens policy and regulation, and promotes collaboration between academia, industry, and policymakers (Uganda Communications Commission, 2026). This project responds to that call by focusing on the weak-service conditions experienced at the edge of GSM-family coverage.

For many rural users, communication still depends on basic and immediate services: voice calls, SMS, USSD, and Mobile Money. These services may be accessed through basic 2G-capable handsets as much as smartphones. A weak-service area should therefore not be judged only by a signal bar or a nominal coverage map. A phone may detect a network but still fail to complete a call setup, delay an SMS, or drop a USSD session when the received signal is weak, noisy, or affected by interference.

UCC has treated signal integrity as a live regulatory concern. It has warned the public about interference caused by network repeaters or boosters and has taken action against illegal and non-compliant broadcasters whose transmissions may interfere with licensed services and public safety communications (Uganda Communications Commission, 2021, 2024). These examples show that weak service is not caused only by distance from a base station. It may also arise from interference, poorly controlled transmission, environmental attenuation, and local equipment conditions.

Conventional responses to weak rural service include new towers, network optimization, backhaul improvements, better power systems, spectrum enforcement, and coverage monitoring. These remain essential. However, they are expensive and may take time to implement. A complementary software-assisted signal interpretation method could help regulators, operators, and university researchers obtain better evidence before deciding whether a weak-service area requires physical expansion, optimization, interference follow-up, or another intervention.

This project therefore investigates GSM-family signal denoising as a practical engineering research problem. It does not attempt to replace licensed network infrastructure or operator planning. Instead, it asks whether a denoising-first signal processing pipeline can recover useful signal structure from noisy GSM-family conditions and improve automatic modulation classification under controlled edge-of-coverage conditions.

## 4. Problem Statement

Rural and underserved areas in Uganda continue to face access, affordability, and user-experience challenges in communication services, and UCC has placed these issues among its priority research areas for academic support (Uganda Communications Commission, 2026). In practical terms, one important part of this problem is weak mobile service: users may have a visible GSM signal but still experience failed calls, delayed SMS, or interrupted USSD and Mobile Money sessions when the received signal is too weak or noisy for reliable use.

The technical problem is made more difficult by interference and signal degradation. UCC has publicly warned against network repeaters or boosters that can create harmful interference, and it has enforced action against illegal or non-compliant broadcasters whose emissions may affect licensed services (Uganda Communications Commission, 2021, 2024). These local regulatory concerns show that weak-service conditions can involve a mixture of low received power, additive noise, interference, fading, and environmental attenuation.

Existing monitoring tools can identify weak-service areas, but they do not fully solve the signal interpretation problem. At the edge of coverage, engineers may know that a service problem exists without having enough low-cost evidence to distinguish whether the issue is mainly coverage weakness, interference, signal distortion, or a combination of these factors. This creates a decision gap for UCC, UCUSAF, operators, and researchers when planning rural connectivity interventions.

The research problem is therefore: how can a denoising-assisted GSM signal processing pipeline improve the recovery and interpretation of weak GSM-family signals under edge-of-coverage conditions relevant to rural Uganda?

The proposed solution is to design and test a denoising autoencoder front end connected to an automatic modulation classification stage. The system will be evaluated offline first and then translated into a low-cost Android-SDR field-support concept for future supervised validation. This gives the project a practical pathway beyond modelling while keeping the present claims evidence-based.

## 5. Aim and Objectives

### 5.1 Aim

The aim of this project is to design and evaluate a denoising-assisted GSM-family signal processing pipeline for improved interpretation of weak, noisy, edge-of-coverage signals in rural Uganda.

### 5.2 Specific Objectives

1. To characterize weak-signal and interference-heavy GSM service conditions relevant to rural Uganda using UCC materials, literature, and controlled signal impairment models.

2. To design a denoising autoencoder that reconstructs cleaner I/Q signal representations from noisy GSM-family samples.

3. To integrate the denoising autoencoder with an automatic modulation classification stage for selected GSM-family or GSM-adjacent modulation classes.

4. To compare the denoising-assisted pipeline with a baseline classifier operating directly on noisy signals.

5. To develop a low-cost Android-SDR field-support concept that can translate the lab workflow into a practical weak-service observation tool.

6. To interpret the results in terms of UCC value, rural service reliability, interference follow-up, coverage planning, and future pilot feasibility.

## 6. Research Questions

1. What weak-signal, noise, and interference conditions are most relevant to GSM-family service interpretation in rural Uganda?

2. Can a denoising autoencoder recover useful signal structure from noisy GSM-family I/Q samples under controlled low-SNR conditions?

3. Does a denoising-assisted modulation classification pipeline perform better than a no-denoising baseline under selected weak-signal conditions?

4. Which evaluation metrics best translate the technical result into practical meaning for UCC, operators, and rural service troubleshooting?

5. How can the denoising pipeline be packaged as a low-cost Android-SDR field-support concept without requiring rural users to own smartphones?

6. What ethical, regulatory, and technical limitations must be addressed before any field validation is attempted?

## 7. Justification and Direct Benefit to UCC

This study is justified because it addresses a practical communications problem at the intersection of infrastructure access, signal processing, artificial intelligence, low-cost prototyping, and rural service quality. UCC's Research Support to Academia call encourages research into new communications technologies and improved access to communication services for all Ugandans (Uganda Communications Commission, 2026). The proposed work fits this call because it investigates a technical method for interpreting weak GSM-family service conditions in areas where basic communication remains important.

For UCC, the direct benefit is a low-cost evidence layer for weak-service interpretation. If the denoising-assisted pipeline improves signal recoverability under controlled low-SNR conditions, the project can support better analysis of marginal service areas before decisions are made about new sites, optimization, interference enforcement, or further field measurements.

For UCUSAF and rural connectivity planning, the project can contribute a technical report and reproducible workflow that explain how software-assisted signal interpretation may support underserved-area planning. The work is not a replacement for infrastructure expansion. It is a complementary diagnostic method that may help prioritize where physical interventions are most needed.

For spectrum and compliance work, the project is relevant because UCC has identified interference from boosters, repeaters, and non-compliant transmitters as a public concern (Uganda Communications Commission, 2021, 2024). A denoising-assisted workflow could help separate weak-signal behaviour from interference-heavy behaviour in controlled experiments and later field-support tests.

For operators, the value is diagnostic. A weak-service area may require coverage optimization, antenna adjustment, interference follow-up, or new infrastructure. A denoising-assisted tool can contribute cleaner signal evidence to this decision process.

For the participating universities, the project builds research capacity in communications engineering, machine learning, RF data handling, ethical field measurement, and practical technology translation. It also gives student researchers a defined role inside a lecturer-led inter-university project, which is consistent with the Lot 3 ambition.

## 8. Literature Review

### 8.1 Rural Connectivity, Access, and User Experience

Connectivity statistics can hide the difference between nominal coverage and usable service. A location may appear covered while still producing failed calls, delayed SMS, or dropped USSD sessions because the received signal is too weak, too noisy, or affected by interference. UCC's Research Support to Academia call identifies access, affordability, and user experience as priority research areas, which makes weak basic-service reliability a relevant local problem (Uganda Communications Commission, 2026).

UCC's strategic framing also emphasizes evidence-based research that can contribute to policy, regulation, and innovation in the communications sector (Uganda Communications Commission, 2026). This supports research that does not merely build a model, but translates model behaviour into evidence that can help decision-makers understand weak-service conditions.

### 8.2 Interference and Signal Integrity

Signal degradation is not caused only by distance from a tower. UCC has issued public warnings about signal interference arising from network repeaters or boosters and has taken enforcement action against illegal or non-compliant broadcasters (Uganda Communications Commission, 2021, 2024). These examples show that signal integrity is a local technical and regulatory concern.

Propagation and attenuation also affect terrestrial radio links. ITU-R recommendations provide standard models for rain attenuation and terrestrial line-of-sight propagation planning (International Telecommunication Union, 2005, 2021). Although GSM user experience depends on many additional factors, these references support the use of controlled attenuation and signal-degradation models when building a laboratory test environment.

### 8.3 Automatic Modulation Classification

Automatic modulation classification identifies the modulation type of a received signal. It is useful in spectrum monitoring, cognitive radio, interference diagnosis, adaptive communication systems, and radio-environment awareness. Traditional feature-based approaches can become less reliable when the signal is weak or distorted. Recent work shows that deep learning can improve modulation classification by learning features directly from I/Q samples. Abd-Elaziz et al. (2023), for example, proposed a CNN-based architecture for automatic modulation classification in cognitive radio networks.

### 8.4 Denoising Autoencoders for Communication Signals

Denoising autoencoders are trained to reconstruct cleaner versions of corrupted inputs. In communication systems, they can serve as front ends that reduce noise before classification or other downstream signal processing. Zhang et al. (2023) proposed a residual denoising autoencoder with channel attention for modulation signals. An and Lee (2023) studied robust modulation classification in low-SNR conditions, while Gao et al. (2026) explored autoencoder-based approaches for improving noise robustness in few-shot modulation classification. These studies support the technical idea that denoising can improve signal interpretation under noisy conditions.

### 8.5 Gap in the Literature

The gap is not simply that denoising autoencoders need another benchmark. The gap is that existing denoising and automatic modulation classification studies are rarely framed around Uganda's rural GSM service problem, UCC's need for weak-service evidence, basic-phone service realities, or low-cost field-support workflows. This project fills that gap by connecting denoising-assisted signal interpretation to rural GSM service, SMS/USSD reliability concerns, interference follow-up, and regulator/operator usefulness.

## 9. Methodology

### 9.1 Research Design

The study will use an experimental research design. The core work will be conducted offline using public I/Q datasets, generated GSM-family test signals where necessary, and controlled signal impairments. This design is appropriate because the first task is to test whether denoising improves weak-signal recoverability before any field-support validation is attempted.

Two processing chains will be compared. The baseline chain will classify noisy I/Q samples directly. The proposed chain will first reconstruct the noisy I/Q sample using a denoising autoencoder and then classify the denoised output using the same classifier family. This keeps the comparison fair because the main difference is the denoising front end.

### 9.2 Data Sources and Signal Scope

The first experiment stage will use public datasets. RadioML 2018.01A will be used because it provides labelled modulation samples with structured SNR information (DeepSig, 2018). RF Signal Data will be considered as a secondary source because it provides SDR-style signal captures that may introduce more practical signal variation (RF Signal Data, 2025). Where a GSM-specific class is not sufficiently represented in a dataset, controlled GSM-family test signals will be generated for the limited experiment scope.

The initial signal scope will focus on selected GSM-family or GSM-adjacent modulation classes such as GMSK, GFSK, and QPSK where available. The scope is deliberately narrow so the study remains defensible within the project period.

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

### 9.6 Experimental Procedure

The experiment will follow these steps:

1. Select and document the dataset, modulation classes, SNR ranges, and preprocessing steps.

2. Segment and normalize I/Q samples into consistent input windows.

3. Generate controlled noisy versions of the samples using additive noise, narrowband interference, wideband disturbance, and attenuation models.

4. Train and test the baseline classifier on noisy samples.

5. Train the denoising autoencoder and inspect reconstruction behaviour.

6. Evaluate the classifier using denoised outputs.

7. Compare the baseline and denoising-assisted pipelines across selected SNR ranges.

8. Measure model size and inference latency for future prototype feasibility.

9. Prepare a low-cost Android-SDR field-support workflow for future supervised validation.

## 10. Data Collection Methods and Tools

The study will use three main data collection tools.

### 10.1 Experiment Log

The experiment log will record dataset name, version, source URL, selected modulation classes, SNR range, preprocessing settings, impairment parameters, model architecture, training settings, random seed, and date of each run. This tool is needed to make the experiment reproducible and to allow supervisors to verify how results were generated.

### 10.2 Results Extraction Sheet

The results extraction sheet will capture reconstruction error, estimated SNR-related measures, classification accuracy, macro F1-score, confusion matrix values, low-SNR performance trends, model size, inference latency, and notes on failure cases. This tool separates raw model output from interpreted findings.

### 10.3 Field Observation Checklist

The field observation checklist will only be used if supervisors approve a limited field-support validation. It will record site type, date, weather, approximate location category, basic-phone service observations, Android-SDR kit setup, antenna placement, visible interference indicators, and supervisor approval notes. It will not record, decode, or store private user communication content.

Copies of the three tools are included in the appendices.

## 11. Data Analysis Plan

Data analysis will be conducted in four stages.

First, denoising quality will be analysed using reconstruction error and SNR-related measures. Lower reconstruction error and improved SNR-related indicators will suggest that the denoising model is recovering useful signal structure.

Second, classification quality will be analysed using accuracy, macro F1-score, and confusion matrices. Macro F1-score will be used because it is more informative than accuracy when classes are not equally easy to classify.

Equation (9): Precision = TP / (TP + FP)

Equation (10): Recall = TP / (TP + FN)

Equation (11): F1 = 2(Precision x Recall) / (Precision + Recall)

Equation (12): Macro F1 = average of F1 across all classes

Third, low-SNR behaviour will be examined separately. The study will compare whether the denoising-assisted pipeline improves performance at weaker SNR levels rather than only improving an overall average.

Fourth, practical feasibility will be analysed through model size, inference latency, and prototype suitability. A model that improves accuracy but is too large or slow for low-cost equipment will be treated as a weaker practical result.

The interpretation will remain evidence-based. Positive offline results will be described as improved weak-signal recoverability under controlled conditions, not as proof of physical coverage extension or live service improvement.

## 12. Ethical and Regulatory Considerations

The core study uses public datasets, generated signals, and controlled impairments. It will not record, decode, or store private subscriber communications. The project will avoid unauthorized transmission and will not interfere with licensed networks.

If field-support validation is approved, the research protocol will be submitted for the required institutional ethical review and regulatory approvals before field activity begins. Where human participants are involved in service-use observations, informed consent will be obtained. Participants will be informed of the study purpose, the voluntary nature of participation, and the fact that private communication content will not be collected.

Any future RF field measurement will be limited to lawful signal observation and engineering validation under supervisor guidance. The Android-SDR kit will be treated as a receive-only research measurement tool unless separate authorization is obtained. The project will also avoid unsafe claims: the denoising model will not be presented as a substitute for licensed operators, infrastructure planning, spectrum regulation, or network optimization.

## 13. Implementation Plan and Gantt-Style Work Schedule

The project will run from May 2026 to November 2026 and will be completed before 30 November 2026, in line with the UCC call requirement for supported research completion (Uganda Communications Commission, 2026).

| Phase | Period | Main activities | Deliverables |
| --- | --- | --- | --- |
| 1 | May 2026 | Project initiation, PI/Co-PI confirmation, dataset selection, instrument refinement, ethical review planning, and experiment protocol approval | Approved scope note, finalized data collection tools, literature matrix, kickoff minutes |
| 2 | June 2026 | Data preparation, I/Q segmentation, normalization, AWGN generation, narrowband interference modelling, wideband disturbance modelling, and attenuation scenario design | Cleaned data workflow, impairment-generation scripts, documented dataset notes |
| 3 | July 2026 | Denoising autoencoder design, model training, reconstruction checks, and parameter refinement | Denoising prototype, reconstruction plots, training log |
| 4 | August 2026 | Baseline classifier training, denoising-assisted pipeline integration, and baseline-versus-proposed comparison | Baseline classifier, denoising-assisted classifier, preliminary result tables, confusion matrices |
| 5 | September 2026 | Model-size and latency assessment, Android-SDR field-support workflow, and supervisor-approved weak-service observation preparation | Prototype workflow, latency notes, field observation checklist, supervisor review notes |
| 6 | October-November 2026 | Final analysis, stakeholder validation, UCC technical brief preparation, final reporting, and dissemination packaging | Final report, UCC technical brief, presentation slides, code/results archive, conference-style manuscript |

## 14. Contribution to Cross-Cutting Issues

### 14.1 Gender and Inclusion

The project focuses on basic mobile services used by many groups, including women, youth, small traders, farmers, community health workers, and households that rely on basic handsets. Any future field observation will aim to include both male and female service-use perspectives where human feedback is collected.

### 14.2 Disability and Marginalised Groups

Reliable voice, SMS, and USSD services matter for users who may not have expensive smartphones, continuous internet, or advanced digital platforms. By focusing on weak basic-service conditions, the project supports a more inclusive understanding of connectivity.

### 14.3 Climate Change and Environmental Responsibility

The project does not claim that software can replace infrastructure. However, better weak-signal evidence can support more efficient planning and reduce poorly targeted field visits or interventions. The research will use low-cost, low-power measurement equipment where practical and will prioritize reusable scripts and reproducible datasets.

### 14.4 Responsible Innovation

The project applies AI to a communications problem in a cautious way. It separates measured results from expected outcomes, avoids inflated coverage claims, and includes ethical controls for RF measurement. This supports trustworthy technology development in a sector where privacy, safety, and regulatory compliance matter.

## 15. Expected Outputs and Deliverables

The expected outputs are:

1. A documented weak-signal GSM-family preprocessing workflow.

2. Controlled signal impairment scripts covering additive noise, narrowband interference, wideband disturbance, and attenuation scenarios.

3. A denoising autoencoder prototype for I/Q signal reconstruction.

4. A baseline automatic modulation classifier.

5. A denoising-assisted modulation classification pipeline.

6. Comparative result tables, SNR-wise plots, reconstruction-error plots, and confusion matrices.

7. A low-cost Android-SDR field-support concept for future lawful weak-service observation.

8. A UCC technical brief explaining how denoising-assisted signal interpretation can support weak-service diagnosis, rural coverage planning, and interference follow-up.

9. A final project report, presentation materials, and a conference-style manuscript.

## 16. Sustainability and Prototype Pathway

The project is designed to move beyond modelling without pretending that deployment has already happened. The practical prototype pathway will use an Android phone, a low-cost SDR receiver, an antenna, and the trained denoising/classification workflow as a field-support kit. This kit is not meant to replace a user's basic phone. Basic phones remain central to the service problem. The Android-SDR kit is an engineering and research tool for observing weak-service conditions and demonstrating whether denoising-assisted interpretation is useful outside the lab.

The long-term pathway has three stages. The first is offline validation using datasets and controlled impairments. The second is supervisor-approved field-support observation at selected weak-service locations, without collecting private communication content. The third is a small pilot with service scenarios such as SMS, USSD, and call setup observations, if the laboratory evidence justifies it.

Sustainability will depend on whether the model is lightweight, reproducible, and useful to real decisions. The project will therefore measure model size and inference latency in addition to accuracy. If the model is too large or slow, the project will recommend further optimization before any field pilot.

## 17. Budget and Budget Justification

The project requests UGX 66,600,000 under Lot 3 Inter-University Research Collaboration, Interdisciplinary. The request stays below the UGX 100,000,000 maximum funding range for interdisciplinary projects described in the UCC call. The budget is in UGX only and avoids tuition, salaries, stipends, remuneration, and infrastructure construction. It focuses on research expenses needed for approvals, experimentation, low-cost prototype measurement, field-support observation, documentation, validation, and dissemination.

| Item | Amount (UGX) | Justification |
| --- | ---: | --- |
| Preliminary approvals and research coordination | 4,800,000 | Supports proposal finalization, protocol preparation, institutional coordination, and required administrative preparation for ethical and regulatory compliance. |
| Low-cost SDR receiver kits and RF accessories | 12,000,000 | SDR receivers, antennas, OTG adapters, RF cables, connectors, power accessories, and replacement parts for controlled signal measurement and prototype testing. |
| Android measurement devices and protective accessories | 8,400,000 | Affordable Android phones and protective accessories for engineering test kits. These are research measurement tools, not a requirement for rural users. |
| Lab consumables, adapters, shielding, and calibration aids | 5,500,000 | Materials needed to keep bench tests repeatable, reduce uncontrolled noise, and connect SDR and antenna components safely. |
| Field transport and site logistics | 12,500,000 | Supervised movement to selected weak-service observation sites, local transport, and basic field logistics for lawful service observation and measurement setup. |
| Data, storage, backup, compute, and connectivity support | 6,500,000 | Internet bundles, external storage, backups, and limited compute support for model training, reproducible experiment storage, and team coordination. |
| Prototype assembly, enclosure fabrication, and testing services | 5,800,000 | Practical assembly, safe packaging, and bench testing of research kits. This is a procurement/service item and not a salary or stipend line. |
| Documentation, printing, and dissemination materials | 3,900,000 | Field forms, consent/approval documents where applicable, report drafts, diagrams, posters, and dissemination materials for UCC and university review. |
| Stakeholder validation and UCC dissemination logistics | 3,200,000 | Modest logistics for technical review sessions and stakeholder validation of results so the final interpretation is useful to UCC and sector actors. |
| Data collection tools, safety, and compliance materials | 1,500,000 | Printed observation tools, site forms, safety items, and responsible fieldwork documentation. |
| Final report production and repository preparation | 1,500,000 | Final report layout, binding, electronic packaging, code/results archive, and preparation of materials for UCC or university repository submission. |
| Contingency and price variation | 1,000,000 | Small provision for unavoidable price changes in equipment, printing, transport, or compliance materials during the project period. |
| **Total** | **66,600,000** |  |

The largest cost areas are prototype measurement equipment and field logistics because the proposal must produce more than an offline ML model. The Android-SDR items are included as an engineering test platform. The ordinary rural user remains represented through basic-phone service scenarios such as SMS, USSD, Mobile Money, and call setup observations.

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
**Project responsibilities:** Support literature review, dataset preparation, denoising model development, experiment logging, result interpretation, documentation, and final reporting.

### 18.4 Kisige Tom Derrick

**Proposed role:** Student researcher / research assistant  
**Institution:** Kyambogo University  
**Department:** Department of Electrical and Electronics Engineering  
**Current level:** Undergraduate engineering student  
**Research focus:** Telecommunications systems, signal quality analysis, low-cost RF prototyping, and practical rural connectivity support.  
**Relevant skills:** Communication networks, electronics prototyping, RF equipment setup, dataset organization, model evaluation support, documentation, and field observation planning.  
**Project responsibilities:** Support signal impairment modelling, baseline classifier testing, prototype setup, field-support planning, result verification, and dissemination preparation.

## 19. References

Abd-Elaziz, O. F., El-Ghandour, A. M., & Ismail, F. H. (2023). Deep learning-based automatic modulation classification using robust CNN architecture for cognitive radio networks. *Sensors, 23*(23), Article 9467. https://doi.org/10.3390/s23239467

An, H., & Lee, B.-M. (2023). Robust automatic modulation classification in low signal-to-noise ratio. *IEEE Access, 11*, 125678-125690. https://doi.org/10.1109/ACCESS.2023.3321108

DeepSig. (2018). *RadioML 2018.01A dataset*. https://www.deepsig.ai/datasets/

Gao, X., Xu, X., Li, D., Liu, X., Yang, J., & Zhai, D. (2026). Enhancing noise robustness in few-shot automatic modulation classification via complex-valued autoencoders. *Electronics, 15*(3), Article 674. https://www.mdpi.com/2079-9292/15/3/674

International Telecommunication Union. (2005). *Recommendation ITU-R P.838-3: Specific attenuation model for rain for use in prediction methods*. https://www.itu.int/rec/R-REC-P.838-3-200503-P/en

International Telecommunication Union. (2021). *Recommendation ITU-R P.530-18: Propagation data and prediction methods required for the design of terrestrial line-of-sight systems*. https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.530-18-202109-I!!PDF-E.pdf

RF Signal Data. (2025). *RF signal data* [Data set]. Kaggle. https://www.kaggle.com/datasets/suraj520/rf-signal-data

Uganda Communications Commission. (2021, July 26). *Public notice: Signal interference arising out of usage of network repeaters - "boosters"*. https://uccinfoblog.com/2021/07/26/public-notice-signal-interference-arising-out-of-usage-of-network-repeaters-boosters/

Uganda Communications Commission. (2024, September 25). *UCC cracks down on illegal and non-compliant broadcasters*. https://www.ucc.co.ug/ucc-cracks-down-on-illegal-and-non-compliant-broadcasters/

Uganda Communications Commission. (2026). *Call for proposals for the Uganda Communications Commission (UCC) research support to academia*. https://www.ucc.co.ug/wp-content/uploads/2026/04/Call-for-Research-Proposals-April-2026.pdf

Zhang, X., et al. (2023). Dual residual denoising autoencoder with channel attention mechanism for modulation of signals. *Sensors, 23*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9861137/

## 20. Appendices: Data Collection Tools

### Appendix A: Experiment Log Template

Fields to be completed for every experiment run: experiment ID; date; researcher; dataset name; dataset version or URL; selected modulation classes; sample length; sampling rate where available; preprocessing method; normalization method; SNR range; impairment model; impairment parameters; model architecture; training epochs; optimizer; loss function; random seed; hardware used; output file path; supervisor review notes.

### Appendix B: Results Extraction Sheet Template

Fields to be completed after each evaluation run: experiment ID; dataset; signal class; SNR level; baseline accuracy; denoising-assisted accuracy; baseline macro F1; denoising-assisted macro F1; reconstruction error; estimated SNR-related improvement; confusion matrix file path; model size; inference latency; observed failure cases; interpretation notes.

### Appendix C: Field Observation Checklist

Fields to be completed only after approval for field-support validation: site code; district or location category; date and time; weather; terrain description; nearby visible tower or obstruction notes; handset type used for service observation; basic service observed, such as SMS, USSD, or call setup; Android-SDR kit ID; antenna position; receive-only confirmation; no private content collected; supervisor approval; safety notes; field remarks.
