# Signal Denoising for Improved Edge-of-Coverage GSM Service in Rural Uganda

## 1. Title Page

**Research study title:** Signal Denoising for Improved Edge-of-Coverage GSM Service in Rural Uganda

**Funding category:** Lot 3 - Inter-University Research Collaboration, Interdisciplinary

**Priority research areas:** Access, Affordability, and User Experience of Communication Services; Digital Innovation, Entrepreneurship, and Emerging Technologies; Next-Generation Telecommunications Infrastructure and Spectrum Management

**Lead student investigators:**  
Ssemujju Sharif Abdukarim, Principal Student Investigator  
Kisige Tom Derrick, Co-Student Investigator

**Department:** Department of Electrical and Electronics Engineering  
**Faculty/School:** Faculty of Engineering  
**Lead institution:** Kyambogo University, Kampala, Uganda  
**Collaborating institution:** Makerere University, College of Computing and Information Sciences, Kampala, Uganda

**Supervisory and collaboration leads:**  
Dr. Dickson Mugerwa  
Dr. Ephrance Eunice Namugenyi

**Proposed duration:** Six months, with completion before 30 November 2026

**Total budget requested:** UGX 66,600,000

**Addressed to:**  
The Executive Director  
Uganda Communications Commission  
UCC House, Plot 42-44 Spring Road, Bugolobi  
P.O. Box 7376, Kampala, Uganda

## 2. Project Summary

Rural mobile service in Uganda is often discussed as a question of whether a location is covered or not covered. In practice, many users experience a more frustrating middle ground: a phone shows signal, but calls fail, SMS messages delay, and USSD or Mobile Money sessions drop before completion. This project investigates a software-side way of improving the interpretation of such weak and noisy GSM-family signals.

The study proposes a denoising-assisted signal processing pipeline that combines a denoising autoencoder with automatic modulation classification. A noisy I/Q signal sample is first reconstructed by the denoising model, then passed to a classifier so that the modulation decision is made from a cleaner signal representation. The research will compare this proposed pipeline with a baseline classifier that operates directly on noisy signals.

The work is student-led but positioned as an inter-university interdisciplinary collaboration between telecommunications engineering and computing. It draws on RF signal processing, machine learning, low-cost prototyping, and rural service-quality concerns. The first stage will use public I/Q datasets and controlled low-SNR impairments to test whether denoising improves weak-signal recoverability. Metrics will include reconstruction error, SNR-related measures, classification accuracy, macro F1-score, confusion matrices, model size, and inference latency.

The project does not claim that software has already extended physical tower coverage. Instead, it asks a defensible research question: can denoising recover more useful signal structure from edge-of-coverage GSM-family conditions? If successful, the results can support better troubleshooting, rural coverage planning, interference investigation, and future low-cost field-support validation for Uganda's communications sector.

## 3. Introduction and Background

Uganda's communications sector continues to expand, but the practical quality of access remains uneven for many users in rural and underserved locations. The Uganda Communications Commission (UCC) Research Support to Academia call emphasizes research that improves access to communication services, supports underserved communities, and contributes to evidence-based policy and regulatory development in the communications sector (Uganda Communications Commission, 2026). This project responds directly to that call by focusing on weak GSM-family service at the edge of coverage.

For many rural users, the most important mobile services are still basic and immediate: voice calls, SMS, USSD, and Mobile Money. These services may depend on ordinary basic phones as much as smartphones. A weak-service area therefore cannot be judged only by the existence of a signal bar or a nominal coverage map. A location may be technically covered but still unreliable when the received signal is weak, noisy, or affected by interference.

UCC has also treated signal integrity as a live regulatory concern. It has warned the public about interference arising from the use of network repeaters or boosters, and it has taken action against illegal and non-compliant broadcasters whose transmissions may interfere with licensed services and public safety communications (Uganda Communications Commission, 2021, 2024). These public actions show that weak service is not caused only by distance from a tower. It may also be shaped by interference, poorly controlled transmission, propagation effects, and local equipment conditions.

Conventional responses to weak rural service include new towers, network optimization, backhaul improvements, better power systems, and coverage monitoring. These remain essential, but they can be costly and slow to implement. A complementary software-side method could help engineers and regulators interpret weak-service areas more clearly before deciding whether a site requires physical expansion, network optimization, interference enforcement, or another intervention.

This project therefore investigates GSM signal denoising as a practical research problem. It does not attempt to replace licensed network infrastructure or operator planning. Instead, it asks whether a denoising-first signal processing pipeline can recover useful structure from noisy GSM-family signals and improve automatic modulation classification under controlled edge-of-coverage conditions. The result is expected to be useful both as an undergraduate engineering research contribution and as an early step toward low-cost field-support tools for rural connectivity planning.

## 4. Problem Statement

Rural and underserved areas in Uganda continue to experience weak and noisy mobile service conditions that reduce the usability of basic communication services such as voice, SMS, USSD, and Mobile Money. These services are especially important because they support communication, payments, local coordination, health alerts, and access to information for users who may rely on basic 2G-capable handsets.

Existing monitoring methods can identify weak-service areas, but they do not fully address the signal recovery problem. At the edge of coverage, GSM-family signals may be degraded by low received power, interference, fading, and environmental attenuation. When the signal is too noisy, both user service and engineering interpretation suffer: users experience failed sessions, while engineers obtain less reliable evidence about what is happening in the radio environment.

The research problem is therefore: how can a denoising-assisted signal processing pipeline improve the recovery and interpretation of weak GSM-family signals under edge-of-coverage conditions relevant to rural Uganda?

## 5. Research Objectives

### 5.1 Main Objective

To design and evaluate a denoising-assisted GSM signal processing pipeline for improved interpretation of weak, noisy, edge-of-coverage signals in rural Uganda.

### 5.2 Specific Objectives

1. To characterize weak-signal and interference-heavy GSM service conditions relevant to rural Uganda using public sources, literature, and controlled signal impairment models.

2. To design a denoising autoencoder that reconstructs cleaner I/Q signal representations from noisy GSM-family samples.

3. To integrate the denoising autoencoder with an automatic modulation classification stage for selected GSM-family or GSM-adjacent modulation classes.

4. To compare the denoising-assisted pipeline against a baseline classifier operating directly on noisy signals.

5. To interpret the results in terms of rural service reliability, weak-signal troubleshooting, regulator/operator usefulness, and future pilot feasibility.

## 6. Research Questions

1. What weak-signal and interference conditions are most relevant to GSM-family service interpretation in rural Uganda?

2. Can a denoising autoencoder recover useful signal structure from noisy GSM-family I/Q samples under controlled low-SNR conditions?

3. Does a denoising-assisted modulation classification pipeline perform better than a no-denoising baseline under selected weak-signal conditions?

4. Which evaluation metrics best translate the technical result into practical meaning for rural service troubleshooting and coverage planning?

5. What technical, ethical, and regulatory limitations must be addressed before the approach can be tested as a field-support tool?

## 7. Justification of the Study

This study is justified because it addresses a practical communications problem at the intersection of infrastructure access, signal processing, artificial intelligence, and rural service quality. UCC's call specifically encourages focused research within priority areas such as access and affordability, digital innovation, emerging technologies, telecommunications infrastructure, and spectrum management (Uganda Communications Commission, 2026). It also provides a strong fit for inter-university collaboration because the work combines communications engineering from Kyambogo University with computing, AI, and applied digital systems expertise from Makerere University CoCIS.

For rural users, the relevance is direct. Weak GSM-family service can affect ordinary communication tasks such as calls, SMS, USSD, and Mobile Money. A research tool that improves interpretation of weak and noisy signal conditions may help expose why service is failing and what kind of intervention is needed.

For operators, the value is diagnostic. A denoising-assisted pipeline could support better distinction between low-signal coverage problems, interference-heavy environments, and conditions that require further field investigation. This does not replace operator measurements, but it can contribute to a lower-cost evidence layer.

For UCC and UCUSAF, the value is policy and regulatory usefulness. The work can contribute evidence for underserved-area planning, interference investigation, service-quality interpretation, and future software-assisted monitoring approaches. It also fits UCC's wider objective of strengthening collaboration between academia, industry, and policymakers through evidence-based research (Uganda Communications Commission, 2026).

For Kyambogo University, Makerere University, and the student investigators, the study builds capacity in communications engineering, machine learning, experimental design, RF data handling, and responsible applied research. It is not a generic machine learning exercise. It is a locally grounded engineering project with a clear Ugandan communications problem.

## 8. Methodology

### 8.1 Research Design

The study will use an experimental research design. The core work will be conducted offline using public I/Q datasets and controlled signal impairments. This design is appropriate because the first task is to test whether denoising improves weak-signal recoverability under controlled conditions before any field-support validation is attempted.

Two processing chains will be compared. The first will be a baseline automatic modulation classifier trained and tested on noisy I/Q samples. The second will be a denoising-assisted pipeline in which a denoising autoencoder reconstructs the signal before the same classification task is performed.

### 8.2 Data Sources

The first experiment stage will use public datasets. RadioML 2018.01A will be used because it provides labelled modulation samples with structured SNR information (DeepSig, 2018). RF Signal Data will be considered as a secondary dataset because it provides SDR-style signal captures that may introduce more practical signal variation (RF Signal Data, 2025).

No private subscriber communication content will be collected during the core study. Any future live RF validation will require supervisor approval, lawful procedures, and strict avoidance of decoding or storing private message or call content.

### 8.3 Signal Scope

The initial signal scope will focus on selected GSM-family or GSM-adjacent modulation classes, including GMSK, GFSK, and QPSK where available in the chosen datasets or generated test signals. The scope is intentionally narrow so that the study remains defensible and manageable within the undergraduate project period.

### 8.4 Signal Impairment Modelling

Controlled weak-signal conditions will be created using additive noise and selected interference models. The planned scenarios include low-SNR reception, wideband disturbance inspired by illegal repeater or booster interference, narrowband harmonic-style disturbance inspired by poorly filtered transmitters, and time-varying attenuation informed by standard propagation recommendations (International Telecommunication Union, 2005, 2021; Uganda Communications Commission, 2021, 2024).

These impairments will be treated as controlled approximations, not as field measurements. Their purpose is to test whether denoising can preserve useful GSM-family signal structure when the input is degraded in ways that resemble difficult rural radio conditions.

### 8.5 Model Design and Experiment Procedure

The denoising model will be a one-dimensional autoencoder designed for I/Q signal windows. The encoder will compress noisy input into a latent representation, and the decoder will reconstruct a cleaner signal estimate. Training will use paired clean and corrupted signal windows where the clean reference is available from the dataset or generated simulation process.

The classifier will operate either on raw noisy inputs or on denoised outputs. The same classifier family will be used for both the baseline and the proposed chain so that the comparison remains fair. A compact convolutional neural network or a similar lightweight sequence classifier will be used, depending on dataset structure and available computing resources.

The experiment will follow these steps:

1. Select and document the dataset, modulation classes, SNR ranges, and preprocessing steps.

2. Segment and normalize I/Q samples into consistent input windows.

3. Create controlled noisy versions of the samples using selected impairment models.

4. Train the baseline classifier on noisy samples and record performance.

5. Train the denoising autoencoder and inspect reconstruction behaviour.

6. Train or evaluate the classifier using denoised outputs.

7. Compare the baseline and denoising-assisted pipelines across SNR ranges.

8. Interpret the results in terms of weak-signal recoverability and rural service relevance.

### 8.6 Evaluation Metrics

The evaluation will include reconstruction error, SNR-related measures, classification accuracy, macro F1-score, confusion matrices, and performance trends across selected low-SNR ranges. Model size and inference latency will also be measured to judge whether the approach could later be considered for low-cost field-support equipment.

The key comparison will be whether the denoising-assisted pipeline performs better than the no-denoising baseline under weak-signal conditions. The study will not claim field-level service improvement unless field validation is separately completed.

### 8.7 Data Collection and Analysis Tools

The study will use three main data collection tools. The first is an experiment log recording dataset version, preprocessing settings, model parameters, training conditions, and evaluation results. The second is a results extraction sheet for SNR-wise metrics, reconstruction error, confusion matrices, model size, and latency. The third is a future field-observation checklist for lawful weak-signal site visits, covering location type, handset service observations, signal measurement setup, weather, nearby interference indicators, and supervisor approvals.

The collected results will be analysed using comparative tables, plots of performance against SNR, confusion matrices, and a short practical interpretation of what the findings mean for weak GSM-family service troubleshooting.

### 8.8 Ethical, Safety, and Regulatory Controls

The core study uses public datasets and controlled impairments. It will not record, decode, or store private user communications. If later field-support validation is approved, the work will be limited to lawful signal measurement and service observation. The team will avoid content interception, unauthorized transmission, or any activity that interferes with licensed networks.

## 9. Implementation Plan

The project will run for six months and will be completed before 30 November 2026, in line with the UCC call requirement for supported research completion (Uganda Communications Commission, 2026).

| Month | Activities | Student and collaboration focus | Deliverables |
| --- | --- | --- | --- |
| Month 1 | Confirm scope, datasets, literature, experiment protocol, collaboration roles, and ethics controls | Literature review, dataset review, experiment design, supervisor alignment | Final scope note, dataset notes, literature matrix |
| Month 2 | Prepare preprocessing pipeline, segmentation, normalization, and impairment generation | Data preparation, reproducibility, computing review | Clean experiment scripts, prepared signal windows |
| Month 3 | Build and train the denoising autoencoder | Model implementation, reconstruction checks, AI review | Denoising prototype, reconstruction plots |
| Month 4 | Build baseline classifier and integrate denoising-assisted classification pipeline | Baseline comparison, pipeline integration, prototype feasibility | Baseline model, proposed pipeline |
| Month 5 | Run evaluation across SNR ranges, analyse errors, and conduct limited weak-service observation if approved | Metrics, confusion matrices, result interpretation, field-support planning | Result tables, SNR curves, error analysis |
| Month 6 | Write final report, prepare dissemination materials, validate findings with supervisors and sector stakeholders | Report writing, presentation, policy/practical interpretation | Final report, presentation, UCC-ready outputs |

Expected project outputs include a documented preprocessing workflow, trained denoising autoencoder prototype, baseline classifier, denoising-assisted classification pipeline, comparative result tables, final project report, conference-style paper or presentation, and policy-relevant notes on weak-service interpretation.

## 10. Contribution to Cross-Cutting Issues

### 10.1 Gender and Inclusion

The project focuses on basic mobile services used by many groups, including women, youth, small traders, farmers, and community health workers. Any future field observation will aim to include both male and female users where service-use experiences are collected. The project will avoid assuming that smartphone ownership is universal; basic phones remain central to the user reality.

### 10.2 Disability and Marginalised Groups

Reliable SMS, voice, and USSD services are important for users who may not have access to expensive devices, continuous internet, or advanced digital platforms. By focusing on weak basic-service conditions, the project supports a more inclusive understanding of connectivity. Field tools and reports will be written in clear language so that findings can be understood beyond a narrow technical audience.

### 10.3 Climate Change and Environmental Responsibility

The project does not claim that software can replace network infrastructure. However, better weak-signal evidence can help avoid poorly targeted interventions and support more efficient planning. The research will use low-cost, low-power measurement equipment where practical and will prioritize reusable scripts, reproducible datasets, and careful field-trip planning to reduce unnecessary movement.

### 10.4 Responsible Innovation

The project applies AI to a communications problem in a cautious way. It separates measured results from expected outcomes, avoids inflated coverage claims, and includes ethical controls for RF measurement. This supports trustworthy technology development in a sector where privacy, safety, and regulatory compliance matter.

## 11. Literature Review (APA 7th Edition)

### 11.1 Rural Connectivity, Access, and Service Quality

Connectivity statistics can hide the difference between nominal coverage and usable service. A place may appear covered but still produce failed calls, delayed messages, or dropped USSD sessions because the received signal is too weak or too noisy for reliable use. UCC's Research Support to Academia call emphasizes access to communication services for all members of society, including underserved communities, and places access, affordability, and user experience among its priority research areas (Uganda Communications Commission, 2026). This strengthens the relevance of studying weak GSM-family service as a practical quality-of-service problem.

UCC's strategic framing also highlights the need for evidence-based research that can support policy and regulatory development in the communications sector (Uganda Communications Commission, 2026). A denoising-assisted weak-signal interpretation method therefore has value beyond model accuracy. It can help translate signal processing results into evidence that is meaningful for rural service planning.

### 11.2 Interference and Signal Integrity

Signal degradation is not caused only by distance from a tower. UCC has issued public warnings about signal interference arising from network repeaters or boosters and has taken enforcement action against illegal or non-compliant broadcasters (Uganda Communications Commission, 2021, 2024). These examples show that signal integrity is a local regulatory and technical concern.

Propagation and attenuation also affect terrestrial radio links. ITU-R recommendations provide standard models for rain attenuation and terrestrial line-of-sight propagation planning (International Telecommunication Union, 2005, 2021). While GSM service at user level involves many more factors than rain attenuation alone, these references support the use of controlled attenuation and degradation models in the experiment.

### 11.3 Automatic Modulation Classification and Denoising

Automatic modulation classification identifies the modulation type of a received signal and is useful in spectrum monitoring, cognitive radio, interference diagnosis, and adaptive communication systems. Traditional feature-based methods can become less reliable when signal quality is poor. Recent work shows that deep learning can improve modulation classification by learning features directly from I/Q samples. Abd-Elaziz et al. (2023), for example, proposed a CNN-based architecture for automatic modulation classification in cognitive radio networks.

Denoising autoencoders are relevant because they can reconstruct cleaner versions of corrupted inputs before classification. Zhang et al. (2023) proposed a residual denoising autoencoder with attention for modulation signals, while An and Lee (2023) studied robust modulation classification in low-SNR conditions. Gao et al. (2026) further explored autoencoder-based approaches for improving noise robustness in few-shot modulation classification. These studies support the technical idea that denoising can improve downstream signal interpretation under noisy conditions.

The gap is that most existing work is not framed around Uganda's rural GSM service problem. This project adapts the denoising-first idea to a local engineering question: whether weak GSM-family signal recovery can support better interpretation of rural edge-of-coverage service conditions.

### References

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

## 12. Budget and Budget Justification

The project requests UGX 66,600,000 under Lot 3 Inter-University Research Collaboration, Interdisciplinary. The request stays below the UGX 100,000,000 ceiling for interdisciplinary Lot 3 projects. The budget avoids tuition, salaries, stipends, remuneration, and infrastructure costs. It focuses on reasonable research expenses needed for controlled experimentation, low-cost prototype measurement, documentation, field observation, collaboration, and dissemination.

| Item | Amount (UGX) | Justification |
| --- | ---: | --- |
| Low-cost SDR receiver kits and RF accessories | 12,800,000 | SDR receivers, antennas, OTG adapters, RF cables, connectors, power accessories, and replacement parts for controlled signal measurement and prototype testing. |
| Android measurement devices and protective accessories | 8,400,000 | Affordable Android phones for engineering test kits, not as a requirement for rural users. These support model-size, latency, and future field-support feasibility checks. |
| Lab consumables, adapters, shielding, and calibration aids | 5,500,000 | Materials needed to keep bench tests repeatable, reduce uncontrolled noise, and connect SDR and antenna components safely. |
| Field transport and site logistics | 14,200,000 | Supervised movement to selected weak-signal observation sites, local transport, and basic field logistics for lawful service observation and measurement setup. |
| Data, storage, backup, and compute support | 6,400,000 | Internet bundles, external storage, backups, and limited cloud or local compute support for model training and reproducible experiment storage. |
| Prototype assembly, enclosure fabrication, and testing services | 5,800,000 | Practical assembly, safe packaging, and bench testing of research kits. This is a service/procurement item and not a salary, stipend, or remuneration line. |
| Documentation, printing, and dissemination materials | 4,700,000 | Field forms, report drafts, diagrams, posters, presentation materials, and documentation needed for UCC and university dissemination. |
| Inter-university coordination and stakeholder validation logistics | 3,200,000 | Modest logistics for supervisor meetings, technical review sessions, and stakeholder validation of result interpretation. |
| Data collection tools, safety, and compliance materials | 2,100,000 | Printed observation tools, consent or approval materials where applicable, safety items, and responsible fieldwork documentation. |
| Final report production and repository preparation | 3,500,000 | Final report layout, binding, electronic packaging, dissemination presentation, and preparation of code/results for submission to the university and UCC repository if required. |
| **Total** | **66,600,000** |  |

The largest budget areas are measurement equipment and field logistics because the project must move beyond a purely theoretical model while still remaining manageable for a student-led inter-university collaboration. The Android-SDR items are included as engineering test equipment, not as a claim that rural users must buy smartphones. Basic phones remain central to the service problem.

The budget is intentionally below the UGX 100,000,000 ceiling for Lot 3 Interdisciplinary support. It is designed to be credible for a six-month applied research collaboration and to produce research outputs that can be reviewed by supervisors, UCC, and future technical partners.

## 13. Curriculum Vitae of the Research Team

### 13.1 Ssemujju Sharif Abdukarim

**Proposed role:** Principal Student Investigator  
**Institution:** Kyambogo University  
**Department:** Department of Electrical and Electronics Engineering  
**Current level:** Undergraduate engineering student  
**Research focus:** Telecommunications engineering, weak-signal GSM service, RF signal processing, and machine learning-assisted signal interpretation.

**Relevant skills:**  
Digital communication systems; signal processing fundamentals; Python/MATLAB-style technical computing; RF measurement concepts; machine learning model experimentation; technical writing; research documentation.

**Project responsibilities:**  
Lead problem framing, literature review, dataset preparation, denoising model development, experiment logging, result interpretation, and final report preparation.

**Expected project outputs:**  
Documented preprocessing workflow, denoising autoencoder prototype, comparative result tables, written report, presentation materials, and submission-ready dissemination outputs.

### 13.2 Kisige Tom Derrick

**Proposed role:** Co-Student Investigator  
**Institution:** Kyambogo University  
**Department:** Department of Electrical and Electronics Engineering  
**Current level:** Undergraduate engineering student  
**Research focus:** Telecommunications systems, signal quality analysis, low-cost RF prototyping, and practical rural connectivity support.

**Relevant skills:**  
Communication networks; electronics prototyping; RF equipment setup; dataset organization; model evaluation support; documentation; field observation planning.

**Project responsibilities:**  
Support dataset review, signal impairment modelling, baseline classifier testing, field-support planning, budget documentation, result verification, and supervisor presentation preparation.

**Expected project outputs:**  
Baseline classifier results, field-support observation tools, implementation records, prototype measurement notes, final report contributions, and dissemination support.

### 13.3 Dr. Dickson Mugerwa

**Proposed role:** Academic supervisor and Kyambogo University technical lead  
**Institution:** Kyambogo University  
**Department/Unit:** Faculty of Engineering / Department of Electrical and Electronics Engineering  
**Relevant public role:** Faculty innovation and engineering education support, including public university innovation and practical-skills activities.

**Supervision contribution:**  
Provide academic supervision, research quality control, engineering feasibility review, access to departmental guidance, and oversight of responsible RF experimentation.

**Project responsibilities:**  
Guide research scope, approve methodology, review experiment outputs, support compliance with university requirements, and supervise final report preparation.

### 13.4 Dr. Ephrance Eunice Namugenyi

**Proposed role:** Supervisory and inter-university collaboration lead  
**Institutional link:** Makerere University College of Computing and Information Sciences / Kyambogo University applied ICT ecosystem  
**Relevant focus:** Communication networks, software engineering, applied digital systems, innovation, and practical ICT deployment.

**Supervision contribution:**  
Provide guidance on computing, AI/ML framing, software-side implementation, practical pilot thinking, and translation of the research into a defensible applied innovation pathway.

**Project responsibilities:**  
Support interdisciplinary alignment, review model and prototype feasibility, advise on field-support pathway, and help position the work for communications-sector relevance.
