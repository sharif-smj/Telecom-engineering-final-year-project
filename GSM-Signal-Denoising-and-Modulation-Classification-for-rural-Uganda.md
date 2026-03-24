# GSM Signal Denoising and Modulation Classification for Rural Uganda

**GSM Signal Denoising and Modulation**  
**Classification for Rural Uganda**

Final Year Project:  
Department of Electrical and Electronics Engineering

**Authors**

- Ssemujju Sharif Abdukarim, `18/U/ETD/181/GV`
- Kisige Tom Derrick, `22/U/ETD/0953/GV`

**Supervisor:** Dr. Dickson Mugerwa

**Kyambogo University**  
Improving Connectivity in Underserved Areas Using Machine Learning

## Proposal Outline

1. Background
2. Problem Statement
3. Objectives
4. Justification
5. Scope of the Project
6. Literature Review
7. Methodology
8. Tools and Equipment
9. Project Timeline
10. Significance
11. Conclusion
12. References

## Background

- UCC reports that network subscriptions reached **33.2 million** by the end of 2022, and mobile data traffic doubled from **217 million GB** in 2020 to **421.5 million GB** in 2022, showing exponential growth in network service demand. [1]
- Unfortunately, network expansion is failing to keep pace with this demand. Only **31% of the population** and **24% of the landmass** have 4G coverage, and **62% of the population** (30 million people) still lacks mobile internet access. UCC's 2024 coverage framework assesses 2G/GSM service using a minimum outdoor signal threshold of `-90 dBm` (`RxLEV`), while UCC and UCUSAF documents still report black spots and underserved sub-counties, showing that some rural locations do not consistently meet this threshold. In such weak-signal conditions, reliable signal identification and decoding become more difficult. [3][4][5][6][27]
- Many people in rural Uganda still use basic phones for calls and Mobile Money, but the signal is often very weak and is further blocked or "noised up" by illegal boosters, unlicensed community radios (Bizindaalo), and heavy tropical rain. [7][23][24][25]

### Background Continuation

- Automatic Modulation Classification (AMC) is crucial for regulators to monitor the spectrum, manage interference, and identify illegal broadcasters that threaten critical services like aviation. [7]
- We propose using a Denoising Autoencoder (DAE) combined with an AMC system that takes a marginal signal near the UCC coverage threshold of `-90 dBm`, and actively cleans it by removing noise before trying to classify or demodulate it. [5][8][15][26][27]
- Instead of just trying to read a noisy signal, our system first cleans the signal to make it clear again, ensuring that even in the most remote sub-counties, the network can still identify and process signals correctly. [8][15][26]

## Problem Statement

- UCC's 2024 coverage framework defines 2G/GSM service using a minimum outdoor signal threshold of `-90 dBm` (`RxLEV`). However, UCC and UCUSAF documents show that black spots and underserved sub-counties still exist, meaning some rural locations do not consistently meet this threshold. In such weak-signal conditions, even current monitoring approaches such as field measurements, drive tests, walk tests, and coverage simulations can confirm weak coverage but still leave reliable signal identification and classification difficult when transmissions are buried in interference and noise. [5][27]
- Signal quality is further degraded by illegal boosters, unlicensed community radios (Bizindaalo), and heavy tropical rain. Illegal boosters raise interference, unlicensed transmitters can inject harmful harmonics, and rain fade attenuates the radio path and microwave backhaul. [7][23][24][25]
- The Uganda Communications Commission (UCC) therefore finds it difficult to detect illegal broadcasters and diagnose interference when both legitimate and harmful signals are buried in low-SNR, interference-heavy conditions. [7][23][27]
- This leads to dropped calls, failed Mobile Money transactions, and poor service in areas where reliable connectivity is most needed. [3][6][27]
- We therefore lack a robust preprocessing mechanism that can clean weak GSM-family signals before modulation classification and further signal analysis. [8][15][26]

## Main Objective

To design, implement, and empirically evaluate a hybrid **Denoising Autoencoder-Automatic Modulation Classification (DAE-AMC) pipeline** that suppresses interference in noisy I/Q streams and maintains accurate modulation decisions across low-SNR channel conditions observed in Ugandan networks.

## Specific Objectives

1. **Design:** Define the hybrid DAE-AMC architecture, select the working datasets, and model Uganda-specific low-SNR interference conditions such as booster oscillation, Bizindaalo harmonics, and rain fade.
2. **Implementation:** Develop and train the Conv1D denoising autoencoder, implement the AMC classifier for GSM-family modulations, and integrate both blocks into a single signal-processing pipeline.
3. **Analysis:** Evaluate the hybrid DAE-AMC pipeline against a standalone AMC baseline using accuracy, F1-score, confusion matrices, and SNR-wise performance trends.

## Justification and Significance

### Meeting UCC Mandates

UCC's 2024 coverage framework defines 2G/GSM service using a minimum outdoor signal threshold of `-90 dBm` (`RxLEV`), while UCC QoS reporting tracks service thresholds such as blocked and dropped call performance. By targeting weak-signal and interference-heavy conditions around this coverage floor, the proposed DAE-AMC pipeline is aligned with the regulator's need for more reliable monitoring and diagnosis in underserved areas. [5][27]

### Spectrum Surveillance

UCC already uses monitoring, measurement, and complaint-handling mechanisms for spectrum oversight. The added value of the proposed system is not basic automation, but low-SNR signal recovery and modulation classification when transmissions are buried in interference and noise. This can strengthen interpretation of weak illegal or harmful transmissions during monitoring and subsequent enforcement analysis, especially for protected services such as aviation and public safety. [7][23][31][32]

### Rural Mobile Connectivity

The direct focus of this project is reliable rural mobile connectivity rather than broadband throughput alone. By improving signal interpretation near the coverage floor, the proposed system supports more stable GSM/2G service conditions in underserved areas where weak coverage and interference coexist. [3][5][6][27]

### Critical Services

More reliable weak-signal interpretation can support voice communication, Mobile Money, and other essential services that depend on stable rural mobile links. [5][6]

### Alignment with National Policy

Our project supports Uganda's connectivity-expansion agenda by complementing conventional feature-engineered signal analysis with a learnable denoising-and-classification pipeline that is better suited to weak-signal, interference-heavy conditions. [4][15][26]

## Scope

### Offline Experimentation

The project is conducted entirely on a computer using pre-existing data, ensuring a controlled environment.

### Public Datasets

DeepSig RadioML 2018.01A benchmark data provides synthetic, well-characterized SNR labels, while the RF Signal Data collection on Kaggle provides real SDR captures that introduce hardware impairment realism. [11][12]

### Noise Modelling

We inject Uganda-specific noise profiles such as:

- Booster oscillation noise as high-power wideband impulses simulating illegal repeater feedback. [23]
- Bizindaalo harmonics as narrowband tones at second and third harmonics of FM frequencies (`200 MHz`, `300 MHz`) leaking into the `900 MHz` band. [7][23]
- Rain-fade attenuation as time-varying SNR reduction modelling tropical storm conditions with rainfall rates above `100 mm/h`. [24][25]

### Model Development

Focus is placed on a Conv1D Denoising Autoencoder (DAE) and a supervised AMC classifier.

### Specific Modulations

The models are trained to recognize GSM-family modulations such as GMSK, GFSK, and QPSK.

### Platform & Tools

Python, PyTorch, Jupyter Notebook, and supporting visualization or cloud tools where needed.

### Timeline

A 6-month dedicated period is allocated for research, model development, and comprehensive testing.

## Chapter Two: Literature Review

### Classical Methods

Early Automatic Modulation Classification (AMC) relied mainly on statistical analysis, likelihood-based decision rules, and manually engineered signal descriptors such as higher-order moments and cyclostationary properties. These approaches are analytically elegant and can perform well in controlled settings, but their performance degrades sharply when channel conditions are unknown or when the received signal is weak and interference-heavy.

**Key limitation**

- Classical AMC methods depend on stable channel assumptions and carefully engineered signal descriptors.
- In congested and interference-affected deployments such as those documented by UCC, weak-signal and noise-heavy conditions make those descriptors less reliable, motivating the transition toward data-driven learning approaches. [7][23][27]

### Machine Learning Is the New Standard

Modern research has shifted to deep learning models such as convolutional neural networks (CNNs), Transformers, and Mixture-of-Experts frameworks. These models learn discriminative signal characteristics directly from raw I/Q data and have shown significant accuracy improvements, especially in low-SNR conditions around `0 dB` and below. [15][16][26]

### GSM Noise Mitigation in African Networks

While DAE-based signal enhancement has been studied in the literature, its use for weak GSM-signal recovery in African interference environments remains limited. [8][26]

#### Regulatory Interventions

UCC's zero-tolerance posture has led to enforcement operations that confiscate illegal boosters and shut down unlicensed Bizindaalo stations. These interventions are important for spectrum control, but they mainly address interference after it has already been detected. [7][23]

#### Current Technical Practice

In practice, operators and regulators rely on coverage measurements, drive tests, walk tests, field troubleshooting, and related monitoring procedures to identify weak coverage and interference sources. These methods are useful for confirming degraded service conditions, but they do not directly recover weak GSM signals once those signals are buried in interference and noise. [27][31][32]

#### Research Gap

Although machine-learning methods for denoising and modulation classification are well studied, there is still limited work applying them to interference mitigation and weak-signal GSM monitoring in African networks, especially for Uganda-specific conditions such as booster oscillation, Bizindaalo harmonics, and tropical rain fade. [7][8][23][24][25][26]

We address that gap by training a DAE on synthetic noise profiles matching documented interference vectors, so that weak GSM-family signals can first be cleaned and then classified more reliably under interference-heavy conditions. [8][15][26]

### Machine Learning in AMC

Deep learning has reshaped AMC by learning discriminative features directly from raw I/Q samples. Recent contributions demonstrate tangible low-SNR gains:

- **Abd-Elaziz et al. (2023):** A robust CNN with parallel asymmetric kernels achieved `96.5%` accuracy at `0 dB` and `86.1%` at `-2 dB` across nine modulations. [15]
- **Zhang et al. (2023):** The MoE-AMC mixture-of-experts framework yielded about `71.8%` average accuracy across `-20 dB` to `18 dB`, about `10%` higher than single-expert models. [16]
- **Meta-Transformer (2024):** Transformer encoders and few-shot learning allow rapid adaptation to unseen modulations while maintaining superior accuracy. [30]
- **Rehman et al. (2025):** DLAMC converts I/Q streams into eye diagrams, overcoming the `10-48%` accuracy ceiling near `-10 dB` SNR.
- **Jagannath et al. (2022):** CNN-based multitask AMC on a USRP SDR testbed demonstrated more than `98%` accuracy in live over-the-air experiments.

### Denoising Autoencoders for Signal Enhancement

Denoising frontends have emerged as an effective countermeasure when raw I/Q features are overwhelmed by interference. [8][26]

- **Zhang et al.:** A dual residual DAE with channel attention improved AMC accuracy by up to `75%` across `-12 dB` to `8 dB` SNR. [8]
- **DenoMAE:** Faysal et al.'s multimodal denoising masked autoencoder sustained `77.5%` accuracy at `-10 dB`, roughly `22%` higher than the non-denoised case. [28]
- **Thresholded AE:** An and Lee's thresholded autoencoder with an SNR predictor delivered about `70%` relative accuracy gains on low-SNR samples. [29]

These findings justify the DAE-AMC architecture explored in this project and provide design cues for gating strategies that conserve energy on SDR deployments. [8][26][28][29]

### Related Works vis-a-vis Proposed Solution

The reviewed studies show strong progress in low-SNR AMC, but most of them focus on general-purpose datasets and broad modulation families rather than the interference conditions motivating this project.

| Study | What was done | Limitation relative to this project | How this project differs |
|---|---|---|---|
| Abd-Elaziz et al. | Built a robust CNN-based AMC model for low-SNR classification. | Focused on classification only, without a dedicated denoising front end tuned to Ugandan interference. | We introduce a denoising block before classification to recover weak GSM-family signals first. |
| Zhang et al. | Used a dual residual DAE to improve AMC performance. | Treated denoising in a general AMC setting rather than a rural Uganda GSM scenario. | We adapt the denoising idea to booster noise, Bizindaalo harmonics, and rain-fade conditions. |
| MoE-AMC | Improved AMC accuracy using mixture-of-experts routing across SNR conditions. | Focused on classifier architecture complexity rather than signal recovery before classification. | We keep the classifier simpler and improve performance by restoring the signal before classification. |
| DenoMAE / Thresholded AE studies | Showed that denoising can improve low-SNR modulation recognition. | Mainly evaluated on synthetic or generic low-SNR conditions. | We apply denoising and classification to a GSM-oriented, Uganda-motivated interference setting. |

From this comparison, the research gap is clear: prior work has shown that denoising and AMC are individually promising, but the literature does not clearly address a Uganda-specific hybrid DAE-AMC pipeline designed around weak rural GSM signals, regulator-facing monitoring needs, and interference patterns documented in the local context. [7][8][23][24][25][26]

## Chapter Three: Methodology

A comprehensive framework is proposed for developing robust Automatic Modulation Classification systems tailored to challenging RF environments through deep learning and denoising techniques.

### Methodology Overview

We adopt an experimental research design anchored in reproducible data processing and quantitative benchmarking. To make the workflow consistent with the main objective, the methodology is organized into three linked parts: **design**, **implementation**, and **analysis**. [11][12][15][16][26]

The technical flow is communicated as a signal chain familiar to communications audiences:

`received noisy GSM I/Q signal -> preprocessing and noise modelling -> denoising autoencoder -> AMC classifier -> modulation decision -> performance analysis`

### Part I: Design

This stage defines the full architecture and experiment setup before training begins.

- Select the datasets to be used, primarily RF Signal Data and RadioML 2018.01A. [11][12]
- Harmonize all signals into a unified tensor format with `1024`-sample windows and normalized amplitude.
- Perform exploratory data analysis (EDA) to inspect class balance, SNR coverage, waveform quality, and impairment patterns before training. [11][12]
- Define the GSM-focused modulation classes: GMSK, GFSK, and QPSK.
- Model Uganda-specific interference using booster oscillation, Bizindaalo harmonics, and rain-fade attenuation. [7][23][24][25]
- Specify the system architecture showing how the denoising block feeds the modulation classifier.

**Expected outcome:** A complete project design package comprising the architecture diagram, dataset specification, low-SNR scenario definitions, and the experimental setup for the DAE-AMC pipeline.

### Part II: Implementation

This stage converts the design into working models and an integrated signal-processing pipeline.

- Train a Conv1D encoder-decoder DAE on noisy and clean I/Q pairs using MSE-based reconstruction loss. [8][26][28]
- Train a supervised AMC classifier on GSM-family modulations using both raw and denoised inputs. [15][16]
- Integrate the DAE and AMC blocks into a hybrid pipeline and compare frozen and fine-tuned configurations. [8][26][28][29]
- Build a lightweight prototype interface for visualizing noisy signals, denoised outputs, and classification predictions.

**Expected outcome:** A functional hybrid DAE-AMC prototype capable of denoising noisy I/Q samples and producing modulation predictions under controlled low-SNR conditions.

### Part III: Analysis and Evaluation

This stage measures whether the proposed pipeline improves modulation recognition under the targeted interference conditions.

- Benchmark the hybrid DAE-AMC system against a standalone AMC baseline. [15][16]
- Evaluate denoising quality and classification quality across held-out SNR ranges. [8][26][28][29]
- Use accuracy curves, macro F1-score, confusion matrices, and SNR-wise comparisons to quantify improvement. [15][16][26]
- Interpret the results in terms of rural Uganda use cases such as weak-signal monitoring, interference diagnosis, and regulator demonstrations.

**Expected outcome:** A clear evidence set showing where the hybrid pipeline improves over baseline AMC, including result plots, comparative metrics, and conclusions about suitability for rural Uganda spectrum-monitoring scenarios.

## Project Budget Overview

The successful execution of this project requires a structured allocation of resources. The total estimated budget is **800,000 UGX**, covering critical areas from software to research.

| Item | Cost (UGX) |
|---|---:|
| Software Licenses | 200,000 |
| Data Curation | 100,000 |
| Virtual Machines | 200,000 |
| Cloud Databases | 100,000 |
| Internet and Research | 50,000 |
| Printing and Documentation | 50,000 |
| Miscellaneous | 100,000 |
| **Total Estimated Cost** | **800,000** |

This budget ensures that all necessary technical infrastructure, data processing capabilities, and operational requirements are adequately funded to support the project's objectives.

## Project Timeline

The project will be executed over a 6-month period, allowing enough time for literature review, dataset preparation, model development, evaluation, documentation, and presentation.

| Phase | Description | Weeks |
|---|---|---|
| Phase 1: Problem Framing and Literature Review | Define the problem, refine objectives, and complete the literature review. | 1-4 |
| Phase 2: Data Acquisition, EDA, and Preprocessing | Gather datasets, inspect signal properties, harmonize samples, and model the interference conditions. | 5-8 |
| Phase 3: Model Design and Initial Training | Design the DAE-AMC pipeline, implement the models, and begin initial training experiments. | 9-14 |
| Phase 4: Integration and Controlled Experiments | Integrate the denoiser and classifier, run hybrid-pipeline experiments, and refine settings. | 15-18 |
| Phase 5: Evaluation and Interpretation | Compare the hybrid system against the baseline and analyze the results across SNR conditions. | 19-22 |
| Phase 6: Documentation and Presentation | Finalize the report, compile findings, and prepare the presentation materials. | 23-24 |

This timeline ensures a methodical approach to each stage, culminating in a robust and well-documented solution for intelligent signal processing over the full 6-month project period.

## References

[1] Uganda Communications Commission, "Telephone Subscriptions Rise to 33.2 Million," *UCC Communications Blog*, 9 June 2023.  
<https://uccinfoblog.com/2023/06/09/telephone-subscriptions-rise-to-33-2-million/>

[2] Atomic Energy Council, "Radiofrequency Radiation in Uganda," 2022.  
<https://www.atomiccouncil.go.ug/non-ionizing-radiation-radiofrequency/>

[3] ChimpReports, "Uganda's Internet users hit 13 million," *ChimpReports News*, 25 March 2024.  
<https://chimpreports.com/ugandas-internet-users-hit-13-million/>

[4] Uganda Communications Commission, "Access Infrastructure Program: Bridging the Digital Divide," *UCC Blog*, February 2024.

[5] TechJaja, "UCUSAF: Why is UCC rolling out own telecom network?" 6 February 2024. Repository capture available at `references/techjaja_ucusaf.html`.

[6] Ghana Chamber of Telecommunications, "Mobile Internet Access Still Limited in Africa, Millions Remain Offline," citing GSMA data, 2024.  
<https://www.telecomschamber.org/industry-news/mobile-internet-access-still-limited-in-africa-millions-remain-offline/>

[7] Uganda Communications Commission, "UCC cracks down on illegal and non-compliant broadcasters," 21 October 2024.  
<https://www.ucc.co.ug/ucc-cracks-down-on-illegal-and-non-compliant-broadcasters/>

[8] Xiaolin Zhang et al., "Dual Residual Denoising Autoencoder with Channel Attention Mechanism for Modulation of Signals," *Sensors*, vol. 23, 2023.  
<https://pmc.ncbi.nlm.nih.gov/articles/PMC9861137/>

[11] RF Signal Data, Kaggle, accessed 2025.  
<https://www.kaggle.com/datasets/suraj520/rf-signal-data>

[12] DeepSig Dataset: RadioML 2018.01A, Kaggle, accessed 2025.  
<https://www.kaggle.com/datasets/pinxau1000/radioml2018>

[13] Ramiro Utrilla, "MIGOU-MOD: A dataset of modulated radio signals acquired with MIGOU, a low-power IoT experimental platform," *Mendeley Data*, V1, 2020.  
<https://data.mendeley.com/datasets/fkwr8mzndr/1>

[14] DeepSig, "RadioML 2016.10A Dataset," accessed 2025.  
<https://www.deepsig.ai/datasets/>

[15] O. F. Abd-Elaziz, A. M. El-Ghandour, and F. H. Ismail, "Deep Learning-Based Automatic Modulation Classification Using Robust CNN Architecture for Cognitive Radio Networks," *Sensors*, vol. 23, no. 23, 2023, Art. 9467.  
doi:10.3390/s23239467

[16] J. Gao, Z. Zhang, and Y. Zhang, "MoE-AMC: Enhancing Automatic Modulation Classification Performance Using Mixture-of-Experts," *arXiv preprint*, 2023.  
<https://arxiv.org/abs/2312.02298>

[23] Uganda Communications Commission, "Public Notice: Signal Interference Arising Out of Usage of Network Repeaters - 'Boosters'," 26 July 2021.  
<https://uccinfoblog.com/2021/07/26/public-notice-signal-interference-arising-out-of-usage-of-network-repeaters-boosters/>

[24] ITU-R, *Recommendation ITU-R P.530-18: Propagation data and prediction methods required for the design of terrestrial line-of-sight systems*, September 2021.  
<https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.530-18-202109-I!!PDF-E.pdf>

[25] ITU-R, *Recommendation ITU-R P.838-3: Specific attenuation model for rain for use in prediction methods*, 2005.  
<https://www.itu.int/rec/R-REC-P.838-3-200503-P/en>

[26] X. Gao, X. Xu, D. Li, X. Liu, J. Yang, and D. Zhai, "Enhancing Noise Robustness in Few-Shot Automatic Modulation Classification via Complex-Valued Autoencoders," *Electronics*, vol. 15, no. 3, Art. 674, 2026.  
<https://www.mdpi.com/2079-9292/15/3/674>

[27] "Quality of Service Findings for Indoor Mobile Voice Telephony and Data Services in Uganda," *UCC/AWS*, Jan. 2026.  
<https://newvision-media.s3.amazonaws.com/cms/b295e426-f529-44b6-928d-330d8b911.pdf>

[28] M. Faysal, J. Chen, and P. Balaprakash, "DenoMAE: A Multimodal Autoencoder for Denoising Modulation Signals," *arXiv preprint*, arXiv:2501.11538, 2025.  
<https://arxiv.org/abs/2501.11538>

[29] H. An and B.-M. Lee, "Robust Automatic Modulation Classification in Low Signal-to-Noise Ratio," *IEEE Access*, vol. 11, pp. 125678-125690, 2023.  
doi:10.1109/ACCESS.2023.3321108

[30] J. Jang, J. Pyo, Y.-i. Yoon, and J. Choi, "Meta-Transformer: A Meta-Learning Framework for Scalable Automatic Modulation Classification," *IEEE Access*, vol. 12, pp. 9267-9276, 2024.  
doi:10.1109/ACCESS.2024.3352634

[31] Uganda Communications Commission, *UCC Strategic Plan 2020/21-2024/25*, 2023 online edition.  
<https://www.ucc.co.ug/wp-content/uploads/2023/10/UCC-Strategic-Plan-202021-202425-ONLINE-VERSION-002_Rev2Final.pdf>

[32] Uganda Communications Commission, *Guidelines on Establishment and Operation of FM Radio Stations in Uganda*, March 2019, Annex 4: Radio Interference Handling Procedures.  
<https://www.ucc.co.ug/wp-content/uploads/2023/10/Guidelines-on-Establishment-and-Operation-of-FM-Radio-Stations-in-Uganda_-March-2019.pdf>

## Closing Note

Thanx for Listening
