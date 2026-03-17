KYAMBOGO UNIVERSITY
FINAL YEAR PROJECT PROPOSAL

TITLE: GSM Signal Denoising and Modulation Classification Using Machine Learning for Rural Uganda Coverage Expansion
NAME: SSEMUJJU SHARIF ABDUKARIM
REG. NO: 18/U/ETD/181/GV
DEPARTMENT: ELECTRICAL AND ELECTRONICS ENGINEERING
SUPERVISOR: Dr. Dickson Mugerwa

---

## Table of Contents

**Chapter One – Introduction**
Executive Summary
1.1 Background of the Study
1.2 Problem Statement
1.3 Main Objective of the Study
1.3.1 Specific Objectives
1.3.2 Contribution Statement
1.4 Scope of the Study
1.5 Significance of the Study
1.6 Justification
1.7 Conceptual Framework

**Chapter Two – Literature Review**
2.0 Introduction
2.1 Overview of Modulation Classification
2.2 Traditional Modulation Classification Approaches
2.3 Machine Learning in Modulation Classification
2.4 Performance Metrics and Datasets
2.5 Research Gaps
2.6 Research Questions
2.7 Denoising Autoencoders for Signal Enhancement
2.8 GSM Noise Mitigation in African Networks
2.9 Related Works vis-a-vis Proposed Solution

**Chapter Three – Methodology**
3.0 Overview of Existing AMC Systems
3.1 Introduction to Methods
3.2 Research Paradigm
3.3 Research Approach
3.4 Design, Implementation, and Analysis Strategy
3.5 Model Selection
3.6 Data Collection and Preprocessing
3.7 Model Development and Training
3.8 Evaluation Metrics
3.9 Model Validation
3.10 User Interface Development
3.11 Deployment and Demonstration
3.12 Ethical Considerations
3.13 Tools and Software Requirements

# List of Abbreviations

| Abbreviation | Description |
| --- | --- |
| AMC | Automatic Modulation Classification |
| DAE | Denoising Autoencoder |
| SDR | Software-Defined Radio |
| ML | Machine Learning |
| DL | Deep Learning |
| SNR | Signal-to-Noise Ratio |
| SINR | Signal-to-Interference-plus-Noise Ratio |
| QoS | Quality of Service |
| UCC | Uganda Communications Commission |
| UCUSAF | Uganda Communications Universal Service and Access Fund |
| NDPIII | Third National Development Plan (2020/21–2024/25) |
| NBP | National Broadband Policy |
| TVWS | Television White Space |
| AWGN | Additive White Gaussian Noise |
| CNN | Convolutional Neural Network |
| RNN | Recurrent Neural Network |
| LSTM | Long Short-Term Memory |
| CLDNN | Convolutional LSTM Dense Neural Network |
| OTA | Over-the-Air |
| USRP | Universal Software Radio Peripheral |
| GUI | Graphical User Interface |
| PSNR | Peak Signal-to-Noise Ratio |
| SSIM | Structural Similarity Index Measure |
| AUC | Area Under the Curve |
| MLflow | Machine Learning Flow (experiment tracking platform) |
| W&B | Weights & Biases (experiment tracking platform) |

---

# CHAPTER ONE

## Executive Summary

Rural Uganda faces a critical connectivity challenge: while 2G/GSM networks remain the lifeline for voice and mobile money services in underserved areas, signal quality in UCUSAF-designated sub-counties frequently falls below the −90 dBm threshold due to persistent interference from illegal signal boosters, unlicensed "Bizindaalo" broadcasters, and tropical rain fade [5], [7], [23], [25], [36], [37]. We propose a Denoising Autoencoder–Automatic Modulation Classification (DAE–AMC) pipeline specifically designed for GSM/GMSK signal recovery in these challenging environments. By training the denoiser on noise profiles characteristic of Ugandan networks—wideband oscillation from illegal boosters, harmonic distortion from unfiltered FM transmitters, and time-varying rain attenuation—our system aims to maintain reliable modulation recognition where conventional classifiers fail [8], [38]. Our contribution is the first documented application of DAE preprocessing to Uganda's unique GSM interference landscape, providing UCC spectrum monitors with a software-based tool to detect weak emitters and support coverage expansion in the country's most underserved communities [7], [23], [25].

---

## 1.1 Background of the Study

Uganda's telecommunications sector has achieved significant milestones, with subscriptions reaching 33.2 million and broadband connections rising to 23.7 million [1]. However, these aggregate numbers mask a critical dependency on legacy 2G/GSM infrastructure that remains the bedrock of rural connectivity. Unlike markets aggressively sunsetting 2G, Uganda has no immediate regulatory roadmap for GSM decommissioning because a significant proportion of the rural user base relies on feature phones for voice communication and USSD-based Mobile Money (MoMo) transactions [23].

The UCUSAF (Uganda Communications Universal Service and Access Fund) program has identified 117 sub-counties where signal strength remains below −90 dBm—the minimum threshold for reliable GSM reception [5]. In these areas, GSM operating on the 900 MHz band is often the only available signal due to its superior propagation characteristics compared to higher frequencies used by 3G/4G [23]. This makes GSM network integrity essential for the 7.5 million Ugandans currently without mobile access [4].

However, GSM networks in Uganda face severe electromagnetic interference that degrades signal quality beyond what coverage maps suggest. The Uganda Communications Commission (UCC) has documented three primary interference vectors [24], [25], [35]:

1. **Illegal Signal Boosters**: Residents in areas with poor indoor coverage purchase cheap, unregulated bi-directional amplifiers. These devices lack proper filtering and frequently enter oscillation states, acting as high-power jammers that broadcast wideband noise across GSM uplink frequencies. The resulting "near-far" problem desensitizes base station receivers, causing dropped calls and blocked access for legitimate users within the affected cell [25], [35].

2. **"Bizindaalo" and Unlicensed FM Transmitters**: Unauthorized community radios operating without technical oversight use cheap transmitters with poor harmonic filtering. A station at 100 MHz can generate harmonics directly in the 900 MHz GSM band, raising the thermal noise floor and forcing the network to use lower, slower modulation schemes [24].

3. **Tropical Rain Fade**: Uganda's equatorial location introduces severe atmospheric attenuation during rainstorms exceeding 100mm/h. Microwave backhaul links lose connectivity, and even the GSM radio path experiences time-varying signal degradation [26], [36], [37].

Automatic Modulation Classification (AMC) is critical for spectrum surveillance and interference mitigation because it enables detection of unauthorized transmitters without prior coordination [7]. Machine Learning approaches have improved AMC accuracy, but performance collapses at low SNR unless the front end is robust to noise [15], [16]. Recent work on Denoising Autoencoders (DAEs), such as the dual-residual architecture by Zhang et al., demonstrated 67–75% classification-accuracy gains across −12 dB to 8 dB SNR [8], [38]. We adapt the DAE–AMC approach specifically for Uganda's GSM interference profile, training on noise patterns characteristic of booster oscillation, harmonic distortion, and rain fade to maintain reliable GMSK recognition in UCUSAF edge environments [25], [35], [36], [37].

---

## 1.2 Problem Statement

GSM networks in rural Uganda operate at the edge of viability. In UCUSAF-designated sub-counties, 2G signal strength frequently hovers at or below the −90 dBm minimum threshold, leaving receivers with marginal Signal-to-Interference-plus-Noise Ratios (SINR) [5]. This already fragile connectivity is further degraded by persistent electromagnetic interference that conventional AMC systems cannot handle [25], [35], [36], [37].

The primary interference source is the proliferation of illegal signal boosters. UCC enforcement operations have documented that these unregulated devices, when they enter oscillation, act as high-power jammers that desensitize base station receivers—a classic "near-far" problem that causes blocked-call rates to spike above the 2% regulatory ceiling [25], [27], [35]. In 2019 QoS audits, over 70% of MTN and Airtel call failures in affected areas were traced to interference, with towns like Jinja experiencing blocked-call rates as high as 34% [27].

Secondary interference comes from unlicensed "Bizindaalo" broadcasters whose FM transmitters generate harmonics in the 900 MHz GSM band, raising the noise floor and forcing the network into lower modulation schemes [24]. During tropical rainstorms exceeding 100mm/h, rain fade introduces additional time-varying attenuation that disrupts both the radio path and microwave backhaul [26], [36], [37].

Conventional AMC approaches—likelihood-ratio tests, cumulant extractors, and even modern neural classifiers—degrade sharply under these conditions [15], [16]. State-of-the-art mixture-of-experts architectures achieve only 71.76% average accuracy across SNR sweeps, far below what UCC's QoS mandates require for reliable spectrum monitoring [16], [27]. Without a front-end that suppresses Uganda-specific noise before classification, enforcement teams cannot reliably detect weak illegal emitters, and network operators cannot diagnose interference affecting UCUSAF coverage expansion sites [7], [25], [35].

We address this gap by designing a DAE–AMC pipeline specifically trained on noise profiles matching illegal booster oscillation, Bizindaalo harmonics, and rain fade—enabling stable GSM/GMSK modulation recognition at the −90 dBm coverage floor where rural Ugandan networks must operate.

---

## 1.3 Main Objective of the Study

To design, implement, and empirically evaluate a hybrid Denoising Autoencoder–Automatic Modulation Classification (DAE–AMC) pipeline that can suppress interference in noisy I/Q streams and maintain accurate modulation decisions across the low-SNR channel conditions observed in Ugandan networks.

### 1.3.1 Specific Objectives

1. **Design objective**: To define a hybrid DAE–AMC architecture, select appropriate GSM-family datasets, and model Uganda-specific interference conditions such as booster oscillation, Bizindaalo harmonics, and rain fade.
2. **Implementation objective**: To develop and train the Conv1D denoising autoencoder and AMC classifier, then integrate them into a single pipeline for GSM-family modulation recognition.
3. **Analysis objective**: To evaluate the hybrid DAE–AMC pipeline against standalone AMC using denoising and classification metrics across low-SNR conditions representative of rural Ugandan deployments, and to interpret the findings for practical UCC monitoring workflows.

### 1.3.2 Contribution Statement

We make the following novel contributions:

1. **Domain-Specific Application**: First documented application of Denoising Autoencoder (DAE) preprocessing to the specific noise profile of Ugandan GSM networks, characterized by illegal booster oscillation, Bizindaalo harmonic interference, and tropical rain fade.

2. **GMSK-Focused Pipeline**: Unlike prior DAE–AMC work that targets diverse modulation families (BPSK, QPSK, 16QAM, 64QAM, etc.), we narrow our focus to GSM/GMSK signals—the de facto standard for rural Uganda where feature phones remain predominant and Mobile Money services are essential.

3. **UCUSAF Edge-Case Design**: The DAE is explicitly trained on SNR conditions matching UCUSAF's −90 dBm coverage floor, addressing the "near-far" interference documented by UCC QoS audits rather than generic low-SNR scenarios.

4. **Practical Deployment Pathway**: Our methodology includes lightweight model architectures (< 500K parameters) suitable for resource-constrained SDR deployments in rural base stations, with documented integration steps for UCC enforcement workflows.

**Distinction from Prior Work**:

| Prior Work | Focus | Our Distinction |
|------------|-------|---------------------------|
| Zhang et al. (2023) [8] | General DAE for AMC across modulations | GMSK-specific DAE for Ugandan noise profile |
| Faysal et al. (DenoMAE) [22] | Multimodal denoising, synthetic datasets | Real-world Ugandan interference patterns |
| An & Lee (2023) [21] | Thresholded AE denoiser for general AMC | Continuous DAE suited to persistent booster noise |
| MoE-AMC (2023) [16] | Mixture-of-experts routing for diverse SNRs | Single-modulation focus removes routing complexity |

---

## 1.4 Scope of the Study

We limit our investigation to offline experimentation using two publicly available I/Q corpora: the DeepSig RadioML 2018.01A benchmark (synthetic, well-characterized SNR labels) and the RF Signal Data collection on Kaggle (real SDR captures providing hardware impairment realism) [11], [12]. No live SDR capture, hardware deployment, or regulatory compliance testing is undertaken.

**Modulation Focus**: We target GSM-family modulations—specifically GMSK (the GSM standard), GFSK (common in Bluetooth/IoT devices sharing similar frequency bands), and QPSK (as a reference comparison). This narrowed scope reflects the rural Ugandan context where 2G/GSM remains the primary connectivity technology.

**Noise Modeling**: Rather than generic AWGN, we inject Uganda-specific noise profiles grounded in Uganda-specific interference and propagation framing [7], [23], [25], [35], [36], [37]:
- **Booster oscillation noise**: High-power wideband impulses simulating illegal repeater feedback
- **Bizindaalo harmonics**: Narrowband tones at 2nd/3rd harmonics of FM frequencies (200 MHz, 300 MHz sidebands leaking into 900 MHz)
- **Rain fade attenuation**: Time-varying SNR reduction modeling tropical storm conditions (100+ mm/h rainfall rates)

**SNR Range**: Training and evaluation target the −90 dBm to −60 dBm signal strength range corresponding to UCUSAF coverage-floor conditions, with controlled sweeps from −12 dB to +12 dB SNR [5].

Findings and metrics reflect these synthetic scenarios rather than live network measurements; our goal is to demonstrate DAE–AMC feasibility for Uganda's specific interference environment before potential field deployment [11], [12], [15], [16], [38].

---

## 1.5 Significance of the Study

By keeping modulation recognition stable when SNR collapses toward UCUSAF's −90 dBm coverage floor, our proposed DAE–AMC pipeline offers a practical response to the congestion, interference, and illegal-transmitter issues highlighted by UCC QoS audits and GSMA connectivity surveys. It bolsters spectrum surveillance, rural broadband, aviation safety, and mobile money services that depend on reliable radio links in Uganda's noisy bands [5], [6], [7], [27].

---

## 1.6 Justification

UCC's mandates to resolve 95% of faults within 24 hours and suppress harmful interference cannot be met if AMC models deliver barely 71.76% accuracy once SNR drops below 0 dB [16], [27]. Pairing a learnable denoiser with the classifier reduces reliance on brittle handcrafted features, aligns with national connectivity and coverage-expansion priorities, and creates an adaptable software upgrade that fits existing SDR monitoring chains without requiring additional spectrum allocations [4], [15], [38].

---

## 1.7 Conceptual Framework

```
Received noisy GSM I/Q signal -> preprocessing -> Denoising Autoencoder (DAE) -> AMC classifier -> predicted modulation -> performance analysis and deployment interpretation
```

---

# CHAPTER TWO – LITERATURE REVIEW

## 2.0 Introduction

This chapter surveys the evolution of Automatic Modulation Classification (AMC) research from classical statistical detectors to recent deep learning architectures and denoising front-ends. Emphasis is placed on studies that quantify performance below 0 dB SNR—conditions that mirror Uganda’s interference-heavy spectrum—and on dataset innovations that support reproducible benchmarking.

## 2.1 Overview of Modulation Classification

AMC underpins spectrum awareness for cognitive radio, electronic warfare, and national regulators because it infers a waveform’s modulation type without a priori coordination [1], [5]. Reliable classification enables dynamic spectrum access, enforcement against illicit transmitters, and automated routing of traffic through increasingly congested infrastructure. Consequently, AMC techniques must perform well even when radios operate at UCUSAF’s −90 dBm edges or experience intentional interference.

## 2.2 Traditional Modulation Classification Approaches

Before the recent wave of deep learning, modulation recognition relied on likelihood-ratio tests, cumulant and cyclostationary feature extraction, or other manually engineered statistics. These detectors remain analytically elegant but degrade sharply when SNR falls below 0 dB or when multipath and oscillator offsets distort the assumed signal model—exactly the impairments documented by UCC in congested Ugandan deployments [25], [27]. Their brittleness under unknown channels motivates the transition toward data-driven feature learning.

## 2.3 Machine Learning in Modulation Classification

Deep learning has reshaped AMC by learning discriminative features directly from raw I/Q samples. Recent contributions demonstrate tangible low-SNR gains:

- Abd-Elaziz et al. (2023) designed a Robust CNN with parallel asymmetric kernels and residual skip connections that achieved 96.5 % accuracy at 0 dB and 86.1 % at −2 dB across nine modulations impaired by AWGN, Rician fading, and clock offsets, substantially outperforming prior CNN baselines [15].
- Zhang et al. (2023) proposed MoE-AMC, a mixture-of-experts framework that routes signals to Transformer-based low-SNR experts or ResNet high-SNR experts via a gating network, yielding ~71.8 % averaged accuracy across −20…18 dB on RadioML2018.01A—about 10 % higher than single-expert models [16].
- Meta-learning approaches such as the 2024 Meta-Transformer leverage transformer encoders and few-shot learning to adapt rapidly to unseen modulations, maintaining superior accuracy across all SNRs on RadioML2018.01A while sharing reproducible code for community validation [17].
- Rehman et al. (2025) introduced DL-AMC, which converts I/Q streams into eye diagrams and classifies them with ResNet variants, overcoming the 10–48 % accuracy ceiling that DBN, RNN, and CLDNN architectures exhibited near −10 dB SNR [18].
- Jagannath et al. (2022) closed the “reality gap” by validating CNN-based multi-task AMC on a USRP SDR testbed, demonstrating >98 % accuracy on seven modulations in live over-the-air experiments and highlighting the importance of heterogeneous training that includes hardware impairments [20].

Collectively, these works show that architectural customization (optimized CNN blocks, expert routing, attention) and domain-adaptive validation are essential for deployments in noisy environments like Uganda’s shared bands.

## 2.4 Performance Metrics and Datasets

Accuracy, F1-score, confusion matrices, and accuracy-vs-SNR curves remain standard evaluation metrics; however, reproducibility now hinges on diverse datasets. The core datasets used in this study are Kaggle RF Signal Data and DeepSig RadioML 2018.01A [11], [12]. Beyond those core datasets, researchers increasingly rely on:

- MIGOU-MOD, which provides over-the-air IoT captures from the MIGOU low-power platform for assessing energy-constrained AMC scenarios [13].
- RadioML 2016.10A, which remains a common legacy benchmark in AMC comparisons [14].
- RML22, a data-centric successor to RadioML that corrects generation artifacts, injects more realistic channel models, and publishes the full Python generation stack so others can regenerate or adapt the benchmark [19].

These datasets enable controlled AWGN sweeps, realistic multipath simulations, and OTA validation, allowing rigorous comparison of DAE–AMC pipelines across signal families.

## 2.5 Research Gaps

Despite progress, several gaps persist. First, even the best-performing architectures suffer accuracy collapses once SNR dips below −5 dB, leaving regulators blind to weak interferers [15], [16], [18]. Second, most published metrics come from simulations; only a handful of OTA demonstrations (e.g., Jagannath et al.) quantify the domain shift introduced by real hardware and channel impairments [20]. Third, few studies integrate denoisers tightly with AMC or explore how denoising impacts regulatory workflows such as UCC’s interference crackdowns [7], [21], [22]. This project addresses the latter by coupling a Conv1D DAE to the classifier and benchmarking the combined pipeline under Ugandan-inspired SNR profiles.

## 2.6 Research Questions

1. How do modern CNN, mixture-of-experts, and transformer architectures extend AMC robustness for GSM/GMSK signals when SNR approaches the −10…0 dB regimes common in UCUSAF-designated areas? [15], [16], [17], [18]
2. Which publicly available datasets (RadioML 2018.01A, Kaggle RF Signal Data) best capture GSM-band impairments including booster interference and harmonic distortion? [11], [12]
3. To what extent does inserting a DAE trained on Uganda-specific noise profiles ahead of the classifier recover low-SNR GSM accuracy relative to standalone AMC models? [8], [21], [22]

## 2.7 Denoising Autoencoders for Signal Enhancement

Denoising front-ends have emerged as an effective countermeasure when raw I/Q features are overwhelmed by interference. Zhang et al.'s dual-residual DAE with channel attention improved AMC accuracy by up to 75% across −12…8 dB SNR [8], demonstrating that reconstructing constellation geometry before classification materially benefits downstream decisions. Faysal et al. (2025) extended this idea with DenoMAE, a multimodal denoising masked autoencoder that treats noise as a separate modality; after fine-tuning, it sustained 77.5% accuracy at −10 dB—roughly 22% higher than the same classifier without denoising pre-training [22]. Complementary work by An and Lee (2023) introduced a thresholded autoencoder denoiser triggered by a lightweight SNR predictor; this combination delivered ~70% relative accuracy gains on low-SNR samples while avoiding unnecessary processing for high-SNR inputs in IEEE Access experiments [21]. These findings justify the DAE–AMC architecture we explore and provide design cues for gating strategies that conserve energy on SDR deployments.

## 2.8 GSM Noise Mitigation in African Networks

While DAE-based signal enhancement has been extensively studied in academic settings, its application to the specific interference environment of African GSM networks remains unexplored. Uganda's telecommunications landscape presents unique challenges that differ from the synthetic noise models used in most AMC research [7], [23], [25], [36], [37]:

**Regulatory Interventions**: The UCC has adopted a zero-tolerance posture toward spectrum pollution, conducting enforcement operations to confiscate illegal boosters and shut down unlicensed Bizindaalo stations [24], [25], [35]. However, the proliferation of these devices continues due to porous borders and high demand from users in coverage-poor areas. The regulator acknowledges that operators are forced to invest in "spectrum cleaning" teams—resources that could otherwise fund network expansion [23].

**Technical Countermeasures**: Modern 4G/5G deployments in Uganda utilize Massive MIMO and beamforming to spatially filter interference [27]. However, these technologies are not available for legacy 2G/GSM infrastructure in rural areas. The primary technical remedy for GSM networks remains the National Backbone Infrastructure (NBI) fiber expansion, which reduces reliance on weather-susceptible microwave backhaul [23], [36]. For the radio access network itself, adaptive power control helps reduce overall noise pollution, but does not address the fundamental challenge of classifying weak signals in high-interference environments [27].

**The Research Gap**: No published work has applied machine learning-based denoising specifically to the GSM interference profile documented by UCC—namely, the combination of booster oscillation (wideband impulse noise), Bizindaalo harmonics (narrowband tones in the 900 MHz band), and tropical rain fade (time-varying attenuation) [7], [23], [25], [35], [36], [37]. We fill that gap by training a DAE on synthetic noise profiles matching these documented interference vectors, enabling GMSK classification at SNR levels where conventional approaches fail [8], [38].

## 2.9 Related Works vis-a-vis Proposed Solution

The studies reviewed in this chapter show that prior work is strong in either classification or denoising, but rarely ties both together in a way that is specific to weak rural GSM signals and interference scenarios relevant to Uganda [7], [23], [25], [36], [37], [38].

| Prior work | Main idea | Strength | Limitation relative to this study | Proposed position of this project |
|------------|-----------|----------|-----------------------------------|-----------------------------------|
| Classical AMC approaches | Use likelihood tests, cumulants, and cyclostationary features for modulation recognition | Interpretable and mathematically grounded | Performance degrades sharply under low SNR, unknown channels, and interference-heavy environments | Replace brittle handcrafted features with a learnable denoising-plus-classification signal chain |
| CNN-based AMC studies such as Abd-Elaziz et al. | Apply deep CNNs directly to raw I/Q data for end-to-end classification | Strong low-SNR classification gains over classical baselines | Classification improves, but noisy signals are still passed directly to the classifier | Introduce an explicit denoising front end before classification |
| MoE-AMC and transformer-based AMC studies | Improve classification accuracy using more expressive classifier architectures | Better generalization across diverse SNR ranges and modulation sets | Architectural complexity increases while signal recovery remains indirect | Keep the classifier lightweight and recover the signal first through denoising |
| Zhang et al., DenoMAE, and thresholded AE studies | Use denoising autoencoders or related denoising models to improve AMC | Show that denoising can materially improve low-SNR classification | Most studies use generic benchmarks and do not focus on Uganda-specific GSM interference | Adapt denoising to booster oscillation, Bizindaalo harmonics, and rain-fade scenarios relevant to rural Uganda |

Viewed together, these studies justify the proposed hybrid DAE–AMC solution. Prior research establishes that classification can improve with better learned features and that denoising can recover weak signals. This project combines those insights in a more deployment-oriented pipeline tailored to the local problem setting: noisy GSM-family signals, rural edge-of-coverage operation, and regulator-facing interference analysis.

---

# CHAPTER THREE – METHODOLOGY

## 3.0 Overview of Existing AMC Systems

Conventional AMC pipelines in Ugandan networks still depend on handcrafted cumulants, likelihood tests, and static DSP filters that assume high SNR and stable oscillators. As laid out in Chapter 2, those assumptions collapse in practice: the UCC documents interference-driven failures, and recent research shows that even sophisticated CNNs lose accuracy near −5 dB unless they incorporate noise-aware architectures [7], [15]. Our methodology therefore replaces hand-engineered features with a learnable denoising preprocessor (DAE) followed by a supervised AMC head, trained and evaluated under the same SNR ranges that stress current deployments.

## 3.1 Introduction to Methods

We adopt an experimental research design anchored in reproducible data processing and quantitative benchmarking. To align the methodology with the main objective, the workflow is communicated in three linked parts: **design**, **implementation**, and **analysis** [11], [12], [15], [16], [38].

The end-to-end signal chain is:

`received noisy GSM I/Q signal -> preprocessing -> denoising autoencoder -> AMC classifier -> modulation decision -> performance analysis`

The three methodological parts are then expanded through the detailed sections that follow.

## 3.2 Research Paradigm

A post-positivist paradigm guides our study: hypotheses about low-SNR robustness are tested empirically, while acknowledging that experimental results are provisional and must be corroborated by replication on additional datasets or field captures. All code, hyperparameters, and preprocessing steps will be version-controlled to enable independent verification.

## 3.3 Research Approach

Our approach is data-driven and follows three connected research actions [15], [16], [38]:

- **Design action** – define the architecture, perform exploratory data analysis (EDA), and formalize the low-SNR and interference scenarios to be studied.
- **Implementation action** – train the denoiser and classifier modules, then integrate them into a hybrid pipeline.
- **Analysis action** – quantify the effect of denoising on classification quality and interpret the results in the context of rural Uganda use cases.

This structure makes it easier to communicate the work to both machine-learning and telecommunications audiences because the solution can be read as a familiar block-based receiver chain rather than as an abstract AI workflow alone.

## 3.4 Design, Implementation, and Analysis Strategy

### 3.4.1 Design Stage

The design stage establishes the system architecture and experiment conditions before model training begins.

1. **Dataset harmonization** – convert RadioML 2018.01A and Kaggle RF Signal Data into a unified tensor format (length-1024 complex samples, normalized amplitude) and split into train/validation/test partitions with stratification by modulation and SNR.
2. **Exploratory data analysis (EDA)** – inspect class balance, SNR distribution, waveform quality, and impairment patterns so that the selected datasets and modulation classes are clearly justified [11], [12].
3. **Uganda-specific noise modeling** – inject three interference types documented by UCC [7], [23], [25], [35], [36], [37]:
   - **Booster oscillation**: High-power wideband impulse noise simulating illegal repeater feedback
   - **Bizindaalo harmonics**: Narrowband tones at 2nd/3rd harmonics of FM frequencies affecting the 900 MHz GSM band
   - **Rain fade attenuation**: Time-varying SNR reduction modeling tropical storm conditions
4. **Architecture definition** – formalize the block-based signal chain linking noisy input, denoising, classification, and result interpretation.

**Expected outcome**: a complete system design comprising the architecture, dataset profile, interference model, and experimental conditions for the hybrid DAE–AMC study.

### 3.4.2 Implementation Stage

The implementation stage converts the design into trainable models and an integrated prototype.

1. **DAE pretraining** – train the Conv1D encoder–decoder on noisy/clean GMSK pairs using mean squared error (MSE) loss until reconstruction PSNR converges [8], [38].
2. **AMC training** – train the baseline AMC on raw I/Q inputs for GSM-family modulations (GMSK, GFSK, QPSK), then re-train with DAE outputs as features to create the hybrid pipeline [15], [16].
3. **Pipeline integration** – compare a frozen-DAE setup against joint fine-tuning so that the denoiser can be treated either as a front-end block or as part of an end-to-end model.
4. **Deployment prototyping** – integrate the trained models into a lightweight inference service for UCC spectrum monitoring demonstrations.

**Expected outcome**: a working DAE model, a baseline AMC model, a hybrid DAE–AMC pipeline, and a prototype demonstration path suitable for research presentation.

### 3.4.3 Analysis Stage

The analysis stage determines whether the proposed architecture produces measurable value under the target interference conditions.

1. **UCUSAF edge evaluation** – assess models on held-out SNR bins matching the −90 dBm to −60 dBm coverage-floor conditions [5].
2. **Comparative benchmarking** – compare raw AMC, denoised AMC, and hybrid variants using denoising and classification metrics [8], [15], [16], [38].
3. **Interpretive analysis** – relate the measured improvements to weak-signal monitoring, interference diagnosis, and regulator-facing demonstrations in rural Uganda.

**Expected outcome**: a defensible evidence set showing whether denoising improves AMC performance in the targeted low-SNR scenarios, together with deployment-oriented conclusions.

## 3.5 Model Selection

- **Denoising Autoencoder (DAE)**: a symmetric Conv1D encoder–decoder with three downsampling and three upsampling blocks, each containing batch normalization, PReLU activations, and residual skip connections inspired by the dual-residual DAEs in literature. [20] The bottleneck dimension is 128, encouraging compact latent representations. Total parameters < 500K to support resource-constrained SDR deployment. Training uses AdamW (learning rate 1e‑3, weight decay 1e‑4) with cosine annealing over 100 epochs.
- **AMC Classifier**: a 1D CNN with four convolutional blocks (kernel sizes 3×1 and 5×1), squeeze-and-excitation modules for channel attention, and a softmax output over three GSM-family modulations (GMSK, GFSK, QPSK). Cross-entropy loss and label smoothing help stabilize training.
- **Baselines**: we maintain a "raw AMC" baseline (no denoiser) and, where feasible, re-implement a thresholded autoencoder denoiser as reported by An & Lee to compare approaches. [20]

## 3.6 Data Collection and Preprocessing

1. **Acquisition**: download and verify checksums for RadioML 2018.01A (synthetic benchmark with well-characterized SNR labels) and Kaggle RF Signal Data (real SDR captures providing hardware impairment realism). [11], [12]
2. **Modulation filtering**: extract samples corresponding to GSM-family modulations (GMSK, GFSK) and include QPSK as a reference comparison class.
3. **Segmentation**: segment each recording into fixed-length windows (1024 samples) with 50% overlap to ensure sufficient training examples per class.
4. **Normalization**: perform per-window zero-mean, unit-variance normalization.
5. **Uganda-specific noise injection** [7], [23], [25], [35], [36], [37]: 
   - **Booster noise**: inject high-power wideband impulses at random intervals (duty cycle ~5%) to simulate illegal repeater oscillation
   - **Bizindaalo harmonics**: add narrowband tones at frequencies corresponding to FM harmonics in the 900 MHz band
   - **Rain fade**: apply time-varying attenuation following a Markov model of tropical storm conditions
6. **SNR range**: target −12 dB to +12 dB with emphasis on the −90 dBm to −60 dBm UCUSAF coverage-floor region. Each sample is tagged with its applied impairments for downstream analysis.

## 3.7 Model Development and Training

**Phase 1 – DAE Pretraining**
- Input: noisy I/Q tensor; Target: clean tensor from the same sample.
- Loss: MSE plus a small L1 penalty on latent activations to discourage trivial copying.
- Optimizer: AdamW with gradient clipping (1.0) and mixed-precision to accelerate training.
- Early stopping monitors validation PSNR at SNR = −6 dB to ensure low-SNR fidelity.

**Phase 2 – AMC Baseline**
- Train the 1D CNN classifier on raw inputs using cross-entropy loss and class-balanced sampling.
- Learning rate: 3e‑4 with cosine decay; batch size 512; training for 80 epochs or until validation accuracy plateaus.

**Phase 3 – Hybrid DAE–AMC**
- Freeze or fine-tune the DAE encoder and feed its denoised outputs into the AMC classifier.
- Compare two settings: (a) frozen DAE (acts as feature preprocessor) and (b) joint fine-tuning (end-to-end backpropagation with a smaller LR on the DAE).

All experiments will log metrics via MLflow/W&B and store checkpoints for reproducibility.

## 3.8 Evaluation Metrics

- **Denoising quality**: Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) between original and reconstructed I/Q waveforms.
- **Classification**: Accuracy, macro F1-score, per-class confusion matrices, and calibration plots across SNR bins.
- **Robustness indices**: accuracy-vs-SNR curves, area-under-curve (AUC) for the −12…0 dB region, and an improvement ratio defined as `Accuracy_hybrid − Accuracy_raw` at each SNR.
- **Computational metrics**: FLOPs, parameter counts, and inference latency on CPU/GPU to validate deployability on SDR hardware.

## 3.9 Model Validation

To mitigate overfitting and quantify generalization:

- **Hold-out splits**: 70/15/15 train/validation/test within each dataset, ensuring that specific SNR bins or modulation classes can be withheld for zero-shot testing.
- **Cross-dataset testing**: train on RadioML 2018.01A and test on RF Signal Data as the primary domain-shift check; where feasible, extend validation with MIGOU-MOD for additional over-the-air comparison. [13], [20]
- **SNR-based k-fold CV**: treat each SNR level as a fold; iteratively leave one SNR out during training to evaluate extrapolation performance.
- **Statistical significance**: run each experiment with three random seeds and report mean ± std accuracy; apply paired t-tests when comparing AMC vs. DAE–AMC.

## 3.10 User Interface Development

A lightweight Streamlit/PyQt GUI will:

1. Load stored I/Q snippets or accept live SDR buffers (future work) and visualize the raw constellation, its denoised counterpart, and power spectra.
2. Display model predictions with probability bars, per-class confusion summaries, and SNR estimates.
3. Offer toggles to compare “raw AMC” vs. “DAE–AMC” outputs, helping regulators understand the benefit of preprocessing during demonstrations.

## 3.11 Deployment and Demonstration

Deployment will target a Python microservice (FastAPI) that serves ONNX-exported versions of the DAE and AMC models. The service will expose REST/WebSocket endpoints for ingesting I/Q chunks and returning modulation predictions plus metadata (SNR estimate, denoising confidence). Demonstrations will simulate UCC monitoring workflows: ingest recorded interference events, visualize denoising improvements, and show automated alerts when low-SNR signals become classifiable. Future work may integrate the service with SDR front-ends (e.g., RTL-SDR or USRP) for live enforcement pilots.

## 3.12 Ethical Considerations

All datasets are open-source and redistributed only under their respective licenses (Kaggle Terms of Service, DeepSig EULA, Mendeley Data licenses). We avoid collecting personal or sensitive information and stress that the resulting models are intended for lawful spectrum monitoring and academic exploration. Any deployment with live SDR captures will require operator consent and compliance with Uganda's Communications Act to prevent inadvertent interception of protected communications.

## 3.13 Tools and Software Requirements

- **Languages/Frameworks**: Python 3.11, PyTorch 2.x (primary DL framework), optional TensorFlow for baseline comparisons, NumPy/SciPy for DSP utilities, scikit-learn for metrics, and Matplotlib/Plotly for visualization.
- **Experiment management**: MLflow or Weights & Biases for logging, DVC/Git LFS for dataset versioning, and Docker for environment reproducibility.
- **Hardware**: Training on NVIDIA GPUs (>= 16 GB VRAM) for efficiency, with CPU-only fallbacks for inference. SDR replay experiments will use GNU Radio or DragonOS Focal for signal capture/streaming.
- **Automation**: Makefiles or Fabric scripts to orchestrate preprocessing, training, evaluation, and report generation so that each experiment can be reproduced end-to-end.

---

# References

[1] Uganda Communications Commission, "Telephone Subscriptions Rise to 33.2 Million," *UCC Communications Blog*, Jun. 9, 2023. [Online]. Available: https://uccinfoblog.com/2023/06/09/telephone-subscriptions-rise-to-33-2-million/

[2] Atomic Energy Council, "Radiofrequency Radiation in Uganda," 2022. [Online]. Available: https://www.atomiccouncil.go.ug/non-ionizing-radiation-radiofrequency/

[3] C. Kiiza, "Uganda's Internet Users Hit 13 Million," *ChimpReports*, Mar. 25, 2024. [Online]. Available: https://chimpreports.com/ugandas-internet-users-hit-13-million/

[4] European Investment Bank, "US$40 million European backing for Uganda rural telecom expansion," Press Release, Apr. 11, 2024. [Online]. Available: https://www.eib.org/en/press/all/2024-097-usd40-million-european-backing-for-uganda-rural-telecom-expansion

[5] TechJaja, "UCUSAF: Why is UCC still rolling out own telecom network?" Feb. 6, 2024. [Online]. Available: https://techjaja.com/ucusaf-why-is-ucc-still-rolling-out-own-telecom-network/

[6] Ghana Chamber of Telecommunications, "Mobile Internet Access Still Limited in Africa, Millions Remain Offline," citing GSMA data, 2024. [Online]. Available: https://www.telecomschamber.org/industry-news/mobile-internet-access-still-limited-in-africa-millions-remain-offline/

[7] Uganda Communications Commission, "UCC cracks down on illegal and non-compliant broadcasters," Oct. 21, 2024. [Online]. Available: https://www.ucc.co.ug/ucc-cracks-down-on-illegal-and-non-compliant-broadcasters/

[8] X. Zhang et al., "Dual Residual Denoising Autoencoder with Channel Attention Mechanism for Modulation of Signals," *Sensors*, vol. 23, no. 2, Art. 1023, 2023. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC9861137/

[11] RF Signal Data, Kaggle, 2025. [Online]. Available: https://www.kaggle.com/datasets/suraj520/rf-signal-data

[12] DeepSig, "RadioML 2018.01A Dataset," Kaggle, 2025. [Online]. Available: https://www.kaggle.com/datasets/pinxau1000/radioml2018

[13] R. Utrilla, "MIGOU-MOD: A dataset of modulated radio signals acquired with MIGOU, a low-power IoT experimental platform," Mendeley Data, V1, 2020. [Online]. Available: https://data.mendeley.com/datasets/fkwr8mzndr/1

[14] DeepSig, "RadioML 2016.10A Dataset," 2025. [Online]. Available: https://www.deepsig.ai/datasets/

[15] O. F. Abd-Elaziz, A. M. El-Ghandour, and F. H. Ismail, "Deep Learning-Based Automatic Modulation Classification Using Robust CNN Architecture for Cognitive Radio Networks," *Sensors*, vol. 23, no. 23, Art. 9467, 2023, doi: 10.3390/s23239467.

[16] J. Gao, Z. Zhang, and Y. Zhang, "MoE-AMC: Enhancing Automatic Modulation Classification Performance Using Mixture-of-Experts," *arXiv preprint*, arXiv:2312.02298, 2023. [Online]. Available: https://arxiv.org/abs/2312.02298

[17] J. Jang, J. Pyo, Y.-i. Yoon, and J. Choi, "Meta-Transformer: A Meta-Learning Framework for Scalable Automatic Modulation Classification," *IEEE Access*, vol. 12, pp. 9267–9276, 2024, doi: 10.1109/ACCESS.2024.3352634.

[18] S. Rehman, H. K. Qureshi, and M. Imran, "DL-AMC: Deep Learning for Automatic Modulation Classification," *arXiv preprint*, arXiv:2504.08011, 2025. [Online]. Available: https://arxiv.org/abs/2504.08011

[19] V. Sathyanarayanan, P. Gerstoft, and A. El Gamal, "RML22: Realistic Dataset Generation for Wireless Modulation Classification," *IEEE Trans. Wireless Commun.*, vol. 22, no. 11, pp. 7663–7675, 2023, doi: 10.1109/TWC.2023.3254490.

[20] A. Jagannath and J. Jagannath, "Multi-Task Learning Approach for Modulation and Wireless Signal Classification for 5G and Beyond: Edge Deployment via Model Compression," *Physical Communication*, vol. 54, Art. 101793, 2022, doi: 10.1016/j.phycom.2022.101793.

[21] H. An and B.-M. Lee, "Robust Automatic Modulation Classification in Low Signal-to-Noise Ratio," *IEEE Access*, vol. 11, pp. 125678–125690, 2023, doi: 10.1109/ACCESS.2023.3321108.

[22] M. Faysal, J. Chen, and P. Balaprakash, "DenoMAE: A Multimodal Autoencoder for Denoising Modulation Signals," *arXiv preprint*, arXiv:2501.11538, 2025. [Online]. Available: https://arxiv.org/abs/2501.11538

[23] UCC, "Market Performance Report," *THE COMMUNICATIONS BLOG*, Jan. 2026. [Online]. Available: https://uccinfoblog.com/tag/market-performance-report/

[24] UCC, "UCC OPERATION TARGETS ILLEGAL RADIO STATIONS, BOOSTERS AND MEGAPHONES," *THE COMMUNICATIONS BLOG*, Jul. 5, 2023. [Online]. Available: https://uccinfoblog.com/2023/07/05/ucc-operation-targets-illegal-radio-stations-boosters-and-megaphones/

[25] UCC, "PUBLIC NOTICE: SIGNAL INTERFERENCE ARISING OUT OF USAGE OF NETWORK REPEATERS – 'BOOSTERS'," *THE COMMUNICATIONS BLOG*, Jul. 26, 2021. [Online]. Available: https://uccinfoblog.com/2021/07/26/public-notice-signal-interference-arising-out-of-usage-of-network-repeaters-boosters/

[26] "Analysis of Rain Attenuation Effects on the Communication System Quality of Satellite VSAT IP Services," *Komdigi*, Jan. 2026. [Online]. Available: https://bpostel.komdigi.go.id/index.php/bpostel/article/view/409/

[27] "Quality of Service Findings for Indoor Mobile Voice Telephony and Data Services in Uganda," *UCC/AWS*, Jan. 2026. [Online]. Available: https://newvision-media.s3.amazonaws.com/cms/b295e426-f529-44b6-928d-330d8b911.pdf

[35] Uganda Communications Commission, "Public Notice: Signal Interference Arising Out of Usage of Network Repeaters - 'Boosters'," Jul. 26, 2021. [Online]. Available: https://uccinfoblog.com/2021/07/26/public-notice-signal-interference-arising-out-of-usage-of-network-repeaters-boosters/

[36] ITU-R, *Recommendation ITU-R P.530-18: Propagation Data and Prediction Methods Required for the Design of Terrestrial Line-of-Sight Systems*, Sep. 2021. [Online]. Available: https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.530-18-202109-I!!PDF-E.pdf

[37] ITU-R, *Recommendation ITU-R P.838-3: Specific Attenuation Model for Rain for Use in Prediction Methods*, 2005. [Online]. Available: https://www.itu.int/rec/R-REC-P.838-3-200503-P/en

[38] X. Gao, X. Xu, D. Li, X. Liu, J. Yang, and D. Zhai, "Enhancing Noise Robustness in Few-Shot Automatic Modulation Classification via Complex-Valued Autoencoders," *Electronics*, vol. 15, no. 3, Art. 674, 2026. [Online]. Available: https://www.mdpi.com/2079-9292/15/3/674
