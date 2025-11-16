KYAMBOGO UNIVERSITY
FINAL YEAR PROJECT PROPOSAL

TITLE: Intelligent Modulation Classification and Signal Denoising Using Machine Learning
NAME: SSEMUJJU SHARIF ABDUKARIM
REG. NO: 18/U/ETD/181/GV
DEPARTMENT: ELECTRICAL AND ELECTRONICS ENGINEERING
SUPERVISOR: Dr. Dickson Mugerwa

---

## Table of Contents

**Chapter One – Introduction**
1.1 Background of the Study
1.2 Problem Statement
1.3 Main Objective of the Study
1.3.1 Specific Objectives
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

**Chapter Three – Methodology**
3.0 Overview of Existing AMC Systems
3.1 Introduction to Methods
3.2 Research Paradigm
3.3 Research Approach
3.4 Research Strategy
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

## 1.1 Background of the Study

Uganda’s wireless ecosystem is expanding at a pace that now tests the limits of its finite spectrum. The Uganda Communications Commission (UCC) reports that fixed and mobile subscriptions climbed to 33.2 million by the end of 2022 (33.1 million of them mobile), pushing national tele-density to 77 lines per 100 people while mobile data traffic doubled from 217 million GB in 2020 to 421.5 million GB in 2022.[^1] Supporting that demand already requires more than 4300 macro base-station sites serving 30.6 million SIM cards, yet most towers must juggle multiple technologies and operators across crowded carrier bands.[^2]

Coverage expansion is lagging behind demand. Even as Kampala and other urban hubs deploy 4G/5G, national coverage remains uneven: only 31 % of the population has 4G service (versus 77 % for 3G) and just 24 % of Uganda’s landmass receives 4G coverage.[^3] The European Investment Bank additionally notes that “only 65% of Uganda has mobile network coverage,” triggering a publicly financed plan to add 506 rural towers, while the UCC’s UCUSAF program has mapped 117 sub-counties where 3G signal strength remains below −90 dBm and is subsidizing new solar-powered sites to lift coverage above the 30 % threshold.[^4][^5]

The usage gap is therefore stark: GSMA estimates that 30 million Ugandans—roughly 62 % of the population—still lack mobile internet access despite this network build-out, keeping many communities in persistently low-SNR, high-interference environments where receivers operate near the sensitivity limit.[^6] These conditions create a technical imperative for radios that can separate overlapping transmissions, compensate for multipath fading, and maintain service quality for both commercial connectivity and public services such as digital payments and e-government.

Automatic Modulation Classification (AMC) is central to that imperative because it enables spectrum surveillance, cognitive radio, and interference mitigation without prior coordination. The UCC’s 2024 enforcement campaign against illegal broadcasters explicitly warned that unauthorized transmitters “affect critical users of frequencies such as aviation and some security services,” underscoring the need for SDR-based monitors that can reliably detect weak or covert emitters even when noise floors rise.[^7]

Machine Learning (ML) and Deep Learning (DL) have improved AMC accuracy by learning representations directly from raw I/Q samples, but their performance still collapses at low SNR unless the front end is robust to noise. Recent work on Denoising Autoencoders (DAEs), such as the dual-residual channel-attention architecture by Zhang et al., demonstrated 67–75 % classification-accuracy gains over conventional denoisers across −12 dB to 8 dB SNR when used ahead of an AMC classifier.[^8] This project therefore proposes an integrated DAE–AMC pipeline tailored to Uganda’s congested spectrum and rural coverage gaps so that receivers can recover intelligible constellations before classification, sustain accuracy across fluctuating channels, and provide regulators with noise-resilient situational awareness.

---

## 1.2 Problem Statement

Licensed networks in Uganda are already operating at the edge of their coverage obligations—UCUSAF targets sub-counties where 3G signal strength hovers around the −90 dBm minimum, meaning receivers routinely experience marginal SINR while sharing congested bands with multiple operators.[^5] UCC QoS drive tests confirm the consequence: in 2019 more than 70 % of MTN and Airtel call failures were traced to same-frequency interference, and blocked-call rates in towns such as Jinja spiked to 34 % against the 2 % regulatory ceiling because radios could not maintain reliable modulation recognition under noise.[^9] The regulator also documented illegal rooftop links and signal boosters that inject additional RF noise, threatening aviation and public-safety channels unless spectrum monitors can correctly classify weak emitters in real time.[^7][^9]

Conventional AMC workflows—built on handcrafted cumulants or unenhanced neural classifiers—struggle in this environment; even mixture-of-expert architectures benchmarked across diverse SNRs only reach about 71.76 % average accuracy, far below what UCC’s fault-repair and QoS mandates require when signals drop below 0 dB.[^10] Without a front-end that denoises I/Q streams before classification, enforcement teams and network operators cannot meet national targets for rapid fault resolution, interference mitigation, or safe spectrum sharing. This project therefore tackles the specific gap by pairing a denoising autoencoder with an AMC head so that modulation decisions remain stable under Uganda’s low-SNR, interference-heavy conditions.

---

## 1.3 Main Objective of the Study

To design, implement, and empirically evaluate a hybrid Denoising Autoencoder–Automatic Modulation Classification (DAE–AMC) pipeline that can suppress interference in noisy I/Q streams and maintain accurate modulation decisions across the low-SNR channel conditions observed in Ugandan networks.

### 1.3.1 Specific Objectives

1. To review AMC and DAE literature with emphasis on low-SNR mitigation strategies relevant to Ugandan channel conditions.
2. To architect and train a Conv1D DAE module that reconstructs clean I/Q samples from noisy inputs representative of UCUSAF –90 dBm thresholds.
3. To implement a supervised AMC classifier that ingests both raw and DAE-cleaned signals and quantify feature preservation.
4. To benchmark the integrated DAE–AMC pipeline against the standalone AMC using accuracy, F1-score, confusion matrices, and robustness across controlled SNR sweeps.
5. To document deployment considerations for embedding the prototype within SDR-based monitoring or enforcement workflows in Uganda.

---

## 1.4 Scope of the Study

This investigation is limited to offline experimentation using publicly available I/Q corpora: the RF Signal Data collection on Kaggle (real SDR captures), the DeepSig RadioML 2018.01A benchmark, the MIGOU-MOD over-the-air IoT dataset, and the historical RadioML 2016.10A archive.[^11][^12][^13][^14] No live SDR capture, hardware deployment, or regulatory compliance testing is undertaken. Instead, controlled Additive White Gaussian Noise (AWGN) levels and SNR sweeps are injected to emulate the −90 dBm edge conditions faced by UCUSAF sites while training two components: (i) an unsupervised Conv1D DAE that reconstructs clean waveforms and (ii) a supervised AMC head that consumes both raw and denoised samples across the BPSK, QPSK, 8PSK, 16QAM, 64QAM, AM, and FM families. Findings and metrics therefore reflect these synthetic SNR scenarios rather than live network measurements; cross-dataset tests quantify how well the pipeline generalizes to other public benchmarks.

---

## 1.5 Significance of the Study

By keeping modulation recognition stable when SNR collapses toward UCUSAF’s −90 dBm coverage floor, the proposed DAE–AMC pipeline offers a practical response to the congestion, interference, and illegal-transmitter issues highlighted by UCC QoS audits and GSMA connectivity surveys. It bolsters spectrum surveillance, rural broadband, aviation safety, and mobile money services that depend on reliable radio links in Uganda’s noisy bands.[^5][^6][^7][^9]

---

## 1.6 Justification

UCC’s mandates to resolve 95 % of faults within 24 hours and suppress harmful interference cannot be met if AMC models deliver barely 71.76 % accuracy once SNR drops below 0 dB.[^9][^10] Pairing a learnable denoiser with the classifier reduces reliance on brittle handcrafted features, aligns with NDPIII/NBP directives to adopt spectrum-efficient techniques, and creates an adaptable software upgrade that fits existing SDR monitoring chains without requiring additional spectrum allocations.[^4][^10]

---

## 1.7 Conceptual Framework

```
RF Signal (I/Q Data) -> Denoising Autoencoder (DAE) -> AMC Classifier -> Predicted Modulation
```

---

# CHAPTER TWO – LITERATURE REVIEW

## 2.0 Introduction

This chapter surveys the evolution of Automatic Modulation Classification (AMC) research from classical statistical detectors to recent deep learning architectures and denoising front-ends. Emphasis is placed on studies that quantify performance below 0 dB SNR—conditions that mirror Uganda’s interference-heavy spectrum—and on dataset innovations that support reproducible benchmarking.

## 2.1 Overview of Modulation Classification

AMC underpins spectrum awareness for cognitive radio, electronic warfare, and national regulators because it infers a waveform’s modulation type without a priori coordination.[^1][^5] Reliable classification enables dynamic spectrum access, enforcement against illicit transmitters, and automated routing of traffic through increasingly congested infrastructure. Consequently, AMC techniques must perform well even when radios operate at UCUSAF’s −90 dBm edges or experience intentional interference.

## 2.2 Traditional Modulation Classification Approaches

Before the recent wave of deep learning, modulation recognition relied on likelihood-ratio tests, cumulant and cyclostationary feature extraction, or other manually engineered statistics. These detectors remain analytically elegant but degrade sharply when SNR falls below 0 dB or when multipath and oscillator offsets distort the assumed signal model—exactly the impairments documented by UCC in congested Ugandan deployments.[^9] Their brittleness under unknown channels motivates the transition toward data-driven feature learning.

## 2.3 Machine Learning in Modulation Classification

Deep learning has reshaped AMC by learning discriminative features directly from raw I/Q samples. Recent contributions demonstrate tangible low-SNR gains:

- Abd-Elaziz et al. (2023) designed a Robust CNN with parallel asymmetric kernels and residual skip connections that achieved 96.5 % accuracy at 0 dB and 86.1 % at −2 dB across nine modulations impaired by AWGN, Rician fading, and clock offsets, substantially outperforming prior CNN baselines.[^15]
- Zhang et al. (2023) proposed MoE-AMC, a mixture-of-experts framework that routes signals to Transformer-based low-SNR experts or ResNet high-SNR experts via a gating network, yielding ~71.8 % averaged accuracy across −20…18 dB on RadioML2018.01A—about 10 % higher than single-expert models.[^16]
- Meta-learning approaches such as the 2024 Meta-Transformer leverage transformer encoders and few-shot learning to adapt rapidly to unseen modulations, maintaining superior accuracy across all SNRs on RadioML2018.01A while sharing reproducible code for community validation.[^17]
- Rehman et al. (2025) introduced DL-AMC, which converts I/Q streams into eye diagrams and classifies them with ResNet variants, overcoming the 10–48 % accuracy ceiling that DBN, RNN, and CLDNN architectures exhibited near −10 dB SNR.[^18]
- Jagannath et al. (2022) closed the “reality gap” by validating CNN-based multi-task AMC on a USRP SDR testbed, demonstrating >98 % accuracy on seven modulations in live over-the-air experiments and highlighting the importance of heterogeneous training that includes hardware impairments.[^20]

Collectively, these works show that architectural customization (optimized CNN blocks, expert routing, attention) and domain-adaptive validation are essential for deployments in noisy environments like Uganda’s shared bands.

## 2.4 Performance Metrics and Datasets

Accuracy, F1-score, confusion matrices, and accuracy-vs-SNR curves remain standard evaluation metrics; however, reproducibility now hinges on diverse datasets. Beyond the Kaggle RF Signal Data and DeepSig RadioML 2016/2018 corpora used in this study,[^11][^12][^14] researchers increasingly rely on:

- MIGOU-MOD, which provides over-the-air IoT captures from the MIGOU low-power platform for assessing energy-constrained AMC scenarios.[^13]
- RML22, a data-centric successor to RadioML that corrects generation artifacts, injects more realistic channel models, and publishes the full Python generation stack so others can regenerate or adapt the benchmark.[^19]

These datasets enable controlled AWGN sweeps, realistic multipath simulations, and OTA validation, allowing rigorous comparison of DAE–AMC pipelines across signal families.

## 2.5 Research Gaps

Despite progress, several gaps persist. First, even the best-performing architectures suffer accuracy collapses once SNR dips below −5 dB, leaving regulators blind to weak interferers.[^15][^16][^18] Second, most published metrics come from simulations; only a handful of OTA demonstrations (e.g., Jagannath et al.) quantify the domain shift introduced by real hardware and channel impairments.[^20] Third, few studies integrate denoisers tightly with AMC or explore how denoising impacts regulatory workflows such as UCC’s interference crackdowns.[^7][^21][^22] This project addresses the latter by coupling a Conv1D DAE to the classifier and benchmarking the combined pipeline under Ugandan-inspired SNR profiles.

## 2.6 Research Questions

1. How do modern CNN, mixture-of-experts, and transformer architectures extend AMC robustness when SNR approaches the −10…0 dB regimes common in Ugandan deployments?[^15][^16][^17][^18]
2. Which publicly available datasets (RadioML 2016/2018, RML22, MIGOU-MOD, Kaggle RF Signal Data) best capture the impairments observed by UCC, and how should they be combined to train denoising front-ends?[^11][^12][^13][^14][^19]
3. To what extent does inserting a DAE ahead of the classifier recover low-SNR accuracy relative to standalone AMC models reported in the literature?[^8][^21][^22]

## 2.7 Denoising Autoencoders for Signal Enhancement

Denoising front-ends have emerged as an effective countermeasure when raw I/Q features are overwhelmed by interference. Zhang et al.’s dual-residual DAE with channel attention improved AMC accuracy by up to 75 % across −12…8 dB SNR,[^8] demonstrating that reconstructing constellation geometry before classification materially benefits downstream decisions. Faysal et al. (2025) extended this idea with DenoMAE, a multimodal denoising masked autoencoder that treats noise as a separate modality; after fine-tuning, it sustained 77.5 % accuracy at −10 dB—roughly 22 % higher than the same classifier without denoising pre-training.[^22] Complementary work by An and Lee (2023) introduced a thresholded autoencoder denoiser triggered by a lightweight SNR predictor; this combination delivered ~70 % relative accuracy gains on low-SNR samples while avoiding unnecessary processing for high-SNR inputs in IEEE Access experiments.[^21] These findings justify the DAE–AMC architecture explored in this project and provide design cues for gating strategies that conserve energy on SDR deployments.

---

# CHAPTER THREE – METHODOLOGY

## 3.0 Overview of Existing AMC Systems

Conventional AMC pipelines in Ugandan networks still depend on handcrafted cumulants, likelihood tests, and static DSP filters that assume high SNR and stable oscillators. As laid out in Chapter 2, those assumptions collapse in practice: the UCC documents interference-driven failures, and recent research shows that even sophisticated CNNs lose accuracy near −5 dB unless they incorporate noise-aware architectures.[^7][^15] Our methodology therefore replaces hand-engineered features with a learnable denoising preprocessor (DAE) followed by a supervised AMC head, trained and evaluated under the same SNR ranges that stress current deployments.

## 3.1 Introduction to Methods

We adopt an experimental research design anchored in reproducible data processing and quantitative benchmarking. The workflow spans four pillars:

1. Curate I/Q datasets representing both simulated and real captures (RF Signal Data, RadioML 2016.10A, RadioML 2018.01A, MIGOU-MOD, and RML22 subsets).[^^11][^12][^13][^14][^19]
2. Generate controlled SNR scenarios (−12…+18 dB) that mimic UCUSAF edge conditions and inject additional impairments such as Rician fading, frequency offsets, and symbol timing jitter.
3. Train a Conv1D denoising autoencoder (unsupervised) and a 1D CNN AMC classifier (supervised) both separately and as a combined pipeline.
4. Compare the hybrid system against a standalone AMC baseline using statistically rigorous metrics (accuracy-vs-SNR curves, F1-score, and confusion matrices) and document findings for eventual SDR deployment.

## 3.2 Research Paradigm

A post-positivist paradigm guides the study: hypotheses about low-SNR robustness are tested empirically, while acknowledging that experimental results are provisional and must be corroborated by replication on additional datasets or field captures. All code, hyperparameters, and preprocessing steps will be version-controlled to enable independent verification.

## 3.3 Research Approach

The approach is data-driven and bifurcated into complementary learning stages:

- **Unsupervised denoising stage** – the DAE learns to reconstruct clean I/Q tensors from synthetically corrupted inputs, capturing noise statistics without label supervision.
- **Supervised classification stage** – the AMC head consumes either raw I/Q data or DAE outputs to predict modulation classes via cross-entropy training. This stage emphasizes generalization across modulation families and SNR bins.

Coupling the two stages allows us to quantify how much low-SNR accuracy is recovered by denoising when the classifier architecture itself remains modest (edge-friendly).

## 3.4 Research Strategy

1. **Dataset harmonization** – convert each dataset into a unified tensor format (length-1024 complex samples, normalized amplitude) and split into train/validation/test partitions with stratification by modulation and SNR.
2. **Noise modeling** – inject AWGN, Rician fading, oscillator offsets, and impulsive noise to emulate the Ugandan RF environment; maintain a metadata log describing each corruption level.
3. **DAE pretraining** – train the Conv1D encoder–decoder on noisy/clean pairs using mean squared error (MSE) loss until reconstruction PSNR converges.
4. **AMC training** – train the baseline AMC on raw I/Q inputs, then re-train with DAE outputs as features to create the hybrid pipeline.
5. **Cross-dataset evaluation** – assess the models on held-out SNR bins, on unseen modulation families (e.g., test on MIGOU-MOD after training on RadioML), and on OTA-style splits to expose domain shift.
6. **Deployment prototyping** – integrate the trained models into a lightweight inference service/GUI for visualization and regulatory demonstrations.

## 3.5 Model Selection

- **Denoising Autoencoder (DAE)**: a symmetric Conv1D encoder–decoder with three downsampling and three upsampling blocks, each containing batch normalization, PReLU activations, and residual skip connections inspired by the dual-residual DAEs in literature.[^8] The bottleneck dimension is 128, encouraging compact latent representations. Training uses AdamW (learning rate 1e‑3, weight decay 1e‑4) with cosine annealing over 100 epochs.
- **AMC Classifier**: a 1D CNN with four convolutional blocks (kernel sizes 3×1 and 5×1), squeeze-and-excitation modules for channel attention, and a softmax output over the target modulation set. Cross-entropy loss and label smoothing help stabilize training. A mixture-of-experts variant (lightweight gating between “low-SNR” and “high-SNR” sub-paths) will also be explored to mirror MoE-AMC’s benefits at extreme SNRs.[^16]
- **Baselines**: we maintain a “raw AMC” baseline (no denoiser) and, where feasible, re-implement a thresholded autoencoder denoiser as reported by An & Lee to compare gating strategies.[^21]

## 3.6 Data Collection and Preprocessing

1. **Acquisition**: download and verify checksums for RF Signal Data (real SDR captures), RadioML 2016.10A/2018.01A (synthetic benchmarks), MIGOU-MOD (IoT OTA traces), and optional RML22 slices for realism.[^^11][^12][^13][^14][^19]
2. **Segmentation**: segment each recording into fixed-length windows (1024 samples) with 50 % overlap to ensure sufficient training examples per class.
3. **Normalization**: perform per-window zero-mean, unit-variance normalization; optionally apply IQ imbalance correction based on dataset metadata.
4. **Label encoding**: unify modulation labels across datasets (e.g., “QPSK,” “OQPSK,” “BPSK,” etc.) and map them to numeric IDs.
5. **Augmentation**: add AWGN at SNR levels {−12, −10, −8, −6, −4, −2, 0, 2, 4, 6, 8, 12, 18 dB}, apply random carrier frequency offsets (±5 ppm), and simulate multipath via tapped-delay lines. Each sample is tagged with its applied impairments for downstream analysis.

## 3.7 Model Development and Training

**Phase 1 – DAE Pretraining**
- Input: noisy I/Q tensor; Target: clean tensor from the same sample.
- Loss: MSE plus a small L1 penalty on latent activations to discourage trivial copying.
- Optimizer: AdamW with gradient clipping (1.0) and mixed-precision to accelerate training.
- Early stopping monitors validation PSNR at SNR = −6 dB to ensure low-SNR fidelity.

**Phase 2 – AMC Baseline**
- Train the 1D CNN classifier on raw inputs using cross-entropy loss and class-balanced sampling.
- Learning rate: 3e‑4 with cosine decay; batch size 512; training for 80 epochs or until validation accuracy plateaus.

**Phase 3 – Hybrid DAE–AMC**
- Freeze or fine-tune the DAE encoder and feed its denoised outputs into the AMC classifier.
- Compare two settings: (a) frozen DAE (acts as feature preprocessor) and (b) joint fine-tuning (end-to-end backpropagation with a smaller LR on the DAE).
- Evaluate mixture-of-experts gating by duplicating the final convolutional block and training SNR-aware experts similar to MoE-AMC.[^16]

All experiments will log metrics via MLflow/W&B and store checkpoints for reproducibility.

## 3.8 Evaluation Metrics

- **Denoising quality**: Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) between original and reconstructed I/Q waveforms.
- **Classification**: Accuracy, macro F1-score, per-class confusion matrices, and calibration plots across SNR bins.
- **Robustness indices**: accuracy-vs-SNR curves, area-under-curve (AUC) for the −12…0 dB region, and an improvement ratio defined as `Accuracy_hybrid − Accuracy_raw` at each SNR.
- **Computational metrics**: FLOPs, parameter counts, and inference latency on CPU/GPU to validate deployability on SDR hardware.

## 3.9 Model Validation

To mitigate overfitting and quantify generalization:

- **Hold-out splits**: 70/15/15 train/validation/test within each dataset, ensuring that specific SNR bins or modulation classes can be withheld for zero-shot testing.
- **Cross-dataset testing**: train on RadioML 2018.01A, test on MIGOU-MOD or RF Signal Data to measure domain shift, echoing the OTA demonstrations outlined in Section 2.[^20]
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

All datasets are open-source and redistributed only under their respective licenses (Kaggle Terms of Service, DeepSig EULA, Mendeley Data licenses). The study avoids collecting personal or sensitive information and stresses that the resulting models are intended for lawful spectrum monitoring and academic exploration. Any deployment with live SDR captures will require operator consent and compliance with Uganda’s Communications Act to prevent inadvertent interception of protected communications.

## 3.13 Tools and Software Requirements

- **Languages/Frameworks**: Python 3.11, PyTorch 2.x (primary DL framework), optional TensorFlow for baseline comparisons, NumPy/SciPy for DSP utilities, scikit-learn for metrics, and Matplotlib/Plotly for visualization.
- **Experiment management**: MLflow or Weights & Biases for logging, DVC/Git LFS for dataset versioning, and Docker for environment reproducibility.
- **Hardware**: Training on NVIDIA GPUs (>= 16 GB VRAM) for efficiency, with CPU-only fallbacks for inference. SDR replay experiments will use GNU Radio or DragonOS Focal for signal capture/streaming.
- **Automation**: Makefiles or Fabric scripts to orchestrate preprocessing, training, evaluation, and report generation so that each experiment can be reproduced end-to-end.

---

# References

[^1]: Uganda Communications Commission, “Telephone Subscriptions Rise to 33.2 Million,” *UCC Communications Blog*, 9 June 2023, https://uccinfoblog.com/2023/06/09/telephone-subscriptions-rise-to-33-2-million/.
[^2]: Atomic Energy Council, “Radiofrequency Radiation in Uganda,” 2022, https://www.atomiccouncil.go.ug/non-ionizing-radiation-radiofrequency/.
[^3]: Christopher Kiiza, “Uganda’s Internet Users Hit 13 Million,” *ChimpReports*, 25 March 2024, https://chimpreports.com/ugandas-internet-users-hit-13-million/.
[^4]: European Investment Bank, “US$40 million European backing for Uganda rural telecom expansion,” Press Release, 11 April 2024, https://www.eib.org/en/press/all/2024-097-usd40-million-european-backing-for-uganda-rural-telecom-expansion.
[^5]: TechJaja, “UCUSAF: Why is UCC still rolling out own telecom network?” 6 February 2024, https://techjaja.com/ucusaf-why-is-ucc-still-rolling-out-own-telecom-network/.
[^6]: Ghana Chamber of Telecommunications, “Mobile Internet Access Still Limited in Africa, Millions Remain Offline,” citing GSMA data, 2024, https://www.telecomschamber.org/industry-news/mobile-internet-access-still-limited-in-africa-millions-remain-offline/.
[^7]: Uganda Communications Commission, “UCC cracks down on illegal and non-compliant broadcasters,” 21 October 2024, https://www.ucc.co.ug/ucc-cracks-down-on-illegal-and-non-compliant-broadcasters/.
[^8]: Xiaolin Zhang et al., “Dual Residual Denoising Autoencoder with Channel Attention Mechanism for Modulation of Signals,” *Sensors* 23, no. 1023 (2023), https://pmc.ncbi.nlm.nih.gov/articles/PMC9861137/.
[^9]: *Low SNR, Interference & Illegal Transmissions in Uganda’s Wireless Networks*, internal research brief, 2025 (synthesizing UCC QoS surveys and enforcement bulletins).
[^10]: *Quantifying the Impact of Low SNR and Interference on Wireless Service Resilience*, internal research brief, 2025 (summarizing NDPIII/NBP mandates and AMC accuracy benchmarks).
[^11]: RF Signal Data, Kaggle, accessed 2025, https://www.kaggle.com/datasets/suraj520/rf-signal-data.
[^12]: DeepSig Dataset: RadioML 2018.01A, Kaggle, accessed 2025, https://www.kaggle.com/datasets/pinxau1000/radioml2018.
[^13]: Ramiro Utrilla, “MIGOU-MOD: A dataset of modulated radio signals acquired with MIGOU, a low-power IoT experimental platform,” Mendeley Data V1, 2020, https://data.mendeley.com/datasets/fkwr8mzndr/1.
[^14]: DeepSig, “RadioML 2016.10A Dataset,” https://www.deepsig.ai/datasets/, accessed 2025.
[^15]: O. F. Abd-Elaziz, A. M. El-Ghandour, and F. H. Ismail, “Deep Learning-Based Automatic Modulation Classification Using Robust CNN Architecture for Cognitive Radio Networks,” *Sensors*, vol. 23, no. 23, 2023, Art. 9467, doi:10.3390/s23239467.
[^16]: J. Gao, Z. Zhang, and Y. Zhang, “MoE-AMC: Enhancing Automatic Modulation Classification Performance Using Mixture-of-Experts,” *arXiv preprint*, 2023, https://arxiv.org/abs/2312.02298.
[^17]: J. Jang, J. Pyo, Y.-i. Yoon, and J. Choi, “Meta-Transformer: A Meta-Learning Framework for Scalable Automatic Modulation Classification,” *IEEE Access*, vol. 12, 2024, pp. 9267–9276, doi:10.1109/ACCESS.2024.3352634.
[^18]: S. Rehman, H. K. Qureshi, and M. Imran, “DL-AMC: Deep Learning for Automatic Modulation Classification,” *arXiv preprint*, 2025, https://arxiv.org/abs/2504.08011.
[^19]: V. Sathyanarayanan, P. Gerstoft, and A. El Gamal, “RML22: Realistic Dataset Generation for Wireless Modulation Classification,” *IEEE Trans. Wireless Commun.*, vol. 22, no. 11, 2023, pp. 7663–7675, doi:10.1109/TWC.2023.3254490.
[^20]: A. Jagannath and J. Jagannath, “Multi-Task Learning Approach for Modulation and Wireless Signal Classification for 5G and Beyond: Edge Deployment via Model Compression,” *Physical Communication*, vol. 54, 2022, Art. 101793, doi:10.1016/j.phycom.2022.101793.
[^21]: H. An and B.-M. Lee, “Robust Automatic Modulation Classification in Low Signal-to-Noise Ratio,” *IEEE Access*, vol. 11, 2023, pp. 125678–125690, doi:10.1109/ACCESS.2023.3321108.
[^22]: M. Faysal, J. Chen, and P. Balaprakash, “DenoMAE: A Multimodal Autoencoder for Denoising Modulation Signals,” *arXiv preprint*, 2025, https://arxiv.org/abs/2501.11538.
