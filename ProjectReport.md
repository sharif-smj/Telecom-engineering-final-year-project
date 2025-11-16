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

This chapter reviews previous studies related to automatic modulation classification, including traditional statistical techniques, modern ML-based models, and denoising autoencoder applications.

## 2.1 Overview of Modulation Classification

AMC enables communication receivers to recognize the modulation format of received signals. It is vital for non-cooperative communication systems, spectrum monitoring, and cognitive radio.

## 2.2 Traditional Modulation Classification Approaches

These include likelihood ratio tests, cumulant-based analysis, and cyclostationary detection. Although mathematically rigorous, they degrade under noise and fading conditions.

## 2.3 Machine Learning in Modulation Classification

Modern systems employ ML and DL architectures, such as CNNs and RNNs, to learn directly from I/Q data. CNNs excel at spatial pattern recognition in constellation maps, while LSTMs capture temporal dependencies.

## 2.4 Performance Metrics and Datasets

Datasets such as RadioML 2016.10A and the Kaggle RF Signal Data provide I/Q samples labeled by modulation type. Key evaluation metrics include accuracy, F1-score, and confusion matrices across varying SNR levels.

## 2.5 Research Gaps

Existing systems rarely handle low-SNR conditions or real hardware impairments. Few works combine denoising and classification in one integrated pipeline.

## 2.6 Research Questions

1. How can ML improve robustness of modulation classification under noise?
2. Which denoising strategy best preserves signal structure for classification?
3. What is the accuracy improvement achieved using DAE-preprocessed data?

## 2.7 Denoising Autoencoders for Signal Enhancement

Autoencoders learn compressed signal representations and can reconstruct clean waveforms from noisy inputs. Denoising Autoencoders (DAEs) trained on noisy/clean pairs significantly improve modulation recognition accuracy, particularly below 0 dB SNR. Studies show CNN-DAE front-ends improve AMC performance by 10–20% under real-world noise conditions.

---

# CHAPTER THREE – METHODOLOGY

## 3.0 Overview of Existing AMC Systems

Existing AMC systems rely on handcrafted features or complex likelihood functions that fail under realistic conditions. This project introduces a supervised ML-based system enhanced with a DAE preprocessor.

## 3.1 Introduction to Methods

An experimental and quantitative approach will be used. The study will train and test the DAE and AMC modules using the RF Signal Data dataset.

## 3.2 Research Paradigm

The study adopts a post-positivist paradigm focused on empirical testing, reproducibility, and quantitative evaluation.

## 3.3 Research Approach

A data-driven approach integrating unsupervised (DAE) and supervised (AMC) learning stages will be used.

## 3.4 Research Strategy

1. Dataset exploration.
2. Noise augmentation and preprocessing.
3. Train DAE for signal reconstruction.
4. Train AMC on clean and denoised data.
5. Evaluate comparative performance.

## 3.5 Model Selection

* DAE: Conv1D encoder-decoder with MSE loss.
* AMC: 1D CNN classifier with cross-entropy loss.
* Comparison between AMC alone and DAE+AMC pipeline to measure improvement.

## 3.6 Data Collection and Preprocessing

The Kaggle RF dataset will be used. Noisy versions will be created synthetically by injecting AWGN at various SNRs. Normalization and windowing will be applied.

## 3.7 Model Development and Training

Phase 1: Train DAE using MSE loss to reconstruct clean signals.
Phase 2: Train AMC using softmax classifier.
Phase 3: Test hybrid system on unseen noisy data.

## 3.8 Evaluation Metrics

Denoising: MSE, PSNR.
Classification: Accuracy, F1-score, confusion matrix.
Combined: Improvement ratio (AMC_DAE – AMC_raw).

## 3.9 Model Validation

Validation uses unseen SNR levels and modulation classes. Cross-validation ensures generalization.

## 3.10 User Interface Development

A simple GUI will visualize noisy vs. denoised constellations and display predicted modulation types.

## 3.11 Deployment and Demonstration

The final model will be deployed in a Python-based app to classify stored or live SDR-captured signals.

## 3.12 Ethical Considerations

All data are open-source and used for academic purposes only. Proper citations will be provided.

## 3.13 Tools and Software Requirements

Python 3.11, TensorFlow/PyTorch, NumPy, scikit-learn, Matplotlib, and Jupyter Notebook.

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
