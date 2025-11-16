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

Traditional AMC systems break down in noisy and multipath environments because they rely on handcrafted features that are easily distorted. Furthermore, their performance significantly degrades when the received signal’s signal-to-noise ratio (SNR) falls below a certain threshold. This study addresses that limitation by introducing a machine learning-based denoising autoencoder (DAE) front-end that cleans the signal before classification, followed by an Automatic Modulation Classifier (AMC) that identifies the modulation type.

---

## 1.3 Main Objective of the Study

To develop a hybrid Denoising Autoencoder and Automatic Modulation Classification (DAE-AMC) system that enhances signal quality and accurately identifies modulation schemes under different SNR and channel environments.

### 1.3.1 Specific Objectives

1. To review existing AMC and DAE-based denoising techniques.
2. To design a DAE network that reduces channel noise and preserves key modulation features in I/Q data.
3. To implement and train an AMC model on both raw and denoised data.
4. To evaluate the combined DAE-AMC system against standard metrics such as accuracy, F1-score, and robustness across varying SNR conditions.

---

## 1.4 Scope of the Study

This study focuses on supervised learning for AMC and unsupervised learning for signal denoising. It will classify modulation schemes such as BPSK, QPSK, 8PSK, 16QAM, 64QAM, AM, and FM. The dataset is the RF Signal Data (Kaggle), containing real I/Q samples from SDR hardware. The DAE module will be trained on synthetically degraded versions of these samples (by adding Gaussian noise) and evaluated based on how much it improves AMC accuracy at low SNRs.

---

## 1.5 Significance of the Study

The proposed hybrid DAE-AMC system improves reliability in signal classification tasks under poor channel conditions. By adding the DAE, the study contributes to robust and noise-resilient RF systems, improving demodulation accuracy, signal clarity, and downstream model confidence. This has practical implications in cognitive radio, spectrum monitoring, and defense communication, where signals are often corrupted by environmental noise.

---

## 1.6 Justification

While most AMC systems assume clean, ideal signals, real-world transmissions suffer from severe distortions. A denoising autoencoder preprocessor allows the model to reconstruct near-clean signals, boosting classifier robustness without manually engineered filters. This hybrid DAE-AMC structure blends unsupervised feature learning (for denoising) and supervised classification (for modulation identification) into one intelligent, adaptive system.

---

## 1.7 Conceptual Framework

```
        RF Signal (I/Q Data)
                │
                ▼
       Denoising Autoencoder (DAE)
                │
                ▼
     Machine Learning Classifier (AMC)
                │
                ▼
      Predicted Modulation Scheme
```

[^1]: Uganda Communications Commission, “Telephone Subscriptions Rise to 33.2 Million,” *UCC Communications Blog*, 9 June 2023, https://uccinfoblog.com/2023/06/09/telephone-subscriptions-rise-to-33-2-million/.
[^2]: Atomic Energy Council, “Radiofrequency Radiation in Uganda,” 2022, https://www.atomiccouncil.go.ug/non-ionizing-radiation-radiofrequency/.
[^3]: Christopher Kiiza, “Uganda’s Internet Users Hit 13 Million,” *ChimpReports*, 25 March 2024, https://chimpreports.com/ugandas-internet-users-hit-13-million/.
[^4]: European Investment Bank, “US$40 million European backing for Uganda rural telecom expansion,” Press Release, 11 April 2024, https://www.eib.org/en/press/all/2024-097-usd40-million-european-backing-for-uganda-rural-telecom-expansion.
[^5]: TechJaja, “UCUSAF: Why is UCC still rolling out own telecom network?” 6 February 2024, https://techjaja.com/ucusaf-why-is-ucc-still-rolling-out-own-telecom-network/.
[^6]: Ghana Chamber of Telecommunications, “Mobile Internet Access Still Limited in Africa, Millions Remain Offline,” citing GSMA data, 2024, https://www.telecomschamber.org/industry-news/mobile-internet-access-still-limited-in-africa-millions-remain-offline/.
[^7]: Uganda Communications Commission, “UCC cracks down on illegal and non-compliant broadcasters,” 21 October 2024, https://www.ucc.co.ug/ucc-cracks-down-on-illegal-and-non-compliant-broadcasters/.
[^8]: Xiaolin Zhang et al., “Dual Residual Denoising Autoencoder with Channel Attention Mechanism for Modulation of Signals,” *Sensors* 23, no. 1023 (2023), https://pmc.ncbi.nlm.nih.gov/articles/PMC9861137/.

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

