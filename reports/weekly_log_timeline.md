# Weekly Log Timeline

This timeline converts the project into an 8-week working schedule for weekly logging.

Assumption: the proposal was presented and accepted on Friday, March 27, 2026, so Week 1 starts on Monday, March 30, 2026, as the first full working week after approval.

Note: this 8-week timeline supersedes the earlier longer draft schedule for weekly reporting purposes.

## Weekly Schedule

| Week | Dates | Phase | Planned Focus | Expected Output |
| --- | --- | --- | --- | --- |
| 1 | Mar 30 - Apr 5, 2026 | Project Framing and Planning | Confirm scope after proposal approval, align objectives, refine weekly work plan, and organize sources | Confirmed project scope and working plan |
| 2 | Apr 6 - Apr 12, 2026 | Literature Review and Evidence Consolidation | Review literature on GSM denoising, AMC, low-SNR classification, and Uganda-specific interference context | Literature review notes and validated source base |
| 3 | Apr 13 - Apr 19, 2026 | Data Acquisition and Exploratory Analysis | Acquire core datasets, inspect modulation classes, SNR coverage, and sample structure | Core datasets collected and EDA summary |
| 4 | Apr 20 - Apr 26, 2026 | Preprocessing and Interference Modelling | Prepare I/Q samples, define train/test splits, and model booster oscillation, harmonic leakage, and rain fade effects | Preprocessed data pipeline and interference modelling plan |
| 5 | Apr 27 - May 3, 2026 | Denoising Model Development | Design and implement the denoising autoencoder and run initial denoising experiments | Working DAE prototype and early denoising outputs |
| 6 | May 4 - May 10, 2026 | Classification Model and Integration | Implement the AMC classifier, train the baseline, and integrate the DAE-AMC pipeline | Baseline AMC and integrated hybrid pipeline |
| 7 | May 11 - May 17, 2026 | Evaluation and Comparative Testing | Compare hybrid and baseline models using accuracy, F1 score, confusion matrices, and SNR-wise trends | Evaluation results and comparative analysis |
| 8 | May 18 - May 24, 2026 | Documentation and Submission Preparation | Consolidate findings, update the report, prepare weekly logs, and finalize presentation/report materials | Final documentation package and submission-ready materials |

## Phase Summary

1. Weeks 1-2: Project Framing, Planning, and Literature Review
2. Weeks 3-4: Data Acquisition, Exploratory Analysis, and Preprocessing
3. Weeks 5-6: Model Development and Pipeline Integration
4. Weeks 7-8: Evaluation, Documentation, and Submission Preparation

## Expanded Weekly Notes

### Week 3: Data Acquisition and Exploratory Analysis

**Dates:** Apr 13 - Apr 19, 2026

**Main objective:** build the working dataset base for the implementation stage and understand the structure of the signal data before preprocessing and modelling.

**Planned activities**

1. Identify and collect the core datasets to be used in the study, especially the primary GSM-family modulation datasets already selected for the proposal.
2. Verify dataset accessibility, file organization, metadata quality, and class labels so the data pipeline starts from a clean and traceable source base.
3. Inspect the modulation classes present in each dataset and confirm which classes are relevant to the project scope and the AMC baseline.
4. Examine the available SNR range in the datasets and note whether the low-SNR cases are sufficient for the weak-signal conditions targeted by the study.
5. Inspect the sample structure of the I/Q data, including sequence length, channel representation, labeling format, and any obvious imbalance or corruption issues.
6. Produce an exploratory data analysis summary covering class distribution, SNR distribution, and any dataset limitations that may affect denoising or classification experiments.
7. Record any dataset gaps that may require augmentation, synthetic interference injection, or restricted scope during implementation.

**Expected outputs**

1. A confirmed list of core datasets for the study.
2. A short EDA summary describing modulation classes, SNR coverage, and sample format.
3. Notes on data quality issues, class imbalance, or low-SNR limitations.
4. A clear handoff into Week 4 preprocessing and interference modelling.

**Suggested weekly log wording**

During Week 3, the project focused on data acquisition and exploratory analysis. Core datasets for GSM-family modulation classification were identified and organized for use in the study. The dataset structure was reviewed to understand the available modulation classes, signal-to-noise-ratio coverage, sample format, and labeling scheme. An exploratory analysis summary was prepared to highlight class distribution, low-SNR availability, and any dataset limitations that may affect preprocessing and subsequent denoising and classification experiments. The outcome of this week was a confirmed dataset base and an evidence-driven understanding of the data characteristics needed for the next implementation stages.

## Suggested Weekly Log Structure

Use this structure each week:

1. Week number and dates
2. Tasks planned
3. Tasks completed
4. Sources or datasets consulted
5. Challenges encountered
6. Next week's plan
