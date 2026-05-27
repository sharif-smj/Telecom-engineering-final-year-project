# Final Year Project Weekly Logbook Entries

Project title: GSM Signal Denoising and Modulation Classification for Rural Uganda

Note: This version covers ten progress-log weeks ending on the 23 May 2026 submission reminder window. The final five weeks correspond to the April-May implementation, NCC/UCC proposal alignment, preliminary simulation, and presentation work.

---

## Week 1

**From:** 16 March 2026  
**To:** 21 March 2026

### Activities and Discussions:

* Reviewed the approved final-year project topic on GSM signal denoising and Automatic Modulation Classification (AMC).
* Discussed the need to frame the work as a telecommunications engineering problem rather than only a machine learning project.
* Identified the main project direction: weak/noisy GSM-family signal interpretation for rural and edge-of-coverage service conditions.
* Started reviewing the project report draft and identifying sections that needed stronger technical grounding.
* Discussed the expected project outputs with the supervisor, including simulation results, presentation slides, and a final technical report.

### Challenges Faced:

* The project scope was broad and could easily be confused with physical network coverage expansion.
* There was uncertainty on how to explain the denoising model to supervisors who may be more familiar with classical telecom blocks than machine learning terms.

### Solutions Found/Proposed:

* Reframed the project as a signal-processing chain: received noisy signal, denoising block, cleaned signal, AMC classifier, and final decision.
* Agreed to keep the work evidence-based and avoid claiming physical tower-range extension unless supported by actual results.

### Next Steps:

* Continue literature review on AMC, low-SNR signal classification, and denoising autoencoders.
* Refine the problem statement and project objectives.

### Supervisor Comments and Signature:

---

## Week 2

**From:** 23 March 2026  
**To:** 28 March 2026

### Activities and Discussions:

* Reviewed literature on GSM signal denoising, low-SNR wireless communication, and deep learning-based AMC.
* Studied GSM-family I/Q signal representation and how modulation classes can be represented in baseband simulations.
* Reviewed Uganda-specific communication service issues, including rural access, weak service, interference, SMS, USSD, and Mobile Money dependence.
* Discussed with the supervisor how the project could remain relevant to Uganda's communications sector without overstating deployment claims.
* Started identifying suitable public I/Q datasets and possible synthetic signal generation methods.

### Challenges Faced:

* Difficulty finding publicly available datasets that directly match Ugandan GSM weak-service conditions.
* Some literature focused on generic modulation classification and did not directly connect to rural service-quality problems.

### Solutions Found/Proposed:

* Proposed using public I/Q datasets where available and synthetic GSM-family/proxy signal generation for controlled testing.
* Decided to keep Uganda-specific service context in the background and justification, while keeping the technical evaluation controlled and reproducible.

### Next Steps:

* Consolidate the literature review into the project report.
* Identify modulation classes and SNR ranges for controlled simulations.

### Supervisor Comments and Signature:

---

## Week 3

**From:** 30 March 2026  
**To:** 4 April 2026

### Activities and Discussions:

* Refined the project aim and specific objectives around denoising-assisted AMC.
* Reviewed possible datasets and signal classes, including GMSK-like, QPSK, 8PSK, and 16QAM examples.
* Discussed the need for a baseline AMC path so that denoising-assisted results can be compared fairly.
* Began outlining the methodology section for the final report and presentation.
* Reviewed safe receive-only boundaries for any future SDR or Android-based demonstration.

### Challenges Faced:

* Some proposed ideas sounded like live network boosting or direct improvement of calls and USSD, which would not be technically or legally defensible at this stage.
* The project needed a clearer difference between final-year-project implementation and future prototype/pilot work.

### Solutions Found/Proposed:

* Defined the current project as controlled simulation and offline I/Q signal analysis.
* Treated Android/SDR work as a future or optional prototype pathway rather than the main evidence base.
* Kept the evaluation focused on signal reconstruction, classification accuracy, macro F1-score, confusion matrices, and SNR-wise trends.

### Next Steps:

* Prepare the controlled simulation plan.
* Begin setting up Python scripts for signal generation and impairment modelling.

### Supervisor Comments and Signature:

---

## Week 4

**From:** 6 April 2026  
**To:** 11 April 2026

### Activities and Discussions:

* Reviewed information on the NCC 2026 conference and how the project could fit into telecommunications research themes.
* Compared the final-year project direction with NCC-style expectations, including technical novelty, practical relevance, and local communication-sector value.
* Discussed with supervisors how the project could be positioned for NCC or UCC-related research opportunities.
* Reviewed previous NCC/UCC-style material and noted the importance of making the work practical, interdisciplinary, and locally grounded.
* Began separating measured project evidence from expected future outcomes so that the project remains academically honest.

### Challenges Faced:

* The NCC conference direction required a concise research-paper style, while supervisors were also interested in a broader practical proposal direction.
* Some proposed claims, such as improved coverage range or field performance, could not be used as completed results without supporting evidence.

### Solutions Found/Proposed:

* Proposed using the NCC direction to strengthen the research framing while keeping the final-year project focused on simulation and implementation.
* Created a working distinction between completed evidence, literature-backed claims, and expected future prototype outcomes.
* Agreed that the project should emphasize weak-signal interpretation and denoising-assisted classification rather than claiming live network improvement.

### Next Steps:

* Continue preparing the NCC/UCC-aligned project narrative.
* Start building the simulation pipeline for noisy GSM-family I/Q samples.

### Supervisor Comments and Signature:

---

## Week 5

**From:** 13 April 2026  
**To:** 18 April 2026

### Activities and Discussions:

* Reviewed UCC Research Support to Academia call requirements and discussed how the project could support access, user experience, digital innovation, and spectrum/interference interpretation.
* Continued discussions with the supervisors on whether the work should be presented as an NCC conference paper, a practical UCC proposal, or both.
* Reviewed the interdisciplinary nature of the project: telecommunications engineering, RF signal processing, machine learning, low-cost prototyping, and rural service-quality analysis.
* Drafted initial wording for the practical project concept while keeping final-year-project evidence separate from future pilot claims.
* Identified possible data collection tools and physical tools, including Android measurement phones, basic GSM phones, receive-only SDR dongles, antennas, OTG adapters, laptops, and field observation checklists.

### Challenges Faced:

* The project direction expanded beyond a normal final-year project because of NCC/UCC proposal discussions.
* There was risk of mixing proposal-level expected outcomes with actual implementation results.

### Solutions Found/Proposed:

* Maintained a clear evidence boundary: completed project work would be reported as simulation or planned implementation, while field prototype claims would remain future work.
* Kept the project title and technical core unchanged: GSM signal denoising and modulation classification for rural Uganda.
* Agreed to keep basic GSM/2G users central to the problem statement while using Android/SDR only as research and measurement tools.

### Next Steps:

* Finalize the simulation design.
* Generate noisy GSM-family signal samples for preliminary testing.

### Supervisor Comments and Signature:

---

## Week 6

**From:** 21 April 2026  
**To:** 25 April 2026

### Activities and Discussions:

* Reviewed literature on GSM signal denoising and Automatic Modulation Classification (AMC).
* Studied GSM-family I/Q signal structures and low-SNR conditions.
* Discussed project scope, NCC conference positioning, and simulation methodology with the supervisors.
* Reviewed supervisor input on a practical GSM signal denoising concept for rural Uganda and identified which claims could be treated as expected outcomes only.
* Started setting up the simulation environment using Python and machine learning/data-processing libraries.

### Challenges Faced:

* Difficulty obtaining suitable GSM low-SNR datasets.
* Limited understanding of preprocessing techniques for noisy I/Q samples.
* Need to balance supervisor expectations for a practical NCC/UCC proposal with what the final-year project could actually prove.

### Solutions Found/Proposed:

* Used synthetic GSM-family signal generation for controlled simulations.
* Reviewed IEEE papers and tutorials on denoising autoencoders and signal classification.
* Proposed a claim-audit approach so that unverified performance figures would not be presented as completed results.

### Next Steps:

* Generate noisy GSM-family signal samples.
* Develop baseline AMC simulation pipeline.
* Continue aligning project wording with supervisor feedback for NCC/UCC discussions.

### Supervisor Comments and Signature:

---

## Week 7

**From:** 28 April 2026  
**To:** 2 May 2026

### Activities and Discussions:

* Generated simulated modulation classes including QPSK, 8PSK, GMSK-like, and 16QAM signals.
* Introduced AWGN noise and interference into GSM-family I/Q samples.
* Tested preliminary modulation classification on noisy signals.
* Held discussions with the supervisors on signal preprocessing approaches and the need for a visible prototype pathway.
* Reviewed feedback from Dr. Dickson and Dr. Ephrance on making the proposal more practical, interdisciplinary, and useful for UCC/NCC-style review.

### Challenges Faced:

* High classification error rates under low SNR conditions.
* Inconsistencies in simulation parameters across modulation classes.
* Supervisor feedback required clearer explanation of how the Android/SDR prototype would work without falsely claiming live network improvement.

### Solutions Found/Proposed:

* Normalized input signal windows and adjusted SNR ranges.
* Standardized simulation parameters for all modulation schemes.
* Defined the prototype pathway as receive-only Android on-device inference using prepared I/Q samples or optional SDR capture, not as a call/USSD booster.

### Next Steps:

* Design denoising front-end architecture.
* Compare noisy and reconstructed signal outputs.
* Update project materials to show the signal-processing pipeline more clearly.

### Supervisor Comments and Signature:

---

## Week 8

**From:** 5 May 2026  
**To:** 9 May 2026

### Activities and Discussions:

* Developed a preliminary denoising model/workflow for GSM-family signal reconstruction.
* Tested denoising performance under different interference conditions.
* Generated waveform and constellation plots for noisy and denoised signals.
* Reviewed denoising-assisted AMC workflow with the supervisors.
* Incorporated feedback from NCC/UCC proposal discussions, including clearer methodology, data collection tools, and practical hardware requirements.

### Challenges Faced:

* Residual interference still appeared in denoised outputs.
* Long simulation processing time during model training and parameter testing.
* The supervisors requested clearer data collection tools, including both research instruments and physical hardware.

### Solutions Found/Proposed:

* Tuned denoising parameters and optimized training/simulation settings.
* Reduced unnecessary processing layers to improve execution speed.
* Listed physical tools such as Android measurement phones, basic GSM phones, receive-only SDR dongles, antennas, USB OTG adapters, power banks, and laptops, while keeping the project receive-only.

### Next Steps:

* Evaluate classification accuracy after denoising.
* Prepare preliminary simulation results.
* Continue refining the methodology and results presentation.

### Supervisor Comments and Signature:

---

## Week 9

**From:** 12 May 2026  
**To:** 16 May 2026

### Activities and Discussions:

* Compared noisy baseline AMC performance with denoising-assisted classification.
* Recorded classification accuracy and macro-F1 metrics at different SNR levels.
* Analyzed confusion matrices and waveform improvements.
* Prepared preliminary presentation slides showing denoising results.
* Discussed with supervisors how the project should be presented for assessment, including NCC/UCC relevance, practical value, and limitations.

### Challenges Faced:

* Some modulation classes remained difficult to distinguish at very low SNR.
* Needed better visualization of denoising improvements.
* The project still needed to avoid overstating preliminary simulation results as completed field validation.

### Solutions Found/Proposed:

* Improved feature extraction and refined signal plotting methods.
* Focused analysis on SNR levels with the greatest denoising improvement.
* Added clear result-status wording: preliminary controlled simulation, not field deployment evidence.

### Next Steps:

* Finalize preliminary simulation results.
* Start drafting discussion and conclusion sections.
* Prepare project slides in the required final-year presentation structure.

### Supervisor Comments and Signature:

---

## Week 10

**From:** 19 May 2026  
**To:** 23 May 2026

### Activities and Discussions:

* Completed preliminary controlled simulations for GSM signal denoising and modulation classification.
* Documented denoising-assisted accuracy improvements over the noisy baseline under controlled SNR conditions.
* Prepared final-year project presentation slides following the required assessment headings: title, background, problem statement, objectives, justification, significance, scope, literature review, methodology, results, discussion, conclusion, recommendations, and references.
* Added APA-style in-text citations and a references slide to strengthen the academic quality of the presentation.
* Discussed future implementation steps, final report structure, and presentation readiness with the supervisors.

### Challenges Faced:

* Need for stronger datasets and more complete DAE training.
* Difficulty validating results under real field conditions within the available time.
* Need to prepare submission materials quickly while keeping claims technically defensible.

### Solutions Found/Proposed:

* Recommended using additional datasets and extended training in the next implementation phase.
* Maintained the project within controlled receive-only simulation scope.
* Used preliminary controlled results to support the project direction while clearly identifying limitations and future work.

### Next Steps:

* Continue improving the denoising model.
* Complete final report writing and presentation rehearsals.
* Prepare for final project assessment and supervisor endorsement.

### Supervisor Comments and Signature:

---
