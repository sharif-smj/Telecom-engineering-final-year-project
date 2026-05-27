# UCC Proposal Feedback Action Matrix - 29 April 2026

Source package: `/Users/sharif/telecom/final-year-project/NCC/feedback/WhatsApp Chat - UCC paper.zip`

Extracted workspace: `/Users/sharif/telecom/final-year-project/NCC/feedback/whatsapp-ucc-paper-extracted/`

## Bottom Line

The supervisors are not asking for small polishing. They are pushing the document toward a more formal UCC grant proposal with:

- Lecturer-led Lot 3 positioning, not student-led Lot 1.
- Stronger problem statement with at least two citations.
- Literature review before methodology.
- Methodology expanded with additive noise, interference models, and equations.
- Separate data collection and data analysis sections.
- Data collection tools placed in appendices.
- A clearer prototype/end-user pathway beyond ML modelling.
- Explicit explanation of how the project benefits UCC.
- Budget and Gantt chart revised using Dr. Dickson's previous proposal as the pattern.
- Dr. Ephrance's one-page CV included.

## Key Feedback and Required Actions

| Feedback source | What they said | Meaning | Required action |
| --- | --- | --- | --- |
| Dr. Dickson, 27 Apr 10:42 | If the principal investigator is undergraduate, it falls under Lot 1; if PI is lecturer, it can fall under Lot 3. | Our current Lot 3 document cannot keep students as principal investigators. | Change PI to lecturer, most likely Dr. Dickson or Dr. Ephrance. Put students as research assistants / student researchers. |
| Dr. Ephrance, 27 Apr 14:54 and Dr. Dickson, 27 Apr 15:02 | We need Lot 3 because Lot 1 has little money; change the PI, students can be research assistants. | Confirms strategic route: ambitious Lot 3. | Revise title page, summary, team section, CV headings, and budget justification to lecturer-led inter-university collaboration. |
| Dr. Dickson, 27 Apr 10:45 | The "student-led but positioned..." sentence cannot be in project summary; summary is more like an abstract. | The project summary should not contain administrative strategy. | Rewrite project summary as a clean abstract: problem, aim, method, expected outputs, UCC relevance. |
| Dr. Dickson, 27 Apr 10:48 | "Get me the actual problem statement with at least two references." | Problem statement is too general and under-cited. | Replace it with a sharper cited problem: access/usage gap, weak GSM service, interference/signal integrity, cost/diagnostic gap. Cite UCC call plus UCC interference/action sources or sector-access sources. |
| Dr. Dickson, 27 Apr 10:50 | "So how are we solving the issue?" | The proposal does not make the intervention concrete enough. | Add a short "Proposed Solution" paragraph after problem statement or in objectives: DAE + AMC + Android-SDR prototype workflow + UCC technical report. |
| Dr. Dickson, 27 Apr 11:11 | Literature cannot come last after methodology. | Current section order is academically weak even if UCC list was loose. | Move Literature Review before Methodology. |
| Dr. Dickson, 27 Apr 11:15 | Discuss additive noise and selected interference models: how they work, why use them, and equations. | Methodology needs engineering depth. | Add equations for received signal model, AWGN, SNR, narrowband interference, wideband disturbance, attenuation/fading, reconstruction loss, classification metrics. |
| Dr. Dickson, 27 Apr 11:15 | "This section looks like literature review." | Some theory inside methodology should be redistributed. | Put background justification in literature review; keep methodology procedural and mathematical. |
| Dr. Dickson, 27 Apr 11:34 | Separate data collection from data analytics/data analysis. | Methodology structure needs clearer research design. | Create separate sections: Data Collection Methods, Data Collection Tools, Data Analysis Plan. |
| Dr. Dickson, 27 Apr 11:34 | Expound the three data collection tools; put copies in appendix if available. | The tools are too abstract. | Add Appendix A: Experiment Log Template; Appendix B: Results Extraction Sheet; Appendix C: Field Observation Checklist. |
| Dr. Dickson, 27 Apr 11:52 | Use previous document for budget, work schedule, methodology, etc. | We need the more formal proposal style from `00003075-temp_...docx`. | Adapt its Gantt chart, budget headings, CV style, REC/UNCST ethics language, and work schedule format. |
| Dr. Ephrance, 27 Apr 14:53 | We must write somewhere how work benefits UCC. | UCC value must be explicit, not implied. | Add a section or subsection: "Direct Benefit to UCC" with policy, spectrum/interference, access planning, UCUSAF, reporting, and prototype outputs. |
| Dr. Ephrance, 28 Apr 18:26 and 29 Apr 11:53 | Are we going only to modelling? ML projects have prototypes/end users/mobile platform beyond the model. | The proposal must show a practical artifact, not just ML experiments. | Add deliverable: low-cost Android + SDR field-support prototype workflow, plus end-user scenario around VHTs/farmer cooperatives/basic phone service observation. |
| Dr. Ephrance, 28 Apr 18:27 | Budget should use the currency used in the call. | Since the UCC call uses UGX, budget should remain UGX. | Keep all budget amounts in UGX only. |
| Dr. Dickson, 28 Apr 21:57 | Edit budget and Gantt chart. | Current budget/workplan needs stronger grant format. | Rebuild the budget and Gantt chart from template logic but keep totals in UGX. |
| Dr. Dickson, 28 Apr 21:58 | Look at problem statement and remove highlighted content. | The highlighted problem-statement wording is likely too broad and not evidence-backed. | Remove weak generic phrasing and replace with cited, direct problem statement. |
| Dr. Ephrance, 28 Apr 18:32 | Shared one-page CV. | Her CV must be included. | Add Ephrance CV using the PDF details. |

## Extracted CV Details To Use

### Dr. Ephrance Eunice Namugenyi

- Location/contact: Kampala, Uganda; email shown in resume.
- Academic profile: postdoctoral-level researcher and lecturer with experience in Data Communications, Software Engineering, Communication Networks, wireless communications, IoT systems, edge intelligence, and adaptive network architectures.
- Education: PhD in Software Engineering, Makerere University, 2021-2026; MSc in Data Communications and Software Engineering, Makerere University, 2015-2019; BSc in Telecommunications Engineering, Makerere University, 2008-2012.
- Appointment: Lecturer, Electrical and Electronics Engineering, Kyambogo University, 2013-present.
- Relevant teaching/supervision: Communication Systems, Satellite Communications, Computing for Engineers, wireless networks, IoT systems, embedded systems.
- Relevant projects: embedded software for Kayoola Bus lighting systems; SMS-based Results Request System for low-connectivity environments; IEEE SIGHT COVID-19 Innovations grant; adaptive network switching and LoRa/energy-aware network research.
- Technical skills: Python, C, C++, ML for network optimization, NS-3, MATLAB, IoT, LPWAN, Wi-Fi, GSM/LTE, edge computing.

### Dr. Dickson Mugerwa

From Dr. Dickson's template CV:

- Role in template: co-PI.
- Institution: Department of Electrical and Electronics Engineering, Kyambogo University.
- Qualifications: PhD in Radio and Communication Engineering; MSc in Radio Science, Electronics and Communication; BSc in Information Technology and Computing; Diploma in Computer Science.
- Experience: Senior ICT Tech (Academics), Kyambogo University; embedded systems consultant; doctoral researcher; ICT technician.
- Publications include IoT LoRa clustering, multi-hop communication, WSN protocols, and computational systems work.

## Revision Architecture For Next Draft

Recommended new section order:

1. Title Page
2. Project Summary
3. Introduction
4. Problem Statement
5. Aim and Objectives
6. Research Questions
7. Justification and Direct Benefit to UCC
8. Literature Review
9. Methodology
10. Data Collection Methods and Tools
11. Data Analysis Plan
12. Ethical and Regulatory Considerations
13. Implementation Plan and Gantt Chart
14. Contribution to Cross-Cutting Issues
15. Expected Outputs and Deliverables
16. Sustainability and Prototype Pathway
17. Budget and Budget Justification
18. Curriculum Vitae of the Research Team
19. References
20. Appendices: Data Collection Tools

## Main Decision To Confirm

To keep the UGX 66,600,000 ambition under Lot 3, the proposal should no longer present Ssemujju Sharif Abdukarim as Principal Investigator. The cleaner structure is:

- Principal Investigator: Dr. Dickson Mugerwa or Dr. Ephrance Eunice Namugenyi.
- Co-Principal Investigator: the other lecturer/supervisor.
- Student researchers/research assistants: Ssemujju Sharif Abdukarim and Kisige Tom Derrick.
- Lead institution: Kyambogo University.
- Collaborating institution: Makerere University CoCIS only if that collaboration is formally acceptable.

If this PI/collaboration structure cannot be confirmed, the safer administrative route is Lot 1 with the lower UGX 20,000,000 ceiling.

