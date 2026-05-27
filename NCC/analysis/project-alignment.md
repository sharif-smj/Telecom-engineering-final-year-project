# Project Alignment

Status date: April 23, 2026

This file converts the NCC and UCC research into a working position for our paper.

## Locked positioning decision

Recommended paper identity:

- Core identity: denoising weak and noisy GSM edge signals to improve effective service reach in rural Uganda
- Technical engine: Denoising Autoencoder plus modulation-recognition pipeline
- User value: better interpretation of weak/noisy signals for operators and regulators
- Policy value: supports underserved-area service quality, interference diagnosis, and smarter coverage-extension decisions

## Track choice

- Primary track: Track 1 - Digital Infrastructure, Connectivity and Future Networks
- Backup track: Track 4 - Artificial Intelligence, Data Science and Trustworthy Technology
- Secondary policy relevance: Track 8 - Policy, Regulation, Cybersecurity and Innovation Ecosystems

Reason:

- The winning framing should center on infrastructure and service reach.
- The AI component should read as the mechanism, not the headline.

## Strongest one-line pitch

This paper shows how signal denoising can recover usable GSM intelligence at the noisy edge of coverage, helping extend the effective reach of rural mobile service in Uganda without immediately relying on new physical infrastructure.

## Strongest grant-winning narrative

Uganda still has weak, interference-prone service zones where conventional monitoring and classification degrade at exactly the point where operators and regulators most need reliable evidence. A denoising-first pipeline can recover more usable signal structure from low-SNR GSM environments, improving troubleshooting, interference interpretation, and cell-edge service decisions in a way that is practical, software-forward, and aligned to inclusive digital access.

## Claims we should make

- improves effective or usable service reach
- improves edge-of-coverage signal interpretation
- recovers more usable information from weak/noisy GSM signals
- supports interference diagnosis and rural coverage decision-making
- offers a software-forward path to better value from existing network assets

## Claims we should avoid

- physically extends radio range
- increases tower coverage footprint
- guarantees better live QoS in deployed networks
- outperforms all existing AMC methods in real-world Uganda field trials
- solves rural connectivity on its own
- claims on-device or no-new-towers performance without verified artifacts

Reason:

- We do not yet have live field deployment evidence.
- The supervisor concept note introduces promising practical claims, but they remain mixed-status until verified.
- Overclaiming will weaken credibility with mixed academic and sector reviewers.

## Supervisor concept note integration

What the note is good for:

- strengthening the paper's practical motivation
- sharpening the future deployment path
- showing why a software-defined coverage story could matter to UCC and operators

What the note is not yet good for:

- populating the NCC results section with device or field numbers
- driving the title toward `on-device` unless device artifacts exist
- justifying `without new towers` as a proven result

Current handling rule:

- Use concept-note deployment language in the discussion, impact paragraph, and future pilot section.
- Keep all unverified supervisor-note numbers out of the title, abstract, results, and conclusion.

Evidence basis:

- `SUP-01`
- [analysis/supervisor-concept-note-claim-ledger.md](/Users/sharif/telecom/final-year-project/NCC/analysis/supervisor-concept-note-claim-ledger.md)
- [analysis/practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md)

## Title direction

Best title shape:

- service or infrastructure outcome first
- method second
- Uganda context explicit

Recommended leading options:

1. Extending Effective GSM Coverage in Rural Uganda Through Signal Denoising
2. Signal Denoising for Improved Edge-of-Coverage GSM Service in Rural Uganda
3. Improving Effective GSM Service Reach in Rural Uganda Using Denoising Autoencoders
4. Denoising Weak GSM Signals to Support Rural Coverage Extension in Uganda
5. Recovering Usable Cell-Edge GSM Signals for Rural Uganda Coverage Expansion

Do not promote into the lead title yet:

- `On-Device GSM Signal Denoising...`
- `...Without New Towers`

Reason:

- Those phrases are strategically attractive, but they currently depend on unverified practical claims from the supervisor note. Source basis: `SUP-01`; [analysis/practical-claim-verification.md](/Users/sharif/telecom/final-year-project/NCC/analysis/practical-claim-verification.md).

## Abstract posture

The abstract should move in this order:

1. Uganda problem
2. Why existing service or monitoring breaks down
3. What we built
4. What data or test setting we used
5. What improved
6. Why operators, regulators, or underserved communities should care

Avoid opening with acronyms or architecture names.

Additional rule:

- The abstract can mention a future on-device or pilot path only in the last sentence, and only as a next-step implication rather than a measured current result. Source basis: `SUP-01`.

## Must-have evidence in the eventual paper

- the Uganda service/interference problem
- why low-SNR matters in the target setting
- the denoising-plus-recognition pipeline in clear block form
- at least one measurable system benefit beyond plain classification accuracy
- a deployment path for UCC / UCUSAF / operator use
- an honest limitations paragraph
- a claim ledger review before the abstract and conclusion are finalized

## Reviewer hooks by audience

- For infrastructure reviewers: software-first improvement of cell-edge utility
- For AI reviewers: denoising tied to a real communications problem
- For UCC reviewers: supports access, QoS interpretation, and compliance-related monitoring
- For operator-minded reviewers: potentially cheaper evidence and troubleshooting in underserved zones
- For innovation ecosystem reviewers: local problem, practical stack, visible adoption path

## What we are really selling

We are not selling a classifier.

We are selling a practical way to get more usable service intelligence out of weak and noisy GSM conditions in Uganda.

That is a stronger NCC 2026 story because it sits directly at the intersection of:

- market-driven innovation
- access and usage gaps
- affordability pressure
- interference and compliance realities
- rural connectivity value

## Sources

- analysis/ncc-landscape.md
- analysis/winner-patterns.md
- analysis/ucc-priority-map.md
- analysis/supervisor-concept-note-claim-ledger.md
- analysis/practical-claim-verification.md
- UCC NCC 2026 launch: https://www.ucc.co.ug/ucc-launches-10th-national-conference-on-communications-to-drive-market-driven-innovation/
- NCC 2025 winners: https://ncc.co.ug/blog/harnessing-digital-innovation-to-power-sustainable-local-solutions-for-ugandas-development-goals-10/harnessing-digital-innovation-to-power-sustainable-local-solutions-for-ugandas-development-goals-7
- SUP-01: sources/background/supervisor-concept-note-2026-04-23.md
