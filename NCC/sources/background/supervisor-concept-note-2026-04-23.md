# Supervisor Practical Concept Note for GSMA/UCC

Status date: April 23, 2026

Citation key:

- `SUP-01`

Source status:

- User-provided in this thread as a supervisor-supplied practical concept note.
- Treat as a primary strategic input for positioning and pilot design.
- Do not treat the numeric claims below as verified NCC evidence until they are backed by artifacts or primary external sources.

## Title

On-Device GSM Signal Denoising for Rural Uganda: Extending 2G Coverage Without New Towers

## Problem Statement

Over 11M Ugandans in rural areas rely on 2G GSM for Mobile Money, health alerts, and agricultural info. UCC 2024 data shows 30-40% USSD/SMS failure rates in districts more than 8 km from towers due to low SNR. Building new towers costs $150k-$250k each and requires grid power. Communities need a solution that works with existing handsets and infrastructure today.

## Our Innovation

We are developing lightweight AI models for GSM signal denoising and modulation classification, optimized to run entirely on low-cost Android phones with a $25 software-defined radio dongle. The system is expected to process raw RF in real time, achieving 3-5 dB SNR gain on weak GSM signals. This is equivalent to a 30% increase in usable cell radius with zero changes to operator infrastructure.

## Evidence to Date

Lab evaluation on captured Ugandan GSM bands shows:

- 4.2 dB average SNR improvement
- burst decode rate increase from 52% to 84% at -102 dBm

Models will be quantized to less than 1.2 MB total and run at less than 40 ms latency on Tecno Spark phones common in rural Uganda. A field pre-test is to be carried out in Wakiso District to show call setup success improvements expected from 9/20 to 16/20 attempts at a known weak-signal site.

## Proposed Pilot: 6-8 Months, 3 Districts

Deploy:

- 20 kits (Android phone + SDR + antenna = $95 per unit) to Village Health Teams and farmer cooperatives in Iganga, Kiryandongo, and Kabale.

Measure:

- USSD completion rate
- SMS delivery time
- user-reported "could communicate" versus baseline

Target:

- 25% reduction in failed sessions

Output:

- open-source Android APK
- training manual in English/Luganda
- technical report for UCC on coverage extension via software

## Scalability and Sustainability

- Software-only after initial kit.
- APK can be side-loaded to any OTG-capable Android.
- Local tech shops can assemble kits from parts available on William Street, Kampala.
- Training should take 2 hours.
- Total cost to equip 1000 VHTs is projected below $100k versus $200k for one new tower.
- Aligns with UCC Universal Access goals and GSMA's "Innovation for Rural Connectivity" focus.

## Request

We request $18,000 USD (+ the rest of the other budget) to fund pilot kits, transport, field allowances, and 1 research engineer for 6 months.

Co-funding:

- Kyambogo University provides lab, staff time, and model development in-kind.

## Team

- Team with 7+ years in wireless DSP + ML.
- Partners: Kyambogo University DEEE / Makerere University COCIS / Rural Health NGO TBD.
- Team has prior experience deploying SIM800/ESP32 systems for agriculture in Eastern Uganda.

## Conclusion

This is a software-defined coverage solution. It turns the phones already in people's pockets into smarter receivers, delivering immediate impact while 4G/5G roll-out continues. The team can demo live in Kampala with 2 weeks' notice.
