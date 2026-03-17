# Noise and Interference Source Pack

Compiled on 2026-03-17 to strengthen the proposal and `ProjectReport.md` with traceable sources on interference, low-SNR operation, denoising, and Uganda-specific network conditions.

## Repository-Available Sources

### Official or captured Uganda context

1. **UCC telephone subscriptions capture**
   - Local file: `references/ucc_blog_telephone.html`
   - Original source: UCC Communications Blog, "Telephone Subscriptions Rise to 33.2 Million," 9 June 2023.
   - Use for: subscriptions growth, market demand, high-level sector context.

2. **ChimpReports internet-users capture**
   - Local file: `references/chimpreports_internet.html`
   - Original source: ChimpReports, "Uganda's Internet users hit 13 million," 25 March 2024.
   - Use for: internet-access gap framing and connectivity growth context.

3. **UCUSAF / UCC rollout capture**
   - Local file: `references/techjaja_ucusaf.html`
   - Original source: TechJaja, "UCUSAF: Why is UCC rolling out own telecom network?" 6 February 2024.
   - Use for: UCUSAF context, rural rollout framing, `-90 dBm` service-floor language already used in drafting.

4. **UCC illegal broadcasters capture**
   - Local file: `references/ucc_illegal_broadcasters.html`
   - Original source: Uganda Communications Commission, "UCC cracks down on illegal and non-compliant broadcasters," 21 October 2024.
   - Use for: Bizindaalo / illegal broadcaster enforcement and interference-risk framing.

### Internal synthesized markdown sources

5. **Uganda telecom background report**
   - Local file: `reports/Uganda-Mobile-Network-Noise-and-Mitigation.md`
   - Use for: Uganda telecom structure, coverage dynamics, interference vectors, GSM reliance, policy context.

6. **Low-SNR AMC and denoising literature synthesis**
   - Local file: `reports/LowSNR_AMC_Denoising_Research.md`
   - Use for: low-SNR AMC benchmarks, denoising front-end rationale, model comparisons, recent literature.

7. **Service resilience / enforcement framing**
   - Local file: `reports/LowSNR_Service_Resilience.md`
   - Use for: spectrum enforcement framing, low-SNR service impact, QoS pressure, UCUSAF interpretation.

## New External Sources Added for Noise / Interference

8. **UCC public notice on network repeaters / boosters**
   - URL: <https://uccinfoblog.com/2021/07/26/public-notice-signal-interference-arising-out-of-usage-of-network-repeaters-boosters/>
   - Publisher: Uganda Communications Commission
   - Date: 26 July 2021
   - Why it matters: explicit official statement that unauthorized boosters amplify the noise environment and degrade mobile-network quality of service.

9. **UCC enforcement operation targeting radios, boosters, and Bizindaalo**
   - URL: <https://uccinfoblog.com/2023/07/05/ucc-operation-targets-illegal-radio-stations-boosters-and-megaphones/>
   - Publisher: Uganda Communications Commission
   - Date: 5 July 2023
   - Why it matters: ties illegal radios, boosters, and Bizindaalo directly to interference and deteriorating QoS.

10. **Recommendation ITU-R P.530-18**
    - URL: <https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.530-18-202109-I!!PDF-E.pdf>
    - Title: *Propagation data and prediction methods required for the design of terrestrial line-of-sight systems*
    - Publisher: International Telecommunication Union
    - Date: September 2021 edition
    - Why it matters: standard propagation reference for terrestrial link impairment, including precipitation effects relevant to microwave backhaul and rural links.

11. **Recommendation ITU-R P.838-3**
    - URL: <https://www.itu.int/rec/R-REC-P.838-3-200503-P/en>
    - Title: *Specific attenuation model for rain for use in prediction methods*
    - Publisher: International Telecommunication Union
    - Date: 8 March 2005 approval page
    - Why it matters: canonical ITU rain-specific attenuation model for linking rainfall to signal degradation.

12. **Complex-valued autoencoder noise-reduction AMC paper**
    - URL: <https://www.mdpi.com/2079-9292/15/3/674>
    - Title: "Enhancing Noise Robustness in Few-Shot Automatic Modulation Classification via Complex-Valued Autoencoders"
    - Venue: *Electronics*
    - Date: 3 February 2026
    - Why it matters: recent denoising-enhanced AMC evidence showing measurable denoising and classification gains in noisy conditions.

13. **Noise-adaptive autoencoder for RF modulation recognition**
    - URL: <https://doi.org/10.1109/IMS40175.2024.10600346>
    - Title: "Noise-Adaptive Auto-Encoder for Modulation Recognition of RF Signal"
    - Venue: IEEE MTT-S International Microwave Symposium Digest
    - Date: 2024
    - Why it matters: directly relevant adaptive-autoencoder source for RF noise suppression before classification.

## Suggested Usage Rules

- Prefer official Uganda or captured Uganda sources when making country-specific claims.
- Prefer ITU recommendations when justifying rain attenuation or terrestrial-link impairment language.
- Prefer the synthesis markdown files only as repository-backed drafting support when the underlying primary source is already reflected or captured in the repo.
- Keep claims about illegal boosters, Bizindaalo, and QoS tied to UCC or captured-Uganda sources whenever possible.
