import { C, addBase, title, bulletBox, metric } from "./deck_helpers.mjs";

export async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Preliminary controlled simulation setup");
  title(slide, ctx, "Preliminary controlled simulation setup", "Results basis");
  metric(slide, ctx, "4", "Modulation classes", "GMSK-like, QPSK, 8PSK, and 16QAM.", 84, 166, 250, 130, C.navy);
  metric(slide, ctx, "4", "SNR levels", "-8, -4, 0, and 4 dB controlled impairment settings.", 376, 166, 250, 130, C.orange);
  metric(slide, ctx, "1,920", "Test windows", "240 held-out windows per SNR and pipeline comparison.", 668, 166, 250, 130, C.teal);
  metric(slide, ctx, "20260524", "Random seed", "The run is deterministic and reproducible.", 960, 166, 250, 130, C.blue);
  bulletBox(slide, ctx, "What was simulated", [
    "Baseband I/Q windows with GSM-family/proxy modulation structure.",
    "Low-SNR AWGN, narrowband interference, mild fading, and phase/frequency offset.",
    "A denoising-assisted path compared against a noisy baseline path."
  ], 94, 340, 520, 220, C.teal);
  bulletBox(slide, ctx, "What was not claimed", [
    "No field capture result is claimed in this slide deck.",
    "No live call, SMS, USSD, or Mobile Money improvement is claimed.",
    "No physical tower range extension is claimed."
  ], 666, 340, 520, 220, C.red);
  return slide;
}
