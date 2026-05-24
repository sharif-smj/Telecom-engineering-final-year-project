import { C, addBase, title, bulletBox, metric } from "./deck_helpers.mjs";

export async function slide12(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Conclusion and recommendations");
  title(slide, ctx, "Conclusion and recommendations", "Final position");
  bulletBox(slide, ctx, "Conclusion", [
    "Weak rural GSM service can be studied through noisy I/Q signal interpretation, not only through coverage-map labels.",
    "The proposed DAE-AMC chain is technically coherent: denoise first, classify second, evaluate across SNR.",
    "Preliminary controlled simulation results support continuing the denoising-assisted AMC implementation."
  ], 76, 168, 540, 250, C.teal);
  bulletBox(slide, ctx, "Recommendations", [
    "Complete the full trained DAE and AMC implementation with stronger dataset coverage.",
    "Report results by SNR band using accuracy, macro F1, confusion matrices, and signal-quality plots.",
    "Use receive-only validation only; do not transmit, decode private content, or claim live-service gains without evidence."
  ], 666, 168, 540, 250, C.blue);
  metric(slide, ctx, "Next 4 weeks", "Implementation focus", "DAE training, AMC benchmarking, final report update, and presentation rehearsal.", 140, 478, 300, 118, C.navy);
  metric(slide, ctx, "Final output", "Expected deliverable", "A reproducible denoising-assisted AMC evaluation for weak GSM-family signals.", 490, 478, 300, 118, C.teal);
  metric(slide, ctx, "Safety", "Operating boundary", "Receive-only, offline-first validation unless formal approval changes the scope.", 840, 478, 300, 118, C.orange);
  return slide;
}
