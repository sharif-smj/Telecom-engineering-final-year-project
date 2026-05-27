import { C, addBase, title, imageCard, metric } from "./deck_helpers.mjs";

const BASE = "/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24";

export async function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Results");
  title(slide, ctx, "Results: Classification Performance", "Controlled simulation");
  await imageCard(slide, ctx, `${BASE}/accuracy_by_snr.png`, "Accuracy by SNR: noisy baseline versus denoising-assisted AMC", 60, 150, 760, 430);
  metric(slide, ctx, "49.4%", "Noisy baseline", "Mean accuracy across controlled SNR levels.", 858, 176, 300, 120, C.orange);
  metric(slide, ctx, "52.0%", "Denoising-assisted", "Mean accuracy with the denoising-assisted path.", 858, 324, 300, 120, C.teal);
  metric(slide, ctx, "+9.2 pts", "Largest observed gain", "At 0 dB in this preliminary controlled run.", 858, 472, 300, 120, C.navy);
  ctx.addText(slide, {
    text: "Result status: preliminary controlled simulation, not final field validation (Ssemujju & Kisige, 2026).",
    x: 80,
    y: 604,
    w: 1120,
    h: 30,
    fontSize: 20,
    bold: true,
    color: C.red,
    align: "center",
  });
  return slide;
}
