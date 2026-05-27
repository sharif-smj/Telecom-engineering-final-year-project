import { C, addBase, title, imageCard } from "./deck_helpers.mjs";

const BASE = "/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24";

export async function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Results");
  title(slide, ctx, "Results: Signal Denoising Output", "Controlled simulation");
  await imageCard(slide, ctx, `${BASE}/signal_comparison.png`, "Noisy, denoised, and clean reference waveform comparison", 62, 166, 556, 430);
  await imageCard(slide, ctx, `${BASE}/constellation_comparison.png`, "Constellation comparison at a low-SNR operating point", 664, 166, 556, 430);
  ctx.addText(slide, {
    text: "Interpretation: this controlled simulation is not final DAE performance; it shows why denoising before AMC is worth completing (An & Lee, 2023; Ssemujju & Kisige, 2026).",
    x: 120,
    y: 604,
    w: 1040,
    h: 34,
    fontSize: 16,
    color: C.ink,
    align: "center",
  });
  return slide;
}
