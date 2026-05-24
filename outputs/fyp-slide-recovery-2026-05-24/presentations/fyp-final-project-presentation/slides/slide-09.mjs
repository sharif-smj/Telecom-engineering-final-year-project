import { C, addBase, title, imageCard } from "./deck_helpers.mjs";

const BASE = "/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24";

export async function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Preliminary results");
  title(slide, ctx, "Preliminary result: denoising makes weak signal structure easier to inspect", "Controlled simulation");
  await imageCard(slide, ctx, `${BASE}/signal_comparison.png`, "Noisy, denoised, and clean reference waveform comparison", 62, 166, 556, 430);
  await imageCard(slide, ctx, `${BASE}/constellation_comparison.png`, "Constellation comparison at a low-SNR operating point", 664, 166, 556, 430);
  ctx.addText(slide, {
    text: "Interpretation: the preliminary denoising stage is not presented as final DAE performance. It is a controlled signal-processing validation layer that shows why denoising before AMC is worth completing.",
    x: 120,
    y: 612,
    w: 1040,
    h: 44,
    fontSize: 18,
    color: C.ink,
    align: "center",
  });
  return slide;
}
