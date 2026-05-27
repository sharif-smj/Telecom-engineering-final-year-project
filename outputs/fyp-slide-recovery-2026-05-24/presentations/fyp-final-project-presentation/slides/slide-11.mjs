import { C, addBase, title, bulletBox, imageCard } from "./deck_helpers.mjs";

const BASE = "/Users/sharif/telecom/final-year-project/reports/preliminary_simulation_2026_05_24";

export async function slide11(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Discussion of results");
  title(slide, ctx, "Discussion of Results", "Interpretation");
  await imageCard(slide, ctx, `${BASE}/confusion_denoised.png`, "Denoising-assisted confusion matrix across controlled SNR levels", 70, 160, 520, 390);
  bulletBox(slide, ctx, "Observed pattern", [
    "The denoising-assisted path improves the controlled run overall (Ssemujju & Kisige, 2026).",
    "The clearest gain appears around the difficult 0 dB condition.",
    "Cleaner SNR conditions do not require aggressive denoising, so adaptive bypass is sensible (An & Lee, 2023)."
  ], 640, 170, 520, 165, C.teal);
  bulletBox(slide, ctx, "Limitations", [
    "The samples are synthetic and should be replaced or supplemented with stronger datasets.",
    "The denoising block is preliminary; final DAE training remains the next technical step.",
    "The results do not prove live service improvement or physical coverage extension."
  ], 640, 378, 520, 172, C.red);
  return slide;
}
