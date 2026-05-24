import { C, addBase, title, bulletBox, metric } from "./deck_helpers.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Introduction / Background");
  title(slide, ctx, "Rural mobile service is often present, but still too weak to depend on", "Background");
  bulletBox(slide, ctx, "Local service reality", [
    "Basic GSM services still matter: voice, SMS, USSD, and Mobile Money.",
    "At the cell edge, a phone may show signal bars but still suffer noisy or unstable service.",
    "Weak I/Q structure makes reliable modulation interpretation difficult under low SNR."
  ], 70, 184, 540, 290, C.teal);
  bulletBox(slide, ctx, "Engineering interpretation", [
    "The project treats weak service as a signal-quality problem, not only a coverage-map problem.",
    "Denoising is placed before classification so the AMC stage sees a cleaner signal representation.",
    "The same logic can support troubleshooting, spectrum monitoring, and rural service planning."
  ], 670, 184, 540, 290, C.blue);
  metric(slide, ctx, "GSM-family I/Q", "Working signal model", "Baseband windows test noisy edge-of-coverage conditions.", 78, 494, 330, 138, C.navy);
  metric(slide, ctx, "DAE + AMC", "Core technical pipeline", "Denoising front end followed by automatic modulation classification.", 475, 494, 330, 138, C.teal);
  metric(slide, ctx, "Rural use case", "Project motivation", "Weak-signal interpretation can support practical network decisions.", 872, 494, 330, 138, C.orange);
  return slide;
}
