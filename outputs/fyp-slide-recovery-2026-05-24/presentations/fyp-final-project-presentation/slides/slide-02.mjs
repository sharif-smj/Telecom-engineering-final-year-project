import { C, addBase, title, bulletBox, metric } from "./deck_helpers.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Introduction / Background");
  title(slide, ctx, "Introduction / Background", "Project context");
  bulletBox(slide, ctx, "Local service reality", [
    "Basic GSM services still matter for ordinary users: voice, SMS, USSD, and Mobile Money (Uganda Communications Commission, 2025).",
    "At the cell edge, a phone may show signal bars but still suffer noisy or unstable service (Uganda Communications Commission, 2022).",
    "Weak I/Q structure makes reliable modulation interpretation difficult under low SNR (O'Shea et al., 2018)."
  ], 70, 184, 540, 290, C.teal);
  bulletBox(slide, ctx, "Engineering interpretation", [
    "The project treats weak service as a signal-quality problem, not only a coverage-map problem.",
    "Denoising is placed before classification so the AMC stage sees a cleaner signal representation (An & Lee, 2023).",
    "The same logic can support troubleshooting, spectrum monitoring, and rural service planning (Uganda Communications Commission, 2024)."
  ], 670, 184, 540, 290, C.blue);
  metric(slide, ctx, "GSM-family I/Q", "Working signal model", "Baseband windows test noisy edge-of-coverage conditions.", 78, 494, 330, 138, C.navy);
  metric(slide, ctx, "DAE + AMC", "Core technical pipeline", "Denoising front end followed by automatic modulation classification.", 475, 494, 330, 138, C.teal);
  metric(slide, ctx, "Rural use case", "Project motivation", "Weak-signal interpretation can support practical network decisions.", 872, 494, 330, 138, C.orange);
  return slide;
}
