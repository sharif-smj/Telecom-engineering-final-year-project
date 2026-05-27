import { C, addBase, title, bulletBox, bodyText } from "./deck_helpers.mjs";

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Problem statement");
  title(slide, ctx, "Problem statement", "What the project addresses");
  ctx.addShape(slide, { x: 80, y: 176, w: 1120, h: 190, fill: C.white, line: ctx.line(C.line, 1) });
  bodyText(slide, ctx, "Rural GSM users may experience unreliable service even where a network signal is visible. UCC service-quality and market evidence shows that access must be judged by usable service, not signal presence alone (Uganda Communications Commission, 2022, 2025). At the edge of coverage, low SNR, interference, fading, and unstable propagation can make AMC decisions unreliable (O'Shea et al., 2018).", 112, 198, 1056, 136, 22, C.ink);
  bulletBox(slide, ctx, "Technical gap", [
    "Classical signal-processing filters are useful, but are usually fixed to known noise assumptions.",
    "Deep-learning AMC can classify raw I/Q samples, but low-SNR interference still reduces reliability (O'Shea et al., 2018).",
    "A denoising front end may recover useful signal structure before classification (An & Lee, 2023)."
  ], 80, 410, 520, 210, C.red);
  bulletBox(slide, ctx, "Project question", [
    "Can denoising improve modulation classification under controlled low-SNR GSM-family conditions?",
    "Which SNR ranges benefit most from denoising-assisted classification?",
    "How should the results be interpreted for rural weak-service analysis?"
  ], 680, 410, 520, 210, C.teal);
  return slide;
}
