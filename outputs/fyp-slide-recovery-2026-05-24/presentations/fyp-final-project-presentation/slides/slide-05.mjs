import { C, addBase, title, bulletBox, metric } from "./deck_helpers.mjs";

export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Justification, significance, and scope");
  title(slide, ctx, "Why this project matters", "Justification and scope");
  bulletBox(slide, ctx, "Justification", [
    "Rural weak service is a practical problem for ordinary users and for network planning.",
    "Denoising before classification gives a measurable way to test whether useful signal structure can be recovered.",
    "The method is software-side and can be evaluated without transmitting or interfering with licensed networks."
  ], 70, 172, 360, 280, C.teal);
  bulletBox(slide, ctx, "Significance", [
    "Supports better interpretation of noisy edge-of-coverage signals.",
    "Gives supervisors and reviewers visible before/after signal evidence.",
    "Creates a foundation for future Android/SDR receive-only prototype work."
  ], 460, 172, 360, 280, C.blue);
  bulletBox(slide, ctx, "Scope", [
    "Controlled simulation and offline I/Q evaluation.",
    "GSM-family/proxy modulation classes used for AMC validation.",
    "No claim of completed field deployment or live service improvement."
  ], 850, 172, 360, 280, C.orange);
  metric(slide, ctx, "Receive-only", "Safety boundary", "The project does not transmit into licensed spectrum.", 118, 500, 300, 110, C.navy);
  metric(slide, ctx, "Offline first", "Evaluation boundary", "The current results are controlled simulation outputs.", 490, 500, 300, 110, C.teal);
  metric(slide, ctx, "Prototype path", "Next step", "Android/SDR demonstration remains a future validation pathway.", 862, 500, 300, 110, C.blue);
  return slide;
}
