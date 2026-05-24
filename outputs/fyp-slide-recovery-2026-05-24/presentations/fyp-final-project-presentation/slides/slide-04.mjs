import { C, addBase, title, bulletBox } from "./deck_helpers.mjs";

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Objectives");
  title(slide, ctx, "Aim and objectives", "Main and specific objectives");
  ctx.addShape(slide, { x: 74, y: 166, w: 1132, h: 112, fill: C.navy, line: ctx.line(C.navy, 0) });
  ctx.addText(slide, {
    text: "Main objective: design and evaluate a denoising-assisted automatic modulation classification pipeline for noisy GSM-family I/Q signals in rural Uganda weak-service scenarios.",
    x: 108,
    y: 196,
    w: 1064,
    h: 58,
    fontSize: 24,
    bold: true,
    color: C.white,
    align: "center",
  });
  bulletBox(slide, ctx, "Specific objectives", [
    "Define GSM-family low-SNR and interference scenarios for controlled evaluation.",
    "Design a denoising front end that reconstructs cleaner I/Q representations.",
    "Train and evaluate an AMC stage on raw noisy and denoised signal features.",
    "Compare baseline and denoising-assisted performance using accuracy, macro F1, confusion behavior, and signal plots.",
    "Interpret the findings for weak-service troubleshooting and rural connectivity planning."
  ], 100, 328, 1080, 278, C.teal);
  return slide;
}
