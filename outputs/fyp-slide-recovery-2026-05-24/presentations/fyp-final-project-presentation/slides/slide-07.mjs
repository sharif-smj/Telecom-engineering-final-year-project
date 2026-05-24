import { C, addBase, title, processNode, arrow, bulletBox } from "./deck_helpers.mjs";

export async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Methodology");
  title(slide, ctx, "Methodology: denoising-assisted AMC signal chain", "System design");
  const y = 222;
  processNode(slide, ctx, 1, "I/Q source", "Synthetic GSM-family or public I/Q windows.", 58, y, 178, 132, C.navy);
  arrow(slide, ctx, 242, y + 42);
  processNode(slide, ctx, 2, "Impairment", "Low SNR, narrowband interference, fading, phase/frequency offset.", 282, y, 190, 132, C.orange);
  arrow(slide, ctx, 478, y + 42);
  processNode(slide, ctx, 3, "Denoising", "Reconstruct cleaner signal representation before classification.", 518, y, 178, 132, C.teal);
  arrow(slide, ctx, 702, y + 42);
  processNode(slide, ctx, 4, "AMC", "Classify modulation from raw/noisy or denoised features.", 742, y, 178, 132, C.blue);
  arrow(slide, ctx, 926, y + 42);
  processNode(slide, ctx, 5, "Metrics", "Accuracy, macro F1, confusion matrix, and signal plots.", 966, y, 210, 132, C.navy);
  bulletBox(slide, ctx, "Evaluation comparison", [
    "Baseline path: noisy I/Q features -> AMC classifier.",
    "Denoising-assisted path: impaired I/Q -> denoising front end -> AMC classifier.",
    "Both paths are evaluated under the same SNR levels and classes.",
    "This makes the difference attributable to the denoising front end."
  ], 120, 432, 1040, 170, C.teal);
  return slide;
}
