import { C, addBase, title, bulletBox, sourceLine } from "./deck_helpers.mjs";

export async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Literature review");
  title(slide, ctx, "Literature review: where the gap sits", "Evidence base");
  bulletBox(slide, ctx, "Classical AMC", [
    "Uses likelihood rules, moments, cyclostationary features, and engineered descriptors.",
    "Can be interpretable, but becomes fragile when assumptions about noise and channel conditions break.",
    "Useful as a baseline for explaining what the ML pipeline improves."
  ], 70, 172, 355, 330, C.navy);
  bulletBox(slide, ctx, "Deep-learning AMC", [
    "Learns discriminative patterns directly from raw or transformed I/Q samples.",
    "Modern CNN and sequence models perform well when trained on representative SNR conditions.",
    "Performance still drops when signal structure is buried under low-SNR and interference."
  ], 462, 172, 355, 330, C.blue);
  bulletBox(slide, ctx, "Denoising front ends", [
    "Denoising autoencoders and related methods reconstruct cleaner signals before downstream tasks.",
    "The key research question is not only whether the signal looks cleaner, but whether classification improves.",
    "This project evaluates denoising as a preprocessing block for AMC under weak-service conditions."
  ], 855, 172, 355, 330, C.teal);
  ctx.addShape(slide, { x: 150, y: 540, w: 980, h: 70, fill: C.white, line: ctx.line(C.line, 1) });
  ctx.addText(slide, {
    text: "Gap: few student-level implementations connect denoising, AMC metrics, and rural GSM weak-service interpretation in one reproducible pipeline.",
    x: 180,
    y: 560,
    w: 920,
    h: 40,
    fontSize: 22,
    bold: true,
    color: C.navy,
    align: "center",
  });
  sourceLine(slide, ctx, "Sources: ProjectReport.md literature review; DeepSig/RadioML datasets; low-SNR AMC and denoising references captured in reports/.");
  return slide;
}
