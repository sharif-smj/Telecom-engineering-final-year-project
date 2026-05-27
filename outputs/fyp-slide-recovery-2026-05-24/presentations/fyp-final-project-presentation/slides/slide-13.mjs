import { C, addBase, title } from "./deck_helpers.mjs";

export async function slide13(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "References");
  title(slide, ctx, "References", "Key sources and provenance");
  ctx.addShape(slide, { x: 70, y: 155, w: 1140, h: 345, fill: C.white, line: ctx.line(C.line, 1) });
  ctx.addShape(slide, { x: 70, y: 155, w: 8, h: 345, fill: C.navy });
  ctx.addText(slide, {
    text: [
      "Uganda Communications Commission. (2025). UCC market report for Q4 2024.",
      "Uganda Communications Commission. (2024). UCC cracks down on illegal and non-compliant broadcasters.",
      "Uganda Communications Commission. (2022). QoS publication: December 2021 mobile voice/data benchmark measurements.",
      "O'Shea, T. J., Roy, T., & Clancy, T. C. (2018). Over-the-air deep learning based radio signal classification. IEEE Journal of Selected Topics in Signal Processing, 12(1), 168-179. https://doi.org/10.1109/JSTSP.2018.2797022",
      "DeepSig. (2018). RadioML 2018.01A [Data set]. Kaggle.",
      "An, T. T., & Lee, B. M. (2023). Robust automatic modulation classification in low signal to noise ratio. IEEE Access, 11, 7860-7872. https://doi.org/10.1109/ACCESS.2023.3238995",
      "Ssemujju, S. A., & Kisige, T. D. (2026). Preliminary controlled DAE-AMC simulation artifacts. final-year-project/reports/preliminary_simulation_2026_05_24."
    ].map((item) => `• ${item}`).join("\n"),
    x: 100,
    y: 178,
    w: 1070,
    h: 294,
    fontSize: 16,
    color: C.ink,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  ctx.addShape(slide, { x: 152, y: 525, w: 976, h: 74, fill: C.mint, line: ctx.line(C.line, 1) });
  ctx.addText(slide, {
    text: "Provenance note: result slides use controlled simulation outputs from the project workspace. They are not presented as field deployment, live-service, or tower-range-extension evidence.",
    x: 180,
    y: 546,
    w: 920,
    h: 34,
    fontSize: 18,
    bold: true,
    color: C.navy,
    align: "center",
  });
  ctx.addText(slide, {
    text: "Thank you.",
    x: 450,
    y: 610,
    w: 380,
    h: 45,
    fontSize: 32,
    bold: true,
    color: C.navy,
    align: "center",
  });
  return slide;
}
