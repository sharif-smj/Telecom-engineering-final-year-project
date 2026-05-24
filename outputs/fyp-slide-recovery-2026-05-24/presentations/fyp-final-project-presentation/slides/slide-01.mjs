import { C, addBase, bodyText } from "./deck_helpers.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Project title");
  ctx.addText(slide, {
    text: "GSM Signal Denoising and Modulation Classification for Rural Uganda",
    x: 80,
    y: 96,
    w: 1120,
    h: 130,
    fontSize: 42,
    bold: true,
    color: C.navy,
    align: "center",
    typeface: "Aptos Display",
  });
  ctx.addShape(slide, { x: 210, y: 252, w: 860, h: 3, fill: C.teal });
  bodyText(slide, ctx, "Final Year Project Presentation", 80, 284, 1120, 40, 26, C.teal);
  ctx.addText(slide, {
    text: "Department of Electrical and Electronics Engineering\nKyambogo University",
    x: 180,
    y: 348,
    w: 920,
    h: 70,
    fontSize: 24,
    color: C.ink,
    align: "center",
  });
  ctx.addShape(slide, { x: 190, y: 456, w: 900, h: 112, fill: C.white, line: ctx.line(C.line, 1) });
  ctx.addText(slide, {
    text: "Ssemujju Sharif Abdukarim  |  18/U/ETD/181/GV\nKisige Tom Derrick  |  22/U/ETD/0953/GV\nSupervisor: Dr. Dickson Mugerwa",
    x: 220,
    y: 478,
    w: 840,
    h: 76,
    fontSize: 22,
    color: C.navy,
    align: "center",
  });
  return slide;
}
