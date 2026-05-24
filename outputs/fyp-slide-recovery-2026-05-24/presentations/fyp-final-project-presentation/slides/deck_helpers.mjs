export const C = {
  navy: "#071D3B",
  blue: "#2563A9",
  teal: "#008A7A",
  mint: "#DDF4F0",
  pale: "#F2F8FA",
  white: "#FFFFFF",
  ink: "#102033",
  muted: "#5C6B75",
  orange: "#E27A1F",
  red: "#B84646",
  line: "#BCD7E2",
};

export function addBase(slide, ctx, section = "") {
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 720, fill: C.pale });
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 14, fill: C.navy });
  ctx.addShape(slide, { x: 0, y: 706, w: 1280, h: 14, fill: C.navy });
  ctx.addText(slide, {
    text: section,
    x: 58,
    y: 660,
    w: 760,
    h: 28,
    fontSize: 15,
    color: C.muted,
    typeface: "Aptos",
  });
  ctx.addText(slide, {
    text: `Final Year Project | ${ctx.slideNumber}`,
    x: 1010,
    y: 660,
    w: 210,
    h: 28,
    fontSize: 15,
    color: C.muted,
    align: "right",
  });
}

export function title(slide, ctx, text, kicker = "") {
  if (kicker) {
    ctx.addText(slide, {
      text: kicker.toUpperCase(),
      x: 58,
      y: 44,
      w: 840,
      h: 24,
      fontSize: 16,
      bold: true,
      color: C.teal,
      typeface: "Aptos",
    });
  }
  ctx.addText(slide, {
    text,
    x: 58,
    y: kicker ? 74 : 54,
    w: 1060,
    h: 80,
    fontSize: 34,
    bold: true,
    color: C.navy,
    typeface: "Aptos Display",
  });
}

export function bodyText(slide, ctx, text, x, y, w, h, size = 22, color = C.ink) {
  return ctx.addText(slide, {
    text,
    x,
    y,
    w,
    h,
    fontSize: size,
    color,
    insets: { left: 8, right: 8, top: 4, bottom: 4 },
  });
}

export function bulletBox(slide, ctx, heading, bullets, x, y, w, h, accent = C.teal) {
  ctx.addShape(slide, { x, y, w, h, fill: C.white, line: ctx.line(C.line, 1) });
  ctx.addShape(slide, { x, y, w: 8, h, fill: accent });
  ctx.addText(slide, {
    text: heading,
    x: x + 24,
    y: y + 18,
    w: w - 44,
    h: 30,
    fontSize: 22,
    bold: true,
    color: C.navy,
  });
  ctx.addText(slide, {
    text: bullets.map((item) => `• ${item}`).join("\n"),
    x: x + 26,
    y: y + 58,
    w: w - 50,
    h: h - 72,
    fontSize: 18,
    color: C.ink,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function metric(slide, ctx, value, label, note, x, y, w, h, accent = C.teal) {
  ctx.addShape(slide, { x, y, w, h, fill: C.white, line: ctx.line(C.line, 1) });
  ctx.addShape(slide, { x, y, w, h: 8, fill: accent });
  ctx.addText(slide, { text: value, x: x + 18, y: y + 20, w: w - 36, h: 40, fontSize: 31, bold: true, color: C.navy });
  ctx.addText(slide, { text: label, x: x + 18, y: y + 64, w: w - 36, h: 28, fontSize: 17, bold: true, color: C.ink });
  ctx.addText(slide, { text: note, x: x + 18, y: y + 94, w: w - 36, h: Math.max(24, h - 102), fontSize: 14, color: C.muted });
}

export function processNode(slide, ctx, index, heading, note, x, y, w = 190, h = 126, accent = C.teal) {
  ctx.addShape(slide, { x, y, w, h, fill: C.white, line: ctx.line(C.line, 1) });
  ctx.addShape(slide, { x: x + 14, y: y + 16, w: 34, h: 34, fill: accent });
  ctx.addText(slide, { text: String(index), x: x + 14, y: y + 19, w: 34, h: 28, fontSize: 18, bold: true, color: C.white, align: "center" });
  ctx.addText(slide, { text: heading, x: x + 58, y: y + 14, w: w - 70, h: 42, fontSize: 18, bold: true, color: C.navy });
  ctx.addText(slide, { text: note, x: x + 18, y: y + 62, w: w - 36, h: h - 74, fontSize: 14, color: C.ink });
}

export function arrow(slide, ctx, x, y) {
  ctx.addText(slide, { text: "→", x, y, w: 34, h: 40, fontSize: 32, bold: true, color: C.navy, align: "center" });
}

export async function imageCard(slide, ctx, imagePath, caption, x, y, w, h) {
  ctx.addShape(slide, { x, y, w, h, fill: C.white, line: ctx.line(C.line, 1) });
  await ctx.addImage(slide, {
    path: imagePath,
    x: x + 10,
    y: y + 10,
    w: w - 20,
    h: h - 54,
    fit: "contain",
    alt: caption,
  });
  ctx.addText(slide, {
    text: caption,
    x: x + 16,
    y: y + h - 38,
    w: w - 32,
    h: 26,
    fontSize: 16,
    color: C.muted,
    align: "center",
  });
}

export function sourceLine(slide, ctx, text) {
  ctx.addText(slide, {
    text,
    x: 58,
    y: 628,
    w: 1120,
    h: 24,
    fontSize: 13,
    color: C.muted,
  });
}
