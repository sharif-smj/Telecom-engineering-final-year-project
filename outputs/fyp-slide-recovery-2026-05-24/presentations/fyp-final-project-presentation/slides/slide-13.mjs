import { C, addBase, title, bulletBox } from "./deck_helpers.mjs";

export async function slide13(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "References");
  title(slide, ctx, "References and artifact provenance", "Sources used");
  bulletBox(slide, ctx, "Selected references", [
    "Uganda Communications Commission reports and service-quality materials referenced in ProjectReport.md.",
    "DeepSig / RadioML I/Q dataset literature for AMC evaluation framing.",
    "Low-SNR AMC and denoising-autoencoder papers summarized in reports/LowSNR_AMC_Denoising_Research.md.",
    "Project report manuscript: ProjectReport.md, final-year project workspace."
  ], 84, 166, 1090, 250, C.navy);
  bulletBox(slide, ctx, "Preliminary simulation provenance", [
    "Script: reports/preliminary_simulation_2026_05_24/run_preliminary_dae_amc.py",
    "Metrics: reports/preliminary_simulation_2026_05_24/results.csv",
    "Figures: signal_comparison.png, constellation_comparison.png, accuracy_by_snr.png, confusion_denoised.png",
    "Status: controlled simulation artifacts generated for presentation submission, not field deployment evidence."
  ], 84, 452, 1090, 150, C.teal);
  ctx.addText(slide, {
    text: "Thank you.",
    x: 450,
    y: 622,
    w: 380,
    h: 45,
    fontSize: 32,
    bold: true,
    color: C.navy,
    align: "center",
  });
  return slide;
}
