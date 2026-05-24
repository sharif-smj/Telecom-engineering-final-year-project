#!/usr/bin/env python3
"""Preliminary controlled DAE-AMC style simulation for FYP presentation.

This script intentionally avoids field claims. It creates synthetic GSM-family
or GSM-adjacent baseband I/Q windows, applies controlled low-SNR impairments,
uses a simple reproducible denoising front end, and compares a clean-reference
nearest-centroid AMC decision on noisy versus denoised features.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SEED = 20260524
CLASSES = ["GMSK-like", "QPSK", "8PSK", "16QAM"]
SNR_LEVELS = [-8, -4, 0, 4]
WINDOW = 256
SYMBOLS = 64
SAMPLES_PER_CLASS_PER_SNR = 120
TRAIN_SAMPLES_PER_CLASS = 180
OUTPUT_DIR = Path(__file__).resolve().parent

NAVY = (9, 31, 66)
TEAL = (0, 128, 112)
BLUE = (52, 111, 191)
ORANGE = (221, 122, 33)
GREY = (96, 105, 112)
LIGHT = (238, 246, 248)
GRID = (210, 226, 232)
WHITE = (255, 255, 255)
RED = (174, 62, 62)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def normalize_power(signal: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(signal) ** 2)
    if power <= 1e-12:
        return signal
    return signal / math.sqrt(power)


def pulse_shape(symbols: np.ndarray) -> np.ndarray:
    repeated = np.repeat(symbols, WINDOW // len(symbols))
    if len(repeated) < WINDOW:
        repeated = np.pad(repeated, (0, WINDOW - len(repeated)), mode="edge")
    shaped = repeated[:WINDOW].astype(np.complex128)
    kernel = np.array([0.08, 0.18, 0.48, 0.18, 0.08])
    real = np.convolve(shaped.real, kernel, mode="same")
    imag = np.convolve(shaped.imag, kernel, mode="same")
    return normalize_power(real + 1j * imag)


def generate_clean_window(label: str, rng: np.random.Generator) -> np.ndarray:
    if label == "GMSK-like":
        bits = rng.choice([-1.0, 1.0], size=SYMBOLS)
        phase = np.cumsum(bits) * (math.pi / 4.0)
        symbols = np.exp(1j * phase)
    elif label == "QPSK":
        phases = rng.integers(0, 4, size=SYMBOLS) * (math.pi / 2.0) + math.pi / 4.0
        symbols = np.exp(1j * phases)
    elif label == "8PSK":
        phases = rng.integers(0, 8, size=SYMBOLS) * (math.pi / 4.0)
        symbols = np.exp(1j * phases)
    elif label == "16QAM":
        levels = np.array([-3.0, -1.0, 1.0, 3.0])
        symbols = rng.choice(levels, size=SYMBOLS) + 1j * rng.choice(levels, size=SYMBOLS)
        symbols = normalize_power(symbols)
    else:
        raise ValueError(f"Unknown label: {label}")

    signal = pulse_shape(symbols)
    t = np.arange(WINDOW)
    freq_offset = rng.uniform(-0.012, 0.012)
    phase_offset = rng.uniform(-math.pi, math.pi)
    fading = 0.92 + 0.08 * np.sin(2 * math.pi * rng.uniform(0.004, 0.014) * t + rng.uniform(0, 2 * math.pi))
    signal = fading * signal * np.exp(1j * (2 * math.pi * freq_offset * t + phase_offset))
    return normalize_power(signal)


def impair(clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(np.abs(clean) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    awgn = math.sqrt(noise_power / 2.0) * (rng.normal(size=WINDOW) + 1j * rng.normal(size=WINDOW))
    t = np.arange(WINDOW)
    tone_freq = rng.uniform(0.045, 0.18)
    tone_phase = rng.uniform(0, 2 * math.pi)
    tone_amp = math.sqrt(signal_power) * (0.18 + 0.03 * max(0, -snr_db))
    interference = tone_amp * np.exp(1j * (2 * math.pi * tone_freq * t + tone_phase))
    dc = (rng.normal(scale=0.035) + 1j * rng.normal(scale=0.035)) * math.sqrt(signal_power)
    return clean + awgn + interference + dc


def denoise(noisy: np.ndarray) -> np.ndarray:
    centered = noisy - np.mean(noisy)
    spectrum = np.fft.fft(centered)
    mag = np.abs(spectrum)
    floor = np.median(mag)
    shrink = np.maximum(mag - 0.85 * floor, 0.0) / (mag + 1e-12)
    cleaned = np.fft.ifft(spectrum * shrink)

    # Smooth only lightly; heavy smoothing would destroy modulation evidence.
    kernel = np.array([0.15, 0.70, 0.15])
    real = np.convolve(cleaned.real, kernel, mode="same")
    imag = np.convolve(cleaned.imag, kernel, mode="same")
    cleaned = normalize_power(real + 1j * imag)
    # Conservative blend: keep modulation structure while reducing obvious noise.
    return normalize_power(0.55 * cleaned + 0.45 * normalize_power(centered))


def features(signal: np.ndarray) -> np.ndarray:
    x = normalize_power(signal)
    sym = normalize_power(x[2::4])
    amp = np.abs(x)
    sym_amp = np.abs(sym)
    phase = np.unwrap(np.angle(sym))
    dphase = np.diff(phase)
    spectrum = np.abs(np.fft.fft(x))
    spectrum = spectrum / (np.sum(spectrum) + 1e-12)
    entropy = -np.sum(spectrum * np.log2(spectrum + 1e-12))
    papr = np.max(amp ** 2) / (np.mean(amp ** 2) + 1e-12)
    fourth = np.mean((amp - np.mean(amp)) ** 4) / ((np.var(amp) + 1e-12) ** 2)
    phase_raw = np.angle(sym)
    phase_lock = [np.abs(np.mean(np.exp(1j * order * phase_raw))) for order in [2, 4, 8, 16]]
    quantiles = np.quantile(sym_amp, [0.1, 0.25, 0.5, 0.75, 0.9])
    return np.array(
        [
            np.mean(amp),
            np.std(amp),
            np.mean(sym_amp),
            np.std(sym_amp),
            *quantiles,
            np.var(x.real),
            np.var(x.imag),
            np.mean(np.cos(dphase)),
            np.mean(np.sin(dphase)),
            np.std(dphase),
            *phase_lock,
            papr,
            fourth,
            entropy,
            np.max(spectrum),
        ],
        dtype=np.float64,
    )


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sigma = train.std(axis=0) + 1e-9
    return (train - mu) / sigma, (test - mu) / sigma


def nearest_centroid_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    centroids = []
    for label in CLASSES:
        centroids.append(train_x[train_y == label].mean(axis=0))
    centroids = np.vstack(centroids)
    dists = ((test_x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.array(CLASSES, dtype=object)[np.argmin(dists, axis=1)]


def knn_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int = 7) -> np.ndarray:
    predictions = []
    for row in test_x:
        dists = np.sum((train_x - row) ** 2, axis=1)
        nearest = np.argsort(dists)[:k]
        labels, counts = np.unique(train_y[nearest], return_counts=True)
        predictions.append(labels[np.argmax(counts)])
    return np.array(predictions, dtype=object)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in CLASSES:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        scores.append(2 * precision * recall / (precision + recall + 1e-12))
    return float(np.mean(scores))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    index = {label: i for i, label in enumerate(CLASSES)}
    for truth, pred in zip(y_true, y_pred):
        matrix[index[truth], index[pred]] += 1
    return matrix


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int, fill=NAVY, bold: bool = False) -> None:
    draw.text(xy, text, fill=fill, font=font(size, bold=bold))


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=18, fill=WHITE, outline=GRID, width=2)
    draw_text(draw, (box[0] + 18, box[1] + 12), title, 28, NAVY, True)


def map_points(values: np.ndarray, box: tuple[int, int, int, int], y_lim: float = 2.8) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    xs = np.linspace(x0, x1, len(values))
    ys = y1 - ((np.clip(values, -y_lim, y_lim) + y_lim) / (2 * y_lim)) * (y1 - y0)
    return list(zip(xs, ys))


def draw_line_plot(draw: ImageDraw.ImageDraw, values: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=GRID, width=1)
    for frac in [0.25, 0.5, 0.75]:
        y = y0 + int((y1 - y0) * frac)
        draw.line((x0, y, x1, y), fill=GRID, width=1)
    points = map_points(values, box)
    draw.line(points, fill=color, width=3)


def draw_scatter(draw: ImageDraw.ImageDraw, values: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=GRID, width=1)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    draw.line((x0, cy, x1, cy), fill=GRID, width=1)
    draw.line((cx, y0, cx, y1), fill=GRID, width=1)
    scale = 0.34 * min(x1 - x0, y1 - y0)
    for point in values[::2]:
        px = cx + float(np.clip(point.real, -2.2, 2.2)) * scale / 2.2
        py = cy - float(np.clip(point.imag, -2.2, 2.2)) * scale / 2.2
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)


def create_signal_figures(clean: np.ndarray, noisy: np.ndarray, cleaned: np.ndarray) -> None:
    img = Image.new("RGB", (1600, 900), LIGHT)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (60, 40), "Preliminary controlled simulation: noisy vs denoised I/Q", 42, NAVY, True)
    panels = [
        ((70, 130, 1530, 330), "Noisy received I component", noisy.real, RED),
        ((70, 370, 1530, 570), "Denoised I component", cleaned.real, TEAL),
        ((70, 610, 1530, 810), "Clean reference I component", clean.real, BLUE),
    ]
    for box, title, values, color in panels:
        draw_panel(draw, box, title)
        draw_line_plot(draw, values, (box[0] + 22, box[1] + 58, box[2] - 22, box[3] - 22), color)
    img.save(OUTPUT_DIR / "signal_comparison.png")

    img = Image.new("RGB", (1600, 900), LIGHT)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (60, 40), "Constellation comparison under low-SNR impairment", 42, NAVY, True)
    panels = [
        ((70, 140, 500, 770), "Clean reference", clean, BLUE),
        ((585, 140, 1015, 770), "Noisy received", noisy, RED),
        ((1100, 140, 1530, 770), "Denoised output", cleaned, TEAL),
    ]
    for box, title, values, color in panels:
        draw_panel(draw, box, title)
        draw_scatter(draw, values, (box[0] + 45, box[1] + 85, box[2] - 45, box[3] - 45), color)
    img.save(OUTPUT_DIR / "constellation_comparison.png")


def draw_accuracy_chart(results: pd.DataFrame) -> None:
    img = Image.new("RGB", (1600, 900), LIGHT)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (60, 40), "Preliminary AMC accuracy by SNR", 44, NAVY, True)
    chart = (160, 150, 1460, 720)
    x0, y0, x1, y1 = chart
    draw.rectangle(chart, fill=WHITE, outline=GRID, width=2)
    for pct in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y1 - int((y1 - y0) * pct)
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        draw_text(draw, (90, y - 14), f"{int(pct * 100)}%", 22, GREY)
    snrs = sorted(results["snr_db"].unique())
    slot = (x1 - x0) / len(snrs)
    bar_w = 54
    for i, snr in enumerate(snrs):
        base_x = x0 + slot * i + slot / 2
        for offset, pipeline, color in [(-34, "Noisy baseline", ORANGE), (34, "Denoising-assisted", TEAL)]:
            row = results[(results["snr_db"] == snr) & (results["pipeline"] == pipeline)].iloc[0]
            acc = float(row["accuracy"])
            h = int((y1 - y0) * acc)
            left = int(base_x + offset - bar_w / 2)
            right = int(base_x + offset + bar_w / 2)
            draw.rounded_rectangle((left, y1 - h, right, y1), radius=8, fill=color)
            draw_text(draw, (left - 2, y1 - h - 30), f"{acc * 100:.0f}%", 20, NAVY, True)
        draw_text(draw, (int(base_x - 24), y1 + 20), f"{snr} dB", 24, NAVY, True)
    draw_text(draw, (620, 775), "Orange: noisy baseline    Green: denoising-assisted", 28, NAVY, True)
    img.save(OUTPUT_DIR / "accuracy_by_snr.png")


def draw_confusion(matrix: np.ndarray) -> None:
    img = Image.new("RGB", (1400, 900), LIGHT)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (60, 40), "Denoising-assisted confusion matrix", 42, NAVY, True)
    x0, y0 = 250, 170
    cell = 135
    max_value = max(1, matrix.max())
    for i, truth in enumerate(CLASSES):
        draw_text(draw, (60, y0 + i * cell + 48), truth, 28, NAVY, True)
    for j, pred in enumerate(CLASSES):
        draw_text(draw, (x0 + j * cell + 10, 120), pred, 24, NAVY, True)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            value = matrix[i, j]
            intensity = int(230 - 150 * (value / max_value))
            fill = (intensity, 238, 234) if i == j else (246, intensity, intensity)
            box = (x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell, y0 + (i + 1) * cell)
            draw.rectangle(box, fill=fill, outline=WHITE, width=3)
            draw_text(draw, (box[0] + 45, box[1] + 45), str(value), 34, NAVY, True)
    draw_text(draw, (x0 + 130, 760), "Predicted class", 30, NAVY, True)
    draw_text(draw, (60, 690), "True class", 30, NAVY, True)
    img.save(OUTPUT_DIR / "confusion_denoised.png")


def write_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", *CLASSES])
        for label, row in zip(CLASSES, matrix):
            writer.writerow([label, *row.tolist()])


def main() -> None:
    rng = np.random.default_rng(SEED)

    records = []
    all_truth = []
    all_pred_baseline = []
    all_pred_denoised = []
    example = None

    for snr in SNR_LEVELS:
        noisy_features = []
        denoised_features = []
        y_true = []
        train_mask = []
        for label in CLASSES:
            for sample_index in range(SAMPLES_PER_CLASS_PER_SNR):
                clean = generate_clean_window(label, rng)
                noisy = impair(clean, snr, rng)
                cleaned = denoise(noisy) if snr <= 0 else noisy
                if example is None and snr == -4 and label == "QPSK":
                    example = (clean, noisy, cleaned)
                noisy_features.append(features(noisy))
                denoised_features.append(features(cleaned))
                y_true.append(label)
                train_mask.append(sample_index % 2 == 0)

        y_true = np.array(y_true, dtype=object)
        train_mask = np.array(train_mask, dtype=bool)
        noisy_features = np.vstack(noisy_features)
        denoised_features = np.vstack(denoised_features)

        train_noisy, test_noisy = standardize(noisy_features[train_mask], noisy_features[~train_mask])
        train_denoised, test_denoised = standardize(denoised_features[train_mask], denoised_features[~train_mask])
        train_y = y_true[train_mask]
        test_y = y_true[~train_mask]

        pred_baseline = knn_predict(train_noisy, train_y, test_noisy)
        pred_denoised = knn_predict(train_denoised, train_y, test_denoised)

        for pipeline, pred in [("Noisy baseline", pred_baseline), ("Denoising-assisted", pred_denoised)]:
            records.append(
                {
                    "snr_db": snr,
                    "pipeline": pipeline,
                    "accuracy": float(np.mean(pred == test_y)),
                    "macro_f1": macro_f1(test_y, pred),
                    "samples": len(test_y),
                }
            )

        all_truth.extend(test_y.tolist())
        all_pred_baseline.extend(pred_baseline.tolist())
        all_pred_denoised.extend(pred_denoised.tolist())

    results = pd.DataFrame(records)
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)
    baseline_matrix = confusion_matrix(np.array(all_truth, dtype=object), np.array(all_pred_baseline, dtype=object))
    denoised_matrix = confusion_matrix(np.array(all_truth, dtype=object), np.array(all_pred_denoised, dtype=object))
    write_confusion_csv(OUTPUT_DIR / "confusion_baseline.csv", baseline_matrix)
    write_confusion_csv(OUTPUT_DIR / "confusion_denoised.csv", denoised_matrix)

    if example is None:
        raise RuntimeError("No example window was captured for figures")
    create_signal_figures(*example)
    draw_accuracy_chart(results)
    draw_confusion(denoised_matrix)

    overall = results.groupby("pipeline", as_index=False).agg({"accuracy": "mean", "macro_f1": "mean"})
    readme = f"""# Preliminary Controlled Simulation Outputs

Generated: 2026-05-24

This folder contains preliminary controlled simulation artifacts for the final-year project presentation on GSM-family signal denoising and automatic modulation classification.

## Scope

- The samples are synthetic baseband I/Q windows, not field captures.
- The impairment model combines AWGN, narrowband interference, phase/frequency offset, and mild fading.
- The denoising stage is a reproducible signal-processing front end used under weak/noisy SNR settings, with bypass at cleaner SNR in this preliminary adaptive pipeline.
- These outputs are suitable for a preliminary results slide, not for final field-performance claims.

## Classes

{", ".join(CLASSES)}

## SNR levels

{", ".join(str(s) + " dB" for s in SNR_LEVELS)}

## Overall preliminary metrics

{overall.to_string(index=False)}

## Evidence rule

Do not describe these artifacts as completed field testing, live call/USSD/SMS improvement, tower range extension, or final trained DAE performance. They are controlled preliminary simulation outputs.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(overall.to_string(index=False))
    print(f"Artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
