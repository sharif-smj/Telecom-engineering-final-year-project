# Preliminary Controlled Simulation Outputs

Generated: 2026-05-24

This folder contains preliminary controlled simulation artifacts for the final-year project presentation on GSM-family signal denoising and automatic modulation classification.

## Scope

- The samples are synthetic baseband I/Q windows, not field captures.
- The impairment model combines AWGN, narrowband interference, phase/frequency offset, and mild fading.
- The denoising stage is a reproducible signal-processing front end used under weak/noisy SNR settings, with bypass at cleaner SNR in this preliminary adaptive pipeline.
- These outputs are suitable for a preliminary results slide, not for final field-performance claims.

## Classes

GMSK-like, QPSK, 8PSK, 16QAM

## SNR levels

-8 dB, -4 dB, 0 dB, 4 dB

## Overall preliminary metrics

          pipeline  accuracy  macro_f1
Denoising-assisted  0.519792  0.519187
    Noisy baseline  0.493750  0.495781

## Evidence rule

Do not describe these artifacts as completed field testing, live call/USSD/SMS improvement, tower range extension, or final trained DAE performance. They are controlled preliminary simulation outputs.
