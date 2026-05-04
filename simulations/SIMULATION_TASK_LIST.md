# DAE-AMC Simulation Suite - Master Task List

This document tracks all simulations to be built for the DAE-AMC (Denoising Autoencoder - Automatic Modulation Classification) project showcase.

---

## Overview

| # | Simulation | Status | Target Audience | Complexity |
|---|------------|--------|-----------------|------------|
| 1 | Signal Denoising Visualizer | ✅ Complete | Engineers, Researchers | Medium |
| 2 | UCC Spectrum Guardian Dashboard | ✅ Complete | Regulators, UCC Officials | High |
| 3 | Coverage Extension Map | ✅ Complete | Stakeholders, Investors | Medium |
| 4 | Modulation Classification | ✅ Complete | Students, Educators | Medium |
| 5 | End-to-End Pipeline | ✅ Complete | Technical Reviewers | High |

---

## Simulation 1: Signal Denoising Visualizer

**Purpose:** An interactive technical demonstration showing how the DAE cleans noisy signals in real-time.

### Features
- [ ] Real-time signal waveform display (time domain)
- [ ] Constellation diagram (I/Q plot) for visual modulation recognition
- [ ] SNR slider control (-10 dB to +20 dB)
- [ ] Side-by-side comparison: Noisy vs. DAE-Cleaned signal
- [ ] Multiple modulation type selector (BPSK, QPSK, 8PSK, 16QAM, 64QAM)
- [ ] Animated DAE processing effect
- [ ] Performance metrics display (MSE, PSNR improvement)
- [ ] "Rural Mode" preset simulating -90 dBm edge-of-coverage scenario

### Technical Stack
- HTML5 Canvas for waveform rendering
- JavaScript for signal generation and animation
- CSS for premium dark theme UI

### Deliverables
- [ ] `signal_denoising_visualizer.html`
- [ ] Supporting assets (icons, animations)

---

## Simulation 2: UCC Spectrum Guardian Dashboard

**Purpose:** A monitoring dashboard demonstrating how UCC can detect unauthorized transmissions using DAE-AMC technology.

### Features
- [ ] Spectrum waterfall display (frequency vs. time heatmap)
- [ ] Real-time signal activity simulation across frequency bands
- [ ] Automated detection of unauthorized signals
- [ ] Alert system with visual/audio notifications
- [ ] Signal classification panel showing modulation identification
- [ ] "Pirate Radio" scenario demonstration
- [ ] "Ghost Interference" scenario demonstration
- [ ] Geolocation estimation display (triangulation visualization)
- [ ] Historical log of detected anomalies

### Technical Stack
- HTML5 Canvas for spectrum visualization
- JavaScript for real-time simulation
- CSS Grid for dashboard layout
- Web Audio API for alert sounds

### Deliverables
- [ ] `spectrum_guardian.html`
- [ ] Scenario presets (pirate radio, ghost interference)
- [ ] Alert sound assets

---

## Simulation 3: Coverage Extension Map

**Purpose:** A geographic visualization showing how DAE technology effectively extends cell tower coverage in Uganda.

### Features
- [ ] Interactive map of Uganda with major regions
- [ ] Cell tower placement visualization
- [ ] Coverage circles (before/after DAE enhancement)
- [ ] Toggle view: Standard Coverage vs. DAE-Enhanced Coverage
- [ ] Population density overlay
- [ ] "Rural Renaissance" scenario animation
- [ ] Statistics panel (% population covered improvement)
- [ ] Click on regions to see local impact metrics

### Technical Stack
- SVG-based map of Uganda
- JavaScript for interactivity
- CSS animations for coverage expansion effect

### Deliverables
- [ ] `coverage_map.html`
- [ ] Uganda SVG map asset
- [ ] Regional data JSON

---

## Bonus: Unified Showcase Portal

After completing all three simulations, we can optionally create:

- [ ] `index.html` - A landing page linking all simulations
- [ ] Consistent branding across all demos
- [ ] Navigation between simulations
- [ ] Export/screenshot functionality

---

## Build Order Recommendation

```
┌─────────────────────────────────────────────────────────────────┐
│  RECOMMENDED BUILD ORDER                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1️⃣  Signal Denoising Visualizer (Foundation)                   │
│      └── Establishes core signal visualization components       │
│          that other simulations can reuse                       │
│                                                                 │
│  2️⃣  UCC Spectrum Guardian (Builds on #1)                       │
│      └── Uses signal visualization + adds dashboard elements    │
│                                                                 │
│  3️⃣  Coverage Extension Map (Independent)                       │
│      └── Geographic focus, different visualization approach     │
│                                                                 │
│  4️⃣  Unified Portal (Final polish)                              │
│      └── Ties everything together                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Progress Tracking

### Current Focus: _Reconciliation and polish_

### Completed Simulations
- Signal Denoising Visualizer
- UCC Spectrum Guardian Dashboard
- Coverage Extension Map
- Modulation Classification Demo
- End-to-End Pipeline Visualizer
- Uganda Terrain/3D Coverage Map
- Unified Showcase Portal (`index.html`)

### Next Steps
1. Cross-check claims/text with `ProjectReport.md` and `docs/WORK_RECONCILIATION.md`
2. Run visual QA across desktop/mobile breakpoints
3. Export final screenshots/media for presentation
4. Keep new simulation assets organized by feature module

---

*Last Updated: 2026-03-05*
