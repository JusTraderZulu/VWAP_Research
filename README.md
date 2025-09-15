# VWAP Reversion EDA — Intraday Crypto Study

This repository presents a clean, reproducible exploratory study of intraday VWAP reversion in crypto markets. The notebook estimates the probability that price reverts to VWAP after sufficiently large deviations and examines how that probability changes across higher-timeframe regimes, deviation size, time-of-day, and volatility. The repo is intentionally minimal: just the research notebook and this README, for easy viewing.

## Research question
- What is the empirical probability that intraday price reverts to VWAP after a sufficiently large deviation?
- How does that probability vary by higher-timeframe Hurst regime (mean-reverting, noisy, trending), deviation size, time-of-day, and volatility level?

## Data & period
- Source: Polygon aggregates (official Python SDK) at 5-minute resolution
- Assets: crypto tickers (normalization supports both `BTC-USD` and `X:BTCUSD` forms)
- Span: 2023-01-01 → 2025-09-01 (editable in notebook)

## Event definition (anchored VWAP reversion)
- Trigger: |close − anchored VWAP| / anchored VWAP ≥ threshold (0.5%, 1.0%, 1.5%). Anchored VWAP resets each UTC day.
- Outcome (success): the signed deviation of CLOSE relative to the (anchored) VWAP flips across zero within N bars (N ∈ {5, 10, 20}).

## Higher-timeframe context (regimes)
- Rolling Hurst (R/S) on 1H returns, smoothed with EMA.
- Regimes: MR ≤ 0.45, NOISY 0.45–0.55, TREND ≥ 0.55. Optional hysteresis is supported for smoother labels.

## What we found (high level)
- Base edge: For large-cap crypto (e.g., BTC, ETH, SOL), P(revert) typically sits around ≈ 8–10% for th=1.0%, N=10, with NOISY dominating regime time and small but clear mass in TREND.
- Structure: P(revert) increases with larger deviation bins (monotonic), especially outside the smallest bin.
- Stability: First vs second half comparisons show small median |Δp| for the kept assets (indicative of robustness). Several smaller alts showed higher instability and were excluded by the per-asset gate.
- Kept assets (example from our run): BTC, ETH, SOL, AVAX. Unstable alts were rejected by the gate (insufficient triggers, weak monotonicity, or higher |Δp|).

These observations are meant as guideposts for whether a modeling step (probability modeling, calibration) is warranted next. The study is EDA-only; it prints a decision (STOP or KEEP) based on frequency, monotonic conditionals, and stability.

## Notebook structure
- Imports & Config → Helpers reload → Data loading (Polygon) → Feature engineering
- Hurst context & regime labeling → Event labeling (anchored VWAP reversion)
- EDA tables (base and conditionals) → Checks (monotonicity, stability)
- Per-asset gate + overall decision (STOP/KEEP)
- Visuals (global + per-asset; heatmaps, reliability, time-of-day, TTR)
- Packaging (omitted in this display repo)

If you’re viewing on GitHub, open `Intraday_Event_Study.ipynb` to step through the analysis.
