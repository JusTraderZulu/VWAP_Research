# VWAP Reversion EDA — Intraday Crypto & FX

This repository presents a clean, reproducible exploratory study of intraday VWAP reversion. The notebook estimates the probability that price reverts to VWAP after sufficiently large deviations and examines how that probability changes across higher-timeframe regimes, deviation size, time-of-day, and volatility.

## Research method (short)
- Data: Polygon 5-minute aggregates; crypto and FX.
- Event (anchored VWAP reversion):
  - Trigger: |close − anchored VWAP| / anchored VWAP ≥ threshold (e.g., 0.5–1.5% for crypto; 0.05–0.20% for FX). Anchored VWAP resets per UTC day.
  - Success: signed deviation flips across zero within N bars (N ∈ {5,10,20}).
- Context: Rolling Hurst (R/S) on 1H returns (EMA-smoothed). Regimes: MR ≤ 0.45, NOISY 0.45–0.55, TREND ≥ 0.55.
- EDA: Base P(revert) by regime & asset; conditional structure (Deviation × Regime × Side); time-of-day & volatility slices; stability (first vs second half).
- Decision gate: STOP if rare/flat/unstable; KEEP if base edge + monotonic conditionals + stability. Per-asset gating prevents weak tickers from blocking the study.

## Results summary

### FX (VWAP_Reversion_FX)
- Study: VWAP Reversion EDA
- Span: 2023-01-01 → 2025-09-01
- Decision: KEEP
- Assets (kept): ['C:EURUSD', 'C:GBPUSD', 'C:USDCHF', 'C:USDJPY']
- Triggers (kept): 350920
- Mean P(revert) (kept): 0.111
- Monotonicity pass rate: 0.89
- Stability median |Δp|: 0.024
- Packaged: studies/20250915_213929_VWAP_Reversion_FX

### Crypto (VWAP_Reversion)
- Study: VWAP Reversion EDA
- Span: 2023-01-01 → 2025-09-01
- Decision: KEEP
- Assets (kept): ['X:AVAXUSD', 'X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD']
- Triggers (kept): 322231
- Mean P(revert) (kept): 0.094
- Monotonicity pass rate: 0.81
- Stability median |Δp|: 0.012
- Packaged: studies/20250915_202810_VWAP_Reversion

## Next steps
- Calibration-ready modeling on kept assets (e.g., logistic with calibration); evaluate AUC, Brier, calibration.
- Regime-aware thresholds for decisioning; size by probability and volatility; time stop = horizon.
- Cost-aware backtest sketch (fees, slippage) with turnover controls; PnL by regime/time-of-day.
- Robustness: rolling/expanding time splits; sensitivity to thresholds/horizons; cross-asset consistency.

## Files
- Intraday_Event_Study.ipynb — crypto study (template flow; also used as a base for FX)
- FX_VWAP_Event_Study.ipynb — FX study (same framework, FX thresholds & tickers)

Open either notebook on GitHub to browse or run locally in a venv (first cell).
