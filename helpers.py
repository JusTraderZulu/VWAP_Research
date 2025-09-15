from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

try:
    from polygon import RESTClient
except Exception:  # polygon may not be installed yet (first run)
    RESTClient = None  # type: ignore


# -----------------------------
# Auth & Client
# -----------------------------

def read_polygon_key(path: Union[str, Path] = "polygon_key.txt") -> str:
    """Read Polygon API key.

    Search order:
    1) Environment variable POLYGON_API_KEY
    2) Provided path
    3) Search upwards from CWD for a file named polygon_key.txt
    """
    env_key = os.environ.get("POLYGON_API_KEY")
    if env_key:
        return env_key.strip()

    p = Path(path)
    if p.exists():
        key = p.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("Polygon API key file is empty")
        return key

    # Walk up parents to find polygon_key.txt
    cur = Path.cwd().resolve()
    for parent in [cur] + list(cur.parents):
        candidate = parent / "polygon_key.txt"
        if candidate.exists():
            key = candidate.read_text(encoding="utf-8").strip()
            if not key:
                raise ValueError("Polygon API key file is empty")
            return key

    raise FileNotFoundError("Polygon API key not found via env or polygon_key.txt in parent dirs")


def get_polygon_client(api_key: str):
    if RESTClient is None:
        raise ImportError(
            "polygon-api-client is not installed yet. Install and re-run the cell."
        )
    return RESTClient(api_key)


# -----------------------------
# Data Fetching
# -----------------------------

def _to_unix_ms(dt: Union[str, datetime]) -> int:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_aggregates(
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    multiplier: int = 5,
    timespan: str = "minute",
    limit: int = 50000,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    Safe Polygon aggregates downloader with pagination and retries.

    Returns tz-aware DataFrame indexed by datetime (UTC) with
    columns: open, high, low, close, volume, vwap.
    """
    api_key = read_polygon_key()
    client = get_polygon_client(api_key)

    # Map common symbols to Polygon format (e.g., BTC-USD -> X:BTCUSD)
    def normalize_polygon_ticker(sym: str) -> str:
        s = str(sym).upper().replace("/", "-")
        if s.startswith(("X:", "C:", "I:", "O:", "A:")):
            return s
        if s.endswith("-USD"):
            base = s.split("-")[0]
            return f"X:{base}USD"
        return s

    ticker = normalize_polygon_ticker(symbol)

    # Prefer ISO date strings for SDK compatibility
    start_str = pd.Timestamp(start).date().isoformat()
    end_str = pd.Timestamp(end).date().isoformat()

    all_rows: List[Dict] = []

    for attempt in range(max_retries):
        try:
            # Newer SDK supports all_pages=True
            resp = client.list_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_str,
                to=end_str,
                limit=limit,
                adjusted=True,
                sort="asc",
                all_pages=True,
            )
            items = list(resp)
            break
        except TypeError:
            # Fallback for older SDKs without all_pages
            resp = client.list_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_str,
                to=end_str,
                limit=limit,
                adjusted=True,
                sort="asc",
            )
            items = list(resp)
            break
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff_seconds * (2 ** attempt))

    for bar in items:
        all_rows.append(
            {
                "ts": pd.to_datetime(getattr(bar, "timestamp", getattr(bar, "t", 0)), unit="ms", utc=True),
                "open": getattr(bar, "open", getattr(bar, "o", np.nan)),
                "high": getattr(bar, "high", getattr(bar, "h", np.nan)),
                "low": getattr(bar, "low", getattr(bar, "l", np.nan)),
                "close": getattr(bar, "close", getattr(bar, "c", np.nan)),
                "volume": getattr(bar, "volume", getattr(bar, "v", np.nan)),
                "vwap": getattr(bar, "vwap", getattr(bar, "vw", np.nan)),
            }
        )

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"]).astype(float)

    df = pd.DataFrame(all_rows).drop_duplicates("ts").sort_values("ts").set_index("ts")

    # Ensure required columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df:
            df[col] = np.nan

    # Compute VWAP if missing or NaN-heavy
    if "vwap" not in df or df["vwap"].isna().mean() > 0.1:
        df["vwap"] = compute_vwap(df)

    return df.tz_localize("UTC") if df.index.tz is None else df


# -----------------------------
# Feature Engineering
# -----------------------------

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    price = (df.get("high") + df.get("low") + df.get("close")) / 3.0
    vol = df.get("volume").replace(0, np.nan)
    return (price * vol).div(vol).fillna(method="ffill").fillna(method="bfill")


def compute_anchored_vwap(df: pd.DataFrame, anchor: str = "utc_day") -> pd.Series:
    """Compute anchored VWAP within each day (UTC) by default.

    Uses per-bar VWAP if available, otherwise typical price.
    anchored_vwap_t = cumsum(price_t * vol_t) / cumsum(vol_t) within each anchor bucket.
    """
    if df.empty:
        return pd.Series(index=df.index, dtype=float)
    vol = df.get("volume").astype(float)
    # Choose price proxy
    if "vwap" in df and df["vwap"].notna().any():
        price = df["vwap"].astype(float)
    else:
        price = ((df.get("high") + df.get("low") + df.get("close")) / 3.0).astype(float)

    # Anchor buckets
    if anchor == "utc_day":
        bucket = df.index.tz_convert("UTC").date if df.index.tz is not None else df.index.tz_localize("UTC").date
    else:
        # fallback: utc day
        bucket = df.index.tz_convert("UTC").date if df.index.tz is not None else df.index.tz_localize("UTC").date

    bucket_series = pd.Series(bucket, index=df.index)
    grouped = bucket_series
    num = (price * vol).groupby(grouped).cumsum()
    den = vol.groupby(grouped).cumsum().replace(0, np.nan)
    avwap = (num / den).astype(float)
    avwap.name = "anchored_vwap"
    return avwap


def rolling_sigma(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window=window, min_periods=max(2, window // 3)).std()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window // 2).mean()


def time_of_day_bucket(index: pd.Index, asset: str) -> pd.Series:
    """Return simple time-of-day labels given any index.

    Coerces to tz-aware UTC before bucketing. Works even if index is RangeIndex.
    """
    # Coerce index to UTC DatetimeIndex robustly
    if not isinstance(index, pd.DatetimeIndex):
        idx = pd.to_datetime(index, utc=True, errors="coerce")
    else:
        idx = index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")

    if any(x in asset.upper() for x in ["BTC", "ETH", "SOL"]):
        # crypto 24/7: split UTC day into thirds
        hours = idx.hour
        bins = pd.cut(hours, bins=[-1, 7, 15, 23], labels=["asia", "eu", "us"])
        return pd.Series(bins, index=idx, name="tod")
    # equities: focus regular session (approx in UTC)
    # 14:30-21:00 UTC corresponds to 9:30-16:00 ET (no DST handling here for simplicity)
    utc_hour_min = idx.hour * 60 + idx.minute
    open_min = 14 * 60 + 30
    close_min = 21 * 60
    labels = []
    for m in utc_hour_min:
        if m < open_min:
            labels.append("pre")
        elif m <= open_min + 60:
            labels.append("open")
        elif m < close_min - 60:
            labels.append("mid")
        elif m <= close_min:
            labels.append("close")
        else:
            labels.append("post")
    return pd.Series(labels, index=idx, name="tod")


# -----------------------------
# Hurst Exponent (R/S)
# -----------------------------

def hurst_rs(x: np.ndarray) -> float:
    """Estimate Hurst exponent via rescaled range (R/S) method.

    H ≈ log(R/S) / log(N) for a window of length N.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if N < 10:
        return np.nan
    s = x.std(ddof=1)
    if not np.isfinite(s) or s == 0:
        return np.nan
    y = x - x.mean()
    z = np.cumsum(y)
    R = float(z.max() - z.min())
    if R <= 0:
        return np.nan
    H = math.log(R / s) / math.log(N)
    # Clamp to [0,1]
    if not np.isfinite(H):
        return np.nan
    return max(0.0, min(1.0, H))


def rolling_hurst(series: pd.Series, win: int = 200, ema_span: int = 8) -> pd.Series:
    vals = series.dropna().values
    if len(vals) == 0:
        return pd.Series(index=series.index, dtype=float)
    roll = series.rolling(window=win, min_periods=win // 2).apply(lambda a: hurst_rs(np.asarray(a)), raw=False)
    if ema_span and ema_span > 1:
        roll = roll.ewm(span=ema_span, adjust=False).mean()
    return roll


def label_h_regime(
    H: pd.Series,
    low: float = 0.45,
    high: float = 0.55,
    hysteresis_k: Optional[int] = None,
) -> pd.Series:
    labels = pd.Series(index=H.index, dtype=object)
    labels[H <= low] = "MR"
    labels[(H > low) & (H < high)] = "NOISY"
    labels[H >= high] = "TREND"

    if hysteresis_k and hysteresis_k > 1:
        labels = labels.copy()
        current = None
        count = 0
        for i, v in enumerate(labels.values):
            if v == current:
                count += 1
            else:
                count = 1
                current = v
            if count < hysteresis_k and i > 0:
                labels.iloc[i] = labels.iloc[i - 1]
    return labels.astype("category")


# -----------------------------
# Event Framework
# -----------------------------

def pct_from_vwap(close: pd.Series, vwap: pd.Series) -> pd.Series:
    return (close / vwap - 1.0) * 100.0


def z_from_ma(series: pd.Series, window: int = 20) -> pd.Series:
    ma = series.rolling(window=window, min_periods=max(2, window // 3)).mean()
    sd = series.rolling(window=window, min_periods=max(2, window // 3)).std()
    return (series - ma) / sd.replace(0, np.nan)


def evaluate_event_trigger(
    df: pd.DataFrame,
    trigger_fn: Callable[[pd.DataFrame, Dict], pd.Series],
    trigger_params: Dict,
) -> pd.Series:
    flags = trigger_fn(df, trigger_params)
    return flags.fillna(False).astype(bool)


def evaluate_event_outcome(
    df: pd.DataFrame,
    trigger_flags: pd.Series,
    outcome_fn: Callable[[pd.DataFrame, pd.Series, Dict], pd.Series],
    outcome_params: Dict,
) -> pd.Series:
    out = outcome_fn(df, trigger_flags, outcome_params)
    return out.fillna(False).astype(bool)


# Example event implementations
def trigger_vwap_reversion(df: pd.DataFrame, params: Dict) -> pd.Series:
    dev_type = params.get("dev_type", "pct_from_vwap")
    thresh = params.get("threshold", 1.0)  # percent
    if dev_type == "pct_from_vwap":
        dev = pct_from_vwap(df["close"], df["vwap"]).abs()
        return dev > thresh
    elif dev_type == "z_from_ma":
        dev = z_from_ma(df["close"], window=params.get("ma_window", 20)).abs()
        return dev > params.get("z", 1.0)
    return pd.Series(False, index=df.index)


def outcome_vwap_reversion(df: pd.DataFrame, flags: pd.Series, params: Dict) -> pd.Series:
    horizon = int(params.get("horizon", 10))
    # success if price returns to VWAP within horizon
    dev = (df["close"] / df["vwap"] - 1.0)
    fut_min_abs = dev.shift(-np.arange(1, horizon + 1)).abs().min(axis=1)
    return (fut_min_abs <= params.get("success_abs_dev", 0.0)).where(flags, False)


def deviation_pct_from_vwap(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """Signed percent deviation from VWAP in percent units."""
    return (close / vwap - 1.0) * 100.0


def label_vwap_reversion(
    df: pd.DataFrame,
    threshold_pct: float,
    horizon: int,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Label VWAP reversion event with explicit side logic and time-to-reversion.

    - Trigger: |dev_pct| >= threshold_pct
    - Success (up): any future low_tau <= vwap_tau within horizon
      Success (down): any future high_tau >= vwap_tau within horizon

    Returns (trigger, success, side, time_to_reversion)
    where side in {'up','down'} and time_to_reversion is number of bars to first reversion (NaN if none).
    """
    if not {"close", "vwap", "high", "low"}.issubset(df.columns):
        raise ValueError("DataFrame must contain close, vwap, high, low columns")

    # Baseline: anchored VWAP (compute if missing)
    baseline = df.get("anchored_vwap")
    if baseline is None or baseline.isna().all():
        baseline = compute_anchored_vwap(df, anchor="utc_day")

    dev_pct = (df["close"] / baseline - 1.0) * 100.0  # signed
    side = pd.Series(np.where(dev_pct >= 0, "up", "down"), index=df.index, dtype=object)
    trigger = dev_pct.abs() >= float(threshold_pct)

    # Build future matrices for deviation sign using CLOSE vs VWAP at each future bar
    fut_close_cols = [df["close"].shift(-k) for k in range(1, horizon + 1)]
    fut_vwap_cols = [baseline.shift(-k) for k in range(1, horizon + 1)]
    if len(fut_close_cols) == 0:
        # No horizon requested
        return (
            pd.Series(False, index=df.index),
            pd.Series(False, index=df.index),
            side.astype("category"),
            pd.Series(np.nan, index=df.index),
        )
    fut_close = pd.concat(fut_close_cols, axis=1)
    fut_vwap = pd.concat(fut_vwap_cols, axis=1)
    if fut_close.shape[0] == 0 or fut_close.shape[1] == 0:
        return (
            pd.Series(False, index=df.index),
            pd.Series(False, index=df.index),
            side.astype("category"),
            pd.Series(np.nan, index=df.index),
        )
    dev_future = (fut_close.values / fut_vwap.values) - 1.0

    # Success if deviation sign flips back across zero within horizon
    up_success_any = (dev_future <= 0).any(axis=1)
    down_success_any = (dev_future >= 0).any(axis=1)

    success = pd.Series(False, index=df.index)
    success.iloc[: len(up_success_any)] = (
        (side.values[: len(up_success_any)] == "up") & up_success_any
    ) | (
        (side.values[: len(down_success_any)] == "down") & down_success_any
    )
    success = success & trigger

    # Time-to-reversion (bars to first hit)
    def first_true(a: np.ndarray) -> float:
        idx = np.argmax(a)
        return float(idx + 1) if a.any() else np.nan

    if dev_future.shape[0] == 0:
        up_first = np.array([])
        down_first = np.array([])
    else:
        up_first = np.apply_along_axis(first_true, 1, (dev_future <= 0))
        down_first = np.apply_along_axis(first_true, 1, (dev_future >= 0))
    ttr = np.where(side.values[: len(up_first)] == "up", up_first, down_first)
    ttr_series = pd.Series(ttr, index=df.index).where(success, np.nan)

    return trigger.astype(bool), success.astype(bool), side.astype("category"), ttr_series


# -----------------------------
# EDA Utilities
# -----------------------------

def conditional_probability_table(
    outcome: pd.Series,
    by: Dict[str, pd.Series],
    min_count: int = 20,
) -> pd.DataFrame:
    df = pd.DataFrame({"outcome": outcome.astype(int)})
    for k, v in by.items():
        df[k] = v
    grouped = df.groupby(list(by.keys()), observed=True)
    counts = grouped.size().rename("n")
    prob = grouped["outcome"].mean().rename("p")
    res = pd.concat([prob, counts], axis=1)
    res.loc[res["n"] < min_count, "p"] = np.nan
    return res.reset_index()


def reliability_curve(probs: pd.Series, outcomes: pd.Series, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": probs, "y": outcomes.astype(int)})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    grp = df.groupby("bin")
    return grp.mean(numeric_only=True).assign(n=grp.size())[["p", "y", "n"]]


def plot_heatmap(table: pd.DataFrame, x: str, y: str, value: str, title: str = "") -> plt.Figure:
    """Simple heatmap using imshow from a pivoted table."""
    pivot = table.pivot(index=y, columns=x, values=value)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=value)
    fig.tight_layout()
    return fig


def split_time_halves(index: pd.DatetimeIndex) -> pd.Series:
    """Return a series with values {'first','second'} for time-based halves."""
    n = len(index)
    labels = np.array(["first"] * (n // 2) + ["second"] * (n - n // 2))
    return pd.Series(labels, index=index, name="half")


def terciles(series: pd.Series, labels: Tuple[str, str, str] = ("low", "mid", "high")) -> pd.Series:
    return pd.qcut(series, q=3, labels=list(labels), duplicates="drop")


# -----------------------------
# Modeling Utilities
# -----------------------------

def time_split_indices(n: int, train_frac: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    split = int(n * train_frac)
    return np.arange(0, split), np.arange(split, n)


def compute_basic_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    metrics = {
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": float(brier_score_loss(y_true, p)),
        "logloss": float(log_loss(y_true, p)),
    }
    return metrics


# -----------------------------
# Packaging
# -----------------------------

def package_study(
    study_name: str,
    cfg: Dict,
    figures: Dict[str, plt.Figure] | List[plt.Figure],
    tables: Dict[str, pd.DataFrame],
    model=None,
    extras: Optional[Dict] = None,
) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in study_name)
    out_dir = Path("studies") / f"{ts}_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Save figures under plots/
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(figures, dict):
        for name, fig in figures.items():
            fig_path = plots_dir / f"{name}.png"
            fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    elif isinstance(figures, list):
        for i, fig in enumerate(figures):
            fig_path = plots_dir / f"fig_{i+1}.png"
            fig.savefig(fig_path, dpi=160, bbox_inches="tight")

    # Save tables
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    # Save model (optional)
    if model is not None:
        try:
            import joblib

            joblib.dump(model, out_dir / "model.pkl")
        except Exception:
            pass

    # Extras
    if extras:
        (out_dir / "extras.json").write_text(json.dumps(extras, indent=2), encoding="utf-8")

    # README
    (out_dir / "README_run.md").write_text(
        "\n".join(
            [
                f"Study: {study_name}",
                f"Timestamp: {ts}",
                "Artifacts: plots/*.png, tables (.csv), model.pkl (optional)",
                "See config.json for full parameters.",
            ]
        ),
        encoding="utf-8",
    )

    return out_dir


DEV_ORDER_DEFAULT: List[str] = ["<t0", "t0-t1", "t1-t2", ">t2"]


def monotonicity_check(
    cond_table: pd.DataFrame,
    dev_col: str = "dev",
    value_col: str = "p",
    group_cols: Tuple[str, ...] = ("asset", "reg", "side"),
    dev_order: Optional[List[str]] = None,
) -> Tuple[float, pd.DataFrame]:
    if dev_order is None:
        dev_order = DEV_ORDER_DEFAULT
    results: List[Dict] = []
    for keys, grp in cond_table.groupby(list(group_cols)):
        seq = grp.set_index(dev_col).reindex(dev_order)[value_col].values
        seq = [x for x in seq if pd.notna(x)]
        mono_inc = True
        if len(seq) > 1:
            mono_inc = all(b >= a for a, b in zip(seq, seq[1:]))
        row = {c: k for c, k in zip(group_cols, keys)}
        row["mono_inc"] = bool(mono_inc)
        results.append(row)
    res_df = pd.DataFrame(results)
    pass_rate = float(res_df["mono_inc"].mean()) if len(res_df) else 0.0
    return pass_rate, res_df


def stability_summary(
    assets_data: Dict[str, pd.DataFrame],
    H_regimes: Dict[str, pd.Series],
    labels: Dict[str, Dict[str, pd.Series]],
    dev_thresholds: List[float],
    dev_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    if dev_order is None:
        dev_order = DEV_ORDER_DEFAULT
    rows: List[Dict] = []
    for sym, df in assets_data.items():
        half = split_time_halves(df.index)
        reg = H_regimes[sym].reindex(df.index, method="ffill")
        trig = labels[sym]["trigger"]
        succ = labels[sym]["success"]
        side = labels[sym]["side"]
        dev_abs = (df["close"] / df["vwap"] - 1).abs() * 100
        dev_bin = pd.cut(
            dev_abs,
            bins=[-1, dev_thresholds[0], dev_thresholds[1], dev_thresholds[2], 1e9],
            labels=dev_order,
        )
        t1 = conditional_probability_table((succ & trig) & (half == "first"), {"dev": dev_bin, "reg": reg, "side": side})
        t2 = conditional_probability_table((succ & trig) & (half == "second"), {"dev": dev_bin, "reg": reg, "side": side})
        m = pd.merge(t1, t2, on=["dev", "reg", "side"], suffixes=("_first", "_second"))
        if len(m) == 0:
            rows.append({"asset": sym, "max_abs_delta_p": np.nan, "median_abs_delta_p": np.nan})
            continue
        m["delta_p"] = (m["p_second"] - m["p_first"]).abs()
        rows.append({
            "asset": sym,
            "max_abs_delta_p": float(m["delta_p"].max()),
            "median_abs_delta_p": float(m["delta_p"].median()),
        })
    return pd.DataFrame(rows)


def decision_gate(
    base_table: pd.DataFrame,
    cond_table: pd.DataFrame,
    stability_df: pd.DataFrame,
    min_triggers: int = 200,
    min_mono_pass: float = 0.6,
    max_median_delta: float = 0.02,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    total_trigs = int(base_table["triggers"].sum()) if "triggers" in base_table else 0
    if total_trigs < min_triggers:
        reasons.append(f"too few triggers: {total_trigs} < {min_triggers}")
    pass_rate, _ = monotonicity_check(cond_table)
    if not (pass_rate >= min_mono_pass):
        reasons.append(f"monotonicity weak: pass_rate={pass_rate:.2f} < {min_mono_pass:.2f}")
    if len(stability_df):
        med = float(stability_df["median_abs_delta_p"].median())
        if not (med <= max_median_delta):
            reasons.append(f"stability weak: median |Δp|={med:.3f} > {max_median_delta:.3f}")
    decision = "KEEP" if len(reasons) == 0 else "STOP"
    return decision, reasons


__all__ = [
    "read_polygon_key",
    "get_polygon_client",
    "fetch_aggregates",
    "compute_vwap",
    "rolling_sigma",
    "atr",
    "time_of_day_bucket",
    "hurst_rs",
    "rolling_hurst",
    "label_h_regime",
    "pct_from_vwap",
    "z_from_ma",
    "evaluate_event_trigger",
    "evaluate_event_outcome",
    "trigger_vwap_reversion",
    "outcome_vwap_reversion",
    "deviation_pct_from_vwap",
    "label_vwap_reversion",
    "conditional_probability_table",
    "reliability_curve",
    "plot_heatmap",
    "split_time_halves",
    "terciles",
    "time_split_indices",
    "compute_basic_metrics",
    "package_study",
    "monotonicity_check",
    "stability_summary",
    "decision_gate",
]


