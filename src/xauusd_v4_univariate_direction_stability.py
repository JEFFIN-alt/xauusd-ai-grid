#!/usr/bin/env python3
"""
XAUUSD V4 univariate directional-signal stability test.

Purpose:
    Determine whether any existing V4 feature has a stable relationship
    with direct future XAUUSD direction, before adding more complexity.

Targets:
    15m / 30m / 60m
    UP = future close > current close
    DOWN/FLAT = future close <= current close

Evaluation:
    2022, 2023, 2024, 2025 walk-forward years are kept separate.
    For every V4 feature, compute per-year ROC-AUC using the feature itself
    as a continuous score.

Interpretation:
    AUC = 0.50 means no ranking information.
    AUC > 0.50 means larger feature values associate with UP.
    AUC < 0.50 means the inverse relationship is stronger.
    We report both raw AUC and absolute directional advantage
    abs(AUC - 0.50), but DO NOT select orientation using validation data.

This is diagnostic only. No P&L and no 2026 use.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

CANONICAL = ROOT / "data/processed/xauusd_m1_2019_2026_canonical.csv"
V4 = ROOT / "data/processed/xauusd_m1_ml_features_v4_session.csv"
REPORT = ROOT / "reports/xauusd_v4_univariate_direction_stability.csv"

HORIZONS = [15, 30, 60]
YEARS = [2022, 2023, 2024, 2025]
STRIDE = 5
CHUNK_SIZE = 250_000

FEATURES = [
    "return_1m", "return_5m", "return_15m", "return_30m", "return_60m",
    "candle_range", "body", "abs_body", "upper_wick", "lower_wick",
    "body_range_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "volatility_15m", "volatility_30m", "volatility_60m", "volatility_240m",
    "momentum_5m", "momentum_15m", "momentum_30m", "momentum_60m",
    "sma_15", "sma_60", "sma_240",
    "distance_sma_15", "distance_sma_60", "distance_sma_240",
    "sma_15_slope", "sma_60_slope", "sma_240_slope",
    "rolling_high_60", "rolling_low_60",
    "rolling_high_240", "rolling_low_240",
    "distance_high_60", "distance_low_60",
    "distance_high_240", "distance_low_240",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
]


def load_data():
    df = pd.read_csv(
        CANONICAL,
        usecols=["timestamp", "close"],
        low_memory=False,
    )
    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(np.int64)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)

    if not np.all(ts[1:] > ts[:-1]):
        raise RuntimeError("Canonical timestamps are not strictly increasing.")

    # Established session-closed proxy.
    flat_prev = close[1:] == close[:-1]
    starts = np.empty(len(close), dtype=np.int64)
    starts[0] = 0
    starts[1:] = np.where(flat_prev, 0, 1)

    run_id = np.cumsum(starts)
    lengths = np.bincount(run_id)
    closed = lengths[run_id] >= 60
    active = ~closed

    print(f"Canonical rows: {len(ts):,}")
    print(f"Closed rows: {int(closed.sum()):,}")
    print(f"Active rows: {int(active.sum()):,}")

    return ts, close, active


def build_target(ts, close, active, horizon):
    n = len(ts)
    target = np.full(n, -1, dtype=np.int8)

    pos = np.arange(n - horizon, dtype=np.int64)
    fut = pos + horizon

    exact = (ts[fut] - ts[pos]) == horizon * 60_000
    valid = active[pos] & active[fut] & exact

    vp = pos[valid]
    vf = fut[valid]

    target[vp] = (close[vf] > close[vp]).astype(np.int8)

    print(
        f"{horizon}m target | valid={int((target>=0).sum()):,} | "
        f"UP={int((target==1).sum()):,} | DOWN={int((target==0).sum()):,}"
    )
    return target


def collect_samples(canonical_ts, target):
    cols = ["timestamp", *FEATURES]
    header = list(pd.read_csv(V4, nrows=0).columns)
    missing = [c for c in cols if c not in header]
    if missing:
        raise RuntimeError(f"V4 missing columns: {missing}")

    X_parts = []
    y_parts = []
    ts_parts = []

    global_row = 0

    for chunk in pd.read_csv(
        V4,
        usecols=cols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk_ts = pd.to_numeric(
            chunk["timestamp"], errors="coerce"
        ).to_numpy(np.int64)

        years = pd.to_datetime(
            chunk_ts, unit="ms", utc=True
        ).year.to_numpy()

        mask = (years >= 2019) & (years <= 2025)
        n_period = int(mask.sum())

        if n_period == 0:
            continue

        chunk = chunk.loc[mask].copy()

        pos = np.arange(len(chunk), dtype=np.int64)
        sampled = ((global_row + pos) % STRIDE) == 0
        chunk = chunk.loc[sampled].copy()

        if chunk.empty:
            global_row += n_period
            continue

        sample_ts = pd.to_numeric(
            chunk["timestamp"], errors="coerce"
        ).to_numpy(np.int64)

        idx = np.searchsorted(
            canonical_ts,
            sample_ts,
            side="left",
        )

        in_bounds = idx < len(canonical_ts)
        exact = np.zeros(len(idx), dtype=bool)
        loc = np.flatnonzero(in_bounds)
        exact[loc] = canonical_ts[idx[loc]] == sample_ts[loc]

        keep = np.zeros(len(idx), dtype=bool)
        loc = np.flatnonzero(exact)
        keep[loc] = target[idx[loc]] >= 0

        if keep.any():
            sub = chunk.loc[keep]
            y = target[idx[keep]].astype(np.int8)
            X = sub[FEATURES].to_numpy(np.float32, copy=True)

            finite = np.isfinite(X).all(axis=1)

            if finite.any():
                X_parts.append(X[finite])
                y_parts.append(y[finite])
                ts_parts.append(sample_ts[keep][finite])

        global_row += n_period
        del chunk
        gc.collect()

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    out_ts = np.concatenate(ts_parts)

    order = np.argsort(out_ts, kind="mergesort")
    X = X[order]
    y = y[order]
    out_ts = out_ts[order]

    unique = np.r_[True, out_ts[1:] != out_ts[:-1]]
    X = X[unique]
    y = y[unique]
    out_ts = out_ts[unique]

    print(f"ML rows: {len(y):,}")
    print(f"X shape: {X.shape}")

    return X, y, out_ts


def analyze(X, y, ts, horizon):
    years = pd.to_datetime(
        ts, unit="ms", utc=True
    ).year.to_numpy()

    rows = []

    for year in YEARS:
        mask = years == year
        yy = y[mask]

        for j, feature in enumerate(FEATURES):
            values = X[mask, j]
            finite = np.isfinite(values)

            if finite.sum() == 0 or np.unique(yy[finite]).size < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(
                    yy[finite],
                    values[finite],
                )

            rows.append({
                "horizon_min": horizon,
                "validation_year": year,
                "feature": feature,
                "auc": auc,
                "advantage_abs": (
                    abs(auc - 0.5) if np.isfinite(auc) else np.nan
                ),
                "n": int(finite.sum()),
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("XAUUSD V4 UNIVARIATE DIRECTIONAL SIGNAL STABILITY")
    print("=" * 78)
    print("Features: existing V4 only")
    print("Horizons: 15m / 30m / 60m")
    print("Walk-forward years: 2022-2025")
    print("2026: COMPLETELY UNTOUCHED")
    print("=" * 78)

    ts, close, active = load_data()

    all_results = []

    for horizon in HORIZONS:
        print("=" * 78)
        print(f"HORIZON {horizon} MINUTES")
        print("=" * 78)

        target = build_target(ts, close, active, horizon)
        X, y, sample_ts = collect_samples(ts, target)

        result = analyze(X, y, sample_ts, horizon)
        all_results.append(result)

        # Print top features by mean absolute directional advantage.
        summary = (
            result.groupby("feature", as_index=False)
            .agg(
                mean_auc=("auc", "mean"),
                std_auc=("auc", "std"),
                min_auc=("auc", "min"),
                max_auc=("auc", "max"),
                mean_advantage=("advantage_abs", "mean"),
            )
            .sort_values(
                ["mean_advantage", "std_auc"],
                ascending=[False, True],
            )
            .head(10)
        )

        print("\nTop 10 univariate features by mean |AUC-0.50|:")
        print(summary.to_string(index=False))

        del target, X, y, sample_ts, result
        gc.collect()

    out = pd.concat(all_results, ignore_index=True)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT, index=False)

    print("=" * 78)
    print("COMPLETED")
    print(f"Saved: {REPORT}")
    print("2026 was not used.")
    print("=" * 78)


if __name__ == "__main__":
    main()
