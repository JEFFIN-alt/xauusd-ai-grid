#!/usr/bin/env python3
"""
CLEAN XAUUSD direct future-return direction walk-forward.

This file intentionally avoids all Pandas timestamp-index/reindex logic.

Task:
    Predict UP (1) vs DOWN/FLAT (0) at 15m, 30m, 60m.

Inputs:
    - Canonical XAUUSD M1: timestamp + close
    - Session-aware V4 feature file: 42 feature columns

Target:
    For row i and horizon h:
        future row = i + h
        require exact timestamp difference = h minutes
        require both endpoints active
        UP if future close > current close
        DOWN/FLAT otherwise

Validation:
    2022, 2023, 2024, 2025
    Training is expanding-window from 2019.
    60-minute purge before each validation year.
    2026 is never used.

No canonical data are modified.
"""

from __future__ import annotations

import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data/processed/xauusd_m1_2019_2026_canonical.csv"
V4 = ROOT / "data/processed/xauusd_m1_ml_features_v4_session.csv"
REPORT = ROOT / "reports/xauusd_direct_direction_walk_forward.csv"

HORIZONS = [15, 30, 60]
VAL_YEARS = [2022, 2023, 2024, 2025]
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
    "rolling_high_60", "rolling_low_60", "rolling_high_240", "rolling_low_240",
    "distance_high_60", "distance_low_60", "distance_high_240", "distance_low_240",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
]


def load_canonical():
    df = pd.read_csv(CANONICAL, usecols=["timestamp", "close"], low_memory=False)

    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(np.int64)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)

    if not np.isfinite(close).all() or (close <= 0).any():
        raise RuntimeError("Invalid canonical close values.")

    if not np.all(ts[1:] > ts[:-1]):
        raise RuntimeError("Canonical timestamps are not strictly increasing.")

    # Same session-closed proxy used by the established session-aware pipeline.
    flat_prev = close[1:] == close[:-1]
    starts = np.empty(len(close), dtype=np.int64)
    starts[0] = 0
    starts[1:] = np.where(flat_prev, 0, 1)

    run_id = np.cumsum(starts)
    run_lengths = np.bincount(run_id)
    closed = run_lengths[run_id] >= 60
    active = ~closed

    print(f"Canonical rows: {len(ts):,}")
    print("Timestamp unit: epoch-ms")
    print(f"Closed-session rows: {int(closed.sum()):,}")
    print(f"Active rows: {int(active.sum()):,}")

    return ts, close, active


def build_target(ts, close, active, horizon):
    n = len(ts)
    target = np.full(n, -1, dtype=np.int8)

    pos = np.arange(n - horizon, dtype=np.int64)
    future = pos + horizon

    exact = (ts[future] - ts[pos]) == horizon * 60_000
    valid = active[pos] & active[future] & exact

    vp = pos[valid]
    vf = future[valid]

    target[vp] = (close[vf] > close[vp]).astype(np.int8)

    good = target >= 0
    up = target == 1
    down = target == 0

    print(
        f"{horizon:>2}m target | valid={int(good.sum()):,} | "
        f"up={int(up.sum()):,} | down_or_zero={int(down.sum()):,}"
    )

    return target


def load_v4_columns():
    cols = list(pd.read_csv(V4, nrows=0).columns)
    required = ["timestamp", *FEATURES]
    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(f"Missing V4 columns: {missing}")
    return required


def collect_samples(canonical_ts, target):
    usecols = load_v4_columns()

    x_parts = []
    y_parts = []
    ts_parts = []

    global_filtered_row = 0

    reader = pd.read_csv(
        V4,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        chunk_ts = pd.to_numeric(
            chunk["timestamp"], errors="coerce"
        ).to_numpy(np.int64)

        years = pd.to_datetime(
            chunk_ts, unit="ms", utc=True
        ).year.to_numpy()

        year_mask = (years >= 2019) & (years <= 2025)
        count_in_period = int(year_mask.sum())

        if count_in_period == 0:
            continue

        chunk = chunk.loc[year_mask].copy()

        local = np.arange(len(chunk), dtype=np.int64)
        sample_mask = (
            (global_filtered_row + local) % STRIDE
        ) == 0

        chunk = chunk.loc[sample_mask].copy()

        if chunk.empty:
            global_filtered_row += count_in_period
            continue

        sampled_ts = pd.to_numeric(
            chunk["timestamp"], errors="coerce"
        ).to_numpy(np.int64)

        # Canonical timestamps are sorted and unique, so searchsorted is safe.
        idx = np.searchsorted(
            canonical_ts,
            sampled_ts,
            side="left",
        )

        in_bounds = idx < len(canonical_ts)
        exact = np.zeros(len(idx), dtype=bool)

        valid_idx = np.flatnonzero(in_bounds)
        exact[valid_idx] = (
            canonical_ts[idx[valid_idx]] == sampled_ts[valid_idx]
        )

        keep = np.zeros(len(idx), dtype=bool)
        exact_idx = np.flatnonzero(exact)

        if len(exact_idx):
            keep[exact_idx] = target[idx[exact_idx]] >= 0

        if keep.any():
            sub = chunk.loc[keep].copy()
            y = target[idx[keep]].astype(np.int8)

            X = sub[FEATURES].to_numpy(
                dtype=np.float32,
                copy=True,
            )

            finite = np.isfinite(X).all(axis=1)

            if finite.any():
                x_parts.append(X[finite])
                y_parts.append(y[finite])
                ts_parts.append(sampled_ts[keep][finite])

        global_filtered_row += count_in_period

        del chunk
        gc.collect()

    if not x_parts:
        raise RuntimeError("No V4 samples aligned to direct target.")

    X = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    out_ts = np.concatenate(ts_parts, axis=0)

    order = np.argsort(out_ts, kind="mergesort")
    X = X[order]
    y = y[order]
    out_ts = out_ts[order]

    # Defensive de-duplication of aligned samples only.
    unique = np.r_[True, out_ts[1:] != out_ts[:-1]]
    X = X[unique]
    y = y[unique]
    out_ts = out_ts[unique]

    print(f"ML rows: {len(y):,}")
    print(f"X shape: {X.shape}")
    print(f"UP %: {(y == 1).mean() * 100:.2f}")
    print(f"DOWN/FLAT %: {(y == 0).mean() * 100:.2f}")

    return X, y, out_ts


def run_fold(X, y, ts, horizon, val_year):
    dates = pd.to_datetime(ts, unit="ms", utc=True)
    years = dates.year.to_numpy()

    val_mask = years == val_year

    val_start = pd.Timestamp(
        f"{val_year}-01-01",
        tz="UTC",
    )

    val_start_ms = int(val_start.timestamp() * 1000)
    train_cutoff_ms = int(
        (val_start - pd.Timedelta(minutes=60)).timestamp()
        * 1000
    )

    train_mask = (
        (years >= 2019)
        & (years < val_year)
        & (ts < train_cutoff_ms)
    )

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    purge_mask = (
        (years >= 2019)
        & (years < val_year)
        & (ts >= train_cutoff_ms)
        & (ts < val_start_ms)
    )

    print("-" * 78)
    print(
        f"{horizon}m | Fold {val_year} | "
        f"train={len(X_train):,} | val={len(X_val):,} | "
        f"purged={int(purge_mask.sum()):,}"
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=200,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(75, verbose=False)
        ],
    )

    prob_up = model.predict_proba(X_val)[:, 1]
    pred = (prob_up >= 0.5).astype(np.int8)

    acc = accuracy_score(y_val, pred)
    bacc = balanced_accuracy_score(y_val, pred)
    auc = roc_auc_score(y_val, prob_up)
    cm = confusion_matrix(y_val, pred, labels=[0, 1])

    print(f"Best iteration: {model.best_iteration_}")
    print(f"Accuracy: {acc * 100:.4f}%")
    print(f"Balanced accuracy: {bacc * 100:.4f}%")
    print(f"ROC-AUC: {auc:.6f}")
    print(f"Actual UP: {y_val.mean() * 100:.2f}%")
    print(f"Predicted UP: {pred.mean() * 100:.2f}%")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    return {
        "horizon_min": horizon,
        "validation_year": val_year,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "purged_rows": int(purge_mask.sum()),
        "best_iteration": int(model.best_iteration_),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "roc_auc": float(auc),
        "actual_up_pct": float(y_val.mean()),
        "pred_up_pct": float(pred.mean()),
    }


def main():
    print("=" * 78)
    print("XAUUSD DIRECT FUTURE-RETURN DIRECTION WALK-FORWARD")
    print("=" * 78)
    print("Task: UP vs DOWN/FLAT without future-event conditioning")
    print("Features: session-aware V4 (42)")
    print(f"Sampling stride: {STRIDE}")
    print("Horizons: 15m / 30m / 60m")
    print("2026: COMPLETELY UNTOUCHED")
    print("=" * 78)

    if not CANONICAL.exists():
        raise FileNotFoundError(CANONICAL)
    if not V4.exists():
        raise FileNotFoundError(V4)

    canonical_ts, close, active = load_canonical()

    results = []

    for horizon in HORIZONS:
        print("=" * 78)
        print(f"HORIZON {horizon} MINUTES")
        print("=" * 78)

        target = build_target(
            canonical_ts,
            close,
            active,
            horizon,
        )

        X, y, out_ts = collect_samples(
            canonical_ts,
            target,
        )

        for year in VAL_YEARS:
            results.append(
                run_fold(
                    X,
                    y,
                    out_ts,
                    horizon,
                    year,
                )
            )

        del target, X, y, out_ts
        gc.collect()

    out = pd.DataFrame(results)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT, index=False)

    summary = (
        out.groupby(
            "horizon_min",
            as_index=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            mean_roc_auc=("roc_auc", "mean"),
            min_roc_auc=("roc_auc", "min"),
            max_roc_auc=("roc_auc", "max"),
        )
    )

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))
    print("=" * 78)
    print(f"Saved: {REPORT}")
    print("2026 was not used for fitting, selection, or evaluation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
