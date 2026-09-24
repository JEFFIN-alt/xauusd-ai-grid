#!/usr/bin/env python3
"""
Leak-safe small-feature directional model for XAUUSD.

Purpose:
    Test whether the weak direct-direction signal is concentrated in a small,
    stable subset of the existing 42 V4 features.

For every outer validation year:
    1. Build the direct UP/DOWN target from canonical XAUUSD.
    2. Use ONLY the training years to rank the 42 V4 features by univariate AUC.
    3. Orient each selected feature so its training relationship points toward UP.
    4. Standardize selected features using training data only.
    5. Fit logistic regression using training data only.
    6. Evaluate on the untouched validation year.

Feature counts tested:
    K = 3, 5, 10

Horizons:
    15m, 30m, 60m

Validation:
    2022, 2023, 2024, 2025
    60-minute purge before each validation-year boundary.

2026:
    COMPLETELY UNTOUCHED.

This is diagnostic/ML research only.
No P&L is calculated and no feature selection uses validation data.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

CANONICAL = ROOT / "data/processed/xauusd_m1_2019_2026_canonical.csv"
V4 = ROOT / "data/processed/xauusd_m1_ml_features_v4_session.csv"
REPORT = ROOT / "reports/xauusd_small_feature_direction_walk_forward.csv"

HORIZONS = [15, 30, 60]
YEARS = [2022, 2023, 2024, 2025]
TOP_KS = [3, 5, 10]

STRIDE = 5
CHUNK_SIZE = 250_000
PURGE_MINUTES = 60

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


def load_canonical():
    df = pd.read_csv(
        CANONICAL,
        usecols=["timestamp", "close"],
        low_memory=False,
    )

    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(np.int64)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)

    if not np.all(ts[1:] > ts[:-1]):
        raise RuntimeError("Canonical timestamps are not strictly increasing.")

    if not np.isfinite(close).all() or (close <= 0).any():
        raise RuntimeError("Canonical close contains invalid values.")

    # Established session-closed proxy:
    # unchanged close for >=60 consecutive rows => closed.
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
        f"{horizon}m target | valid={int((target >= 0).sum()):,} | "
        f"UP={int((target == 1).sum()):,} | "
        f"DOWN={int((target == 0).sum()):,}"
    )

    return target


def load_v4_header():
    cols = list(pd.read_csv(V4, nrows=0).columns)
    required = ["timestamp", *FEATURES]
    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(f"V4 missing columns: {missing}")
    return required


def collect_samples(canonical_ts, target):
    usecols = load_v4_header()

    xs = []
    ys = []
    timestamps = []

    global_row = 0

    for chunk in pd.read_csv(
        V4,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk_ts = pd.to_numeric(
            chunk["timestamp"], errors="coerce"
        ).to_numpy(np.int64)

        years = pd.to_datetime(
            chunk_ts, unit="ms", utc=True
        ).year.to_numpy()

        period = (years >= 2019) & (years <= 2025)
        n_period = int(period.sum())

        if n_period == 0:
            continue

        chunk = chunk.loc[period].copy()

        local = np.arange(len(chunk), dtype=np.int64)
        sample = ((global_row + local) % STRIDE) == 0

        chunk = chunk.loc[sample].copy()

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
        exact[loc] = (
            canonical_ts[idx[loc]] == sample_ts[loc]
        )

        keep = np.zeros(len(idx), dtype=bool)
        loc = np.flatnonzero(exact)
        keep[loc] = target[idx[loc]] >= 0

        if keep.any():
            sub = chunk.loc[keep]
            y = target[idx[keep]].astype(np.int8)

            X = sub[FEATURES].to_numpy(
                dtype=np.float32,
                copy=True,
            )

            finite = np.isfinite(X).all(axis=1)

            if finite.any():
                xs.append(X[finite])
                ys.append(y[finite])
                timestamps.append(sample_ts[keep][finite])

        global_row += n_period
        del chunk
        gc.collect()

    if not xs:
        raise RuntimeError("No samples collected.")

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    out_ts = np.concatenate(timestamps, axis=0)

    order = np.argsort(out_ts, kind="mergesort")
    X = X[order]
    y = y[order]
    out_ts = out_ts[order]

    # Defensive deduplication after V4 alignment.
    unique = np.r_[True, out_ts[1:] != out_ts[:-1]]
    X = X[unique]
    y = y[unique]
    out_ts = out_ts[unique]

    print(f"ML rows: {len(y):,}")
    print(f"X shape: {X.shape}")

    return X, y, out_ts


def rank_features(X_train, y_train):
    """
    Rank V4 features using training data only.
    If AUC < 0.5, the feature is conceptually inverted.
    """
    ranked = []

    for j, feature in enumerate(FEATURES):
        values = X_train[:, j]
        finite = np.isfinite(values)

        if finite.sum() < 100 or np.unique(y_train[finite]).size < 2:
            continue

        auc = roc_auc_score(
            y_train[finite],
            values[finite],
        )

        # An inverse relationship is still information; store its
        # training orientation separately.
        advantage = abs(auc - 0.5)

        ranked.append(
            {
                "index": j,
                "feature": feature,
                "train_auc_raw": float(auc),
                "train_advantage": float(advantage),
                "sign": 1.0 if auc >= 0.5 else -1.0,
            }
        )

    ranked.sort(
        key=lambda r: r["train_advantage"],
        reverse=True,
    )

    return ranked


def run_model(X_train, y_train, X_val, y_val, ranked, k):
    selected = ranked[:k]
    indices = [r["index"] for r in selected]

    A = X_train[:, indices].astype(np.float64)
    B = X_val[:, indices].astype(np.float64)

    signs = np.array(
        [r["sign"] for r in selected],
        dtype=np.float64,
    )

    # Orient using training data only.
    A *= signs
    B *= signs

    scaler = StandardScaler()
    A = scaler.fit_transform(A)
    B = scaler.transform(B)

    model = LogisticRegression(
        C=0.1,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )

    model.fit(A, y_train)

    score = model.predict_proba(B)[:, 1]
    pred = (score >= 0.5).astype(np.int8)

    acc = accuracy_score(y_val, pred)
    bacc = balanced_accuracy_score(y_val, pred)
    auc = roc_auc_score(y_val, score)

    return acc, bacc, auc, selected


def main():
    print("=" * 78)
    print("XAUUSD LEAK-SAFE SMALL-FEATURE DIRECTION WALK-FORWARD")
    print("=" * 78)
    print("Feature selection: TRAINING DATA ONLY")
    print("Models: logistic regression")
    print("Top K: 3 / 5 / 10")
    print("Horizons: 15m / 30m / 60m")
    print("2026: COMPLETELY UNTOUCHED")
    print("=" * 78)

    if not CANONICAL.exists():
        raise FileNotFoundError(CANONICAL)
    if not V4.exists():
        raise FileNotFoundError(V4)

    canonical_ts, close, active = load_canonical()

    all_results = []

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

        X, y, sample_ts = collect_samples(
            canonical_ts,
            target,
        )

        years = pd.to_datetime(
            sample_ts,
            unit="ms",
            utc=True,
        ).year.to_numpy()

        for year in YEARS:
            val_mask = years == year

            val_start = pd.Timestamp(
                f"{year}-01-01",
                tz="UTC",
            )

            cutoff_ms = int(
                (
                    val_start
                    - pd.Timedelta(minutes=PURGE_MINUTES)
                ).timestamp() * 1000
            )

            train_mask = (
                (years >= 2019)
                & (years < year)
                & (sample_ts < cutoff_ms)
            )

            X_train = X[train_mask]
            y_train = y[train_mask]

            X_val = X[val_mask]
            y_val = y[val_mask]

            print("-" * 78)
            print(
                f"{horizon}m | Fold {year} | "
                f"train={len(y_train):,} | val={len(y_val):,}"
            )

            ranked = rank_features(
                X_train,
                y_train,
            )

            print("Top 10 TRAIN-ONLY features:")
            for r in ranked[:10]:
                print(
                    f"  {r['feature']:<24} "
                    f"train_auc={r['train_auc_raw']:.5f} "
                    f"adv={r['train_advantage']:.5f} "
                    f"sign={int(r['sign']):+d}"
                )

            for k in TOP_KS:
                acc, bacc, auc, selected = run_model(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    ranked,
                    k,
                )

                names = "|".join(
                    r["feature"] for r in selected
                )

                print(
                    f"K={k:>2} | "
                    f"accuracy={acc*100:.4f}% | "
                    f"balanced_acc={bacc*100:.4f}% | "
                    f"ROC-AUC={auc:.6f}"
                )

                all_results.append(
                    {
                        "horizon_min": horizon,
                        "validation_year": year,
                        "top_k": k,
                        "accuracy": float(acc),
                        "balanced_accuracy": float(bacc),
                        "roc_auc": float(auc),
                        "selected_features": names,
                    }
                )

        del target, X, y, sample_ts
        gc.collect()

    out = pd.DataFrame(all_results)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT, index=False)

    summary = (
        out.groupby(
            ["horizon_min", "top_k"],
            as_index=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
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
    print("All feature selection used training data only.")
    print("2026 was not used for fitting, selection, or evaluation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
