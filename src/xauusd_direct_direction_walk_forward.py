#!/usr/bin/env python3
"""
XAUUSD direct future-return direction walk-forward.

Predict:
    1 = future return > 0
    0 = future return <= 0

Horizons:
    15m, 30m, 60m

Features:
    Existing session-aware V4 features only (42 features).

No future-event conditioning, no P&L.
Walk-forward validation:
    2022, 2023, 2024, 2025
2026 remains completely untouched.

The script computes future close-to-close direction directly from the
canonical XAUUSD M1 data, then aligns targets to V4 feature timestamps.
Targets require the exact future timestamp in the same active session segment.
"""

from __future__ import annotations

import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
V4_PATH = ROOT / "data/processed/xauusd_m1_ml_features_v4_session.csv"
CANONICAL_PATH = ROOT / "data/processed/xauusd_m1_2019_2026_canonical.csv"
REPORT_PATH = ROOT / "reports/xauusd_direct_direction_walk_forward.csv"

HORIZONS = [15, 30, 60]
VAL_YEARS = [2022, 2023, 2024, 2025]
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
    "rolling_high_60", "rolling_low_60", "rolling_high_240", "rolling_low_240",
    "distance_high_60", "distance_low_60", "distance_high_240", "distance_low_240",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
]


def detect_epoch_unit(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    med = float(x.dropna().abs().median())
    if med >= 1e17:
        return "ns"
    if med >= 1e14:
        return "us"
    if med >= 1e11:
        return "ms"
    return "s"


def load_canonical():
    cols = list(pd.read_csv(CANONICAL_PATH, nrows=0).columns)
    ts_col = "timestamp" if "timestamp" in cols else cols[0]
    close_col = "close"
    if close_col not in cols:
        raise RuntimeError(f"Canonical close column not found. Columns: {cols}")

    df = pd.read_csv(
        CANONICAL_PATH,
        usecols=[ts_col, close_col],
        low_memory=False,
    )
    ts = pd.to_numeric(df[ts_col], errors="coerce")
    unit = detect_epoch_unit(ts)
    ts_ms = pd.to_datetime(ts, unit=unit, utc=True).astype("int64") // 1_000_000
    close = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=np.float64)
    ts_ms = ts_ms.to_numpy(dtype=np.int64)

    # Active-session proxy: same definition used in the session-aware target work.
    flat = np.isclose(close[1:], close[:-1], rtol=0.0, atol=0.0)
    run_start = np.empty(len(close), dtype=np.int64)
    run_start[0] = 0
    run_start[1:] = np.where(flat, 0, 1)
    seg = np.cumsum(run_start)

    # A row belongs to a closed market run if its consecutive identical-close
    # run is at least 60 minutes long.
    counts = np.bincount(seg)
    closed = counts[seg] >= 60
    active = ~closed

    # A target is valid only when the exact horizon endpoint exists and remains
    # inside the same active segment.
    segment_id = seg

    print(f"Canonical rows: {len(ts_ms):,}")
    print(f"Timestamp unit: epoch-{unit}")
    print(f"Closed-session rows: {int(closed.sum()):,}")
    print(f"Active rows: {int(active.sum()):,}")

    return ts_ms, close, active, segment_id


def load_v4_header():
    cols = list(pd.read_csv(V4_PATH, nrows=0).columns)
    need = ["timestamp", *FEATURES]
    missing = [c for c in need if c not in cols]
    if missing:
        raise RuntimeError(f"Missing V4 columns: {missing}")
    return need


def build_target_maps(ts_ms, close, active, segment_id):
    index = pd.Series(np.arange(len(ts_ms), dtype=np.int64), index=ts_ms)

    maps = {}
    for h in HORIZONS:
        expected = ts_ms + h * 60_000

        # Exact timestamp lookup.
        future_idx = index.reindex(expected).to_numpy()
        valid_idx = future_idx >= 0

        target_ret = np.full(len(ts_ms), np.nan, dtype=np.float64)
        valid = active & valid_idx

        pos = np.flatnonzero(valid)
        fidx = future_idx[pos].astype(np.int64)

        same_segment = segment_id[pos] == segment_id[fidx]
        valid_pos = pos[same_segment]
        valid_fidx = fidx[same_segment]

        target_ret[valid_pos] = (
            close[valid_fidx] / close[valid_pos] - 1.0
        )

        # Exact zero is classified DOWN/NOT-UP to keep binary definition deterministic.
        target = np.full(len(ts_ms), -1, dtype=np.int8)
        good = np.isfinite(target_ret)
        target[good] = (target_ret[good] > 0.0).astype(np.int8)

        maps[h] = target

        print(
            f"{h:>2}m target | valid={int(good.sum()):,} | "
            f"up={int((target[good] == 1).sum()):,} | "
            f"down_or_zero={int((target[good] == 0).sum()):,}"
        )

    return maps


def collect_v4_samples(target, ts_lookup, horizon):
    usecols = load_v4_header()
    target_series = pd.Series(target, index=ts_lookup)
    parts_x, parts_y, parts_ts = [], [], []
    global_row = 0

    reader = pd.read_csv(
        V4_PATH,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        ts = pd.to_numeric(chunk["timestamp"], errors="coerce").astype("int64")
        dt = pd.to_datetime(ts, unit="ms", utc=True)
        years = dt.dt.year.to_numpy()

        mask = (years >= 2019) & (years <= 2025)
        n = int(mask.sum())
        if n == 0:
            global_row += len(chunk)
            continue

        chunk = chunk.loc[mask].copy()

        pos = np.arange(len(chunk), dtype=np.int64)
        sm = ((global_row + pos) % STRIDE) == 0
        chunk = chunk.loc[sm].copy()

        if chunk.empty:
            global_row += n
            continue

        ts_int = pd.to_numeric(chunk["timestamp"], errors="coerce").astype("int64")
        labels = target_series.reindex(ts_int.to_numpy()).to_numpy()

        valid = np.isin(labels, [0, 1])
        if valid.any():
            sub = chunk.loc[valid].copy()
            y = labels[valid].astype(np.int8)
            X = sub[FEATURES].to_numpy(dtype=np.float32, copy=True)
            finite = np.isfinite(X).all(axis=1)

            if finite.any():
                parts_x.append(X[finite])
                parts_y.append(y[finite])
                parts_ts.append(
                    pd.to_numeric(sub["timestamp"], errors="coerce")
                    .to_numpy(dtype=np.int64)[finite]
                )

        global_row += n

    X = np.concatenate(parts_x)
    y = np.concatenate(parts_y)
    ts = np.concatenate(parts_ts)

    order = np.argsort(ts, kind="mergesort")
    X, y, ts = X[order], y[order], ts[order]

    print(f"ML rows: {len(y):,}")
    print(f"X shape: {X.shape}")
    print(f"UP %: {(y == 1).mean()*100:.2f}")
    print(f"DOWN/0 %: {(y == 0).mean()*100:.2f}")

    return X, y, ts


def run_fold(X, y, ts, year, horizon):
    dt = pd.to_datetime(ts, unit="ms", utc=True)
    years = dt.year.to_numpy()

    val_mask = years == year

    val_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    val_start_ms = int(val_start.timestamp() * 1000)
    cutoff = int(
        (val_start - pd.Timedelta(minutes=PURGE_MINUTES)).timestamp() * 1000
    )

    train_mask = (
        (years >= 2019)
        & (years < year)
        & (ts < cutoff)
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print("-" * 78)
    print(
        f"{horizon}m | Fold {year} | train={len(X_train):,} | "
        f"val={len(X_val):,}"
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
        callbacks=[lgb.early_stopping(75, verbose=False)],
    )

    p = model.predict_proba(X_val)[:, 1]
    pred = (p >= 0.5).astype(np.int8)

    acc = accuracy_score(y_val, pred)
    bacc = balanced_accuracy_score(y_val, pred)
    auc = roc_auc_score(y_val, p)
    cm = confusion_matrix(y_val, pred, labels=[0, 1])

    print(f"Best iteration: {model.best_iteration_}")
    print(f"Accuracy: {acc*100:.4f}%")
    print(f"Balanced accuracy: {bacc*100:.4f}%")
    print(f"ROC-AUC: {auc:.6f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    return {
        "horizon_min": horizon,
        "validation_year": year,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "best_iteration": int(model.best_iteration_),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "roc_auc": float(auc),
        "up_val_pct": float(y_val.mean()),
        "pred_up_pct": float(pred.mean()),
    }


def main():
    print("=" * 78)
    print("XAUUSD DIRECT FUTURE-RETURN DIRECTION WALK-FORWARD")
    print("=" * 78)
    print("Task: UP vs DOWN/0 without future-event conditioning")
    print("Features: session-aware V4 (42)")
    print(f"Sampling stride: {STRIDE}")
    print("Horizons: 15m / 30m / 60m")
    print("2026: COMPLETELY UNTOUCHED")
    print("=" * 78)

    for path in [V4_PATH, CANONICAL_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    ts_ms, close, active, segment_id = load_canonical()
    target_maps = build_target_maps(ts_ms, close, active, segment_id)

    results = []

    for horizon in HORIZONS:
        print("=" * 78)
        print(f"HORIZON {horizon} MINUTES")
        print("=" * 78)

        X, y, ts = collect_v4_samples(
            target_maps[horizon],
            ts_ms,
            horizon,
        )

        for year in VAL_YEARS:
            results.append(run_fold(X, y, ts, year, horizon))

        del X, y, ts
        gc.collect()

    out = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_PATH, index=False)

    summary = (
        out.groupby("horizon_min", as_index=False)
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
    print(f"Saved: {REPORT_PATH}")
    print("2026 was not used for fitting, selection, or evaluation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
