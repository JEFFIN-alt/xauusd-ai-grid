#!/usr/bin/env python3
"""
Session-aware dynamic-barrier EVENT vs TIMEOUT walk-forward baseline.

Predict whether a 60-minute dynamic barrier event occurs at all.

Labels:
  tb_label_60m == 0 (SHORT) or 2 (LONG) -> EVENT = 1
  tb_label_60m == 1 (TIMEOUT)             -> NO EVENT = 0
  -1 ambiguous and -2 invalid are excluded.

Walk-forward:
  train: 2019..validation_year-1
  validation: 2022, 2023, 2024, 2025
  2026 completely untouched.

No P&L is calculated here. This isolates whether event occurrence itself
is predictable before combining it with directional prediction.
"""

from __future__ import annotations

import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

V4_PATH = ROOT / "data/processed/xauusd_m1_ml_features_v4_session.csv"
TARGET_TEMPLATE = ROOT / "data/processed/xauusd_m1_session_dynamic_tb_{cfg}.csv"
REPORT_PATH = ROOT / "reports/session_dynamic_tb_event_walk_forward.csv"

CONFIGS = ["0p50", "0p75", "1p00"]
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


def epoch_to_datetime(values):
    x = pd.to_numeric(values, errors="coerce")
    med = float(x.dropna().abs().median())
    if med >= 1e17:
        unit = "ns"
    elif med >= 1e14:
        unit = "us"
    elif med >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(x, unit=unit, utc=True)


def load_target(target_path):
    return pd.read_csv(
        target_path,
        usecols=["timestamp", "tb_label_60m"],
        dtype={"timestamp": "int64", "tb_label_60m": "int8"},
    )


def load_v4_header():
    cols = list(pd.read_csv(V4_PATH, nrows=0).columns)
    needed = ["timestamp", *FEATURES]
    missing = [c for c in needed if c not in cols]
    if missing:
        raise RuntimeError(f"Missing V4 columns: {missing}")
    return needed


def collect_samples(target_df):
    """Return V4 samples for valid SHORT/TIMEOUT/LONG labels.

    EVENT=1 for SHORT/LONG, NO EVENT=0 for TIMEOUT.
    """
    usecols = load_v4_header()
    target_map = target_df.set_index("timestamp")["tb_label_60m"]

    X_parts, y_parts, ts_parts = [], [], []
    global_row = 0

    reader = pd.read_csv(
        V4_PATH,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        raw_ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
        years = epoch_to_datetime(raw_ts).dt.year.to_numpy()

        year_mask = (years >= 2019) & (years <= 2025)
        n_kept_year = int(year_mask.sum())

        if n_kept_year == 0:
            global_row += len(chunk)
            continue

        chunk = chunk.loc[year_mask].copy()

        positions = np.arange(len(chunk), dtype=np.int64)
        stride_mask = ((global_row + positions) % STRIDE) == 0
        chunk = chunk.loc[stride_mask].copy()

        if chunk.empty:
            global_row += n_kept_year
            continue

        # Rebuild timestamps after stride sampling.
        ts_int = pd.to_numeric(chunk["timestamp"], errors="coerce").astype("int64")
        labels = target_map.reindex(ts_int.to_numpy()).to_numpy()

        valid = np.isin(labels, [0, 1, 2])
        if not valid.any():
            global_row += n_kept_year
            continue

        sub = chunk.loc[valid]
        lab = labels[valid].astype(np.int8)

        feats = sub[FEATURES].to_numpy(dtype=np.float32, copy=True)
        finite = np.isfinite(feats).all(axis=1)

        if finite.any():
            X_parts.append(feats[finite])
            y_parts.append((lab[finite] != 1).astype(np.int8))
            ts_parts.append(
                pd.to_numeric(sub["timestamp"], errors="coerce")
                .to_numpy(dtype=np.int64)[finite]
            )

        global_row += n_kept_year
        gc.collect()

    if not X_parts:
        raise RuntimeError("No event/timeout samples collected.")

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    ts = np.concatenate(ts_parts)

    order = np.argsort(ts, kind="mergesort")
    X, y, ts = X[order], y[order], ts[order]

    print(f"ML rows: {len(y):,}")
    print(f"X shape: {X.shape}")
    print(f"EVENT %: {(y == 1).mean() * 100:.2f}")
    print(f"NO EVENT %: {(y == 0).mean() * 100:.2f}")

    return X, y, ts


def run_fold(X, y, ts, val_year, cfg):
    dates = pd.to_datetime(ts, unit="ms", utc=True)
    years = dates.year.to_numpy()

    val_mask = years == val_year

    val_start = pd.Timestamp(f"{val_year}-01-01", tz="UTC")
    val_start_ms = int(val_start.timestamp() * 1000)
    train_cutoff = int(
        (val_start - pd.Timedelta(minutes=PURGE_MINUTES)).timestamp() * 1000
    )

    train_mask = (
        (years >= 2019)
        & (years < val_year)
        & (ts < train_cutoff)
    )

    purge_mask = (
        (years >= 2019)
        & (years < val_year)
        & (ts >= train_cutoff)
        & (ts < val_start_ms)
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print("-" * 78)
    print(
        f"Fold {val_year} | train={len(X_train):,} | "
        f"val={len(X_val):,} | purged_train={int(purge_mask.sum()):,}"
    )
    print(
        f"Actual train EVENT={y_train.mean()*100:.2f}% | "
        f"val EVENT={y_val.mean()*100:.2f}%"
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.9,
        colsample_bytree=0.9,
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
    pred = (p >= 0.50).astype(np.int8)

    acc = accuracy_score(y_val, pred)
    bacc = balanced_accuracy_score(y_val, pred)
    auc = roc_auc_score(y_val, p)
    ap = average_precision_score(y_val, p)
    brier = brier_score_loss(y_val, p)
    cm = confusion_matrix(y_val, pred, labels=[0, 1])

    print(f"Best iteration: {model.best_iteration_}")
    print(f"Accuracy: {acc*100:.4f}%")
    print(f"Balanced accuracy: {bacc*100:.4f}%")
    print(f"ROC-AUC: {auc:.6f}")
    print(f"PR-AUC: {ap:.6f}")
    print(f"Brier score: {brier:.6f}")
    print(
        f"Pred NO-EVENT/EVENT: "
        f"[{(pred==0).mean()*100:.2f}, {(pred==1).mean()*100:.2f}]"
    )
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    print("Probability deciles:")
    bins = np.linspace(0, 1, 11)
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & ((p <= hi) if i == 9 else (p < hi))
        if m.any():
            print(
                f"  {lo:.1f}-{hi:.1f} | n={int(m.sum()):>6} | "
                f"pred_event={p[m].mean()*100:>6.2f}% | "
                f"actual_event={y_val[m].mean()*100:>6.2f}%"
            )

    return {
        "config": f"dynamic_{cfg}x",
        "barrier_multiplier": float(cfg.replace("p", ".")),
        "validation_year": val_year,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "purged_rows": int(purge_mask.sum()),
        "best_iteration": int(model.best_iteration_),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "roc_auc": float(auc),
        "pr_auc": float(ap),
        "brier_score": float(brier),
        "actual_event_pct": float(y_val.mean()),
        "pred_event_pct": float(pred.mean()),
    }


def main():
    print("=" * 78)
    print("SESSION-AWARE DYNAMIC-BARRIER EVENT/TIMEOUT WALK-FORWARD")
    print("=" * 78)
    print("Task: EVENT (SHORT/LONG) vs NO EVENT (TIMEOUT)")
    print("Features: session-aware V4")
    print(f"Sampling stride: {STRIDE}")
    print("2026: COMPLETELY UNTOUCHED")
    print("=" * 78)

    if not V4_PATH.exists():
        raise FileNotFoundError(V4_PATH)

    results = []

    for cfg in CONFIGS:
        target_path = Path(str(TARGET_TEMPLATE).format(cfg=cfg))
        if not target_path.exists():
            raise FileNotFoundError(target_path)

        print("=" * 78)
        print(f"Loading target: {target_path.name}")

        target_df = load_target(target_path)
        counts = target_df["tb_label_60m"].value_counts().to_dict()

        print(
            f"Target rows: {len(target_df):,} | "
            f"SHORT={counts.get(0,0):,} | "
            f"TIMEOUT={counts.get(1,0):,} | "
            f"LONG={counts.get(2,0):,} | "
            f"AMBIG={counts.get(-1,0):,} | "
            f"INVALID={counts.get(-2,0):,}"
        )

        print("Streaming V4 features and aligning event/timeout target...")
        X, y, ts = collect_samples(target_df)

        for year in VAL_YEARS:
            results.append(run_fold(X, y, ts, year, cfg))

        del X, y, ts, target_df
        gc.collect()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(results)
    out.to_csv(REPORT_PATH, index=False)

    summary = (
        out.groupby("config", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_brier=("brier_score", "mean"),
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
