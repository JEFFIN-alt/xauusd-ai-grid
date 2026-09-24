#!/usr/bin/env python3
"""
Conditional directional model for session-aware dynamic-barrier targets.

Purpose:
    Measure whether the existing V4 features can distinguish LONG vs SHORT
    AFTER conditioning on the fact that a dynamic barrier event actually occurs.

Important:
    This is a predictive-skill experiment, NOT a deployable trading backtest.
    Filtering to labels {SHORT, LONG} uses the realized future event, so any
    P&L calculation on this filtered sample would be selection-biased.

Walk-forward:
    train: 2019..(validation_year-1)
    validate: validation_year
    validation years: 2022..2025
    2026 is completely untouched.

Targets:
    tb_label == 0 -> SHORT -> binary 0
    tb_label == 2 -> LONG  -> binary 1
    ambiguous (-1), invalid (-2), timeout (1), and session-closed rows excluded.

Features:
    Same 42 V4 features already used in the current baseline.
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
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

V4_PATH = ROOT / "data/processed/xauusd_m1_ml_features_v4_session.csv"
TARGET_TEMPLATE = ROOT / "data/processed/xauusd_m1_session_dynamic_tb_{cfg}.csv"
REPORT_PATH = ROOT / "reports/session_dynamic_tb_directional_walk_forward.csv"

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


def epoch_to_datetime(values: pd.Series) -> pd.Series:
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


def load_target_label(target_path: Path) -> pd.DataFrame:
    hdr = pd.read_csv(target_path, nrows=0)
    cols = list(hdr.columns)

    ts_col = next((c for c in ["timestamp", "timestamp_ms", "time", "datetime", "date"] if c in cols), None)
    label_col = next((c for c in ["tb_label_60m", "tb_label", "label", "target", "class"] if c in cols), None)

    if ts_col is None or label_col is None:
        raise RuntimeError(
            f"Could not detect timestamp/label columns in {target_path.name}. "
            f"Available columns: {cols}"
        )

    df = pd.read_csv(
        target_path,
        usecols=[ts_col, label_col],
        dtype={ts_col: "int64", label_col: "int8"},
    )
    df = df.rename(columns={ts_col: "timestamp", label_col: "tb_label"})
    return df


def load_v4_header() -> list[str]:
    hdr = pd.read_csv(V4_PATH, nrows=0)
    cols = list(hdr.columns)
    needed = ["timestamp", *FEATURES]
    missing = [c for c in needed if c not in cols]
    if missing:
        raise RuntimeError(f"Missing V4 columns: {missing}")
    return needed


def collect_aligned_samples(target_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stream V4 rows for 2019-2025, stride-sample, align target labels by timestamp,
    and keep only realized barrier-hit rows (SHORT/LONG).

    Returns:
        X: float32 feature matrix
        y: int8 binary direction (0=SHORT, 1=LONG)
        ts: int64 epoch-ms timestamps
    """
    usecols = load_v4_header()

    target_idx = target_df.set_index("timestamp")["tb_label"].sort_index()

    X_parts = []
    y_parts = []
    ts_parts = []

    global_row = 0
    kept = 0

    reader = pd.read_csv(
        V4_PATH,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
        dt = epoch_to_datetime(ts)
        years = dt.dt.year.to_numpy()

        # Keep only 2019-2025. Never load/use 2026 for fitting or validation.
        year_mask = (years >= 2019) & (years <= 2025)
        if not year_mask.any():
            global_row += len(chunk)
            continue

        chunk = chunk.loc[year_mask].copy()
        ts_int = pd.to_numeric(chunk["timestamp"], errors="coerce").astype("int64")

        # Deterministic global stride.
        local_positions = np.arange(len(chunk), dtype=np.int64)
        stride_mask = ((global_row + local_positions) % STRIDE) == 0
        chunk = chunk.loc[stride_mask].copy()

        if chunk.empty:
            global_row += int(year_mask.sum())
            continue

        # Rebuild timestamps AFTER stride sampling so labels and chunk
        # have exactly the same number of rows.
        ts_int = pd.to_numeric(chunk["timestamp"], errors="coerce").astype("int64")
        labels = target_idx.reindex(ts_int.to_numpy()).to_numpy()
        event_mask = np.isin(labels, [0, 2])

        if not event_mask.any():
            global_row += int(year_mask.sum())
            continue

        sub = chunk.loc[event_mask]
        lab = labels[event_mask].astype(np.int8)

        feats = sub[FEATURES].to_numpy(dtype=np.float32, copy=True)
        finite_mask = np.isfinite(feats).all(axis=1)

        if finite_mask.any():
            X_parts.append(feats[finite_mask])
            y_parts.append((lab[finite_mask] == 2).astype(np.int8))
            ts_parts.append(
                pd.to_numeric(sub["timestamp"], errors="coerce")
                .to_numpy(dtype=np.int64)[finite_mask]
            )
            kept += int(finite_mask.sum())

        global_row += int(year_mask.sum())

        if len(X_parts) % 4 == 0:
            gc.collect()

    if not X_parts:
        raise RuntimeError("No aligned directional samples were collected.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    ts = np.concatenate(ts_parts, axis=0)

    order = np.argsort(ts, kind="mergesort")
    X = X[order]
    y = y[order]
    ts = ts[order]

    print(f"Directional event rows: {len(y):,}")
    print(f"X shape: {X.shape}")
    print(f"SHORT %: {(y == 0).mean()*100:.2f}")
    print(f"LONG  %: {(y == 1).mean()*100:.2f}")

    return X, y, ts


def train_one_fold(
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    val_year: int,
    cfg: str,
) -> dict:
    dt = pd.to_datetime(ts, unit="ms", utc=True)
    years = dt.year.to_numpy()

    val_mask = years == val_year

    # Conservative 60-minute purge: every dynamic target can look forward
    # for up to 60 minutes, so remove the final 60 minutes of the training era.
    val_start = pd.Timestamp(f"{val_year}-01-01", tz="UTC")
    train_cutoff = int((val_start - pd.Timedelta(minutes=PURGE_MINUTES)).timestamp() * 1000)

    train_mask = (years >= 2019) & (years < val_year) & (ts < train_cutoff)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError(f"Empty fold for {cfg} / {val_year}")

    print("-" * 78)
    purge_mask = (
        (years >= 2019)
        & (years < val_year)
        & (ts >= train_cutoff)
        & (ts < int(val_start.timestamp() * 1000))
    )
    purged_train = int(purge_mask.sum())

    print(
        f"Fold {val_year} | train={len(X_train):,} | val={len(X_val):,} "
        f"| purged_train={purged_train:,}"
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
    pred = (p >= 0.5).astype(np.int8)

    acc = accuracy_score(y_val, pred)
    bacc = balanced_accuracy_score(y_val, pred)
    auc = roc_auc_score(y_val, p)

    cm = confusion_matrix(y_val, pred, labels=[0, 1])

    print(f"Actual SHORT/LONG: [{(y_val==0).mean()*100:.2f}, {(y_val==1).mean()*100:.2f}]")
    print(f"Pred   SHORT/LONG: [{(pred==0).mean()*100:.2f}, {(pred==1).mean()*100:.2f}]")
    print(f"Best iteration: {model.best_iteration_}")
    print(f"Accuracy: {acc*100:.4f}%")
    print(f"Balanced accuracy: {bacc*100:.4f}%")
    print(f"ROC-AUC: {auc:.6f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    # Probability calibration / monotonicity diagnostic:
    # Does larger predicted LONG probability correspond to more actual LONGs?
    bins = np.linspace(0, 1, 11)
    bucket_rows = []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        if i == 9:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if m.any():
            bucket_rows.append(
                {
                    "prob_bin": f"{lo:.1f}-{hi:.1f}",
                    "n": int(m.sum()),
                    "mean_p_long": float(p[m].mean()),
                    "actual_long_pct": float(y_val[m].mean() * 100),
                }
            )

    print("Probability deciles:")
    for r in bucket_rows:
        print(
            f"  {r['prob_bin']:>9} | n={r['n']:>6} | "
            f"pred={r['mean_p_long']:.3f} | actual_long={r['actual_long_pct']:.2f}%"
        )

    return {
        "config": f"dynamic_{cfg}x",
        "barrier_multiplier": float(cfg.replace("p", ".")) if "p" in cfg else np.nan,
        "validation_year": val_year,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "best_iteration": int(model.best_iteration_),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "roc_auc": auc,
        "short_val_pct": float((y_val == 0).mean()),
        "long_val_pct": float((y_val == 1).mean()),
        "pred_short_pct": float((pred == 0).mean()),
        "pred_long_pct": float((pred == 1).mean()),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def main() -> None:
    print("=" * 78)
    print("SESSION-AWARE DYNAMIC-BARRIER CONDITIONAL DIRECTIONAL WALK-FORWARD")
    print("=" * 78)
    print("Task: SHORT vs LONG only, conditioned on a realized barrier hit")
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
        target_df = load_target_label(target_path)
        print(
            f"Target rows: {len(target_df):,} | "
            f"SHORT={int((target_df.tb_label==0).sum()):,} | "
            f"TIMEOUT={int((target_df.tb_label==1).sum()):,} | "
            f"LONG={int((target_df.tb_label==2).sum()):,} | "
            f"AMBIG={int((target_df.tb_label==-1).sum()):,} | "
            f"INVALID={int((target_df.tb_label==-2).sum()):,}"
        )

        print("Streaming V4 features and aligning realized directional events...")
        X, y, ts = collect_aligned_samples(target_df)

        for val_year in VAL_YEARS:
            results.append(train_one_fold(X, y, ts, val_year, cfg))

        del X, y, ts, target_df
        gc.collect()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(REPORT_PATH, index=False)

    summary = (
        df.groupby("config", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            min_balanced_accuracy=("balanced_accuracy", "min"),
            max_balanced_accuracy=("balanced_accuracy", "max"),
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
