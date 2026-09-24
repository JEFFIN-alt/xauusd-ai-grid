#!/usr/bin/env python3
"""
Session-aware directional V5 walk-forward.

Goal:
    Re-test LONG vs SHORT prediction after expanding the current V4
    price-only feature set with additional directional/market-structure
    features.

Important:
    - Conditioning on SHORT/LONG realized event is used ONLY to define the
      diagnostic target. No P&L is calculated here.
    - 2026 remains completely untouched.
    - Existing V4 definitions are preserved; V5 adds features on top.
    - We focus on 0.75x and 1.00x dynamic 60-minute barriers because the
      0.50x event target is extremely event-heavy.

New V5 features:
    - lagged 1m returns
    - rolling return mean / dispersion
    - volatility ratios
    - RSI-style momentum
    - rolling close range position
    - normalized distances to 20/30/120m highs/lows
    - up-candle / down-candle fractions
    - body-direction imbalance
    - range-normalized momentum
    - corrected 24h cyclical time
    - day-of-week cyclical time
    - London / New York / overlap session flags
"""

from __future__ import annotations

import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

V4_PATH = ROOT / "data/processed/xauusd_m1_ml_features_v4_session.csv"
TARGET_TEMPLATE = ROOT / "data/processed/xauusd_m1_session_dynamic_tb_{cfg}.csv"
REPORT_PATH = ROOT / "reports/session_dynamic_tb_directional_v5_walk_forward.csv"

CONFIGS = ["0p75", "1p00"]
VAL_YEARS = [2022, 2023, 2024, 2025]

STRIDE = 5
CHUNK_SIZE = 250_000
HISTORY = 120

BASE_FEATURES = [
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

NEW_FEATURES = [
    "ret_1_lag1", "ret_1_lag2", "ret_1_lag3", "ret_1_lag5",
    "ret_1_lag10", "ret_1_lag15", "ret_1_lag30", "ret_1_lag60",
    "ret_mean_5", "ret_mean_15", "ret_mean_30", "ret_mean_60",
    "ret_std_15", "ret_std_30", "ret_std_60",
    "vol_ratio_15_60", "vol_ratio_30_240",
    "rsi_14", "rsi_30",
    "close_pos_20", "close_pos_30", "close_pos_60", "close_pos_120",
    "dist_high_20", "dist_low_20", "dist_high_30", "dist_low_30",
    "dist_high_120", "dist_low_120",
    "up_frac_10", "up_frac_30", "up_frac_60",
    "body_imbalance_10", "body_imbalance_30", "body_imbalance_60",
    "range_norm_mom_5", "range_norm_mom_15", "range_norm_mom_30",
    "hour2_sin", "hour2_cos", "dow_sin", "dow_cos",
    "session_london", "session_newyork", "session_overlap",
]

FEATURES = BASE_FEATURES + NEW_FEATURES


def load_target(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=["timestamp", "tb_label_60m"],
        dtype={"timestamp": "int64", "tb_label_60m": "int8"},
    )


def load_v4_columns() -> list[str]:
    cols = list(pd.read_csv(V4_PATH, nrows=0).columns)
    need = [
        "timestamp", "open", "high", "low", "close",
        *BASE_FEATURES,
    ]
    missing = [c for c in need if c not in cols]
    if missing:
        raise RuntimeError(f"Missing required V4 columns: {missing}")
    return need


def epoch_to_dt(x: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.to_numeric(x, errors="coerce"), unit="ms", utc=True)


def add_v5_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute added directional features on continuous V4 segments."""
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    seg = ts.diff().ne(60_000).cumsum()

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    ret = pd.to_numeric(df["return_1m"], errors="coerce")
    body = pd.to_numeric(df["body"], errors="coerce")

    def shift(s, n):
        return s.groupby(seg, sort=False).shift(n)

    def roll_mean(s, n):
        return (
            s.groupby(seg, sort=False)
            .rolling(n, min_periods=n)
            .mean()
            .reset_index(level=0, drop=True)
        )

    def roll_std(s, n):
        return (
            s.groupby(seg, sort=False)
            .rolling(n, min_periods=n)
            .std()
            .reset_index(level=0, drop=True)
        )

    def roll_sum(s, n):
        return (
            s.groupby(seg, sort=False)
            .rolling(n, min_periods=n)
            .sum()
            .reset_index(level=0, drop=True)
        )

    def roll_max(s, n):
        return (
            s.groupby(seg, sort=False)
            .rolling(n, min_periods=n)
            .max()
            .reset_index(level=0, drop=True)
        )

    def roll_min(s, n):
        return (
            s.groupby(seg, sort=False)
            .rolling(n, min_periods=n)
            .min()
            .reset_index(level=0, drop=True)
        )

    for n in [1, 2, 3, 5, 10, 15, 30, 60]:
        df[f"ret_1_lag{n}"] = shift(ret, n)

    for n in [5, 15, 30, 60]:
        df[f"ret_mean_{n}"] = roll_mean(ret, n)

    for n in [15, 30, 60]:
        df[f"ret_std_{n}"] = roll_std(ret, n)

    v15 = pd.to_numeric(df["volatility_15m"], errors="coerce")
    v30 = pd.to_numeric(df["volatility_30m"], errors="coerce")
    v60 = pd.to_numeric(df["volatility_60m"], errors="coerce")
    v240 = pd.to_numeric(df["volatility_240m"], errors="coerce")
    df["vol_ratio_15_60"] = v15 / v60.replace(0, np.nan)
    df["vol_ratio_30_240"] = v30 / v240.replace(0, np.nan)

    # RSI-style relative strength from 1m returns.
    up = ret.clip(lower=0)
    down = (-ret.clip(upper=0))
    for n in [14, 30]:
        up_mean = roll_mean(up, n)
        down_mean = roll_mean(down, n)
        rs = up_mean / down_mean.replace(0, np.nan)
        df[f"rsi_{n}"] = 100.0 - (100.0 / (1.0 + rs))
        df.loc[(down_mean == 0) & (up_mean > 0), f"rsi_{n}"] = 100.0
        df.loc[(up_mean == 0) & (down_mean > 0), f"rsi_{n}"] = 0.0

    for n in [20, 30, 60, 120]:
        rh = roll_max(high, n)
        rl = roll_min(low, n)
        span = (rh - rl).replace(0, np.nan)
        df[f"close_pos_{n}"] = (close - rl) / span

    for n in [20, 30, 120]:
        rh = roll_max(high, n)
        rl = roll_min(low, n)
        df[f"dist_high_{n}"] = close / rh.replace(0, np.nan) - 1.0
        df[f"dist_low_{n}"] = close / rl.replace(0, np.nan) - 1.0

    for n in [10, 30, 60]:
        up_frac = (
            (body > 0).astype(np.float64)
            .groupby(seg, sort=False)
            .rolling(n, min_periods=n)
            .mean()
            .reset_index(level=0, drop=True)
        )
        body_sum = roll_sum(body, n)
        abs_body_sum = roll_sum(body.abs(), n)
        df[f"up_frac_{n}"] = up_frac
        df[f"body_imbalance_{n}"] = body_sum / abs_body_sum.replace(0, np.nan)

    safe_range = roll_mean((high - low).replace(0, np.nan), 20)
    for n in [5, 15, 30]:
        df[f"range_norm_mom_{n}"] = (
            pd.to_numeric(df[f"momentum_{n}m"], errors="coerce")
            / safe_range.replace(0, np.nan)
        )

    dt = epoch_to_dt(df["timestamp"])
    hour_float = (
        dt.dt.hour.to_numpy(dtype=np.float64)
        + dt.dt.minute.to_numpy(dtype=np.float64) / 60.0
    )
    dow = dt.dt.dayofweek.to_numpy(dtype=np.float64)

    df["hour2_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    df["hour2_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    # UTC trading-session proxies, deliberately simple and deterministic.
    h = dt.dt.hour.to_numpy()
    df["session_london"] = ((h >= 7) & (h < 16)).astype(np.int8)
    df["session_newyork"] = ((h >= 13) & (h < 21)).astype(np.int8)
    df["session_overlap"] = ((h >= 13) & (h < 16)).astype(np.int8)

    return df


def collect(cfg: str):
    target = load_target(Path(str(TARGET_TEMPLATE).format(cfg=cfg)))
    target_map = target.set_index("timestamp")["tb_label_60m"]

    usecols = load_v4_columns()
    parts_x, parts_y, parts_ts = [], [], []

    carry = pd.DataFrame()
    global_row = 0

    reader = pd.read_csv(
        V4_PATH,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for raw in reader:
        years = epoch_to_dt(raw["timestamp"]).dt.year.to_numpy()
        mask_year = (years >= 2019) & (years <= 2025)
        chunk = raw.loc[mask_year].copy()

        if chunk.empty:
            global_row += int(mask_year.sum())
            continue

        new_len = len(chunk)

        if not carry.empty:
            work = pd.concat([carry, chunk], ignore_index=True)
            carry_len = len(carry)
        else:
            work = chunk.reset_index(drop=True)
            carry_len = 0

        work = add_v5_features(work)

        # Only the newly-read rows can enter the current chunk output.
        new = work.iloc[carry_len:].copy()

        positions = np.arange(new_len, dtype=np.int64)
        sample_mask = ((global_row + positions) % STRIDE) == 0
        new = new.loc[sample_mask].copy()

        if not new.empty:
            ts = pd.to_numeric(new["timestamp"], errors="coerce").astype("int64")
            labels = target_map.reindex(ts.to_numpy()).to_numpy()

            # Only realized SHORT/LONG events.
            event_mask = np.isin(labels, [0, 2])

            if event_mask.any():
                sub = new.loc[event_mask].copy()
                lab = labels[event_mask].astype(np.int8)

                feats = sub[FEATURES].to_numpy(dtype=np.float32, copy=True)
                finite = np.isfinite(feats).all(axis=1)

                if finite.any():
                    parts_x.append(feats[finite])
                    parts_y.append((lab[finite] == 2).astype(np.int8))
                    parts_ts.append(
                        pd.to_numeric(sub["timestamp"], errors="coerce")
                        .to_numpy(dtype=np.int64)[finite]
                    )

        # Preserve enough history for the next chunk.
        carry = work.tail(HISTORY).copy()
        global_row += new_len

        del raw, chunk, work, new
        gc.collect()

    X = np.concatenate(parts_x)
    y = np.concatenate(parts_y)
    ts = np.concatenate(parts_ts)

    order = np.argsort(ts, kind="mergesort")
    X, y, ts = X[order], y[order], ts[order]

    print(f"V5 directional rows: {len(y):,}")
    print(f"X shape: {X.shape}")
    print(f"SHORT %: {(y==0).mean()*100:.2f}")
    print(f"LONG  %: {(y==1).mean()*100:.2f}")
    return X, y, ts


def run_fold(X, y, ts, year, cfg):
    dt = pd.to_datetime(ts, unit="ms", utc=True)
    years = dt.year.to_numpy()

    val_mask = years == year

    val_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    val_start_ms = int(val_start.timestamp() * 1000)
    train_cutoff = int(
        (val_start - pd.Timedelta(minutes=60)).timestamp() * 1000
    )

    train_mask = (
        (years >= 2019)
        & (years < year)
        & (ts < train_cutoff)
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print("-" * 78)
    print(
        f"{cfg} | Fold {year} | train={len(X_train):,} | "
        f"val={len(X_val):,}"
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

    print(f"Best iteration: {model.best_iteration_}")
    print(f"Accuracy: {acc*100:.4f}%")
    print(f"Balanced accuracy: {bacc*100:.4f}%")
    print(f"ROC-AUC: {auc:.6f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    return {
        "config": f"dynamic_{cfg}x",
        "validation_year": year,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "best_iteration": int(model.best_iteration_),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "roc_auc": float(auc),
        "pred_long_pct": float(pred.mean()),
    }


def main():
    print("=" * 78)
    print("SESSION-AWARE DIRECTIONAL V5 WALK-FORWARD")
    print("=" * 78)
    print("Configs: 0.75x / 1.00x")
    print("Task: SHORT vs LONG, conditioned on realized barrier event")
    print("Features: V4 + V5 directional / market-structure features")
    print(f"Sampling stride: {STRIDE}")
    print("2026: COMPLETELY UNTOUCHED")
    print("=" * 78)

    results = []

    for cfg in CONFIGS:
        print("=" * 78)
        print(f"COLLECTING CONFIG {cfg}")
        X, y, ts = collect(cfg)

        for year in VAL_YEARS:
            results.append(run_fold(X, y, ts, year, cfg))

        del X, y, ts
        gc.collect()

    out = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_PATH, index=False)

    summary = (
        out.groupby("config", as_index=False)
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
