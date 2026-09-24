#!/usr/bin/env python3
"""
SESSION-AWARE DYNAMIC-BARRIER LIGHTGBM WALK-FORWARD BASELINE

Configs tested:
    0.50x / 60m
    0.75x / 60m
    1.00x / 60m

Data:
    Features:
        data/processed/xauusd_m1_ml_features_v4_session.csv

    Targets:
        data/processed/xauusd_m1_session_dynamic_tb_0p50.csv
        data/processed/xauusd_m1_session_dynamic_tb_0p75.csv
        data/processed/xauusd_m1_session_dynamic_tb_1p00.csv

Method:
    - Exact 42 V3-compatible features from session-aware V4 dataset.
    - 3-class LightGBM: SHORT=0, TIMEOUT=1, LONG=2.
    - Sampling stride = 5, matching the previous baseline family.
    - Walk-forward folds:
        2019-2021 -> 2022
        2019-2022 -> 2023
        2019-2023 -> 2024
        2019-2024 -> 2025
    - 60-minute target-end purge is applied to training at each year boundary.
    - 2026 is completely excluded.
    - AMBIGUOUS target rows are excluded.
    - No confidence threshold optimization here.
    - Simulator uses ACTUAL dynamic barrier_return at entry:
        correct side -> +actual barrier
        wrong side   -> -actual barrier
        timeout      -> exact close-to-close return at actual exit
    - Positions are non-overlapping.
    - Validation trades whose actual exit crosses the validation-year end
      are skipped.

Outputs:
    reports/session_dynamic_tb_lgbm_walk_forward.csv
"""

from pathlib import Path
import gc
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


BASE = Path("data/processed")
REPORT = Path("reports")

FEATURE_FILE = BASE / "xauusd_m1_ml_features_v4_session.csv"
CANONICAL_FILE = BASE / "xauusd_m1_2019_2026_canonical.csv"

CONFIGS = [
    {
        "multiplier": 0.50,
        "target_file": BASE / "xauusd_m1_session_dynamic_tb_0p50.csv",
        "tag": "dynamic_0p50x",
    },
    {
        "multiplier": 0.75,
        "target_file": BASE / "xauusd_m1_session_dynamic_tb_0p75.csv",
        "tag": "dynamic_0p75x",
    },
    {
        "multiplier": 1.00,
        "target_file": BASE / "xauusd_m1_session_dynamic_tb_1p00.csv",
        "tag": "dynamic_1p00x",
    },
]

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

TARGET_USECOLS = [
    "timestamp",
    "tb_label_60m",
    "tb_exit_index_60m",
    "barrier_return_60m",
]

CHUNK_SIZE = 250_000
STRIDE = 5

YEARS = [2022, 2023, 2024, 2025]

MODEL_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    colsample_bytree=0.8,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=2,
    verbosity=-1,
)

COSTS_BPS = [0, 1, 2, 5]


def detect_epoch_unit(values):
    # Accept pandas Series or NumPy arrays.
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    med = float(x.dropna().abs().median())
    if med >= 1e17:
        return "ns"
    if med >= 1e14:
        return "us"
    if med >= 1e11:
        return "ms"
    return "s"


def load_canonical_close():
    print("=" * 78)
    print("Loading canonical timestamp/close arrays...")
    print("=" * 78)

    parts_ts = []
    parts_close = []

    for chunk in pd.read_csv(
        CANONICAL_FILE,
        usecols=["timestamp", "close"],
        chunksize=CHUNK_SIZE,
    ):
        ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
        close = pd.to_numeric(chunk["close"], errors="coerce")

        if ts.isna().any() or close.isna().any():
            raise ValueError("Invalid timestamp/close in canonical file.")

        parts_ts.append(ts.astype(np.int64).to_numpy())
        parts_close.append(close.astype(np.float64).to_numpy())

    ts = np.concatenate(parts_ts)
    close = np.concatenate(parts_close)

    if not np.all(ts[1:] > ts[:-1]):
        raise ValueError("Canonical timestamps are not strictly increasing.")

    print(f"Canonical rows: {len(ts):,}")
    print(f"Timestamp unit: epoch-{detect_epoch_unit(ts)}")
    return ts, close


def load_target_arrays(path):
    print(f"Loading target: {path}")

    if not path.exists():
        raise FileNotFoundError(path)

    head = pd.read_csv(path, nrows=2)
    missing = [c for c in TARGET_USECOLS if c not in head.columns]
    if missing:
        raise KeyError(f"Missing target columns in {path}: {missing}")

    parts = []
    for chunk in pd.read_csv(
        path,
        usecols=TARGET_USECOLS,
        chunksize=CHUNK_SIZE,
    ):
        ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
        label = pd.to_numeric(chunk["tb_label_60m"], errors="coerce")
        exit_idx = pd.to_numeric(chunk["tb_exit_index_60m"], errors="coerce")
        barrier = pd.to_numeric(chunk["barrier_return_60m"], errors="coerce")

        if ts.isna().any() or label.isna().any() or exit_idx.isna().any():
            raise ValueError(f"Invalid target fields in {path}")

        # Barrier may be NaN on session-closed/invalid rows, which is expected.
        parts.append(
            (
                ts.astype(np.int64).to_numpy(),
                label.astype(np.int16).to_numpy(),
                exit_idx.astype(np.int64).to_numpy(),
                barrier.astype(np.float64).to_numpy(),
            )
        )

    target_ts = np.concatenate([p[0] for p in parts])
    labels = np.concatenate([p[1] for p in parts])
    exits = np.concatenate([p[2] for p in parts])
    barriers = np.concatenate([p[3] for p in parts])

    del parts
    gc.collect()

    if not np.all(target_ts[1:] > target_ts[:-1]):
        raise ValueError(f"Target timestamps are not strictly increasing: {path}")

    print(
        f"Target rows: {len(target_ts):,} | "
        f"valid labels: {np.isin(labels, [0,1,2]).sum():,} | "
        f"ambiguous: {(labels == -1).sum():,} | "
        f"invalid: {(labels == -2).sum():,} | "
        f"session-closed: {(labels == -3).sum():,}"
    )

    return target_ts, labels, exits, barriers


def load_v4_features(target_ts, target_labels, target_exits, target_barriers):
    """
    Stream V4 and align target arrays by timestamp.

    Returns compact NumPy arrays for only the 2019-2025 ML development period.
    """
    print("Streaming V4 features and aligning dynamic target...")

    required = ["timestamp", "close", *FEATURES]
    head = pd.read_csv(FEATURE_FILE, nrows=2)
    missing = [c for c in required if c not in head.columns]
    if missing:
        raise KeyError(f"Missing V4 feature columns: {missing}")

    xs = []
    timestamps = []
    closes = []
    labels = []
    exits = []
    barriers = []
    years_list = []

    total = 0
    kept = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            FEATURE_FILE,
            usecols=required,
            chunksize=CHUNK_SIZE,
        )
    ):
        ts = pd.to_numeric(chunk["timestamp"], errors="coerce").to_numpy(
            dtype=np.int64,
        )

        if not np.all(ts[1:] > ts[:-1]):
            raise ValueError(f"V4 timestamp order broken in chunk {chunk_no}")

        idx = np.searchsorted(target_ts, ts)

        if np.any(idx >= len(target_ts)):
            raise ValueError(f"V4 timestamp beyond target range in chunk {chunk_no}")

        if not np.all(target_ts[idx] == ts):
            bad = int(np.sum(target_ts[idx] != ts))
            raise ValueError(
                f"{bad} V4 timestamps failed target alignment in chunk {chunk_no}"
            )

        chunk_labels = target_labels[idx]
        chunk_exits = target_exits[idx]
        chunk_barriers = target_barriers[idx]

        years = pd.to_datetime(ts, unit="ms", utc=True).year.to_numpy(
            dtype=np.int16
        )

        # Keep development period only. 2026 never enters the ML arrays.
        mask = (
            np.isin(chunk_labels, [0, 1, 2])
            & (years >= 2019)
            & (years <= 2025)
        )

        pos = np.flatnonzero(mask)

        if pos.size:
            # Deterministic chronological stride, matching prior baseline.
            pos = pos[(pos % STRIDE) == 0]

        if pos.size:
            feat = chunk.iloc[pos][FEATURES].to_numpy(
                dtype=np.float32,
                copy=True,
            )

            if not np.isfinite(feat).all():
                raise ValueError(f"Non-finite V4 feature at chunk {chunk_no}")

            close = pd.to_numeric(
                chunk.iloc[pos]["close"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            if not np.isfinite(close).all():
                raise ValueError(f"Invalid close in V4 chunk {chunk_no}")

            xs.append(feat)
            timestamps.append(ts[pos].copy())
            closes.append(close)
            labels.append(chunk_labels[pos].copy())
            exits.append(chunk_exits[pos].copy())
            barriers.append(chunk_barriers[pos].copy())
            years_list.append(years[pos].copy())

            kept += len(pos)

        total += len(chunk)

        if (chunk_no + 1) % 5 == 0:
            print(
                f"Chunks: {chunk_no+1} | "
                f"V4 rows: {total:,} | "
                f"sampled kept: {kept:,}"
            )

    X = np.concatenate(xs, axis=0)
    ts = np.concatenate(timestamps, axis=0)
    close = np.concatenate(closes, axis=0)
    y = np.concatenate(labels, axis=0)
    exits = np.concatenate(exits, axis=0)
    barriers = np.concatenate(barriers, axis=0)
    years = np.concatenate(years_list, axis=0)

    del xs, timestamps, closes, labels, years_list
    gc.collect()

    if not np.all(ts[1:] > ts[:-1]):
        raise ValueError("Aligned ML timestamps are not strictly increasing.")

    print(
        f"Aligned ML rows: {len(X):,} | "
        f"X shape: {X.shape} | "
        f"years: {years.min()}-{years.max()}"
    )

    return X, ts, close, y, exits, barriers, years


def simulate(
    pred,
    sample_ts,
    entry_close,
    actual_label,
    exit_index,
    barrier_return,
    canonical_ts,
    canonical_close,
    validation_year,
    cost_bps,
):
    """
    Simulate non-overlapping trades.

    Predicted TIMEOUT = no trade.

    For LONG/SHORT predictions:
      actual same side -> +dynamic barrier_return
      actual opposite  -> -dynamic barrier_return
      actual timeout   -> exact close-to-close return to tb_exit_index

    AMBIGUOUS/INVALID are excluded before this function.
    """
    year_end = int(
        pd.Timestamp(f"{validation_year}-12-31 23:59:59", tz="UTC").value
        // 1_000_000
    )

    trades = []
    i = 0
    n = len(pred)

    while i < n:
        side = int(pred[i])

        if side == 1:  # TIMEOUT => no trade
            i += 1
            continue

        eidx = int(exit_index[i])
        if eidx < 0 or eidx >= len(canonical_ts):
            i += 1
            continue

        exit_ts = int(canonical_ts[eidx])
        if exit_ts > year_end:
            i += 1
            continue

        entry = float(entry_close[i])
        br = float(barrier_return[i])
        actual = int(actual_label[i])

        if (
            not np.isfinite(entry)
            or entry <= 0.0
            or not np.isfinite(br)
            or br <= 0.0
        ):
            i += 1
            continue

        if actual == side:
            gross = br
        elif actual in (0, 2):
            gross = -br
        elif actual == 1:
            exit_price = float(canonical_close[eidx])
            if not np.isfinite(exit_price) or exit_price <= 0.0:
                i += 1
                continue

            if side == 2:
                gross = exit_price / entry - 1.0
            else:
                gross = entry / exit_price - 1.0
        else:
            i += 1
            continue

        net = gross - cost_bps / 10000.0

        trades.append(
            (
                i,
                eidx,
                gross,
                net,
                actual,
                side,
                (exit_ts - int(sample_ts[i])) / 60000.0,
            )
        )

        # Move to first sampled feature after actual exit.
        next_i = int(np.searchsorted(sample_ts, exit_ts, side="right"))
        i = max(i + 1, next_i)

    if not trades:
        return {
            "trades": 0,
            "compound": 0.0,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "max_drawdown": 0.0,
            "avg_hold_minutes": np.nan,
            "directional_accuracy": np.nan,
        }

    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    holds = []
    directional_hits = 0

    for (
        entry_i,
        exit_i,
        gross,
        net,
        actual,
        side,
        hold_min,
    ) in trades:
        equity *= max(0.0, 1.0 + net)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)

        if net > 0:
            wins += 1
            gross_profit += net
        elif net < 0:
            gross_loss += -net

        if actual == side:
            directional_hits += 1

        holds.append(hold_min)

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
        if gross_profit > 0
        else np.nan
    )

    return {
        "trades": len(trades),
        "compound": equity - 1.0,
        "profit_factor": pf,
        "win_rate": wins / len(trades),
        "max_drawdown": max_dd,
        "avg_hold_minutes": float(np.mean(holds)),
        "directional_accuracy": directional_hits / len(trades),
    }


def run_config(cfg, canonical_ts, canonical_close):
    target_ts, target_labels, target_exits, target_barriers = load_target_arrays(
        cfg["target_file"]
    )

    X, ts, close, y, exits, barriers, years = load_v4_features(
        target_ts,
        target_labels,
        target_exits,
        target_barriers,
    )

    rows = []

    print("\n" + "=" * 78)
    print(f"CONFIG: {cfg['tag']} | barrier multiplier={cfg['multiplier']:.2f}x")
    print("=" * 78)

    for val_year in YEARS:
        # Purge training rows whose actual target exit reaches the
        # validation year. This prevents a training label from using future
        # prices from inside the validation period.
        validation_start_ms = int(
            pd.Timestamp(f"{val_year}-01-01 00:00:00", tz="UTC").value
            // 1_000_000
        )

        exit_ts_train = np.full(len(exits), np.iinfo(np.int64).max, dtype=np.int64)
        valid_exit_mask = (exits >= 0) & (exits < len(canonical_ts))
        exit_ts_train[valid_exit_mask] = canonical_ts[exits[valid_exit_mask]]

        train_mask = (
            (years >= 2019)
            & (years < val_year)
            & (exit_ts_train < validation_start_ms)
        )
        val_mask = years == val_year

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(f"Empty fold for {cfg['tag']} {val_year}")

        print("-" * 78)
        purged_rows = int(((years >= 2019) & (years < val_year) & ~train_mask).sum())
        print(
            f"Fold validation {val_year} | "
            f"train={len(X_train):,} | val={len(X_val):,} | "
            f"purged_train_rows={purged_rows:,}"
        )

        actual_counts = np.bincount(y_val.astype(np.int64), minlength=3)
        actual_pct = actual_counts / actual_counts.sum() * 100.0

        model = LGBMClassifier(**MODEL_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )

        pred = model.predict(X_val).astype(np.int8)

        pred_counts = np.bincount(pred.astype(np.int64), minlength=3)
        pred_pct = pred_counts / pred_counts.sum() * 100.0

        acc = accuracy_score(y_val, pred)
        bal = balanced_accuracy_score(y_val, pred)
        macro = f1_score(y_val, pred, average="macro")
        best_iter = int(model.best_iteration_ or MODEL_PARAMS["n_estimators"])

        print(
            f"Actual %: [{actual_pct[0]:.2f}, {actual_pct[1]:.2f}, {actual_pct[2]:.2f}]"
        )
        print(
            f"Pred %:   [{pred_pct[0]:.2f}, {pred_pct[1]:.2f}, {pred_pct[2]:.2f}]"
        )
        print(f"Best iteration: {best_iter}")
        print(f"Accuracy: {acc*100:.4f}%")
        print(f"Balanced accuracy: {bal*100:.4f}%")
        print(f"Macro F1: {macro:.4f}")

        val_ts = ts[val_mask]
        val_close = close[val_mask]
        val_y = y_val
        val_exits = exits[val_mask]
        val_barriers = barriers[val_mask]

        for cost in COSTS_BPS:
            sim = simulate(
                pred=pred,
                sample_ts=val_ts,
                entry_close=val_close,
                actual_label=val_y,
                exit_index=val_exits,
                barrier_return=val_barriers,
                canonical_ts=canonical_ts,
                canonical_close=canonical_close,
                validation_year=val_year,
                cost_bps=cost,
            )

            print(
                f"{cost}bps | trades={sim['trades']} | "
                f"compound={sim['compound']*100:.4f}% | "
                f"PF={sim['profit_factor']:.4f} | "
                f"win={sim['win_rate']*100 if np.isfinite(sim['win_rate']) else np.nan:.4f}% | "
                f"MDD={sim['max_drawdown']*100:.4f}%"
            )

            rows.append(
                {
                    "config": cfg["tag"],
                    "multiplier": cfg["multiplier"],
                    "horizon_min": 60,
                    "validation_year": val_year,
                    "best_iteration": best_iter,
                    "train_rows": len(X_train),
                    "validation_rows": len(X_val),
                    "purged_train_rows": purged_rows,
                    "actual_short_pct": actual_pct[0],
                    "actual_timeout_pct": actual_pct[1],
                    "actual_long_pct": actual_pct[2],
                    "pred_short_pct": pred_pct[0],
                    "pred_timeout_pct": pred_pct[1],
                    "pred_long_pct": pred_pct[2],
                    "accuracy": acc,
                    "balanced_accuracy": bal,
                    "macro_f1": macro,
                    "cost_bps": cost,
                    "trades": sim["trades"],
                    "compound_return": sim["compound"],
                    "profit_factor": sim["profit_factor"],
                    "win_rate": sim["win_rate"],
                    "max_drawdown": sim["max_drawdown"],
                    "avg_hold_minutes": sim["avg_hold_minutes"],
                    "directional_accuracy": sim["directional_accuracy"],
                }
            )

        del model, X_train, y_train, X_val, y_val, pred
        gc.collect()

    del target_ts, target_labels, target_exits, target_barriers
    del X, ts, close, y, exits, barriers, years
    gc.collect()

    return rows


def main():
    print("=" * 78)
    print("SESSION-AWARE DYNAMIC-BARRIER LIGHTGBM WALK-FORWARD BASELINE")
    print("=" * 78)
    print("Configs: 0.50x / 0.75x / 1.00x")
    print("Features: session-aware V4")
    print("Sampling stride: 5")
    print("2026: COMPLETELY UNTOUCHED")
    print("=" * 78)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)
    if not CANONICAL_FILE.exists():
        raise FileNotFoundError(CANONICAL_FILE)

    canonical_ts, canonical_close = load_canonical_close()

    all_rows = []
    for cfg in CONFIGS:
        all_rows.extend(run_config(cfg, canonical_ts, canonical_close))

    REPORT.mkdir(parents=True, exist_ok=True)
    out = REPORT / "session_dynamic_tb_lgbm_walk_forward.csv"

    df = pd.DataFrame(all_rows)
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("SESSION-AWARE DYNAMIC-BARRIER ML COMPLETE")
    print("=" * 78)
    print(f"Saved: {out}")
    print("2026 was not used for fitting, selection, or evaluation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
