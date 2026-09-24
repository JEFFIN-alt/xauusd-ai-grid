#!/usr/bin/env python3
"""
Cost-aware, non-overlapping LightGBM sanity backtest for the existing
walk-forward XAUUSD experiment.

Important:
- Uses the same expanding walk-forward periods as walk_forward_horizon_test.py.
- 2026 is NEVER used.
- Re-trains each fold/horizon model with early stopping on that fold's validation
  set, so this is an exploratory sanity check rather than a pristine final OOS test.
- No threshold/cost is optimized; a fixed sensitivity grid is reported.
- Trades are close-to-close, one position at a time, held for the target horizon.
- Spread/slippage are abstracted as round-trip return costs in basis points (bps).
  This is NOT broker-specific execution modeling yet.
"""

from pathlib import Path
import gc
import json
import math

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/processed/xauusd_m1_ml_features_v3.csv")
OUTPUT_FILE = Path("reports/cost_aware_backtest_summary.csv")

HORIZONS = [5, 15, 30, 60]
STRIDE = 5
MAX_HORIZON_MIN = 60
FEATURE_WARMUP_MIN = 480
CHUNK_SIZE = 100_000

# Fixed sensitivity grid. These are NOT tuned to pick a winner.
# 0.0001 = 1 basis point of return.
THRESHOLDS = [0.0, 0.0001, 0.0002, 0.0005]
COST_BPS_GRID = [0.0, 1.0, 2.0, 5.0]

FEATURES = [
    "return_1m",
    "return_5m",
    "return_15m",
    "return_30m",
    "return_60m",
    "candle_range",
    "body",
    "abs_body",
    "upper_wick",
    "lower_wick",
    "body_range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "volatility_15m",
    "volatility_30m",
    "volatility_60m",
    "volatility_240m",
    "momentum_5m",
    "momentum_15m",
    "momentum_30m",
    "momentum_60m",
    "sma_15",
    "sma_60",
    "sma_240",
    "distance_sma_15",
    "distance_sma_60",
    "distance_sma_240",
    "sma_15_slope",
    "sma_60_slope",
    "sma_240_slope",
    "rolling_high_60",
    "rolling_low_60",
    "rolling_high_240",
    "rolling_low_240",
    "distance_high_60",
    "distance_low_60",
    "distance_high_240",
    "distance_low_240",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
]

DTYPES = {
    "timestamp": "int64",
    "close": "float32",
    **{c: "float32" for c in FEATURES},
}

FOLDS = [
    (1, 2022),
    (2, 2023),
    (3, 2024),
    (4, 2025),
]

PARAMS = dict(
    objective="regression",
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=2,
    verbosity=-1,
)


# ============================================================
# HELPERS
# ============================================================


def epoch_ms(dt: str) -> int:
    return int(pd.Timestamp(dt, tz="UTC").timestamp() * 1000)


def resolve_target_columns(header_columns):
    """Support the likely target naming variants used during development."""
    candidates = {}
    for h in HORIZONS:
        possible = [
            f"future_return_{h}m",
            f"target_return_{h}m",
            f"target_{h}m",
            f"return_{h}m_target",
        ]
        hit = next((c for c in possible if c in header_columns), None)
        if hit is None:
            raise RuntimeError(
                f"Could not find target column for {h}m. "
                f"Tried {possible}. Available columns containing 'return' or 'target': "
                f"{[c for c in header_columns if 'return' in c.lower() or 'target' in c.lower()]}"
            )
        candidates[h] = hit
    return candidates


def fold_windows(validation_year: int):
    """Match the existing walk-forward script's calendar windows."""
    train_start = epoch_ms("2019-01-01 00:00:00")
    val_year_start = pd.Timestamp(f"{validation_year}-01-01", tz="UTC")
    train_end = int((val_year_start - pd.Timedelta(minutes=MAX_HORIZON_MIN + 1)).timestamp() * 1000)
    val_start = int((val_year_start + pd.Timedelta(minutes=FEATURE_WARMUP_MIN)).timestamp() * 1000)
    val_end = int(pd.Timestamp(f"{validation_year + 1}-01-01", tz="UTC").timestamp() * 1000)
    return train_start, train_end, val_start, val_end


def load_fold(input_file, target_col, train_start, train_end, val_start, val_end):
    """Load only one fold. Float32 keeps RAM manageable on the VM."""
    usecols = ["timestamp", "close", target_col] + FEATURES
    train_parts = []
    val_parts = []
    chunks = 0

    for chunk in pd.read_csv(
        input_file,
        usecols=usecols,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
    ):
        chunks += 1
        chunk = chunk.sort_values("timestamp")
        train_mask = (
            (chunk["timestamp"] >= train_start)
            & (chunk["timestamp"] <= train_end)
            & chunk[target_col].notna()
        )
        val_mask = (
            (chunk["timestamp"] >= val_start)
            & (chunk["timestamp"] < val_end)
            & chunk[target_col].notna()
        )

        if train_mask.any():
            train_parts.append(chunk.loc[train_mask, FEATURES + [target_col]].copy())
        if val_mask.any():
            val_parts.append(chunk.loc[val_mask, ["timestamp", "close", FEATURES[0], *FEATURES[1:], target_col]].copy())

    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    valid = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()

    if train.empty or valid.empty:
        raise RuntimeError(
            f"Empty split. chunks={chunks}, train_rows={len(train):,}, val_rows={len(valid):,}"
        )

    # Reproduce the existing stride-5 training sampling.
    train = train.iloc[::STRIDE].reset_index(drop=True)
    valid = valid.sort_values("timestamp").reset_index(drop=True)

    return train, valid, chunks


def simulate_non_overlapping(timestamp_ms, actual_return, prediction, horizon_minutes, threshold, cost_bps):
    """
    One position at a time. A trade opened at t is held for horizon_minutes.
    The target already guarantees an exact future endpoint on the same segment.
    After exit, the next eligible signal is the first available row at/after exit.
    """
    ts = np.asarray(timestamp_ms, dtype=np.int64)
    y = np.asarray(actual_return, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)

    n = len(ts)
    if n == 0:
        return {
            "trades": 0,
            "longs": 0,
            "shorts": 0,
            "gross_return_sum": 0.0,
            "net_return_sum": 0.0,
            "compound_return": 0.0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
        }

    cost = cost_bps / 10_000.0
    hold_ms = horizon_minutes * 60_000

    trade_returns = []
    longs = 0
    shorts = 0
    i = 0

    while i < n:
        p = pred[i]
        if p > threshold:
            side = 1
            longs += 1
        elif p < -threshold:
            side = -1
            shorts += 1
        else:
            i += 1
            continue

        realized = side * y[i]
        net = realized - cost
        trade_returns.append(net)

        # Prevent overlapping positions.
        exit_time = ts[i] + hold_ms
        i = int(np.searchsorted(ts, exit_time, side="left"))
        if i <= 0:
            i += 1

    if not trade_returns:
        return {
            "trades": 0,
            "longs": 0,
            "shorts": 0,
            "gross_return_sum": 0.0,
            "net_return_sum": 0.0,
            "compound_return": 0.0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
        }

    r = np.asarray(trade_returns, dtype=np.float64)
    equity = np.cumprod(1.0 + r)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0

    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = wins / losses if losses > 0 else np.inf

    return {
        "trades": int(len(r)),
        "longs": int(longs),
        "shorts": int(shorts),
        "gross_return_sum": float(np.sum(side * y[i] if False else 0.0)),  # replaced below
        "net_return_sum": float(np.sum(r)),
        "compound_return": float(equity[-1] - 1.0),
        "win_rate": float(np.mean(r > 0)),
        "profit_factor": float(pf),
        "max_drawdown": float(np.min(drawdowns)),
    }


def simulate_with_gross(timestamp_ms, actual_return, prediction, horizon_minutes, threshold, cost_bps):
    """Same simulator, but keeps gross and net trade returns explicitly."""
    ts = np.asarray(timestamp_ms, dtype=np.int64)
    y = np.asarray(actual_return, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    cost = cost_bps / 10_000.0
    hold_ms = horizon_minutes * 60_000

    gross = []
    net = []
    longs = 0
    shorts = 0
    i = 0
    while i < len(ts):
        if pred[i] > threshold:
            side = 1
            longs += 1
        elif pred[i] < -threshold:
            side = -1
            shorts += 1
        else:
            i += 1
            continue
        g = side * y[i]
        gross.append(g)
        net.append(g - cost)
        exit_time = ts[i] + hold_ms
        i = int(np.searchsorted(ts, exit_time, side="left"))
        if i <= 0:
            i += 1

    if not net:
        return {
            "trades": 0,
            "longs": 0,
            "shorts": 0,
            "gross_return_sum": 0.0,
            "net_return_sum": 0.0,
            "compound_return": 0.0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
            "avg_trade_net": np.nan,
        }

    gross = np.asarray(gross, dtype=np.float64)
    net = np.asarray(net, dtype=np.float64)
    equity = np.cumprod(1.0 + net)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    wins = net[net > 0].sum()
    losses = -net[net < 0].sum()
    pf = wins / losses if losses > 0 else np.inf

    return {
        "trades": int(len(net)),
        "longs": int(longs),
        "shorts": int(shorts),
        "gross_return_sum": float(gross.sum()),
        "net_return_sum": float(net.sum()),
        "compound_return": float(equity[-1] - 1.0),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": float(pf),
        "max_drawdown": float(drawdowns.min()),
        "avg_trade_net": float(net.mean()),
    }


# ============================================================
# MAIN
# ============================================================


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    print("=" * 78)
    print("XAUUSD COST-AWARE WALK-FORWARD SANITY BACKTEST")
    print("=" * 78)
    print(f"Input: {INPUT_FILE}")
    print("2026 TEST SET: NOT USED")
    print("Training stride: every 5th V3 row")
    print("Trade model: close-to-close, one position at a time")
    print("Cost model: abstract round-trip return cost in bps")
    print()

    header = pd.read_csv(INPUT_FILE, nrows=0)
    target_columns = resolve_target_columns(header.columns)
    print("Resolved targets:")
    for h, c in target_columns.items():
        print(f"  {h}m -> {c}")
    print()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for fold_id, val_year in FOLDS:
        print("=" * 78)
        print(f"FOLD {fold_id} | VALIDATION {val_year}")
        print("=" * 78)
        train_start, train_end, val_start, val_end = fold_windows(val_year)
        print(
            f"Train: {pd.to_datetime(train_start, unit='ms', utc=True)} -> "
            f"{pd.to_datetime(train_end, unit='ms', utc=True)}"
        )
        print(
            f"Validation: {pd.to_datetime(val_start, unit='ms', utc=True)} -> "
            f"{pd.to_datetime(val_end, unit='ms', utc=True) - pd.Timedelta(minutes=1)}"
        )

        for h in HORIZONS:
            target_col = target_columns[h]
            print("-" * 78)
            print(f"Fold {fold_id} | Horizon {h}m")

            train, valid, chunk_count = load_fold(
                INPUT_FILE, target_col, train_start, train_end, val_start, val_end
            )
            print(f"Chunks read: {chunk_count}")
            print(f"Train rows: {len(train):,}")
            print(f"Validation rows: {len(valid):,}")

            model = LGBMRegressor(**PARAMS)
            model.fit(
                train[FEATURES],
                train[target_col],
                eval_set=[(valid[FEATURES], valid[target_col])],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            best_iteration = model.best_iteration_ or PARAMS["n_estimators"]
            pred = model.predict(valid[FEATURES], num_iteration=best_iteration)

            base = valid[["timestamp", "close", target_col]].copy()
            ts = base["timestamp"].to_numpy(dtype=np.int64)
            actual = base[target_col].to_numpy(dtype=np.float64)
            pred = np.asarray(pred, dtype=np.float64)

            print(f"Best iteration: {best_iteration}")
            print(f"Prediction mean: {pred.mean():.8e}")
            print(f"Prediction std:  {pred.std():.8e}")
            print()

            for threshold in THRESHOLDS:
                for cost_bps in COST_BPS_GRID:
                    stats = simulate_with_gross(
                        ts, actual, pred, h, threshold, cost_bps
                    )
                    results.append({
                        "fold": fold_id,
                        "validation_year": val_year,
                        "horizon_min": h,
                        "threshold": threshold,
                        "cost_bps_round_trip": cost_bps,
                        "train_rows": len(train),
                        "validation_rows": len(valid),
                        "best_iteration": best_iteration,
                        **stats,
                    })

            del model, train, valid, base, pred, ts, actual
            gc.collect()

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 78)
    print("SUMMARY: COMPOUNDED NET RETURN")
    print("=" * 78)
    summary = (
        out.groupby(["horizon_min", "threshold", "cost_bps_round_trip"], as_index=False)
        .agg(
            folds=("fold", "count"),
            total_trades=("trades", "sum"),
            mean_compound_return=("compound_return", "mean"),
            median_compound_return=("compound_return", "median"),
            mean_win_rate=("win_rate", "mean"),
            mean_profit_factor=("profit_factor", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
        )
    )
    print(summary.to_string(index=False))

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("2026 remains untouched.")


if __name__ == "__main__":
    main()
