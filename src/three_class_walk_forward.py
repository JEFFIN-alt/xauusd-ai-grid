#!/usr/bin/env python3
"""
XAUUSD 3-class classification baseline.

Goal:
    Replace raw-return regression with:
        SHORT = future return < -deadzone
        FLAT  = abs(future return) <= deadzone
        LONG  = future return > +deadzone

Initial deadzone:
    2 bps = 0.0002 round-trip-cost scale

This is a baseline experiment:
- 4 walk-forward folds
- 5/15/30/60 minute horizons
- LightGBM multiclass
- stride-5 training sample
- 2026 is NEVER used
- validation is used for early stopping only; this is not the final nested test

The script reports:
- class balance
- accuracy
- balanced accuracy
- macro F1
- LONG/SHORT precision and recall
- non-flat directional accuracy
- predicted class coverage
- simple close-to-close trading result at 2 bps round-trip cost
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )
except ImportError as exc:
    raise RuntimeError(
        "scikit-learn is required for this script. "
        "Install it in ai-lab with: conda install -c conda-forge scikit-learn"
    ) from exc


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/processed/xauusd_m1_ml_features_v3.csv")
OUTPUT_FILE = Path("reports/three_class_walk_forward_summary.csv")

HORIZONS = [5, 15, 30, 60]

# 2 bps expressed as decimal return.
DEADZONE = 0.0002
ROUND_TRIP_COST_BPS = 2.0

STRIDE = 5
MAX_HORIZON_MIN = 60
FEATURE_WARMUP_MIN = 480
CHUNK_SIZE = 100_000

# Class mapping:
# 0 = SHORT
# 1 = FLAT
# 2 = LONG
CLASS_NAMES = ["SHORT", "FLAT", "LONG"]

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

PARAMS = dict(
    objective="multiclass",
    num_class=3,
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

DTYPES = {
    "timestamp": "int64",
    **{c: "float32" for c in FEATURES},
}

for h in HORIZONS:
    DTYPES[f"future_return_{h}m"] = "float32"


# ============================================================
# DATE HELPERS
# ============================================================

def ts_ms(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").timestamp() * 1000)


def year_start(year: int) -> int:
    return ts_ms(f"{year}-01-01 00:00:00")


def training_end_before(year: int) -> int:
    # Conservative 60-minute purge before validation year.
    return int(
        (
            pd.Timestamp(f"{year}-01-01", tz="UTC")
            - pd.Timedelta(minutes=MAX_HORIZON_MIN + 1)
        ).timestamp()
        * 1000
    )


def validation_range(year: int):
    start = int(
        (
            pd.Timestamp(f"{year}-01-01", tz="UTC")
            + pd.Timedelta(minutes=FEATURE_WARMUP_MIN)
        ).timestamp()
        * 1000
    )
    end = year_start(year + 1)
    return start, end


# ============================================================
# LABELING
# ============================================================

def make_labels(y: np.ndarray) -> np.ndarray:
    """
    SHORT = 0
    FLAT  = 1
    LONG  = 2

    Deadzone is fixed at 2 bps.
    """
    labels = np.ones(len(y), dtype=np.int8)
    labels[y < -DEADZONE] = 0
    labels[y > DEADZONE] = 2
    return labels


# ============================================================
# DATA LOADING
# ============================================================

def load_split(target_col, train_start, train_end, valid_start, valid_end):
    usecols = ["timestamp", target_col] + FEATURES

    train_parts = []
    valid_parts = []

    chunks = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=usecols,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
    ):
        chunks += 1

        train_mask = (
            (chunk["timestamp"] >= train_start)
            & (chunk["timestamp"] <= train_end)
            & chunk[target_col].notna()
        )

        valid_mask = (
            (chunk["timestamp"] >= valid_start)
            & (chunk["timestamp"] < valid_end)
            & chunk[target_col].notna()
        )

        if train_mask.any():
            train_parts.append(
                chunk.loc[train_mask, FEATURES + [target_col]].copy()
            )

        if valid_mask.any():
            valid_parts.append(
                chunk.loc[
                    valid_mask,
                    ["timestamp", target_col] + FEATURES
                ].copy()
            )

    if not train_parts or not valid_parts:
        raise RuntimeError(
            f"Empty split for {target_col}: "
            f"train_parts={len(train_parts)}, valid_parts={len(valid_parts)}"
        )

    train = pd.concat(train_parts, ignore_index=True)
    valid = pd.concat(valid_parts, ignore_index=True)

    # Match the existing ML experiments: systematic stride-5 training sample.
    train = train.iloc[::STRIDE].reset_index(drop=True)

    valid = valid.sort_values("timestamp").reset_index(drop=True)

    return train, valid, chunks


# ============================================================
# TRADING SIMULATION
# ============================================================

def simulate_trades(
    timestamp_ms,
    actual_return,
    predicted_class,
    horizon_minutes,
):
    """
    One position at a time.
    Predicted FLAT => no trade.
    Predicted LONG/SHORT => hold for the stated horizon.

    Cost is an abstract fixed round-trip return cost.
    """
    ts = np.asarray(timestamp_ms, dtype=np.int64)
    y = np.asarray(actual_return, dtype=np.float64)
    cls = np.asarray(predicted_class, dtype=np.int8)

    cost = ROUND_TRIP_COST_BPS / 10_000.0
    hold_ms = horizon_minutes * 60_000

    trade_net = []
    trade_gross = []
    longs = 0
    shorts = 0

    i = 0

    while i < len(ts):
        if cls[i] == 2:
            side = 1.0
            longs += 1
        elif cls[i] == 0:
            side = -1.0
            shorts += 1
        else:
            i += 1
            continue

        gross = side * y[i]
        net = gross - cost

        trade_gross.append(gross)
        trade_net.append(net)

        exit_time = ts[i] + hold_ms
        next_i = int(np.searchsorted(ts, exit_time, side="left"))

        i = max(next_i, i + 1)

    if not trade_net:
        return {
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "gross_sum": 0.0,
            "net_sum": 0.0,
            "compound_return": 0.0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
        }

    gross = np.asarray(trade_gross, dtype=np.float64)
    net = np.asarray(trade_net, dtype=np.float64)

    equity = np.cumprod(1.0 + net)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0

    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()

    return {
        "trades": int(len(net)),
        "long_trades": int(longs),
        "short_trades": int(shorts),
        "gross_sum": float(gross.sum()),
        "net_sum": float(net.sum()),
        "compound_return": float(equity[-1] - 1.0),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": (
            float(gains / losses)
            if losses > 0
            else np.inf
        ),
        "max_drawdown": float(drawdown.min()),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    header = pd.read_csv(INPUT_FILE, nrows=0)

    for h in HORIZONS:
        col = f"future_return_{h}m"
        if col not in header.columns:
            raise RuntimeError(f"Missing target column: {col}")

    folds = [
        (1, 2022),
        (2, 2023),
        (3, 2024),
        (4, 2025),
    ]

    results = []

    print("=" * 78)
    print("XAUUSD 3-CLASS WALK-FORWARD CLASSIFICATION BASELINE")
    print("=" * 78)
    print(f"Input: {INPUT_FILE}")
    print(f"Deadzone: ±{DEADZONE:.4%} (2 bps)")
    print(f"Trading cost: {ROUND_TRIP_COST_BPS:.1f} bps round trip")
    print("2026 TEST SET: NOT USED")
    print()

    for fold_id, valid_year in folds:
        train_start = year_start(2019)
        train_end = training_end_before(valid_year)
        valid_start, valid_end = validation_range(valid_year)

        print("=" * 78)
        print(f"FOLD {fold_id} | VALIDATION {valid_year}")
        print("=" * 78)
        print(
            "Train:",
            pd.to_datetime(train_start, unit="ms", utc=True),
            "->",
            pd.to_datetime(train_end, unit="ms", utc=True),
        )
        print(
            "Validation:",
            pd.to_datetime(valid_start, unit="ms", utc=True),
            "->",
            pd.to_datetime(valid_end, unit="ms", utc=True)
            - pd.Timedelta(minutes=1),
        )
        print()

        for horizon in HORIZONS:
            target_col = f"future_return_{horizon}m"

            print("-" * 78)
            print(f"Fold {fold_id} | Horizon {horizon}m")
            print("-" * 78)

            train, valid, chunk_count = load_split(
                target_col,
                train_start,
                train_end,
                valid_start,
                valid_end,
            )

            y_train = make_labels(
                train[target_col].to_numpy(np.float64)
            )
            y_valid = make_labels(
                valid[target_col].to_numpy(np.float64)
            )

            print(f"Chunks read: {chunk_count}")
            print(f"Train rows: {len(train):,}")
            print(f"Validation rows: {len(valid):,}")

            print(
                "Train class %:",
                np.round(
                    np.bincount(y_train, minlength=3)
                    / len(y_train)
                    * 100.0,
                    2,
                ).tolist()
            )
            print(
                "Valid class %:",
                np.round(
                    np.bincount(y_valid, minlength=3)
                    / len(y_valid)
                    * 100.0,
                    2,
                ).tolist()
            )

            model = LGBMClassifier(**PARAMS)

            model.fit(
                train[FEATURES],
                y_train,
                eval_set=[(valid[FEATURES], y_valid)],
                eval_metric="multi_logloss",
                callbacks=[
                    lgb.early_stopping(30, verbose=False)
                ],
            )

            best_iteration = model.best_iteration_ or PARAMS["n_estimators"]

            proba = model.predict_proba(
                valid[FEATURES],
                num_iteration=best_iteration,
            )

            pred = np.argmax(proba, axis=1).astype(np.int8)

            acc = accuracy_score(y_valid, pred)
            bal_acc = balanced_accuracy_score(y_valid, pred)
            macro_f1 = f1_score(
                y_valid,
                pred,
                average="macro",
                zero_division=0,
            )

            precision = precision_score(
                y_valid,
                pred,
                labels=[0, 2],
                average=None,
                zero_division=0,
            )
            recall = recall_score(
                y_valid,
                pred,
                labels=[0, 2],
                average=None,
                zero_division=0,
            )

            actual_nonflat = y_valid != 1
            pred_direction = pred != 1
            nonflat_direction_accuracy = (
                float(np.mean(pred[actual_nonflat] == y_valid[actual_nonflat]))
                if actual_nonflat.any()
                else np.nan
            )

            predicted_trade_coverage = float(np.mean(pred != 1))

            trading = simulate_trades(
                valid["timestamp"].to_numpy(np.int64),
                valid[target_col].to_numpy(np.float64),
                pred,
                horizon,
            )

            cm = confusion_matrix(
                y_valid,
                pred,
                labels=[0, 1, 2],
            )

            print(f"Best iteration: {best_iteration}")
            print(f"Accuracy: {acc:.4%}")
            print(f"Balanced accuracy: {bal_acc:.4%}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(
                f"SHORT precision: {precision[0]:.4%} | "
                f"LONG precision: {precision[1]:.4%}"
            )
            print(
                f"SHORT recall: {recall[0]:.4%} | "
                f"LONG recall: {recall[1]:.4%}"
            )
            print(
                f"Non-flat actual direction accuracy: "
                f"{nonflat_direction_accuracy:.4%}"
            )
            print(
                f"Predicted trade coverage: "
                f"{predicted_trade_coverage:.4%}"
            )
            print(
                f"2bps cost-aware trades: {trading['trades']:,} | "
                f"compound: {trading['compound_return']:.4%} | "
                f"PF: {trading['profit_factor']:.4f}"
            )
            print("Confusion matrix [SHORT, FLAT, LONG]:")
            print(cm)

            results.append({
                "fold": fold_id,
                "validation_year": valid_year,
                "horizon_min": horizon,
                "deadzone": DEADZONE,
                "best_iteration": best_iteration,
                "train_rows": len(train),
                "validation_rows": len(valid),
                "train_short_pct": float(np.mean(y_train == 0)),
                "train_flat_pct": float(np.mean(y_train == 1)),
                "train_long_pct": float(np.mean(y_train == 2)),
                "valid_short_pct": float(np.mean(y_valid == 0)),
                "valid_flat_pct": float(np.mean(y_valid == 1)),
                "valid_long_pct": float(np.mean(y_valid == 2)),
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "macro_f1": float(macro_f1),
                "short_precision": float(precision[0]),
                "long_precision": float(precision[1]),
                "short_recall": float(recall[0]),
                "long_recall": float(recall[1]),
                "nonflat_direction_accuracy": nonflat_direction_accuracy,
                "predicted_trade_coverage": predicted_trade_coverage,
                "trades_at_2bps": trading["trades"],
                "compound_return_at_2bps": trading["compound_return"],
                "win_rate_at_2bps": trading["win_rate"],
                "profit_factor_at_2bps": trading["profit_factor"],
                "max_drawdown_at_2bps": trading["max_drawdown"],
                "cm_short_short": int(cm[0, 0]),
                "cm_short_flat": int(cm[0, 1]),
                "cm_short_long": int(cm[0, 2]),
                "cm_flat_short": int(cm[1, 0]),
                "cm_flat_flat": int(cm[1, 1]),
                "cm_flat_long": int(cm[1, 2]),
                "cm_long_short": int(cm[2, 0]),
                "cm_long_flat": int(cm[2, 1]),
                "cm_long_long": int(cm[2, 2]),
            })

            del model, proba, pred, train, valid, y_train, y_valid
            gc.collect()

    out = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 78)
    print("3-CLASS WALK-FORWARD SUMMARY")
    print("=" * 78)

    print(
        out[
            [
                "validation_year",
                "horizon_min",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "short_precision",
                "long_precision",
                "nonflat_direction_accuracy",
                "predicted_trade_coverage",
                "trades_at_2bps",
                "compound_return_at_2bps",
                "profit_factor_at_2bps",
                "max_drawdown_at_2bps",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("2026 remains untouched.")
    print("=" * 78)


if __name__ == "__main__":
    main()
