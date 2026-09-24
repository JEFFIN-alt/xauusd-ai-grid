#!/usr/bin/env python3
"""
XAUUSD Triple-Barrier LightGBM Walk-Forward Baseline V2.

This is the same classifier experiment as train_triple_barrier_lgbm.py,
but the trading simulator is corrected.

CRITICAL CORRECTION
-------------------
The old simulator used fixed-horizon future_return_* values after a
triple-barrier prediction. That is not the same payoff definition as the
triple-barrier target.

This V2 simulator uses the actual triple-barrier outcome and exit:
    predicted LONG + actual LONG    -> +5 bps gross
    predicted LONG + actual SHORT   -> -5 bps gross
    predicted SHORT + actual SHORT  -> +5 bps gross
    predicted SHORT + actual LONG   -> -5 bps gross
    predicted LONG + TIMEOUT        -> close_exit / close_entry - 1
    predicted SHORT + TIMEOUT       -> close_entry / close_exit - 1

Ambiguous (-1) and invalid (-2) targets are excluded before evaluation.

Positions are non-overlapping: after entering, the next signal is allowed
only after the actual target exit.

A 2 bps transaction cost is subtracted from every completed trade in the
baseline check. The simulator also reports 0 / 1 / 2 / 5 bps sensitivity.

Validation folds:
    2019-2021 -> 2022
    2019-2022 -> 2023
    2019-2023 -> 2024
    2019-2024 -> 2025

2026 is kept completely untouched. Validation entries whose actual target
exit reaches 2026 are excluded so that the 2026 holdout is not used by the
trading evaluation.

This remains a BASELINE experiment. No threshold, barrier, or model
hyperparameter optimization is performed here.
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# ============================================================
# FILES
# ============================================================

FEATURE_FILE = Path("data/processed/xauusd_m1_ml_features_v3.csv")
TARGET_FILE = Path("data/processed/xauusd_m1_triple_barrier_targets.csv")
CANONICAL_FILE = Path("data/processed/xauusd_m1_2019_2026_canonical.csv")
OUTPUT_FILE = Path("reports/triple_barrier_walk_forward_summary_v2.csv")


# ============================================================
# CONFIG
# ============================================================

HORIZONS = [15, 30, 60]
BARRIER_RETURN = 0.0005  # +/- 5 bps
COST_SENSITIVITY_BPS = [0.0, 1.0, 2.0, 5.0]

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

STRIDE = 5
CHUNK_SIZE = 100_000
MAX_HORIZON = 60
WARMUP_MINUTES = 480

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

FEATURE_DTYPES = {
    "timestamp": "int64",
    **{c: "float32" for c in FEATURES},
}


# ============================================================
# TIME HELPERS
# ============================================================

def year_start_ms(year: int) -> int:
    return int(pd.Timestamp(f"{year}-01-01", tz="UTC").timestamp() * 1000)


def train_end_ms(validation_year: int) -> int:
    return int(
        (
            pd.Timestamp(f"{validation_year}-01-01", tz="UTC")
            - pd.Timedelta(minutes=MAX_HORIZON + 1)
        ).timestamp() * 1000
    )


def validation_start_ms(year: int) -> int:
    return int(
        (
            pd.Timestamp(f"{year}-01-01", tz="UTC")
            + pd.Timedelta(minutes=WARMUP_MINUTES)
        ).timestamp() * 1000
    )


# ============================================================
# LOAD TARGET INDEX
# ============================================================

def load_targets():
    cols = ["timestamp", "close"]
    for h in HORIZONS:
        cols.extend([
            f"tb_label_{h}m",
            f"tb_exit_index_{h}m",
        ])

    dtype = {
        "timestamp": "int64",
        "close": "float32",
    }
    for h in HORIZONS:
        dtype[f"tb_label_{h}m"] = "int8"
        dtype[f"tb_exit_index_{h}m"] = "int64"

    print("Loading triple-barrier target index...")
    target = pd.read_csv(
        TARGET_FILE,
        usecols=cols,
        dtype=dtype,
    )

    if target["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps in triple-barrier target file.")

    target = target.set_index("timestamp")
    print(f"Triple-barrier index rows: {len(target):,}")
    return target


# ============================================================
# LOAD CANONICAL CLOSE / TIMESTAMP ARRAYS
# ============================================================

def load_canonical_arrays():
    print("Loading canonical timestamp/close arrays for exact exits...")
    canonical = pd.read_csv(
        CANONICAL_FILE,
        usecols=["timestamp", "close"],
        dtype={"timestamp": "int64", "close": "float32"},
    )

    ts = canonical["timestamp"].to_numpy(np.int64)
    close = canonical["close"].to_numpy(np.float32)

    if len(ts) == 0:
        raise RuntimeError("Canonical dataset is empty.")

    return ts, close


# ============================================================
# LOAD ONE FOLD / HORIZON
# ============================================================

def load_split(target_index, label_col, exit_col, train_start, train_end, val_start, val_end):
    usecols = ["timestamp"] + FEATURES

    train_parts = []
    val_parts = []
    chunks = 0

    for chunk in pd.read_csv(
        FEATURE_FILE,
        usecols=usecols,
        dtype=FEATURE_DTYPES,
        chunksize=CHUNK_SIZE,
    ):
        chunks += 1

        labels = target_index.reindex(chunk["timestamp"].to_numpy())
        chunk[label_col] = labels[label_col].to_numpy(dtype=np.int16)
        chunk[exit_col] = labels[exit_col].to_numpy(dtype=np.int64)

        train_mask = (
            (chunk["timestamp"] >= train_start)
            & (chunk["timestamp"] <= train_end)
            & (chunk[label_col] >= 0)
        )

        val_mask = (
            (chunk["timestamp"] >= val_start)
            & (chunk["timestamp"] < val_end)
            & (chunk[label_col] >= 0)
            & (chunk[exit_col] >= 0)
        )

        if train_mask.any():
            train_parts.append(
                chunk.loc[train_mask, FEATURES + [label_col]].copy()
            )

        if val_mask.any():
            val_parts.append(
                chunk.loc[val_mask, ["timestamp"] + FEATURES + [label_col, exit_col]].copy()
            )

    if not train_parts or not val_parts:
        raise RuntimeError(
            f"Empty split for {label_col}: train_parts={len(train_parts)}, "
            f"val_parts={len(val_parts)}"
        )

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)

    train = train.iloc[::STRIDE].reset_index(drop=True)
    val = val.sort_values("timestamp").reset_index(drop=True)

    return train, val, chunks


# ============================================================
# BARRIER-AWARE TRADING SIMULATOR
# ============================================================


def run_barrier_simulation(
    timestamp,
    entry_close,
    actual_label,
    exit_index,
    predicted_class,
    canonical_timestamp,
    canonical_close,
    cost_bps,
    val_end_ms,
):
    """Barrier-aware, non-overlapping simulator."""
    ts = np.asarray(timestamp, dtype=np.int64)
    entry_close = np.asarray(entry_close, dtype=np.float64)
    actual = np.asarray(actual_label, dtype=np.int8)
    exit_idx = np.asarray(exit_index, dtype=np.int64)
    pred = np.asarray(predicted_class, dtype=np.int8)

    cost = cost_bps / 10_000.0

    net_returns = []
    gross_returns = []
    durations = []

    long_predictions = 0
    short_predictions = 0
    correct_direction = 0
    direction_predictions = 0

    i = 0
    skipped_beyond_validation = 0

    while i < len(ts):
        if pred[i] == 1:
            i += 1
            continue

        direction_predictions += 1
        if pred[i] == 2:
            long_predictions += 1
        elif pred[i] == 0:
            short_predictions += 1

        j = int(exit_idx[i])
        if j < 0 or j >= len(canonical_close):
            i += 1
            continue

        exit_ts = int(canonical_timestamp[j])

        # Keep the final holdout clean: a validation trade may not use a
        # target exit that reaches the next calendar year.
        if exit_ts >= val_end_ms:
            skipped_beyond_validation += 1
            i += 1
            continue

        label = int(actual[i])

        if pred[i] == 2:
            side = 1.0
            if label == 2:
                gross = BARRIER_RETURN
                correct_direction += 1
            elif label == 0:
                gross = -BARRIER_RETURN
            elif label == 1:
                gross = float(canonical_close[j] / entry_close[i] - 1.0)
            else:
                i += 1
                continue
        else:  # pred[i] == 0
            side = -1.0
            if label == 0:
                gross = BARRIER_RETURN
                correct_direction += 1
            elif label == 2:
                gross = -BARRIER_RETURN
            elif label == 1:
                gross = float(entry_close[i] / canonical_close[j] - 1.0)
            else:
                i += 1
                continue

            # `side` exists explicitly for readability and future extension.
            _ = side

        net = gross - cost
        gross_returns.append(gross)
        net_returns.append(net)

        durations.append((exit_ts - ts[i]) / 60_000.0)

        # Do not allow a second signal while the current position is open.
        next_ts = exit_ts
        i = int(np.searchsorted(ts, next_ts, side="right"))

    if not net_returns:
        return {
            "trades": 0,
            "compound_return": 0.0,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "max_drawdown": 0.0,
            "avg_hold_minutes": np.nan,
            "directional_accuracy": np.nan,
            "long_predictions": long_predictions,
            "short_predictions": short_predictions,
            "skipped_beyond_validation": skipped_beyond_validation,
        }

    gross_returns = np.asarray(gross_returns, dtype=np.float64)
    net_returns = np.asarray(net_returns, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)

    equity = np.cumprod(1.0 + net_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0

    gross_profit = net_returns[net_returns > 0].sum()
    gross_loss = -net_returns[net_returns < 0].sum()

    return {
        "trades": int(len(net_returns)),
        "compound_return": float(equity[-1] - 1.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        "win_rate": float(np.mean(net_returns > 0)),
        "max_drawdown": float(drawdown.min()),
        "avg_hold_minutes": float(durations.mean()),
        "directional_accuracy": float(correct_direction / direction_predictions)
            if direction_predictions > 0 else np.nan,
        "long_predictions": int(long_predictions),
        "short_predictions": int(short_predictions),
        "skipped_beyond_validation": int(skipped_beyond_validation),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    for path in [FEATURE_FILE, TARGET_FILE, CANONICAL_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("=" * 78)
    print("XAUUSD TRIPLE-BARRIER LIGHTGBM WALK-FORWARD BASELINE V2")
    print("=" * 78)
    print(f"Features:   {FEATURE_FILE}")
    print(f"Targets:    {TARGET_FILE}")
    print(f"Canonical:  {CANONICAL_FILE}")
    print("Barrier:    +/- 5 bps")
    print("Simulator:  ACTUAL triple-barrier exit")
    print("2026 TEST:  NOT USED")
    print()

    target_index = load_targets()
    canonical_ts, canonical_close = load_canonical_arrays()

    folds = [
        (1, 2022),
        (2, 2023),
        (3, 2024),
        (4, 2025),
    ]

    rows = []

    for fold_id, val_year in folds:
        tr_start = year_start_ms(2019)
        tr_end = train_end_ms(val_year)
        va_start = validation_start_ms(val_year)
        va_end = year_start_ms(val_year + 1)

        print("=" * 78)
        print(f"FOLD {fold_id} | VALIDATION {val_year}")
        print("=" * 78)

        for horizon in HORIZONS:
            label_col = f"tb_label_{horizon}m"
            exit_col = f"tb_exit_index_{horizon}m"

            print("-" * 78)
            print(f"Fold {fold_id} | Horizon {horizon}m")

            train, val, chunks = load_split(
                target_index,
                label_col,
                exit_col,
                tr_start,
                tr_end,
                va_start,
                va_end,
            )

            y_train = train[label_col].to_numpy(np.int8)
            y_val = val[label_col].to_numpy(np.int8)

            # Entry close comes from the triple-barrier target file, which is
            # aligned to the same timestamp as each V3 feature row.
            target_info = target_index.reindex(val["timestamp"].to_numpy())
            entry_close = target_info["close"].to_numpy(np.float64)
            exit_index = val[exit_col].to_numpy(np.int64)

            if np.isnan(entry_close).any():
                raise RuntimeError(
                    f"Missing entry close after target alignment for fold {fold_id}, "
                    f"horizon {horizon}m"
                )

            model = LGBMClassifier(**PARAMS)

            model.fit(
                train[FEATURES],
                y_train,
                eval_set=[(val[FEATURES], y_val)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            best_iteration = model.best_iteration_ or PARAMS["n_estimators"]

            proba = model.predict_proba(
                val[FEATURES],
                num_iteration=best_iteration,
            )
            pred = np.argmax(proba, axis=1).astype(np.int8)

            acc = accuracy_score(y_val, pred)
            bal = balanced_accuracy_score(y_val, pred)
            f1 = f1_score(y_val, pred, average="macro", zero_division=0)
            coverage = float(np.mean(pred != 1))

            print(f"Chunks read: {chunks}")
            print(f"Train rows: {len(train):,}")
            print(f"Validation rows: {len(val):,}")
            print(
                "Actual class %:",
                np.round(
                    np.bincount(y_val, minlength=3) / len(y_val) * 100.0,
                    2,
                ).tolist(),
            )
            print(
                "Predicted class %:",
                np.round(
                    np.bincount(pred, minlength=3) / len(pred) * 100.0,
                    2,
                ).tolist(),
            )
            print(f"Best iteration: {best_iteration}")
            print(f"Accuracy: {acc:.4%}")
            print(f"Balanced accuracy: {bal:.4%}")
            print(f"Macro F1: {f1:.4f}")
            print(f"Predicted trade coverage: {coverage:.4%}")

            sim_results = {}
            for cost_bps in COST_SENSITIVITY_BPS:
                sim = run_barrier_simulation(
                    val["timestamp"].to_numpy(np.int64),
                    entry_close,
                    y_val,
                    exit_index,
                    pred,
                    canonical_ts,
                    canonical_close,
                    cost_bps,
                    va_end,
                )
                sim_results[cost_bps] = sim
                print(
                    f"{cost_bps:g}bps | trades={sim['trades']:,} | "
                    f"compound={sim['compound_return']:.4%} | "
                    f"PF={sim['profit_factor']:.4f} | "
                    f"win={sim['win_rate']:.4%} | "
                    f"MDD={sim['max_drawdown']:.4%}"
                )

            base = sim_results[2.0]

            rows.append({
                "fold": fold_id,
                "validation_year": val_year,
                "horizon_min": horizon,
                "barrier_return": BARRIER_RETURN,
                "best_iteration": best_iteration,
                "train_rows": len(train),
                "validation_rows": len(val),
                "valid_short_pct": float(np.mean(y_val == 0)),
                "valid_timeout_pct": float(np.mean(y_val == 1)),
                "valid_long_pct": float(np.mean(y_val == 2)),
                "pred_short_pct": float(np.mean(pred == 0)),
                "pred_timeout_pct": float(np.mean(pred == 1)),
                "pred_long_pct": float(np.mean(pred == 2)),
                "accuracy": float(acc),
                "balanced_accuracy": float(bal),
                "macro_f1": float(f1),
                "predicted_trade_coverage": coverage,
                "trades_at_0bps": sim_results[0.0]["trades"],
                "compound_return_at_0bps": sim_results[0.0]["compound_return"],
                "profit_factor_at_0bps": sim_results[0.0]["profit_factor"],
                "max_drawdown_at_0bps": sim_results[0.0]["max_drawdown"],
                "trades_at_1bps": sim_results[1.0]["trades"],
                "compound_return_at_1bps": sim_results[1.0]["compound_return"],
                "profit_factor_at_1bps": sim_results[1.0]["profit_factor"],
                "max_drawdown_at_1bps": sim_results[1.0]["max_drawdown"],
                "trades_at_2bps": base["trades"],
                "compound_return_at_2bps": base["compound_return"],
                "profit_factor_at_2bps": base["profit_factor"],
                "win_rate_at_2bps": base["win_rate"],
                "max_drawdown_at_2bps": base["max_drawdown"],
                "avg_hold_minutes_at_2bps": base["avg_hold_minutes"],
                "directional_accuracy_at_2bps": base["directional_accuracy"],
                "trades_at_5bps": sim_results[5.0]["trades"],
                "compound_return_at_5bps": sim_results[5.0]["compound_return"],
                "profit_factor_at_5bps": sim_results[5.0]["profit_factor"],
                "max_drawdown_at_5bps": sim_results[5.0]["max_drawdown"],
                "skipped_beyond_validation_exit_at_2bps": base["skipped_beyond_validation"],
            })

            del (
                train,
                val,
                y_train,
                y_val,
                target_info,
                entry_close,
                exit_index,
                model,
                proba,
                pred,
                sim_results,
            )
            gc.collect()

    out = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 78)
    print("TRIPLE-BARRIER WALK-FORWARD V2 SUMMARY")
    print("=" * 78)
    print(out.to_string(index=False))
    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("2026 remains untouched; validation exits are also kept inside each fold's calendar year.")
    print("The simulator now uses actual barrier outcomes and exact timeout exits.")
    print("=" * 78)


if __name__ == "__main__":
    main()
