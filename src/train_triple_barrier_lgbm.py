#!/usr/bin/env python3
"""
XAUUSD Triple-Barrier LightGBM Walk-Forward Baseline.

Uses the fixed triple-barrier targets already built:
    +/- 5 bps horizontal barriers
    15m / 30m / 60m vertical horizons

Labels:
    0 = SHORT
    1 = TIMEOUT
    2 = LONG
    -1 = AMBIGUOUS   -> excluded
    -2 = INVALID     -> excluded

Features:
    The existing 42 V3 causal features.

Validation:
    2019-2021 -> 2022
    2019-2022 -> 2023
    2019-2023 -> 2024
    2019-2024 -> 2025

2026 is completely untouched.

This is a BASELINE experiment, not nested model selection.
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

FEATURE_FILE = Path(
    "data/processed/xauusd_m1_ml_features_v3.csv"
)

TARGET_FILE = Path(
    "data/processed/xauusd_m1_triple_barrier_targets.csv"
)

OUTPUT_FILE = Path(
    "reports/triple_barrier_walk_forward_summary.csv"
)

# ============================================================
# CONFIG
# ============================================================

HORIZONS = [15, 30, 60]

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

# Target file only needs timestamp + labels.
TARGET_DTYPES = {
    "timestamp": "int64",
    **{f"tb_label_{h}m": "int8" for h in HORIZONS},
}


# ============================================================
# TIME HELPERS
# ============================================================

def year_start_ms(year: int) -> int:
    return int(
        pd.Timestamp(f"{year}-01-01", tz="UTC").timestamp() * 1000
    )


def train_end_ms(validation_year: int) -> int:
    return int(
        (
            pd.Timestamp(f"{validation_year}-01-01", tz="UTC")
            - pd.Timedelta(minutes=MAX_HORIZON + 1)
        ).timestamp()
        * 1000
    )


def validation_start_ms(year: int) -> int:
    return int(
        (
            pd.Timestamp(f"{year}-01-01", tz="UTC")
            + pd.Timedelta(minutes=WARMUP_MINUTES)
        ).timestamp()
        * 1000
    )


# ============================================================
# LOAD TARGET LABELS ONCE
# ============================================================

def load_targets():
    cols = ["timestamp"] + [f"tb_label_{h}m" for h in HORIZONS]

    print("Loading triple-barrier label index...")
    target = pd.read_csv(
        TARGET_FILE,
        usecols=cols,
        dtype=TARGET_DTYPES,
    )

    if target["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps in triple-barrier target file.")

    target = target.set_index("timestamp")
    print(f"Triple-barrier index rows: {len(target):,}")

    return target


# ============================================================
# LOAD ONE FOLD / HORIZON
# ============================================================

def load_split(target_index, label_col, train_start, train_end, val_start, val_end):
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

        # Match labels by timestamp.
        labels = target_index.reindex(chunk["timestamp"].to_numpy())
        labels = labels[label_col].to_numpy(dtype=np.int16)

        chunk[label_col] = labels

        train_mask = (
            (chunk["timestamp"] >= train_start)
            & (chunk["timestamp"] <= train_end)
            & (chunk[label_col] >= 0)
        )

        val_mask = (
            (chunk["timestamp"] >= val_start)
            & (chunk["timestamp"] < val_end)
            & (chunk[label_col] >= 0)
        )

        if train_mask.any():
            train_parts.append(
                chunk.loc[
                    train_mask,
                    FEATURES + [label_col]
                ].copy()
            )

        if val_mask.any():
            val_parts.append(
                chunk.loc[
                    val_mask,
                    ["timestamp"] + FEATURES + [label_col]
                ].copy()
            )

    if not train_parts or not val_parts:
        raise RuntimeError(
            f"Empty split for {label_col}: "
            f"train_parts={len(train_parts)}, val_parts={len(val_parts)}"
        )

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)

    # Same computational sampling policy as the earlier ML experiments.
    train = train.iloc[::STRIDE].reset_index(drop=True)
    val = val.sort_values("timestamp").reset_index(drop=True)

    return train, val, chunks


# ============================================================
# SIMPLE COST-AWARE TRADING CHECK
# ============================================================

def simulate_trades(timestamp, actual_future_return, predicted_class, horizon_minutes,
                    cost_bps=2.0):
    """
    LONG prediction -> take future return as-is.
    SHORT prediction -> take negative future return.
    TIMEOUT -> no trade.

    One position at a time.
    This is a sanity check only.
    """
    ts = np.asarray(timestamp, dtype=np.int64)
    ret = np.asarray(actual_future_return, dtype=np.float64)
    pred = np.asarray(predicted_class, dtype=np.int8)

    cost = cost_bps / 10_000.0
    hold_ms = horizon_minutes * 60_000

    net = []

    i = 0
    while i < len(ts):
        if pred[i] == 2:
            side = 1.0
        elif pred[i] == 0:
            side = -1.0
        else:
            i += 1
            continue

        net.append(side * ret[i] - cost)

        exit_time = ts[i] + hold_ms
        j = int(np.searchsorted(ts, exit_time, side="left"))
        i = max(j, i + 1)

    if not net:
        return {
            "trades": 0,
            "compound_return": 0.0,
            "profit_factor": np.nan,
            "win_rate": np.nan,
        }

    net = np.asarray(net, dtype=np.float64)
    equity = np.cumprod(1.0 + net)

    wins = net[net > 0].sum()
    losses = -net[net < 0].sum()

    return {
        "trades": len(net),
        "compound_return": float(equity[-1] - 1.0),
        "profit_factor": float(wins / losses) if losses > 0 else np.inf,
        "win_rate": float(np.mean(net > 0)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)

    if not TARGET_FILE.exists():
        raise FileNotFoundError(TARGET_FILE)

    print("=" * 78)
    print("XAUUSD TRIPLE-BARRIER LIGHTGBM WALK-FORWARD BASELINE")
    print("=" * 78)
    print(f"Features: {FEATURE_FILE}")
    print(f"Targets:  {TARGET_FILE}")
    print("Barrier:  +/- 5 bps")
    print("2026 TEST SET: NOT USED")
    print()

    target_index = load_targets()

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

            print("-" * 78)
            print(f"Fold {fold_id} | Horizon {horizon}m")

            train, val, chunks = load_split(
                target_index,
                label_col,
                tr_start,
                tr_end,
                va_start,
                va_end,
            )

            y_train = train[label_col].to_numpy(np.int8)
            y_val = val[label_col].to_numpy(np.int8)

            # Base close-to-close return target from the V3 feature file.
            # It is used ONLY for the simple trading sanity check.
            raw_return_col = f"future_return_{horizon}m"

            # Reload only timestamp + actual return for this validation range.
            # These are joined from the V3 dataset in the same chunk loop.
            actual_return_parts = []

            for chunk in pd.read_csv(
                FEATURE_FILE,
                usecols=["timestamp", raw_return_col],
                dtype={"timestamp": "int64", raw_return_col: "float32"},
                chunksize=CHUNK_SIZE,
            ):
                mask = (
                    (chunk["timestamp"] >= va_start)
                    & (chunk["timestamp"] < va_end)
                    & chunk[raw_return_col].notna()
                )
                if mask.any():
                    actual_return_parts.append(
                        chunk.loc[mask, ["timestamp", raw_return_col]]
                    )

            actual_returns = (
                pd.concat(actual_return_parts, ignore_index=True)
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            val = val.sort_values("timestamp").reset_index(drop=True)

            # Ensure exact alignment.
            aligned = val[["timestamp"]].merge(
                actual_returns,
                on="timestamp",
                how="left",
                validate="one_to_one",
            )

            if aligned[raw_return_col].isna().any():
                raise RuntimeError(
                    f"Missing actual returns after alignment for fold {fold_id}, "
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
            f1 = f1_score(
                y_val,
                pred,
                average="macro",
                zero_division=0,
            )

            valid_trade = pred != 1
            coverage = float(np.mean(valid_trade))

            trading = simulate_trades(
                val["timestamp"].to_numpy(np.int64),
                aligned[raw_return_col].to_numpy(np.float64),
                pred,
                horizon,
                cost_bps=2.0,
            )

            print(f"Chunks read: {chunks}")
            print(f"Train rows: {len(train):,}")
            print(f"Validation rows: {len(val):,}")
            print(
                "Valid class %:",
                np.round(
                    np.bincount(y_val, minlength=3)
                    / len(y_val)
                    * 100.0,
                    2,
                ).tolist(),
            )
            print(f"Best iteration: {best_iteration}")
            print(f"Accuracy: {acc:.4%}")
            print(f"Balanced accuracy: {bal:.4%}")
            print(f"Macro F1: {f1:.4f}")
            print(f"Predicted trade coverage: {coverage:.4%}")
            print(
                f"2bps trades: {trading['trades']:,} | "
                f"compound: {trading['compound_return']:.4%} | "
                f"PF: {trading['profit_factor']:.4f}"
            )

            rows.append({
                "fold": fold_id,
                "validation_year": val_year,
                "horizon_min": horizon,
                "barrier_return": 0.0005,
                "best_iteration": best_iteration,
                "train_rows": len(train),
                "validation_rows": len(val),
                "valid_short_pct": float(np.mean(y_val == 0)),
                "valid_timeout_pct": float(np.mean(y_val == 1)),
                "valid_long_pct": float(np.mean(y_val == 2)),
                "accuracy": float(acc),
                "balanced_accuracy": float(bal),
                "macro_f1": float(f1),
                "predicted_trade_coverage": coverage,
                "trades_at_2bps": trading["trades"],
                "compound_return_at_2bps": trading["compound_return"],
                "profit_factor_at_2bps": trading["profit_factor"],
                "win_rate_at_2bps": trading["win_rate"],
            })

            del (
                train,
                val,
                y_train,
                y_val,
                model,
                proba,
                pred,
                actual_return_parts,
                actual_returns,
                aligned,
            )
            gc.collect()

    out = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 78)
    print("TRIPLE-BARRIER WALK-FORWARD SUMMARY")
    print("=" * 78)
    print(out.to_string(index=False))
    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("2026 remains untouched.")
    print("=" * 78)


if __name__ == "__main__":
    main()
