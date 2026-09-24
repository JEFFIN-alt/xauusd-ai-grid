#!/usr/bin/env python3
"""
Nested walk-forward ML/trading test for XAUUSD.

Purpose:
- Use an INNER validation year to choose the horizon, LightGBM stopping point,
  and signal threshold.
- Evaluate the frozen choice on the OUTER validation year.
- Keep 2026 completely untouched.
- Use a fixed 2 bps round-trip cost ONLY for inner selection.
- Report outer performance at 0/1/2/5 bps as sensitivity analysis.

Outer folds:
  2019-2021 -> 2022
  2019-2022 -> 2023
  2019-2023 -> 2024
  2019-2024 -> 2025

For each outer fold:
  inner train = data before the inner validation year
  inner valid = immediately preceding year
  outer train = all data before outer validation year
  outer valid = outer validation year

No outer validation information is used to select the horizon/threshold.
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/processed/xauusd_m1_ml_features_v3.csv")
OUTPUT_SELECTION = Path("reports/nested_walk_forward_selections.csv")
OUTPUT_RESULTS = Path("reports/nested_walk_forward_outer_results.csv")

HORIZONS = [5, 15, 30, 60]
THRESHOLDS = [0.0, 0.0001, 0.0002, 0.0005]
OUTER_COST_GRID = [0.0, 1.0, 2.0, 5.0]

# Fixed cost used ONLY during inner selection.
SELECTION_COST_BPS = 2.0

STRIDE = 5
MAX_HORIZON_MIN = 60
FEATURE_WARMUP_MIN = 480
CHUNK_SIZE = 100_000

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

# Only the columns actually needed by this script.
DTYPES = {
    "timestamp": "int64",
    **{c: "float32" for c in FEATURES},
}
for h in HORIZONS:
    DTYPES[f"future_return_{h}m"] = "float32"


# ============================================================
# DATE HELPERS
# ============================================================

def ts_ms(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def year_start(year: int) -> int:
    return ts_ms(f"{year}-01-01 00:00:00")


def range_before_validation_year(validation_year: int):
    start = year_start(2019)
    val_start = year_start(validation_year)
    train_end = int(
        (pd.Timestamp(f"{validation_year}-01-01", tz="UTC")
         - pd.Timedelta(minutes=MAX_HORIZON_MIN + 1)).timestamp() * 1000
    )
    return start, train_end


def validation_range(year: int):
    start = int(
        (pd.Timestamp(f"{year}-01-01", tz="UTC")
         + pd.Timedelta(minutes=FEATURE_WARMUP_MIN)).timestamp() * 1000
    )
    end = year_start(year + 1)
    return start, end


# ============================================================
# TARGET COLUMN RESOLUTION
# ============================================================

def resolve_targets():
    header = pd.read_csv(INPUT_FILE, nrows=0)
    targets = {}
    for h in HORIZONS:
        name = f"future_return_{h}m"
        if name not in header.columns:
            raise RuntimeError(f"Missing required target column: {name}")
        targets[h] = name
    return targets


# ============================================================
# DATA LOADING
# ============================================================

def load_range(target_col, train_start, train_end, val_start, val_end):
    """
    Load one train and one validation range for a single horizon.
    Training is sampled every 5th row to match the existing experiments.
    """
    usecols = ["timestamp", target_col] + FEATURES

    train_parts = []
    valid_parts = []
    chunk_count = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=usecols,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
    ):
        chunk_count += 1

        train_mask = (
            (chunk["timestamp"] >= train_start)
            & (chunk["timestamp"] <= train_end)
            & chunk[target_col].notna()
        )

        valid_mask = (
            (chunk["timestamp"] >= val_start)
            & (chunk["timestamp"] < val_end)
            & chunk[target_col].notna()
        )

        if train_mask.any():
            train_parts.append(
                chunk.loc[train_mask, FEATURES + [target_col]].copy()
            )

        if valid_mask.any():
            valid_parts.append(
                chunk.loc[valid_mask, ["timestamp", target_col] + FEATURES].copy()
            )

    if not train_parts or not valid_parts:
        raise RuntimeError(
            f"Empty split: target={target_col}, "
            f"train={len(train_parts)}, valid={len(valid_parts)}"
        )

    train = pd.concat(train_parts, ignore_index=True).iloc[::STRIDE].reset_index(drop=True)
    valid = pd.concat(valid_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    return train, valid, chunk_count


# ============================================================
# TRADING SIMULATION
# ============================================================

def simulate(timestamp_ms, actual_return, prediction, horizon_minutes, threshold, cost_bps):
    ts = np.asarray(timestamp_ms, dtype=np.int64)
    y = np.asarray(actual_return, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)

    cost = cost_bps / 10_000.0
    hold_ms = horizon_minutes * 60_000

    gross = []
    net = []
    longs = 0
    shorts = 0

    i = 0
    while i < len(ts):
        if p[i] > threshold:
            side = 1.0
            longs += 1
        elif p[i] < -threshold:
            side = -1.0
            shorts += 1
        else:
            i += 1
            continue

        g = side * y[i]
        gross.append(g)
        net.append(g - cost)

        exit_time = ts[i] + hold_ms
        next_i = int(np.searchsorted(ts, exit_time, side="left"))
        i = max(next_i, i + 1)

    if not net:
        return {
            "trades": 0,
            "longs": 0,
            "shorts": 0,
            "gross_sum": 0.0,
            "net_sum": 0.0,
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
    drawdown = equity / peaks - 1.0

    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()
    pf = gains / losses if losses > 0 else np.inf

    return {
        "trades": int(len(net)),
        "longs": int(longs),
        "shorts": int(shorts),
        "gross_sum": float(gross.sum()),
        "net_sum": float(net.sum()),
        "compound_return": float(equity[-1] - 1.0),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": float(pf),
        "max_drawdown": float(drawdown.min()),
        "avg_trade_net": float(net.mean()),
    }


# ============================================================
# INNER MODEL
# ============================================================

def fit_inner_model(train, valid, target_col):
    model = LGBMRegressor(**PARAMS)
    model.fit(
        train[FEATURES],
        train[target_col],
        eval_set=[(valid[FEATURES], valid[target_col])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    best_iteration = model.best_iteration_ or PARAMS["n_estimators"]
    prediction = model.predict(valid[FEATURES], num_iteration=best_iteration)
    return model, best_iteration, prediction


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    targets = resolve_targets()

    outer_folds = [
        (1, 2022, 2021),
        (2, 2023, 2022),
        (3, 2024, 2023),
        (4, 2025, 2024),
    ]

    print("=" * 78)
    print("NESTED WALK-FORWARD XAUUSD ML/TRADING TEST")
    print("=" * 78)
    print(f"Input: {INPUT_FILE}")
    print(f"Inner-selection cost: {SELECTION_COST_BPS:.1f} bps round trip")
    print("2026 TEST SET: NOT USED")
    print()

    selections = []
    outer_results = []

    for outer_id, outer_year, inner_year in outer_folds:
        print("=" * 78)
        print(f"OUTER FOLD {outer_id}")
        print(f"Outer validation year: {outer_year}")
        print(f"Inner validation year: {inner_year}")
        print("=" * 78)

        outer_train_start, outer_train_end = range_before_validation_year(outer_year)
        outer_val_start, outer_val_end = validation_range(outer_year)

        inner_train_start, inner_train_end = range_before_validation_year(inner_year)
        inner_val_start, inner_val_end = validation_range(inner_year)

        print(
            "Outer train:",
            pd.to_datetime(outer_train_start, unit="ms", utc=True),
            "->",
            pd.to_datetime(outer_train_end, unit="ms", utc=True),
        )
        print(
            "Outer valid:",
            pd.to_datetime(outer_val_start, unit="ms", utc=True),
            "->",
            pd.to_datetime(outer_val_end, unit="ms", utc=True) - pd.Timedelta(minutes=1),
        )
        print()

        # ----------------------------------------------------
        # INNER SELECTION
        # ----------------------------------------------------
        inner_candidates = []

        for h in HORIZONS:
            target_col = targets[h]

            print("-" * 78)
            print(f"INNER | Horizon {h}m")

            inner_train, inner_valid, chunks = load_range(
                target_col,
                inner_train_start,
                inner_train_end,
                inner_val_start,
                inner_val_end,
            )

            print(f"Chunks read: {chunks}")
            print(f"Inner train rows: {len(inner_train):,}")
            print(f"Inner validation rows: {len(inner_valid):,}")

            model, best_iteration, pred = fit_inner_model(
                inner_train, inner_valid, target_col
            )

            print(f"Best iteration: {best_iteration}")
            print(f"Prediction std: {np.std(pred):.8e}")

            for threshold in THRESHOLDS:
                stats = simulate(
                    inner_valid["timestamp"].to_numpy(np.int64),
                    inner_valid[target_col].to_numpy(np.float64),
                    pred,
                    h,
                    threshold,
                    SELECTION_COST_BPS,
                )

                inner_candidates.append({
                    "outer_fold": outer_id,
                    "outer_validation_year": outer_year,
                    "inner_validation_year": inner_year,
                    "horizon_min": h,
                    "threshold": threshold,
                    "best_iteration": best_iteration,
                    **stats,
                })

            del model, inner_train, inner_valid, pred
            gc.collect()

        inner_df = pd.DataFrame(inner_candidates)

        # Require at least 20 trades to avoid selecting a configuration
        # based on a tiny number of observations.
        eligible = inner_df[inner_df["trades"] >= 20].copy()
        if eligible.empty:
            eligible = inner_df.copy()

        # Primary: highest inner compound return.
        # Tie-break 1: higher profit factor.
        # Tie-break 2: higher trade count.
        eligible = eligible.sort_values(
            by=["compound_return", "profit_factor", "trades"],
            ascending=[False, False, False],
        )

        selected = eligible.iloc[0].to_dict()
        selections.append(selected)

        selected_h = int(selected["horizon_min"])
        selected_threshold = float(selected["threshold"])
        selected_iteration = int(selected["best_iteration"])

        print()
        print("SELECTED FROM INNER ONLY")
        print(f"  Horizon:   {selected_h}m")
        print(f"  Threshold: {selected_threshold:.6f}")
        print(f"  Iterations:{selected_iteration}")
        print(f"  Inner compound return @ {SELECTION_COST_BPS:.1f} bps: "
              f"{selected['compound_return']:.4%}")
        print(f"  Inner trades: {int(selected['trades'])}")
        print()

        # ----------------------------------------------------
        # OUTER EVALUATION
        # ----------------------------------------------------
        target_col = targets[selected_h]

        outer_train, outer_valid, chunks = load_range(
            target_col,
            outer_train_start,
            outer_train_end,
            outer_val_start,
            outer_val_end,
        )

        print("-" * 78)
        print(f"OUTER | Frozen {selected_h}m model/rule")
        print(f"Chunks read: {chunks}")
        print(f"Outer train rows: {len(outer_train):,}")
        print(f"Outer validation rows: {len(outer_valid):,}")

        # IMPORTANT: no outer early stopping.
        # The iteration count was selected inside the nested inner split.
        model = LGBMRegressor(
            **{**PARAMS, "n_estimators": selected_iteration}
        )
        model.fit(
            outer_train[FEATURES],
            outer_train[target_col],
        )

        pred = model.predict(outer_valid[FEATURES])

        print(f"Frozen iterations: {selected_iteration}")
        print(f"Outer prediction std: {np.std(pred):.8e}")

        ts = outer_valid["timestamp"].to_numpy(np.int64)
        actual = outer_valid[target_col].to_numpy(np.float64)

        for cost_bps in OUTER_COST_GRID:
            stats = simulate(
                ts,
                actual,
                pred,
                selected_h,
                selected_threshold,
                cost_bps,
            )

            outer_results.append({
                "outer_fold": outer_id,
                "outer_validation_year": outer_year,
                "inner_validation_year": inner_year,
                "selected_horizon_min": selected_h,
                "selected_threshold": selected_threshold,
                "selected_iterations": selected_iteration,
                "evaluation_cost_bps": cost_bps,
                **stats,
            })

        del model, outer_train, outer_valid, pred, ts, actual
        gc.collect()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    OUTPUT_SELECTION.parent.mkdir(parents=True, exist_ok=True)

    selections_df = pd.DataFrame(selections)
    results_df = pd.DataFrame(outer_results)

    selections_df.to_csv(OUTPUT_SELECTION, index=False)
    results_df.to_csv(OUTPUT_RESULTS, index=False)

    print()
    print("=" * 78)
    print("NESTED WALK-FORWARD SUMMARY")
    print("=" * 78)

    print("\nSelections (chosen without seeing outer validation):")
    print(
        selections_df[
            [
                "outer_fold",
                "outer_validation_year",
                "inner_validation_year",
                "horizon_min",
                "threshold",
                "best_iteration",
                "compound_return",
                "trades",
                "profit_factor",
            ]
        ].to_string(index=False)
    )

    print("\nOuter results by cost:")
    print(
        results_df[
            [
                "outer_fold",
                "outer_validation_year",
                "selected_horizon_min",
                "selected_threshold",
                "selected_iterations",
                "evaluation_cost_bps",
                "trades",
                "compound_return",
                "win_rate",
                "profit_factor",
                "max_drawdown",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved selections: {OUTPUT_SELECTION}")
    print(f"Saved outer results: {OUTPUT_RESULTS}")
    print("2026 remains untouched.")
    print("=" * 78)


if __name__ == "__main__":
    main()
