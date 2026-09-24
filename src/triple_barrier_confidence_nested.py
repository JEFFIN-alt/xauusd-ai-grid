#!/usr/bin/env python3
"""
XAUUSD Triple-Barrier LightGBM: Nested Walk-Forward Confidence Filter.

Purpose
-------
The baseline classifier produced useful classification metrics but traded on
nearly every directional prediction. This experiment tests whether requiring
higher prediction confidence reduces turnover enough to improve cost-aware
trading results.

Nested design
-------------
For each outer validation year:
    inner year -> choose confidence threshold
    outer training period -> retrain classifier
    outer validation year -> frozen threshold evaluation

Folds:
    outer 2022: train 2019-2020 -> select on 2021 -> retrain 2019-2021 -> test 2022
    outer 2023: train 2019-2021 -> select on 2022 -> retrain 2019-2022 -> test 2023
    outer 2024: train 2019-2022 -> select on 2023 -> retrain 2019-2023 -> test 2024
    outer 2025: train 2019-2023 -> select on 2024 -> retrain 2019-2024 -> test 2025

2026 is completely untouched.

Confidence rule
---------------
Use argmax class prediction among SHORT(0), TIMEOUT(1), LONG(2).
A directional trade is allowed only when the winning directional class
probability is >= the selected threshold. TIMEOUT predictions are always
NO TRADE.

Selection objective
-------------------
Primary: highest 2 bps cost-aware compound return on the inner year,
subject to at least MIN_TRADE_FRACTION of the unfiltered directional trades.
This prevents a threshold with only a handful of trades from winning purely
by chance. The threshold grid is fixed in advance and is not changed per fold.

The trading simulator is barrier-aware: barrier hits use +/-5 bps and timeout
uses the exact canonical close at the stored exit index. Positions are
non-overlapping.
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
SUMMARY_FILE = Path("reports/triple_barrier_confidence_nested_summary.csv")
THRESHOLD_FILE = Path("reports/triple_barrier_confidence_nested_thresholds.csv")


# ============================================================
# CONFIG
# ============================================================

HORIZONS = [15, 30, 60]
BARRIER_RETURN = 0.0005
COST_BPS_PRIMARY = 2.0
COST_SENSITIVITY_BPS = [0.0, 1.0, 2.0, 5.0]

# Fixed in advance. 0.333... is the theoretical minimum for 3-class argmax.
CONFIDENCE_THRESHOLDS = [0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MIN_TRADE_FRACTION = 0.01

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
# DATA LOADERS
# ============================================================

def load_targets():
    cols = ["timestamp", "close"]
    for h in HORIZONS:
        cols.extend([f"tb_label_{h}m", f"tb_exit_index_{h}m"])

    dtype = {"timestamp": "int64", "close": "float32"}
    for h in HORIZONS:
        dtype[f"tb_label_{h}m"] = "int8"
        dtype[f"tb_exit_index_{h}m"] = "int64"

    print("Loading triple-barrier target index...")
    target = pd.read_csv(TARGET_FILE, usecols=cols, dtype=dtype)
    if target["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps in triple-barrier target file.")
    target = target.set_index("timestamp")
    print(f"Triple-barrier index rows: {len(target):,}")
    return target


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


def load_split(
    target_index,
    label_col,
    exit_col,
    train_start,
    train_end,
    val_start,
    val_end,
):
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
            train_parts.append(chunk.loc[train_mask, FEATURES + [label_col]].copy())

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
# BARRIER-AWARE SIMULATOR
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
    ts = np.asarray(timestamp, dtype=np.int64)
    entry_close = np.asarray(entry_close, dtype=np.float64)
    actual = np.asarray(actual_label, dtype=np.int8)
    exit_idx = np.asarray(exit_index, dtype=np.int64)
    pred = np.asarray(predicted_class, dtype=np.int8)

    cost = cost_bps / 10_000.0
    net_returns = []
    durations = []
    correct_direction = 0
    direction_predictions = 0
    long_predictions = 0
    short_predictions = 0
    skipped_beyond_validation = 0

    i = 0
    while i < len(ts):
        if pred[i] == 1:
            i += 1
            continue

        direction_predictions += 1
        if pred[i] == 2:
            long_predictions += 1
        elif pred[i] == 0:
            short_predictions += 1
        else:
            i += 1
            continue

        j = int(exit_idx[i])
        if j < 0 or j >= len(canonical_close):
            i += 1
            continue

        exit_ts = int(canonical_timestamp[j])
        if exit_ts >= val_end_ms:
            skipped_beyond_validation += 1
            i += 1
            continue

        label = int(actual[i])
        if pred[i] == 2:
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
        else:  # SHORT
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

        net_returns.append(gross - cost)
        durations.append((exit_ts - ts[i]) / 60_000.0)
        i = int(np.searchsorted(ts, exit_ts, side="right"))

    if not net_returns:
        return {
            "trades": 0,
            "compound_return": 0.0,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "max_drawdown": 0.0,
            "avg_hold_minutes": np.nan,
            "directional_accuracy": np.nan,
            "long_predictions": int(long_predictions),
            "short_predictions": int(short_predictions),
            "skipped_beyond_validation": int(skipped_beyond_validation),
        }

    net = np.asarray(net_returns, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    equity = np.cumprod(1.0 + net)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0

    gross_profit = net[net > 0].sum()
    gross_loss = -net[net < 0].sum()

    return {
        "trades": int(len(net)),
        "compound_return": float(equity[-1] - 1.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        "win_rate": float(np.mean(net > 0)),
        "max_drawdown": float(drawdown.min()),
        "avg_hold_minutes": float(durations.mean()),
        "directional_accuracy": float(correct_direction / direction_predictions)
            if direction_predictions > 0 else np.nan,
        "long_predictions": int(long_predictions),
        "short_predictions": int(short_predictions),
        "skipped_beyond_validation": int(skipped_beyond_validation),
    }


# ============================================================
# CONFIDENCE FILTER
# ============================================================

def confidence_filter(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Return SHORT=0, TIMEOUT=1, LONG=2 predictions after filtering."""
    pred = np.argmax(proba, axis=1).astype(np.int8)
    confidence = np.max(proba, axis=1)
    directional = pred != 1
    pred[(directional) & (confidence < threshold)] = 1
    return pred


def evaluate_thresholds(
    val,
    proba,
    canonical_ts,
    canonical_close,
    label_col,
    exit_col,
    val_end_ms,
    context,
):
    y_val = val[label_col].to_numpy(np.int8)
    target_info = target_index_global.reindex(val["timestamp"].to_numpy())
    entry_close = target_info["close"].to_numpy(np.float64)
    exit_index = val[exit_col].to_numpy(np.int64)

    base_pred = np.argmax(proba, axis=1).astype(np.int8)
    base_sim = run_barrier_simulation(
        val["timestamp"].to_numpy(np.int64),
        entry_close,
        y_val,
        exit_index,
        base_pred,
        canonical_ts,
        canonical_close,
        COST_BPS_PRIMARY,
        val_end_ms,
    )

    baseline_trade_count = base_sim["trades"]
    min_trades = max(1, int(np.ceil(baseline_trade_count * MIN_TRADE_FRACTION)))

    candidates = []
    for threshold in CONFIDENCE_THRESHOLDS:
        pred = confidence_filter(proba, threshold)
        sim = run_barrier_simulation(
            val["timestamp"].to_numpy(np.int64),
            entry_close,
            y_val,
            exit_index,
            pred,
            canonical_ts,
            canonical_close,
            COST_BPS_PRIMARY,
            val_end_ms,
        )
        row = {
            **context,
            "threshold": threshold,
            "inner_trades": sim["trades"],
            "inner_compound_2bps": sim["compound_return"],
            "inner_profit_factor_2bps": sim["profit_factor"],
            "inner_win_rate_2bps": sim["win_rate"],
            "inner_mdd_2bps": sim["max_drawdown"],
            "inner_avg_hold_minutes_2bps": sim["avg_hold_minutes"],
            "inner_directional_accuracy_2bps": sim["directional_accuracy"],
            "min_required_trades": min_trades,
            "eligible": sim["trades"] >= min_trades,
        }
        candidates.append(row)

    eligible = [r for r in candidates if r["eligible"]]
    if not eligible:
        # Safe fallback if all thresholds prune too aggressively.
        selected = min(
            candidates,
            key=lambda r: abs(r["threshold"] - 0.50),
        )
        selected["selection_reason"] = "fallback_no_threshold_met_min_trades"
    else:
        selected = max(
            eligible,
            key=lambda r: (
                r["inner_compound_2bps"],
                r["inner_profit_factor_2bps"] if np.isfinite(r["inner_profit_factor_2bps"]) else -np.inf,
                r["inner_trades"],
            ),
        )
        selected["selection_reason"] = "max_inner_2bps_compound_return_with_min_trades"

    return base_sim, candidates, selected


# Global target index used by evaluate_thresholds; populated in main.
target_index_global = None


# ============================================================
# MAIN
# ============================================================

def main():
    global target_index_global

    for path in [FEATURE_FILE, TARGET_FILE, CANONICAL_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("=" * 78)
    print("XAUUSD TRIPLE-BARRIER LIGHTGBM NESTED CONFIDENCE FILTER")
    print("=" * 78)
    print(f"Features:              {FEATURE_FILE}")
    print(f"Targets:               {TARGET_FILE}")
    print(f"Canonical:             {CANONICAL_FILE}")
    print("Barrier:               +/- 5 bps")
    print(f"Primary cost:          {COST_BPS_PRIMARY:g} bps")
    print(f"Confidence grid:       {CONFIDENCE_THRESHOLDS}")
    print(f"Min trade fraction:    {MIN_TRADE_FRACTION:.2%} of unfiltered trades")
    print("2026 TEST:             NOT USED")
    print()

    target_index_global = load_targets()
    canonical_ts, canonical_close = load_canonical_arrays()

    folds = [
        (1, 2022, 2021),
        (2, 2023, 2022),
        (3, 2024, 2023),
        (4, 2025, 2024),
    ]

    summary_rows = []
    threshold_rows = []

    for fold_id, outer_year, inner_year in folds:
        print("=" * 78)
        print(f"OUTER FOLD {fold_id} | INNER {inner_year} -> OUTER TEST {outer_year}")
        print("=" * 78)

        for horizon in HORIZONS:
            label_col = f"tb_label_{horizon}m"
            exit_col = f"tb_exit_index_{horizon}m"

            # -------------------------
            # 1) Inner model
            # -------------------------
            print("-" * 78)
            print(f"Horizon {horizon}m | INNER selection on {inner_year}")

            inner_train, inner_val, inner_chunks = load_split(
                target_index_global,
                label_col,
                exit_col,
                year_start_ms(2019),
                train_end_ms(inner_year),
                validation_start_ms(inner_year),
                year_start_ms(inner_year + 1),
            )

            inner_model = LGBMClassifier(**PARAMS)
            inner_model.fit(
                inner_train[FEATURES],
                inner_train[label_col].to_numpy(np.int8),
                eval_set=[(inner_val[FEATURES], inner_val[label_col].to_numpy(np.int8))],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            inner_best = inner_model.best_iteration_ or PARAMS["n_estimators"]
            inner_proba = inner_model.predict_proba(inner_val[FEATURES], num_iteration=inner_best)

            context = {
                "outer_fold": fold_id,
                "inner_year": inner_year,
                "outer_year": outer_year,
                "horizon_min": horizon,
            }
            _, inner_candidates, selected = evaluate_thresholds(
                inner_val,
                inner_proba,
                canonical_ts,
                canonical_close,
                label_col,
                exit_col,
                year_start_ms(inner_year + 1),
                context,
            )

            threshold_rows.extend(inner_candidates)

            selected_threshold = float(selected["threshold"])
            print(f"Inner best iteration: {inner_best}")
            print(f"Selected confidence threshold: {selected_threshold:.2f}")
            print(f"Inner trades at 2bps: {selected['inner_trades']:,}")
            print(f"Inner compound at 2bps: {selected['inner_compound_2bps']:.4%}")
            print(f"Inner PF at 2bps: {selected['inner_profit_factor_2bps']:.4f}")
            print(f"Selection: {selected['selection_reason']}")

            del inner_model, inner_train, inner_val, inner_proba
            gc.collect()

            # -------------------------
            # 2) Outer retrain + test
            # -------------------------
            print(f"Horizon {horizon}m | OUTER test on {outer_year}")

            outer_train, outer_val, outer_chunks = load_split(
                target_index_global,
                label_col,
                exit_col,
                year_start_ms(2019),
                train_end_ms(outer_year),
                validation_start_ms(outer_year),
                year_start_ms(outer_year + 1),
            )

            y_outer_train = outer_train[label_col].to_numpy(np.int8)
            y_outer = outer_val[label_col].to_numpy(np.int8)

            outer_model = LGBMClassifier(**PARAMS)
            outer_model.fit(
                outer_train[FEATURES],
                y_outer_train,
                eval_set=[(outer_val[FEATURES], y_outer)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            outer_best = outer_model.best_iteration_ or PARAMS["n_estimators"]
            outer_proba = outer_model.predict_proba(outer_val[FEATURES], num_iteration=outer_best)

            outer_argmax = np.argmax(outer_proba, axis=1).astype(np.int8)
            outer_filtered = confidence_filter(outer_proba, selected_threshold)

            outer_acc = accuracy_score(y_outer, outer_argmax)
            outer_bal = balanced_accuracy_score(y_outer, outer_argmax)
            outer_f1 = f1_score(y_outer, outer_argmax, average="macro", zero_division=0)
            outer_trade_coverage = float(np.mean(outer_filtered != 1))

            target_info = target_index_global.reindex(outer_val["timestamp"].to_numpy())
            entry_close = target_info["close"].to_numpy(np.float64)
            exit_index = outer_val[exit_col].to_numpy(np.int64)

            sims = {}
            for cost_bps in COST_SENSITIVITY_BPS:
                sims[cost_bps] = run_barrier_simulation(
                    outer_val["timestamp"].to_numpy(np.int64),
                    entry_close,
                    y_outer,
                    exit_index,
                    outer_filtered,
                    canonical_ts,
                    canonical_close,
                    cost_bps,
                    year_start_ms(outer_year + 1),
                )
                print(
                    f"{cost_bps:g}bps | trades={sims[cost_bps]['trades']:,} | "
                    f"compound={sims[cost_bps]['compound_return']:.4%} | "
                    f"PF={sims[cost_bps]['profit_factor']:.4f} | "
                    f"win={sims[cost_bps]['win_rate']:.4%} | "
                    f"MDD={sims[cost_bps]['max_drawdown']:.4%}"
                )

            selected_outer_2 = sims[COST_BPS_PRIMARY]
            base_sim = run_barrier_simulation(
                outer_val["timestamp"].to_numpy(np.int64),
                entry_close,
                y_outer,
                exit_index,
                outer_argmax,
                canonical_ts,
                canonical_close,
                COST_BPS_PRIMARY,
                year_start_ms(outer_year + 1),
            )

            print(f"Outer best iteration: {outer_best}")
            print(f"Outer accuracy (unfiltered): {outer_acc:.4%}")
            print(f"Outer balanced accuracy: {outer_bal:.4%}")
            print(f"Outer macro F1: {outer_f1:.4f}")
            print(f"Outer filtered coverage: {outer_trade_coverage:.4%}")
            print(
                f"2bps | FILTERED trades={selected_outer_2['trades']:,} | "
                f"compound={selected_outer_2['compound_return']:.4%} | "
                f"PF={selected_outer_2['profit_factor']:.4f}"
            )
            print(
                f"2bps | UNFILTERED trades={base_sim['trades']:,} | "
                f"compound={base_sim['compound_return']:.4%} | "
                f"PF={base_sim['profit_factor']:.4f}"
            )

            summary_rows.append({
                "outer_fold": fold_id,
                "inner_year": inner_year,
                "outer_year": outer_year,
                "horizon_min": horizon,
                "selected_threshold": selected_threshold,
                "inner_best_iteration": inner_best,
                "outer_best_iteration": outer_best,
                "inner_trades_at_2bps": selected["inner_trades"],
                "inner_compound_at_2bps": selected["inner_compound_2bps"],
                "inner_pf_at_2bps": selected["inner_profit_factor_2bps"],
                "inner_mdd_at_2bps": selected["inner_mdd_2bps"],
                "outer_accuracy_unfiltered": float(outer_acc),
                "outer_balanced_accuracy_unfiltered": float(outer_bal),
                "outer_macro_f1_unfiltered": float(outer_f1),
                "outer_trade_coverage_filtered": outer_trade_coverage,
                "outer_trades_filtered_0bps": sims[0.0]["trades"],
                "outer_compound_filtered_0bps": sims[0.0]["compound_return"],
                "outer_pf_filtered_0bps": sims[0.0]["profit_factor"],
                "outer_mdd_filtered_0bps": sims[0.0]["max_drawdown"],
                "outer_trades_filtered_1bps": sims[1.0]["trades"],
                "outer_compound_filtered_1bps": sims[1.0]["compound_return"],
                "outer_pf_filtered_1bps": sims[1.0]["profit_factor"],
                "outer_mdd_filtered_1bps": sims[1.0]["max_drawdown"],
                "outer_trades_filtered_2bps": sims[2.0]["trades"],
                "outer_compound_filtered_2bps": sims[2.0]["compound_return"],
                "outer_pf_filtered_2bps": sims[2.0]["profit_factor"],
                "outer_win_filtered_2bps": sims[2.0]["win_rate"],
                "outer_mdd_filtered_2bps": sims[2.0]["max_drawdown"],
                "outer_avg_hold_filtered_2bps": sims[2.0]["avg_hold_minutes"],
                "outer_directional_accuracy_filtered_2bps": sims[2.0]["directional_accuracy"],
                "outer_trades_unfiltered_2bps": base_sim["trades"],
                "outer_compound_unfiltered_2bps": base_sim["compound_return"],
                "outer_pf_unfiltered_2bps": base_sim["profit_factor"],
                "outer_mdd_unfiltered_2bps": base_sim["max_drawdown"],
                "outer_trades_filtered_5bps": sims[5.0]["trades"],
                "outer_compound_filtered_5bps": sims[5.0]["compound_return"],
                "outer_pf_filtered_5bps": sims[5.0]["profit_factor"],
                "outer_mdd_filtered_5bps": sims[5.0]["max_drawdown"],
                "skipped_beyond_validation_2bps": sims[2.0]["skipped_beyond_validation"],
            })

            del outer_model, outer_train, outer_val, outer_proba
            gc.collect()

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_FILE, index=False)
    pd.DataFrame(threshold_rows).to_csv(THRESHOLD_FILE, index=False)

    print()
    print("=" * 78)
    print("NESTED CONFIDENCE FILTER COMPLETE")
    print("=" * 78)
    print(f"Saved summary:   {SUMMARY_FILE}")
    print(f"Saved thresholds:{THRESHOLD_FILE}")
    print("2026 remains untouched.")


if __name__ == "__main__":
    main()
