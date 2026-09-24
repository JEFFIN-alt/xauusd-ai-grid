#!/usr/bin/env python3
"""
Triple-barrier barrier-size ML comparison.

Compares:
  5 bps / 60m   (existing control)
 10 bps / 60m
 15 bps / 60m
 10 bps / 30m

Method:
- Same 42 V3 features
- LightGBM 3-class classifier: SHORT / TIMEOUT / LONG
- Same 5-row stride used by the prior V3 baseline family
- Walk-forward folds:
    2019-2021 -> 2022
    2019-2022 -> 2023
    2019-2023 -> 2024
    2019-2024 -> 2025
- 2026 is completely excluded from model fitting/selection/evaluation.
- No confidence threshold optimization here; argmax prediction is used.
- Trading simulator uses the actual triple-barrier exit:
    matching barrier -> +barrier
    opposite barrier -> -barrier
    timeout -> exact close-to-close return at tb_exit_index
- Validation trades whose actual exit occurs after the validation calendar year
  are skipped, matching the prior simulator convention.

Memory design:
- Target index is loaded as compact NumPy arrays.
- Features are streamed in chunks and sampled at stride 5.
- Only valid labels (0/1/2) from 2019-2025 are retained.
"""

from pathlib import Path
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


BASE = Path("data/processed")
REPORT = Path("reports")

FEATURE_FILE = BASE / "xauusd_m1_ml_features_v3.csv"
CANONICAL_FILE = BASE / "xauusd_m1_2019_2026_canonical.csv"

CONFIGS = [
    {
        "barrier_bps": 5,
        "horizon": 60,
        "target_file": BASE / "xauusd_m1_triple_barrier_targets.csv",
        "tag": "5bps_60m_control",
    },
    {
        "barrier_bps": 10,
        "horizon": 60,
        "target_file": BASE / "xauusd_m1_triple_barrier_10bps.csv",
        "tag": "10bps_60m",
    },
    {
        "barrier_bps": 15,
        "horizon": 60,
        "target_file": BASE / "xauusd_m1_triple_barrier_15bps.csv",
        "tag": "15bps_60m",
    },
    {
        "barrier_bps": 10,
        "horizon": 30,
        "target_file": BASE / "xauusd_m1_triple_barrier_10bps.csv",
        "tag": "10bps_30m",
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

CHUNK_SIZE = 250_000
STRIDE = 5
YEARS = [2022, 2023, 2024, 2025]
TRAIN_START = 2019
TRAIN_END_BEFORE_VALIDATION = 2021

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
    s = pd.to_numeric(values, errors="coerce")
    med = float(s.dropna().abs().median())
    if med >= 1e17:
        return "ns"
    if med >= 1e14:
        return "us"
    if med >= 1e11:
        return "ms"
    return "s"


def load_target_index(path, horizon):
    """Load compact target timestamp, label, and exit-index arrays."""
    print(f"Loading target index: {path}")
    head = pd.read_csv(path, nrows=2)

    label_col = f"tb_label_{horizon}m"
    exit_candidates = [
        f"tb_exit_index_{horizon}m",
        f"tb_exit_{horizon}m",
        f"exit_index_{horizon}m",
    ]
    exit_col = next((c for c in exit_candidates if c in head.columns), None)

    if label_col not in head.columns:
        raise KeyError(f"Missing {label_col} in {path}")
    if exit_col is None:
        raise KeyError(
            f"Could not find exit-index column for {horizon}m in {path}. "
            f"Columns include: {list(head.columns)}"
        )

    usecols = ["timestamp", label_col, exit_col]
    parts = []
    unit = None
    total = 0

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE):
        total += len(chunk)
        raw_ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
        if raw_ts.isna().any():
            raise ValueError(f"Invalid numeric timestamp in target file {path}")

        if unit is None:
            unit = detect_epoch_unit(raw_ts)

        labels = pd.to_numeric(chunk[label_col], errors="coerce").fillna(-99).astype(np.int8)
        exits = pd.to_numeric(chunk[exit_col], errors="coerce").fillna(-1).astype(np.int64)

        parts.append(
            np.column_stack(
                [
                    raw_ts.to_numpy(dtype=np.int64, copy=False),
                    labels.to_numpy(dtype=np.int8, copy=False),
                    exits.to_numpy(dtype=np.int64, copy=False),
                ]
            )
        )

    arr = np.concatenate(parts, axis=0)
    del parts
    gc.collect()

    ts = arr[:, 0].astype(np.int64, copy=False)
    labels = arr[:, 1].astype(np.int8, copy=False)
    exits = arr[:, 2].astype(np.int64, copy=False)

    if not np.all(ts[1:] >= ts[:-1]):
        raise ValueError(f"Target timestamps are not sorted: {path}")

    print(f"Target rows: {len(ts):,} | timestamp unit: epoch-{unit}")
    print(
        f"Target valid labels: {(np.isin(labels, [0,1,2])).sum():,} | "
        f"ambiguous: {(labels == -1).sum():,} | invalid: {(labels == -2).sum():,}"
    )
    return ts, labels, exits, unit


def load_canonical_close():
    """Load canonical timestamp and close arrays once for exact exits."""
    print("Loading canonical timestamp/close arrays...")
    head = pd.read_csv(CANONICAL_FILE, nrows=2)
    usecols = ["timestamp", "close"]

    parts_ts = []
    parts_close = []
    unit = None

    for chunk in pd.read_csv(CANONICAL_FILE, usecols=usecols, chunksize=CHUNK_SIZE):
        raw_ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
        if raw_ts.isna().any():
            raise ValueError("Invalid timestamp in canonical file")
        if unit is None:
            unit = detect_epoch_unit(raw_ts)

        parts_ts.append(raw_ts.to_numpy(dtype=np.int64, copy=False))
        parts_close.append(
            pd.to_numeric(chunk["close"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        )

    ts = np.concatenate(parts_ts)
    close = np.concatenate(parts_close)

    if not np.all(ts[1:] > ts[:-1]):
        raise ValueError("Canonical timestamps are not strictly increasing")

    print(f"Canonical rows: {len(ts):,} | timestamp unit: epoch-{unit}")
    return ts, close


def load_sampled_dataset(target_ts, target_labels, target_exits, unit):
    """
    Stream V3 features once, align by timestamp using searchsorted,
    and retain every STRIDE-th valid row from 2019-2025.

    Returns:
      X float32
      timestamp int64 (same epoch unit)
      close float64
      label int8
      exit_index int64
      year int16
    """
    print("Streaming V3 features and aligning target labels...")
    head = pd.read_csv(FEATURE_FILE, nrows=2)

    required = ["timestamp", "close"] + FEATURES
    missing = [c for c in required if c not in head.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    collected = []
    total_seen = 0
    total_kept = 0

    usecols = required

    for chunk_no, chunk in enumerate(
        pd.read_csv(FEATURE_FILE, usecols=usecols, chunksize=CHUNK_SIZE)
    ):
        raw_ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
        if raw_ts.isna().any():
            raise ValueError("Invalid timestamp in feature file")

        ts = raw_ts.to_numpy(dtype=np.int64, copy=False)
        close = pd.to_numeric(chunk["close"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

        idx = np.searchsorted(target_ts, ts)

        valid_match = (idx < len(target_ts))
        valid_match[valid_match] &= target_ts[idx[valid_match]] == ts[valid_match]

        # The V3 dataset should align exactly with target timestamps.
        if not np.all(valid_match):
            missing_count = int((~valid_match).sum())
            raise ValueError(
                f"{missing_count} feature timestamps did not match target index "
                f"in chunk {chunk_no}."
            )

        labels = target_labels[idx]
        exits = target_exits[idx]

        # Keep only model-valid labels and years 2019-2025.
        dt = pd.to_datetime(ts, unit=unit, utc=True)
        years = dt.year.to_numpy(dtype=np.int16)

        mask = (
            np.isin(labels, [0, 1, 2])
            & (years >= 2019)
            & (years <= 2025)
        )

        positions = np.flatnonzero(mask)
        # Deterministic 5-row stride within the already sorted feature dataset.
        if positions.size:
            positions = positions[(positions % STRIDE) == 0]

        if positions.size:
            feats = chunk.iloc[positions][FEATURES].to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(feats).all():
                raise ValueError(f"Non-finite feature encountered in chunk {chunk_no}")

            collected.append(
                (
                    feats,
                    ts[positions].copy(),
                    close[positions].copy(),
                    labels[positions].copy(),
                    exits[positions].copy(),
                    years[positions].copy(),
                )
            )
            total_kept += len(positions)

        total_seen += len(chunk)
        if (chunk_no + 1) % 5 == 0:
            print(
                f"Chunks: {chunk_no+1} | feature rows: {total_seen:,} | "
                f"sampled kept: {total_kept:,}"
            )

    X = np.concatenate([x[0] for x in collected], axis=0)
    timestamps = np.concatenate([x[1] for x in collected], axis=0)
    close = np.concatenate([x[2] for x in collected], axis=0)
    labels = np.concatenate([x[3] for x in collected], axis=0)
    exits = np.concatenate([x[4] for x in collected], axis=0)
    years = np.concatenate([x[5] for x in collected], axis=0)

    del collected
    gc.collect()

    print(f"Sampled ML rows: {len(X):,} | features: {X.shape[1]}")
    return X, timestamps, close, labels, exits, years


def simulate_trades(
    pred,
    sample_ts,
    sample_close,
    actual_label,
    exit_index,
    canonical_ts,
    canonical_close,
    target_ts,
    barrier_return,
    validation_year,
    cost_bps,
):
    """
    Non-overlapping, actual triple-barrier simulator.

    Predicted class mapping:
      0 SHORT, 1 TIMEOUT, 2 LONG

    A TIMEOUT prediction is NO TRADE.

    For a trade:
      actual label == predicted side:
          +barrier_return
      actual label == opposite side:
          -barrier_return
      actual label == TIMEOUT:
          exact close-to-close return to tb_exit_index

    Trades exiting after the validation calendar year are skipped.
    """
    year_end_ts = int(
        pd.Timestamp(f"{validation_year}-12-31 23:59:59", tz="UTC").value
    )

    # Convert year-end nanoseconds to the target's timestamp unit.
    # This simulator supports ms/us/ns/s.
    target_unit = "ms"  # target and features in this project are epoch-ms
    if target_unit == "ms":
        year_end_ts //= 1_000_000
    elif target_unit == "us":
        year_end_ts //= 1_000
    elif target_unit == "s":
        year_end_ts //= 1_000_000_000

    entry_indices = []
    i = 0
    n = len(pred)

    while i < n:
        side = int(pred[i])
        if side == 1:  # TIMEOUT = no trade
            i += 1
            continue

        # Entry belongs to this validation year by construction.
        xi = i
        eidx = int(exit_index[xi])
        if eidx < 0 or eidx >= len(canonical_ts):
            i += 1
            continue

        exit_ts = int(canonical_ts[eidx])
        if exit_ts > year_end_ts:
            i += 1
            continue

        entry_price = float(sample_close[xi])
        if not np.isfinite(entry_price) or entry_price <= 0:
            i += 1
            continue

        actual = int(actual_label[xi])

        if actual == side:
            gross = barrier_return
        elif actual in (0, 2):
            gross = -barrier_return
        elif actual == 1:
            exit_price = float(canonical_close[eidx])
            if not np.isfinite(exit_price) or exit_price <= 0:
                i += 1
                continue
            if side == 2:  # LONG
                gross = exit_price / entry_price - 1.0
            else:  # SHORT
                gross = entry_price / exit_price - 1.0
        else:
            i += 1
            continue

        cost = cost_bps / 10000.0
        net = gross - cost

        entry_indices.append((i, eidx, gross, net, actual, side))

        # Advance to the first sampled feature timestamp strictly after exit.
        next_i = int(np.searchsorted(sample_ts, exit_ts, side="right"))
        i = max(i + 1, next_i)

    if not entry_indices:
        return {
            "trades": 0,
            "compound": 0.0,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "max_drawdown": 0.0,
            "avg_hold_minutes": np.nan,
            "directional_accuracy": np.nan,
        }

    gross_list = np.array([t[2] for t in entry_indices], dtype=np.float64)
    net_list = np.array([t[3] for t in entry_indices], dtype=np.float64)
    actual_list = np.array([t[4] for t in entry_indices], dtype=np.int8)
    side_list = np.array([t[5] for t in entry_indices], dtype=np.int8)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    holds = []

    for (entry_i, exit_i, gross, net, actual, side) in entry_indices:
        equity *= max(0.0, 1.0 + net)
        peak = max(peak, equity)
        dd = equity / peak - 1.0
        max_dd = min(max_dd, dd)

        if net > 0:
            wins += 1
        if net > 0:
            gross_profit += net
        elif net < 0:
            gross_loss += -net

        # Approximate hold time from canonical timestamps.
        delta = int(canonical_ts[exit_i]) - int(sample_ts[entry_i])
        holds.append(delta / 60000.0)  # epoch-ms -> minutes

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
        if gross_profit > 0
        else np.nan
    )

    return {
        "trades": len(entry_indices),
        "compound": equity - 1.0,
        "profit_factor": pf,
        "win_rate": wins / len(entry_indices),
        "max_drawdown": max_dd,
        "avg_hold_minutes": float(np.mean(holds)),
        "directional_accuracy": float(np.mean(actual_list == side_list)),
    }


def fit_and_evaluate_one_config(cfg, X, timestamps, close, labels, exits, years,
                                target_ts, canonical_ts, canonical_close):
    barrier = cfg["barrier_bps"] / 10000.0
    horizon = cfg["horizon"]

    print("\n" + "=" * 78)
    print(f"CONFIG: {cfg['tag']}")
    print(f"Barrier: +/- {cfg['barrier_bps']} bps | Horizon: {horizon}m")
    print("=" * 78)

    results = []

    for val_year in YEARS:
        train_mask = (years >= TRAIN_START) & (years < val_year)
        val_mask = years == val_year

        X_train = X[train_mask]
        y_train = labels[train_mask]
        X_val = X[val_mask]
        y_val = labels[val_mask]

        print("-" * 78)
        print(f"Fold | VALIDATION {val_year}")
        print(f"Train rows: {len(X_train):,} | Validation rows: {len(X_val):,}")

        counts = np.bincount(y_val.astype(np.int64), minlength=3)
        actual_pct = counts / max(1, counts.sum()) * 100.0
        print(f"Actual class %: [{actual_pct[0]:.2f}, {actual_pct[1]:.2f}, {actual_pct[2]:.2f}]")

        model = LGBMClassifier(**MODEL_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )

        pred = model.predict(X_val).astype(np.int8)
        best_iter = int(model.best_iteration_ or MODEL_PARAMS["n_estimators"])

        pred_counts = np.bincount(pred.astype(np.int64), minlength=3)
        pred_pct = pred_counts / max(1, pred_counts.sum()) * 100.0

        acc = accuracy_score(y_val, pred)
        bal = balanced_accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average="macro")

        print(
            f"Predicted class %: [{pred_pct[0]:.2f}, {pred_pct[1]:.2f}, {pred_pct[2]:.2f}]"
        )
        print(f"Best iteration: {best_iter}")
        print(f"Accuracy: {acc*100:.4f}%")
        print(f"Balanced accuracy: {bal*100:.4f}%")
        print(f"Macro F1: {f1:.4f}")

        val_timestamps = timestamps[val_mask]
        val_close = close[val_mask]
        val_labels = y_val
        val_exits = exits[val_mask]

        for cost_bps in COSTS_BPS:
            sim = simulate_trades(
                pred=pred,
                sample_ts=val_timestamps,
                sample_close=val_close,
                actual_label=val_labels,
                exit_index=val_exits,
                canonical_ts=canonical_ts,
                canonical_close=canonical_close,
                target_ts=target_ts,
                barrier_return=barrier,
                validation_year=val_year,
                cost_bps=cost_bps,
            )

            print(
                f"{cost_bps}bps | trades={sim['trades']} | "
                f"compound={sim['compound']*100:.4f}% | "
                f"PF={sim['profit_factor']:.4f} | "
                f"win={sim['win_rate']*100 if np.isfinite(sim['win_rate']) else np.nan:.4f}% | "
                f"MDD={sim['max_drawdown']*100:.4f}%"
            )

            results.append({
                "tag": cfg["tag"],
                "barrier_bps": cfg["barrier_bps"],
                "horizon_min": horizon,
                "validation_year": val_year,
                "best_iteration": best_iter,
                "train_rows": len(X_train),
                "validation_rows": len(X_val),
                "actual_short_pct": actual_pct[0],
                "actual_timeout_pct": actual_pct[1],
                "actual_long_pct": actual_pct[2],
                "pred_short_pct": pred_pct[0],
                "pred_timeout_pct": pred_pct[1],
                "pred_long_pct": pred_pct[2],
                "accuracy": acc,
                "balanced_accuracy": bal,
                "macro_f1": f1,
                "cost_bps": cost_bps,
                "trades": sim["trades"],
                "compound_return": sim["compound"],
                "profit_factor": sim["profit_factor"],
                "win_rate": sim["win_rate"],
                "max_drawdown": sim["max_drawdown"],
                "avg_hold_minutes": sim["avg_hold_minutes"],
                "directional_accuracy": sim["directional_accuracy"],
            })

        del model, X_train, y_train, X_val, y_val, pred
        gc.collect()

    return results


def main():
    print("=" * 78)
    print("TRIPLE-BARRIER BARRIER-SIZE ML WALK-FORWARD COMPARISON")
    print("=" * 78)
    print("Configs: 5/60 control, 10/60, 15/60, 10/30")
    print("Sampling stride: 5")
    print("2026: COMPLETELY UNTOUCHED")
    print("=" * 78)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)
    if not CANONICAL_FILE.exists():
        raise FileNotFoundError(CANONICAL_FILE)

    canonical_ts, canonical_close = load_canonical_close()

    all_results = []

    for cfg in CONFIGS:
        if not cfg["target_file"].exists():
            raise FileNotFoundError(cfg["target_file"])

        target_ts, target_labels, target_exits, target_unit = load_target_index(
            cfg["target_file"], cfg["horizon"]
        )

        X, timestamps, close, labels, exits, years = load_sampled_dataset(
            target_ts, target_labels, target_exits, target_unit
        )

        # Ensure no 2026 rows entered the model dataset.
        assert years.max() <= 2025, f"Unexpected 2026 row in ML dataset for {cfg['tag']}"

        results = fit_and_evaluate_one_config(
            cfg, X, timestamps, close, labels, exits, years,
            target_ts, canonical_ts, canonical_close
        )
        all_results.extend(results)

        del target_ts, target_labels, target_exits
        del X, timestamps, close, labels, exits, years
        gc.collect()

    df = pd.DataFrame(all_results)
    REPORT.mkdir(parents=True, exist_ok=True)

    out = REPORT / "triple_barrier_barrier_ml_comparison.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("BARRIER-SIZE ML COMPARISON COMPLETE")
    print("=" * 78)
    print(f"Saved: {out}")
    print("2026 was not used for model fitting, selection, or reported evaluation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
