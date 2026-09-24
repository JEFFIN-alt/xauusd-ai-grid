#!/usr/bin/env python3
"""
XAUUSD OOS economic-value test for the small-feature directional model.

This is NOT a final MT5 backtest. It is a walk-forward economic sanity check.

Model:
    - Direct UP vs DOWN/FLAT target
    - Existing 42 V4 features
    - TRAIN-ONLY univariate feature selection
    - Logistic regression
    - Top K = 3, 5, 10

Horizons:
    15m, 30m, 60m

Walk-forward:
    2022, 2023, 2024, 2025
    2019..previous year for training
    60-minute purge before each validation year
    2026 completely untouched

Trading rules:
    p >= 0.55 -> LONG
    p <= 0.45 -> SHORT
    otherwise -> NO TRADE

Additional fixed threshold pairs:
    0.50 / 0.50 (always take predicted side)
    0.55 / 0.45
    0.60 / 0.40

Only one trade can be open at a time. After entering at t and exiting at
t+h, all candidate entries before that exit are skipped.

Return:
    Long:  exit/entry - 1
    Short: entry/exit - 1

Synthetic round-trip costs:
    0, 1, 2, 5 bps
    These are deductions, not measured XAUUSD bid/ask costs. The source
    dataset is bid-only, so this cannot reproduce the true MT5 execution cost.

Important:
    - Thresholds are fixed before seeing validation results.
    - No parameter is selected using 2026.
    - This experiment is exploratory; a later final strategy should use
      nested inner validation for threshold/model selection.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

CANONICAL = ROOT / "data/processed/xauusd_m1_2019_2026_canonical.csv"
V4 = ROOT / "data/processed/xauusd_m1_ml_features_v4_session.csv"
REPORT = ROOT / "reports/xauusd_small_feature_oos_economic_value.csv"

HORIZONS = [15, 30, 60]
YEARS = [2022, 2023, 2024, 2025]
TOP_KS = [3, 5, 10]
THRESHOLD_PAIRS = [(0.50, 0.50), (0.55, 0.45), (0.60, 0.40)]
COST_BPS = [0, 1, 2, 5]

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
    "distance_high_60", "distance_low_60",
    "distance_high_240", "distance_low_240",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
]


def load_canonical():
    df = pd.read_csv(
        CANONICAL,
        usecols=["timestamp", "close"],
        low_memory=False,
    )

    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(np.int64)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)

    if not np.all(ts[1:] > ts[:-1]):
        raise RuntimeError("Canonical timestamps are not strictly increasing.")

    # Established session-closed proxy.
    flat_prev = close[1:] == close[:-1]
    starts = np.empty(len(close), dtype=np.int64)
    starts[0] = 0
    starts[1:] = np.where(flat_prev, 0, 1)

    run_id = np.cumsum(starts)
    lengths = np.bincount(run_id)
    closed = lengths[run_id] >= 60
    active = ~closed

    print(f"Canonical rows: {len(ts):,}")
    print(f"Closed rows: {int(closed.sum()):,}")
    print(f"Active rows: {int(active.sum()):,}")

    return ts, close, active


def build_target_and_exit(ts, close, active, horizon):
    """
    Return:
      target: -1 invalid, 0 DOWN/FLAT, 1 UP
      exit_idx: canonical positional index for valid target
    """
    n = len(ts)
    target = np.full(n, -1, dtype=np.int8)
    exit_idx = np.full(n, -1, dtype=np.int64)

    pos = np.arange(n - horizon, dtype=np.int64)
    future = pos + horizon

    exact = (ts[future] - ts[pos]) == horizon * 60_000
    valid = active[pos] & active[future] & exact

    vp = pos[valid]
    vf = future[valid]

    target[vp] = (close[vf] > close[vp]).astype(np.int8)
    exit_idx[vp] = vf

    print(
        f"{horizon}m target | valid={int((target>=0).sum()):,} | "
        f"UP={int((target==1).sum()):,} | DOWN={int((target==0).sum()):,}"
    )

    return target, exit_idx


def collect_samples(canonical_ts, target, exit_idx):
    cols = ["timestamp", *FEATURES]
    header = list(pd.read_csv(V4, nrows=0).columns)
    missing = [c for c in cols if c not in header]
    if missing:
        raise RuntimeError(f"V4 missing columns: {missing}")

    xs, ys, sample_ts, sample_exit = [], [], [], []
    global_row = 0

    for chunk in pd.read_csv(
        V4,
        usecols=cols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk_ts = pd.to_numeric(
            chunk["timestamp"], errors="coerce"
        ).to_numpy(np.int64)

        years = pd.to_datetime(
            chunk_ts, unit="ms", utc=True
        ).year.to_numpy()

        period = (years >= 2019) & (years <= 2025)
        n_period = int(period.sum())

        if n_period == 0:
            continue

        chunk = chunk.loc[period].copy()

        local = np.arange(len(chunk), dtype=np.int64)
        sampled = ((global_row + local) % STRIDE) == 0
        chunk = chunk.loc[sampled].copy()

        if chunk.empty:
            global_row += n_period
            continue

        ts_part = pd.to_numeric(
            chunk["timestamp"], errors="coerce"
        ).to_numpy(np.int64)

        idx = np.searchsorted(
            canonical_ts,
            ts_part,
            side="left",
        )

        in_bounds = idx < len(canonical_ts)
        exact = np.zeros(len(idx), dtype=bool)
        loc = np.flatnonzero(in_bounds)
        exact[loc] = canonical_ts[idx[loc]] == ts_part[loc]

        keep = np.zeros(len(idx), dtype=bool)
        loc = np.flatnonzero(exact)
        keep[loc] = target[idx[loc]] >= 0

        if keep.any():
            sub = chunk.loc[keep]
            cidx = idx[keep]

            X = sub[FEATURES].to_numpy(np.float32, copy=True)
            y = target[cidx].astype(np.int8)
            ex = exit_idx[cidx].astype(np.int64)

            finite = np.isfinite(X).all(axis=1)

            if finite.any():
                xs.append(X[finite])
                ys.append(y[finite])
                sample_ts.append(ts_part[keep][finite])
                sample_exit.append(ex[finite])

        global_row += n_period
        del chunk
        gc.collect()

    X = np.concatenate(xs)
    y = np.concatenate(ys)
    out_ts = np.concatenate(sample_ts)
    out_exit = np.concatenate(sample_exit)

    order = np.argsort(out_ts, kind="mergesort")
    X = X[order]
    y = y[order]
    out_ts = out_ts[order]
    out_exit = out_exit[order]

    unique = np.r_[True, out_ts[1:] != out_ts[:-1]]
    X = X[unique]
    y = y[unique]
    out_ts = out_ts[unique]
    out_exit = out_exit[unique]

    print(f"ML rows: {len(y):,}")
    print(f"X shape: {X.shape}")

    return X, y, out_ts, out_exit


def rank_features(X_train, y_train):
    ranked = []

    for j, feature in enumerate(FEATURES):
        values = X_train[:, j]
        finite = np.isfinite(values)

        if finite.sum() < 100:
            continue
        if np.unique(y_train[finite]).size < 2:
            continue

        auc = roc_auc_score(y_train[finite], values[finite])
        advantage = abs(auc - 0.5)

        ranked.append(
            {
                "index": j,
                "feature": feature,
                "auc": float(auc),
                "advantage": float(advantage),
                "sign": 1.0 if auc >= 0.5 else -1.0,
            }
        )

    ranked.sort(key=lambda r: r["advantage"], reverse=True)
    return ranked


def fit_predict(X_train, y_train, X_val, y_val, ranked, k):
    selected = ranked[:k]
    idx = [r["index"] for r in selected]
    signs = np.array([r["sign"] for r in selected], dtype=np.float64)

    A = X_train[:, idx].astype(np.float64) * signs
    B = X_val[:, idx].astype(np.float64) * signs

    scaler = StandardScaler()
    A = scaler.fit_transform(A)
    B = scaler.transform(B)

    model = LogisticRegression(
        C=0.1,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(A, y_train)

    p = model.predict_proba(B)[:, 1]
    auc = roc_auc_score(y_val, p)

    return p, auc, selected


def simulate(
    val_positions,
    prob,
    canonical_ts,
    close,
    exit_idx_by_sample,
    long_threshold,
    short_threshold,
    cost_bps,
):
    """
    Non-overlapping fixed-horizon simulation.
    Signals at entries before the active trade exits are skipped.
    """
    cost = cost_bps / 10_000.0
    rows = []

    next_allowed_ts_index = -1

    for i in range(len(prob)):
        entry_idx = int(val_positions[i])

        if entry_idx < next_allowed_ts_index:
            continue

        p = float(prob[i])

        if p >= long_threshold:
            side = 1
        elif p <= short_threshold:
            side = -1
        else:
            continue

        exit_idx = int(exit_idx_by_sample[i])

        if exit_idx <= entry_idx:
            continue

        entry = float(close[entry_idx])
        exit_price = float(close[exit_idx])

        if side == 1:
            gross = exit_price / entry - 1.0
        else:
            gross = entry / exit_price - 1.0

        net = gross - cost

        rows.append(
            {
                "entry_idx": entry_idx,
                "entry_ts": canonical_ts[entry_idx],
                "exit_idx": exit_idx,
                "exit_ts": canonical_ts[exit_idx],
                "side": side,
                "prob_long": p,
                "gross_return": gross,
                "net_return": net,
            }
        )

        next_allowed_ts_index = exit_idx

    if not rows:
        return {
            "trades": 0,
            "compound": 0.0,
            "pf": np.nan,
            "win_rate": np.nan,
            "mdd": 0.0,
        }

    r = np.array([x["net_return"] for x in rows], dtype=np.float64)

    equity = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0

    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    return {
        "trades": len(rows),
        "compound": float(equity[-1] - 1.0),
        "pf": float(pf),
        "win_rate": float((r > 0).mean()),
        "mdd": float(dd.min()),
    }


def main():
    print("=" * 78)
    print("XAUUSD SMALL-FEATURE OOS ECONOMIC-VALUE TEST")
    print("=" * 78)
    print("Model: train-only feature selection + logistic regression")
    print("Top K:", TOP_KS)
    print("Thresholds:", THRESHOLD_PAIRS)
    print("Costs:", COST_BPS, "bps")
    print("Non-overlapping fixed-horizon trades")
    print("2026: COMPLETELY UNTOUCHED")
    print("=" * 78)

    canonical_ts, close, active = load_canonical()
    all_results = []

    for horizon in HORIZONS:
        print("=" * 78)
        print(f"HORIZON {horizon} MINUTES")
        print("=" * 78)

        target, exit_idx = build_target_and_exit(
            canonical_ts,
            close,
            active,
            horizon,
        )

        X, y, sample_ts, sample_exit = collect_samples(
            canonical_ts,
            target,
            exit_idx,
        )

        # Canonical entry positions for simulation.
        sample_positions = np.searchsorted(
            canonical_ts,
            sample_ts,
            side="left",
        )

        years = pd.to_datetime(
            sample_ts,
            unit="ms",
            utc=True,
        ).year.to_numpy()

        for val_year in YEARS:
            val_mask = years == val_year

            val_start = pd.Timestamp(
                f"{val_year}-01-01",
                tz="UTC",
            )
            cutoff_ms = int(
                (
                    val_start
                    - pd.Timedelta(minutes=PURGE_MINUTES)
                ).timestamp() * 1000
            )

            train_mask = (
                (years >= 2019)
                & (years < val_year)
                & (sample_ts < cutoff_ms)
            )

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]

            pos_val = sample_positions[val_mask]
            exit_val = sample_exit[val_mask]

            print("-" * 78)
            print(
                f"{horizon}m | Fold {val_year} | "
                f"train={len(y_train):,} | val={len(y_val):,}"
            )

            ranked = rank_features(
                X_train,
                y_train,
            )

            for k in TOP_KS:
                prob, auc, selected = fit_predict(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    ranked,
                    k,
                )

                print(
                    f"K={k} | OOS ROC-AUC={auc:.6f} | "
                    f"features={','.join(r['feature'] for r in selected)}"
                )

                for long_t, short_t in THRESHOLD_PAIRS:
                    for cost_bps in COST_BPS:
                        sim = simulate(
                            pos_val,
                            prob,
                            canonical_ts,
                            close,
                            exit_val,
                            long_t,
                            short_t,
                            cost_bps,
                        )

                        print(
                            f"  K={k} "
                            f"thr={long_t:.2f}/{short_t:.2f} "
                            f"cost={cost_bps}bps | "
                            f"trades={sim['trades']} | "
                            f"compound={sim['compound']*100:.3f}% | "
                            f"PF={sim['pf']:.3f} | "
                            f"win={sim['win_rate']*100 if np.isfinite(sim['win_rate']) else np.nan:.2f}% | "
                            f"MDD={sim['mdd']*100:.3f}%"
                        )

                        all_results.append(
                            {
                                "horizon_min": horizon,
                                "validation_year": val_year,
                                "top_k": k,
                                "long_threshold": long_t,
                                "short_threshold": short_t,
                                "cost_bps": cost_bps,
                                "auc": float(auc),
                                "trades": sim["trades"],
                                "compound": sim["compound"],
                                "profit_factor": sim["pf"],
                                "win_rate": sim["win_rate"],
                                "max_drawdown": sim["mdd"],
                                "selected_features": "|".join(
                                    r["feature"] for r in selected
                                ),
                            }
                        )

        del target, exit_idx, X, y, sample_ts, sample_exit
        gc.collect()

    out = pd.DataFrame(all_results)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT, index=False)

    # Summary over outer OOS years. We do NOT select a final configuration
    # here; this is a diagnostic sensitivity report.
    summary = (
        out.groupby(
            [
                "horizon_min",
                "top_k",
                "long_threshold",
                "short_threshold",
                "cost_bps",
            ],
            as_index=False,
        )
        .agg(
            mean_compound=("compound", "mean"),
            mean_pf=("profit_factor", "mean"),
            mean_trades=("trades", "mean"),
            mean_win_rate=("win_rate", "mean"),
            mean_mdd=("max_drawdown", "mean"),
            mean_auc=("auc", "mean"),
        )
    )

    print("=" * 78)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 78)
    print(summary.sort_values(
        ["mean_compound"],
        ascending=False,
    ).head(30).to_string(index=False))

    print("=" * 78)
    print(f"Saved: {REPORT}")
    print("2026 was not used.")
    print("No final configuration was selected from these OOS results.")
    print("=" * 78)


if __name__ == "__main__":
    main()
