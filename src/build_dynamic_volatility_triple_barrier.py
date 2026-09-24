#!/usr/bin/env python3
"""
XAUUSD dynamic volatility-adjusted triple-barrier target generator.

Purpose:
    Replace fixed +/- bps barriers with barriers based on volatility known
    at the ENTRY time only.

First experiment:
    - Horizon: 60 minutes
    - Volatility: standard deviation of 1-minute log returns over the
      previous 60 continuous bars, annualization NOT applied.
    - Barrier distance:
          barrier_return = multiplier * volatility_60m * sqrt(60)
      This converts the 1-minute return volatility estimate into an
      approximate 60-minute move scale.
    - Multipliers: 0.50, 0.75, 1.00, 1.25, 1.50

Important:
    - No future prices are used to compute the barrier.
    - Continuous segments are respected; history never crosses a gap.
    - Same-bar touch of both barriers is labeled AMBIGUOUS (-1).
    - No complete horizon is labeled INVALID (-2).
    - SHORT=0, TIMEOUT=1, LONG=2.
    - 2026 is generated for diagnostics only and must remain untouched
      during later ML selection/testing.

The output stores:
    timestamp, open, high, low, close,
    volatility_60m,
    barrier_return_<multiplier>,
    tb_label_60m,
    tb_exit_index_60m

The exit index is the canonical row index, allowing the later simulator
to use the exact exit close for TIMEOUT trades.
"""

from pathlib import Path
import math
import time
import numpy as np
import pandas as pd
from numba import njit


BASE = Path("data/processed")
INPUT = BASE / "xauusd_m1_2019_2026_canonical.csv"

OUTPUTS = {
    0.50: BASE / "xauusd_m1_dynamic_tb_0p50.csv",
    0.75: BASE / "xauusd_m1_dynamic_tb_0p75.csv",
    1.00: BASE / "xauusd_m1_dynamic_tb_1p00.csv",
    1.25: BASE / "xauusd_m1_dynamic_tb_1p25.csv",
    1.50: BASE / "xauusd_m1_dynamic_tb_1p50.csv",
}

SUMMARY = BASE / "xauusd_m1_dynamic_tb_summary.csv"

HORIZON = 60
VOL_LOOKBACK = 60
SQRT_HORIZON = math.sqrt(HORIZON)

# Label encoding.
SHORT = np.int8(0)
TIMEOUT = np.int8(1)
LONG = np.int8(2)
AMBIGUOUS = np.int8(-1)
INVALID = np.int8(-2)


@njit(cache=True)
def build_dynamic_tb(high, low, close, segment_starts, segment_ends, multiplier):
    n = len(close)
    labels = np.full(n, INVALID, dtype=np.int8)
    exits = np.full(n, -1, dtype=np.int64)
    vol = np.full(n, np.nan, dtype=np.float64)
    barriers = np.full(n, np.nan, dtype=np.float64)

    for seg in range(len(segment_starts)):
        s = int(segment_starts[seg])
        e = int(segment_ends[seg])

        # Need 60 previous returns and 60 future minutes.
        first = s + VOL_LOOKBACK
        last = e - HORIZON - 1

        if last < first:
            continue

        # Rolling volatility of 1-minute log returns, using only history
        # strictly before the entry close.
        returns = np.empty(VOL_LOOKBACK, dtype=np.float64)

        for i in range(first, last + 1):
            # returns[k] = log(close[i-VOL_LOOKBACK+k] /
            #                 close[i-VOL_LOOKBACK+k-1])
            ok = True
            for k in range(VOL_LOOKBACK):
                p0 = close[i - VOL_LOOKBACK + k - 1]
                p1 = close[i - VOL_LOOKBACK + k]
                if p0 <= 0.0 or p1 <= 0.0:
                    ok = False
                    break
                returns[k] = math.log(p1 / p0)

            if not ok or not np.isfinite(returns).all():
                continue

            mean_r = 0.0
            for k in range(VOL_LOOKBACK):
                mean_r += returns[k]
            mean_r /= VOL_LOOKBACK

            var = 0.0
            for k in range(VOL_LOOKBACK):
                d = returns[k] - mean_r
                var += d * d
            # Sample standard deviation.
            var /= (VOL_LOOKBACK - 1)

            sigma_1m = math.sqrt(max(var, 0.0))
            barrier = multiplier * sigma_1m * SQRT_HORIZON

            if not np.isfinite(barrier) or barrier <= 0.0:
                continue

            entry = close[i]
            if not np.isfinite(entry) or entry <= 0.0:
                continue

            upper = entry * (1.0 + barrier)
            lower = entry * (1.0 - barrier)

            vol[i] = sigma_1m
            barriers[i] = barrier

            outcome = TIMEOUT
            exit_i = i + HORIZON

            for j in range(i + 1, i + HORIZON + 1):
                hit_up = high[j] >= upper
                hit_down = low[j] <= lower

                if hit_up and hit_down:
                    outcome = AMBIGUOUS
                    exit_i = j
                    break
                elif hit_down:
                    outcome = SHORT
                    exit_i = j
                    break
                elif hit_up:
                    outcome = LONG
                    exit_i = j
                    break

            labels[i] = outcome
            exits[i] = exit_i

    return labels, exits, vol, barriers


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


def load_canonical():
    print(f"Input: {INPUT}")
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    frames = []
    for chunk in pd.read_csv(
        INPUT,
        usecols=["timestamp", "open", "high", "low", "close"],
        chunksize=500_000,
    ):
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    del frames

    ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts_num.isna().any():
        raise ValueError("Invalid timestamps found.")

    unit = detect_epoch_unit(ts_num)
    df["timestamp"] = ts_num.astype(np.int64)

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)

    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("NaN OHLC values found.")

    if not np.all(df["timestamp"].to_numpy()[1:] > df["timestamp"].to_numpy()[:-1]):
        raise ValueError("Canonical timestamps are not strictly increasing.")

    print(f"Rows loaded: {len(df):,}")
    print(f"Timestamp unit: epoch-{unit}")
    return df, unit


def find_segments(timestamp, unit):
    ts = timestamp.to_numpy(np.int64, copy=False)

    expected = {
        "s": 60,
        "ms": 60_000,
        "us": 60_000_000,
        "ns": 60_000_000_000,
    }[unit]

    delta = np.diff(ts)
    breaks = np.flatnonzero(delta != expected) + 1

    starts = np.r_[0, breaks].astype(np.int64)
    ends = np.r_[breaks, len(ts)].astype(np.int64)

    return starts, ends


def run_one(df, starts, ends, multiplier):
    high = df["high"].to_numpy(np.float64, copy=False)
    low = df["low"].to_numpy(np.float64, copy=False)
    close = df["close"].to_numpy(np.float64, copy=False)

    labels, exits, vol, barriers = build_dynamic_tb(
        high, low, close, starts, ends, multiplier
    )

    return labels, exits, vol, barriers


def summarize(df, labels, barriers, multiplier):
    valid = np.isin(labels, [SHORT, TIMEOUT, LONG])
    valid_n = int(valid.sum())
    ambiguous_n = int((labels == AMBIGUOUS).sum())
    invalid_n = int((labels == INVALID).sum())

    short_n = int((labels == SHORT).sum())
    timeout_n = int((labels == TIMEOUT).sum())
    long_n = int((labels == LONG).sum())

    valid_barrier = barriers[valid]
    mean_barrier_bps = (
        float(np.nanmean(valid_barrier)) * 10_000
        if valid_n else np.nan
    )

    return {
        "multiplier": multiplier,
        "horizon_min": HORIZON,
        "vol_lookback_min": VOL_LOOKBACK,
        "total_rows": len(df),
        "valid_rows": valid_n,
        "ambiguous_rows": ambiguous_n,
        "invalid_rows": invalid_n,
        "short_rows": short_n,
        "timeout_rows": timeout_n,
        "long_rows": long_n,
        "short_pct_valid": short_n / valid_n * 100 if valid_n else np.nan,
        "timeout_pct_valid": timeout_n / valid_n * 100 if valid_n else np.nan,
        "long_pct_valid": long_n / valid_n * 100 if valid_n else np.nan,
        "mean_barrier_bps": mean_barrier_bps,
    }


def save_output(df, labels, exits, vol, barriers, path):
    out = df.copy()
    out["volatility_60m"] = vol
    out["barrier_return_60m"] = barriers
    out["tb_label_60m"] = labels
    out["tb_exit_index_60m"] = exits
    out.to_csv(path, index=False)


def main():
    print("=" * 78)
    print("XAUUSD DYNAMIC VOLATILITY-ADJUSTED TRIPLE-BARRIER TARGETS")
    print("=" * 78)
    print("Horizon: 60 minutes")
    print("Volatility lookback: previous 60 continuous 1-minute returns")
    print("Barrier = multiplier × sigma_1m × sqrt(60)")
    print("Multipliers: [0.50, 0.75, 1.00, 1.25, 1.50]")
    print("No future data is used for barrier calculation.")
    print("2026: diagnostic only; excluded later from model selection.")
    print("=" * 78)

    t0 = time.time()

    df, unit = load_canonical()
    starts, ends = find_segments(df["timestamp"], unit)
    print(f"Continuous segments: {len(starts):,}")

    # Compile once using first multiplier.
    print("\nCompiling Numba engine...")
    _ = run_one(df, starts, ends, 1.0)
    print("Numba compilation complete.")

    summaries = []

    for multiplier, path in OUTPUTS.items():
        print("\n" + "=" * 78)
        print(f"MULTIPLIER {multiplier:.2f}x")
        print(f"Output: {path}")
        print("=" * 78)

        t1 = time.time()
        labels, exits, vol, barriers = run_one(
            df, starts, ends, multiplier
        )

        summary = summarize(df, labels, barriers, multiplier)
        summaries.append(summary)

        print(f"Valid:          {summary['valid_rows']:,}")
        print(f"Ambiguous:      {summary['ambiguous_rows']:,}")
        print(f"Invalid:        {summary['invalid_rows']:,}")
        print(f"SHORT:          {summary['short_rows']:,}")
        print(f"TIMEOUT:        {summary['timeout_rows']:,}")
        print(f"LONG:           {summary['long_rows']:,}")
        print(
            "Valid class %:  "
            f"[{summary['short_pct_valid']:.2f}, "
            f"{summary['timeout_pct_valid']:.2f}, "
            f"{summary['long_pct_valid']:.2f}]"
        )
        print(f"Mean barrier:   {summary['mean_barrier_bps']:.2f} bps")
        print(f"Elapsed:        {time.time() - t1:.2f} sec")

        save_output(df, labels, exits, vol, barriers, path)
        print(f"Saved: {path}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SUMMARY, index=False)

    print("\n" + "=" * 78)
    print("DYNAMIC TRIPLE-BARRIER TARGET GENERATION COMPLETE")
    print("=" * 78)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {SUMMARY}")
    print(f"Total wall time: {time.time() - t0:.1f} sec")
    print("2026 is included only for diagnostic inspection.")
    print("=" * 78)


if __name__ == "__main__":
    main()
