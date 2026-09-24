#!/usr/bin/env python3
"""
BUILD XAUUSD TRIPLE-BARRIER TARGETS FOR A BARRIER-SIZE SWEEP

Purpose
-------
Create separate triple-barrier target files for larger one-sided barriers:
    +/- 10 bps
    +/- 15 bps
    +/- 20 bps

Vertical horizons:
    30 minutes
    60 minutes

The existing +/- 5 bps target remains the control and is NOT rebuilt here.

Labels:
    0  = SHORT       -> lower barrier hit first
    1  = TIMEOUT     -> neither barrier hit before vertical barrier
    2  = LONG        -> upper barrier hit first
   -1  = AMBIGUOUS   -> both barriers touched in the same M1 candle
   -2  = INVALID     -> no complete vertical horizon within a segment

Ambiguous rows are kept as -1 so they can be excluded from ML later.
No intrabar ordering is guessed.

Important
---------
- Targets are built directly from the canonical M1 dataset.
- Continuous segments are enforced; targets never cross data/session gaps.
- Future HIGH/LOW are used only for target construction, never as ML inputs.
- 2026 is present in the target files but MUST remain excluded from model
  selection/training/testing decisions later.
- This script is a research sweep, not a claim that any barrier is optimal.

Dependencies:
    pandas, numpy, numba
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:
    print("NUMBA IS REQUIRED")
    print("Install with:")
    print("conda install -c conda-forge numba")
    sys.exit(1)


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/processed/xauusd_m1_2019_2026_canonical.csv")
OUTPUT_DIR = Path("data/processed")
SUMMARY_FILE = OUTPUT_DIR / "xauusd_m1_triple_barrier_barrier_sweep_summary.csv"

BARRIER_BPS = [10, 15, 20]
HORIZONS = [30, 60]

USECOLS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
]


def output_file_for(barrier_bps: int) -> Path:
    return OUTPUT_DIR / f"xauusd_m1_triple_barrier_{barrier_bps}bps.csv"


# ============================================================
# NUMBA ENGINE
# ============================================================

@njit(cache=True)
def triple_barrier_labels(
    timestamps,
    high,
    low,
    close,
    segment_id,
    horizon,
    barrier_return,
):
    n = len(close)

    labels = np.full(n, -2, dtype=np.int8)
    barrier_hit = np.full(n, -1, dtype=np.int8)
    exit_index = np.full(n, -1, dtype=np.int64)
    elapsed_minutes = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        last_j = i + horizon

        if last_j >= n:
            continue

        if segment_id[last_j] != segment_id[i]:
            continue

        entry = close[i]
        upper = entry * (1.0 + barrier_return)
        lower = entry * (1.0 - barrier_return)

        found = False

        for j in range(i + 1, last_j + 1):
            if segment_id[j] != segment_id[i]:
                break

            hit_upper = high[j] >= upper
            hit_lower = low[j] <= lower

            if hit_upper and hit_lower:
                labels[i] = -1
                barrier_hit[i] = 3
                exit_index[i] = j
                elapsed_minutes[i] = (
                    (timestamps[j] - timestamps[i]) / 60000.0
                )
                found = True
                break

            if hit_upper:
                labels[i] = 2
                barrier_hit[i] = 2
                exit_index[i] = j
                elapsed_minutes[i] = (
                    (timestamps[j] - timestamps[i]) / 60000.0
                )
                found = True
                break

            if hit_lower:
                labels[i] = 0
                barrier_hit[i] = 0
                exit_index[i] = j
                elapsed_minutes[i] = (
                    (timestamps[j] - timestamps[i]) / 60000.0
                )
                found = True
                break

        if not found:
            labels[i] = 1
            barrier_hit[i] = 1
            exit_index[i] = last_j
            elapsed_minutes[i] = (
                (timestamps[last_j] - timestamps[i]) / 60000.0
            )

    return labels, barrier_hit, exit_index, elapsed_minutes


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing canonical dataset: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("XAUUSD TRIPLE-BARRIER BARRIER-SIZE SWEEP")
    print("=" * 78)
    print(f"Input:          {INPUT_FILE}")
    print(f"Barriers:       +/- {BARRIER_BPS} bps")
    print(f"Horizons:       {HORIZONS} minutes")
    print("Existing 5bps:  CONTROL (not rebuilt)")
    print("2026 TEST:       NOT USED LATER FOR MODEL SELECTION")
    print("=" * 78)

    start = time.time()

    df = pd.read_csv(
        INPUT_FILE,
        usecols=USECOLS,
        dtype={
            "timestamp": "int64",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
        },
    )

    n = len(df)
    print(f"Rows loaded: {n:,}")

    if df["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps detected.")

    ts_diff = df["timestamp"].diff()
    backward = int((ts_diff < 0).sum())
    if backward:
        raise RuntimeError(f"Backward timestamp transitions detected: {backward}")

    invalid_ohlc = int(
        (
            (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
            | (df["high"] < df["low"])
        ).sum()
    )
    if invalid_ohlc:
        raise RuntimeError(f"Invalid OHLC rows detected: {invalid_ohlc}")

    new_segment = ts_diff.ne(60_000).astype(np.int32)
    new_segment.iloc[0] = 0
    segment_id = new_segment.cumsum().to_numpy(np.int32)

    print(f"Continuous segments: {int(segment_id[-1]) + 1:,}")

    timestamps = df["timestamp"].to_numpy(np.int64)
    high = df["high"].to_numpy(np.float32)
    low = df["low"].to_numpy(np.float32)
    close = df["close"].to_numpy(np.float32)

    # Compile once before the sweep.
    print("\nCompiling Numba engine...")
    _ = triple_barrier_labels(
        timestamps[:1000],
        high[:1000],
        low[:1000],
        close[:1000],
        segment_id[:1000],
        30,
        0.0010,
    )
    print("Numba compilation complete.")

    summary_rows = []

    for barrier_bps in BARRIER_BPS:
        barrier_return = barrier_bps / 10000.0
        output_file = output_file_for(barrier_bps)

        print("\n" + "=" * 78)
        print(f"BARRIER +/- {barrier_bps} BPS")
        print(f"Output: {output_file}")
        print("=" * 78)

        results = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": df["open"].to_numpy(np.float32),
                "high": high,
                "low": low,
                "close": close,
            }
        )

        for horizon in HORIZONS:
            t0 = time.time()
            print(f"\nHorizon {horizon}m...")

            labels, barrier_hit, exit_index, elapsed = triple_barrier_labels(
                timestamps,
                high,
                low,
                close,
                segment_id,
                horizon,
                barrier_return,
            )

            results[f"tb_label_{horizon}m"] = labels
            results[f"tb_barrier_{horizon}m"] = barrier_hit
            results[f"tb_exit_index_{horizon}m"] = exit_index
            results[f"tb_elapsed_minutes_{horizon}m"] = elapsed

            valid = labels >= 0
            ambiguous = labels == -1
            invalid = labels == -2

            valid_count = int(valid.sum())
            ambiguous_count = int(ambiguous.sum())
            invalid_count = int(invalid.sum())
            short_count = int((labels == 0).sum())
            timeout_count = int((labels == 1).sum())
            long_count = int((labels == 2).sum())

            print(f"Valid:          {valid_count:,}")
            print(f"Ambiguous:      {ambiguous_count:,}")
            print(f"Invalid:        {invalid_count:,}")
            print(f"SHORT:          {short_count:,}")
            print(f"TIMEOUT:        {timeout_count:,}")
            print(f"LONG:           {long_count:,}")

            if valid_count:
                print(
                    "Valid class %: ",
                    np.round(
                        np.array([short_count, timeout_count, long_count], dtype=np.float64)
                        / valid_count
                        * 100.0,
                        2,
                    ).tolist(),
                )

            print(f"Elapsed:        {time.time() - t0:.1f} sec")

            summary_rows.append(
                {
                    "barrier_bps": barrier_bps,
                    "barrier_return": barrier_return,
                    "horizon_min": horizon,
                    "total_rows": n,
                    "valid_rows": valid_count,
                    "ambiguous_rows": ambiguous_count,
                    "invalid_rows": invalid_count,
                    "short_rows": short_count,
                    "timeout_rows": timeout_count,
                    "long_rows": long_count,
                    "short_pct_valid": short_count / valid_count * 100 if valid_count else np.nan,
                    "timeout_pct_valid": timeout_count / valid_count * 100 if valid_count else np.nan,
                    "long_pct_valid": long_count / valid_count * 100 if valid_count else np.nan,
                }
            )

        # Write one barrier file only after all horizons succeed.
        print(f"\nWriting {output_file}...")
        results.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n" + "=" * 78)
    print("BARRIER-SIZE SWEEP COMPLETE")
    print("=" * 78)
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {SUMMARY_FILE}")
    print(f"Total wall time: {time.time() - start:.1f} sec")
    print("\nNext step: inspect target distributions before any ML training.")
    print("2026 remains untouched for later final evaluation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
