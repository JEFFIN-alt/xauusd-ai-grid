#!/usr/bin/env python3
"""
BUILD XAUUSD TRIPLE-BARRIER TARGETS

Purpose
-------
Create a trading-oriented target from the canonical XAUUSD M1 dataset.

For each entry bar:
    + upper barrier = +0.05%
    - lower barrier = -0.05%
    vertical barrier = 15 / 30 / 60 minutes

Labels:
    0 = SHORT       -> lower barrier hit first
    1 = TIMEOUT     -> neither barrier hit before vertical barrier
    2 = LONG        -> upper barrier hit first
   -1 = AMBIGUOUS   -> both barriers touched in the same M1 bar

Ambiguous rows are marked invalid and should be excluded from ML.
The target uses future HIGH/LOW to determine barrier touches and therefore
must only be used as the label, never as an input feature.

Important:
- Uses the canonical M1 dataset directly.
- Does not cross session/weekend gaps.
- 2026 data is included in the file for target construction but MUST remain
  excluded from model selection/testing later.
- This is an intentionally fixed baseline configuration. Do not optimize
  the 0.05% barrier after looking at results.

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
    print("=" * 78)
    print("NUMBA IS REQUIRED")
    print("=" * 78)
    print()
    print("Install it inside the ai-lab Conda environment with:")
    print()
    print("conda install -c conda-forge numba")
    print()
    sys.exit(1)


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/processed/xauusd_m1_2019_2026_canonical.csv"
)

OUTPUT_FILE = Path(
    "data/processed/xauusd_m1_triple_barrier_targets.csv"
)

HORIZONS = [15, 30, 60]

# Fixed one-sided barrier:
# +0.05% / -0.05% = +/- 5 bps.
BARRIER_RETURN = 0.0005

# Columns needed from canonical M1.
USECOLS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
]


# ============================================================
# NUMBA TRIPLE-BARRIER ENGINE
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

    labels = np.full(n, -1, dtype=np.int8)
    barrier_hit = np.full(n, -1, dtype=np.int8)
    exit_index = np.full(n, -1, dtype=np.int64)
    elapsed_minutes = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        # Need the complete vertical horizon.
        last_j = i + horizon
        if last_j >= n:
            continue

        # Do not cross a session/data gap.
        if segment_id[last_j] != segment_id[i]:
            continue

        entry = close[i]

        upper = entry * (1.0 + barrier_return)
        lower = entry * (1.0 - barrier_return)

        found = False

        for j in range(i + 1, last_j + 1):
            # Stop if a discontinuity appears before the target horizon.
            if segment_id[j] != segment_id[i]:
                break

            hit_upper = high[j] >= upper
            hit_lower = low[j] <= lower

            # Both barriers touched within the same M1 candle.
            # OHLC data does not reveal which happened first.
            if hit_upper and hit_lower:
                labels[i] = -1
                barrier_hit[i] = 3  # ambiguous
                exit_index[i] = j
                elapsed_minutes[i] = (
                    (timestamps[j] - timestamps[i]) / 60000.0
                )
                found = True
                break

            if hit_upper:
                labels[i] = 2  # LONG
                barrier_hit[i] = 2
                exit_index[i] = j
                elapsed_minutes[i] = (
                    (timestamps[j] - timestamps[i]) / 60000.0
                )
                found = True
                break

            if hit_lower:
                labels[i] = 0  # SHORT
                barrier_hit[i] = 0
                exit_index[i] = j
                elapsed_minutes[i] = (
                    (timestamps[j] - timestamps[i]) / 60000.0
                )
                found = True
                break

        if not found:
            # Neither barrier was touched:
            # vertical barrier / TIMEOUT.
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
        raise FileNotFoundError(
            f"Missing canonical dataset:\n{INPUT_FILE}"
        )

    print("=" * 78)
    print("XAUUSD TRIPLE-BARRIER TARGET BUILD")
    print("=" * 78)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Barriers: +/- {BARRIER_RETURN:.4%} (+/- 5 bps)")
    print(f"Vertical horizons: {HORIZONS} minutes")
    print()
    print("2026 WILL BE PRESENT IN THE TARGET FILE BUT MUST")
    print("REMAIN EXCLUDED FROM MODEL TRAINING/SELECTION/TESTING.")
    print("=" * 78)

    start_time = time.time()

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

    print(f"\nRows loaded: {len(df):,}")

    # --------------------------------------------------------
    # Validate basic order before target construction.
    # --------------------------------------------------------

    if df["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps detected.")

    ts_diff = df["timestamp"].diff()

    backward = (ts_diff < 0).sum()

    if backward:
        raise RuntimeError(
            f"Backward timestamp transitions detected: {backward}"
        )

    invalid_ohlc = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    ).sum()

    if invalid_ohlc:
        raise RuntimeError(
            f"Invalid OHLC rows detected: {invalid_ohlc}"
        )

    # --------------------------------------------------------
    # Build continuous segment IDs.
    # A new segment begins whenever timestamps are not exactly
    # one minute apart.
    # --------------------------------------------------------

    new_segment = ts_diff.ne(60_000).astype(np.int32)

    # First row begins segment 0.
    new_segment.iloc[0] = 0

    segment_id = new_segment.cumsum().to_numpy(np.int32)

    print(
        f"Continuous segments: {int(segment_id[-1]) + 1:,}"
    )

    # Convert once for the numba engine.
    timestamps = df["timestamp"].to_numpy(np.int64)
    high = df["high"].to_numpy(np.float32)
    low = df["low"].to_numpy(np.float32)
    close = df["close"].to_numpy(np.float32)

    results = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": df["open"].to_numpy(np.float32),
            "high": high,
            "low": low,
            "close": close,
        }
    )

    # --------------------------------------------------------
    # Warm up Numba on a tiny call so compile time is separate
    # from the actual dataset timing.
    # --------------------------------------------------------

    print("\nCompiling triple-barrier engine...")
    _ = triple_barrier_labels(
        timestamps[:1000],
        high[:1000],
        low[:1000],
        close[:1000],
        segment_id[:1000],
        min(15, 999),
        BARRIER_RETURN,
    )

    print("Numba compilation complete.")

    summary_rows = []

    for horizon in HORIZONS:
        print("\n" + "-" * 78)
        print(f"HORIZON {horizon} MINUTES")
        print("-" * 78)

        t0 = time.time()

        labels, barrier_hit, exit_index, elapsed = (
            triple_barrier_labels(
                timestamps,
                high,
                low,
                close,
                segment_id,
                horizon,
                BARRIER_RETURN,
            )
        )

        results[f"tb_label_{horizon}m"] = labels
        results[f"tb_barrier_{horizon}m"] = barrier_hit
        results[f"tb_exit_index_{horizon}m"] = exit_index
        results[f"tb_elapsed_minutes_{horizon}m"] = elapsed

        valid = labels >= 0
        ambiguous = labels == -1

        valid_count = int(valid.sum())
        ambiguous_count = int(ambiguous.sum())

        short_count = int((labels == 0).sum())
        timeout_count = int((labels == 1).sum())
        long_count = int((labels == 2).sum())

        print(f"Valid targets: {valid_count:,}")
        print(f"Ambiguous:     {ambiguous_count:,}")
        print(f"SHORT:         {short_count:,}")
        print(f"TIMEOUT:       {timeout_count:,}")
        print(f"LONG:          {long_count:,}")

        if valid_count:
            print(
                "Valid class %:",
                np.round(
                    np.array(
                        [
                            short_count,
                            timeout_count,
                            long_count,
                        ],
                        dtype=np.float64,
                    )
                    / valid_count
                    * 100.0,
                    2,
                ).tolist(),
            )

        print(
            f"Elapsed wall time: {time.time() - t0:.1f} sec"
        )

        summary_rows.append(
            {
                "horizon_min": horizon,
                "barrier_return": BARRIER_RETURN,
                "total_rows": len(df),
                "valid_rows": valid_count,
                "ambiguous_rows": ambiguous_count,
                "short_rows": short_count,
                "timeout_rows": timeout_count,
                "long_rows": long_count,
                "short_pct_valid": (
                    short_count / valid_count * 100
                    if valid_count else np.nan
                ),
                "timeout_pct_valid": (
                    timeout_count / valid_count * 100
                    if valid_count else np.nan
                ),
                "long_pct_valid": (
                    long_count / valid_count * 100
                    if valid_count else np.nan
                ),
            }
        )

    # --------------------------------------------------------
    # Save only after every horizon succeeds.
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = pd.DataFrame(summary_rows)

    summary_file = OUTPUT_FILE.with_name(
        "xauusd_m1_triple_barrier_summary.csv"
    )

    summary.to_csv(summary_file, index=False)

    print("\n" + "=" * 78)
    print("TRIPLE-BARRIER TARGET BUILD COMPLETE")
    print("=" * 78)

    print("\nSummary:")
    print(summary.to_string(index=False))

    print(f"\nSaved targets:")
    print(OUTPUT_FILE)

    print("\nSaved summary:")
    print(summary_file)

    print(
        f"\nTotal wall time: {time.time() - start_time:.1f} sec"
    )

    print("\nIMPORTANT:")
    print(
        "- -1 labels are ambiguous M1 candles where both barriers "
        "were touched."
    )
    print(
        "- Exclude -1 rows from ML rather than guessing the intrabar order."
    )
    print(
        "- Do NOT tune the 5 bps barrier after seeing these results."
    )
    print(
        "- 2026 remains a future holdout for later ML evaluation."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
