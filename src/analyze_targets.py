import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# XAUUSD TARGET ANALYSIS
# Corrected + memory-safe version
# ============================================================

INPUT_FILE = Path("data/processed/xauusd_m1_2019_2026.csv")

CHUNK_SIZE = 100_000

HORIZONS = [5, 15, 30, 60]

PERCENTILES = [
    1,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    99
]


print("=" * 70)
print("XAUUSD TARGET ANALYSIS")
print("=" * 70)

print("\nLoading dataset in chunks...")


# ============================================================
# Storage for valid target returns
# ============================================================

returns = {
    horizon: []
    for horizon in HORIZONS
}


# Last 60 rows from previous chunk
# Needed because a future target can cross a chunk boundary.
carry = pd.DataFrame(
    columns=["timestamp", "close"]
)


total_rows = 0
total_valid_rows = {
    horizon: 0
    for horizon in HORIZONS
}


# ============================================================
# PROCESS DATA
# ============================================================

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=["timestamp", "close"],
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    total_rows += len(chunk)

    # --------------------------------------------------------
    # Add previous rows
    # --------------------------------------------------------

    carry_length = len(carry)

    if carry_length > 0:

        working = pd.concat(
            [carry, chunk],
            ignore_index=True
        )

    else:

        working = chunk.copy()


    # --------------------------------------------------------
    # Make sure timestamp is numeric
    # --------------------------------------------------------

    working["timestamp"] = pd.to_numeric(
        working["timestamp"],
        errors="coerce"
    )


    close = working["close"].to_numpy(
        dtype=np.float64
    )

    timestamp = working["timestamp"].to_numpy(
        dtype=np.int64
    )


    # --------------------------------------------------------
    # Only analyze rows belonging to CURRENT chunk
    #
    # This prevents double-counting carried rows.
    # --------------------------------------------------------

    current_start = carry_length
    current_end = len(working)

    current_indices = np.arange(
        current_start,
        current_end
    )


    # ========================================================
    # FUTURE RETURNS
    # ========================================================

    for horizon in HORIZONS:

        future_indices = current_indices + horizon

        # Rows near the end of the dataset/chunk area
        # without enough future rows
        valid_position = (
            future_indices < len(working)
        )

        current_idx = current_indices[
            valid_position
        ]

        future_idx = future_indices[
            valid_position
        ]


        if len(current_idx) == 0:
            continue


        # ----------------------------------------------------
        # Check actual elapsed time
        #
        # We require exactly N minutes between the current
        # observation and future observation.
        #
        # This prevents a weekend/session/holiday gap from
        # being incorrectly treated as a normal 5/15/30/60m
        # target.
        # ----------------------------------------------------

        expected_ms = (
            horizon * 60 * 1000
        )

        actual_ms = (
            timestamp[future_idx]
            - timestamp[current_idx]
        )


        valid_time = (
            actual_ms == expected_ms
        )


        current_idx = current_idx[
            valid_time
        ]

        future_idx = future_idx[
            valid_time
        ]


        if len(current_idx) == 0:
            continue


        # ----------------------------------------------------
        # Calculate future return
        # ----------------------------------------------------

        current_close = close[current_idx]

        future_close = close[future_idx]


        future_return = (
            future_close / current_close
        ) - 1.0


        # Remove invalid values
        valid_return = np.isfinite(
            future_return
        )

        future_return = future_return[
            valid_return
        ]


        if len(future_return) == 0:
            continue


        returns[horizon].append(
            future_return
        )


        total_valid_rows[horizon] += len(
            future_return
        )


    # ========================================================
    # CARRY LAST 60 ROWS
    # ========================================================

    carry = working.tail(60)[
        ["timestamp", "close"]
    ].copy()


    print(
        f"Processed chunk {chunk_number} | "
        f"Rows: {total_rows:,}"
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

print("\nCombining target values...")


final_returns = {}

for horizon in HORIZONS:

    if returns[horizon]:

        final_returns[horizon] = np.concatenate(
            returns[horizon]
        )

    else:

        final_returns[horizon] = np.array(
            [],
            dtype=np.float64
        )


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TARGET DISTRIBUTIONS")
print("=" * 70)


for horizon in HORIZONS:

    values = final_returns[horizon]


    if len(values) == 0:

        print(
            f"\nNo valid values for {horizon} minute target."
        )

        continue


    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    mean = np.mean(values)

    std = np.std(
        values,
        ddof=1
    )

    minimum = np.min(values)

    maximum = np.max(values)

    median = np.median(values)


    positive = np.sum(
        values > 0
    )

    negative = np.sum(
        values < 0
    )

    zero = np.sum(
        values == 0
    )


    positive_pct = (
        positive / len(values)
    ) * 100

    negative_pct = (
        negative / len(values)
    ) * 100

    zero_pct = (
        zero / len(values)
    ) * 100


    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    percentile_values = np.percentile(
        values,
        PERCENTILES
    )


    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "-" * 70)

    print(
        f"FUTURE {horizon} MINUTE RETURN"
    )

    print("-" * 70)

    print(
        f"Valid observations: {len(values):,}"
    )

    print(
        f"Mean:               {mean:.8f}"
    )

    print(
        f"Std:                {std:.8f}"
    )

    print(
        f"Median:             {median:.8f}"
    )

    print(
        f"Minimum:            {minimum:.8f}"
    )

    print(
        f"Maximum:            {maximum:.8f}"
    )


    print("\nDirection:")

    print(
        f"Positive:           {positive_pct:.2f}%"
    )

    print(
        f"Negative:           {negative_pct:.2f}%"
    )

    print(
        f"Zero:               {zero_pct:.2f}%"
    )


    print("\nPercentiles:")

    for percentile, value in zip(
        PERCENTILES,
        percentile_values
    ):

        print(
            f"P{percentile:02d}:               "
            f"{value:.8f} "
            f"({value * 100:.4f}%)"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TARGET ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nDataset rows processed: {total_rows:,}"
)

print("\nValid target observations:")

for horizon in HORIZONS:

    print(
        f"{horizon:>2} minutes: "
        f"{len(final_returns[horizon]):,}"
    )

print("\n" + "=" * 70)
