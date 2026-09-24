import pandas as pd
import numpy as np


CANONICAL_FILE = (
    "data/processed/"
    "xauusd_m1_2019_2026_canonical.csv"
)

TARGET_FILE = (
    "data/processed/"
    "xauusd_m1_horizon_targets.csv"
)

CHUNK_SIZE = 100_000

HORIZONS = [5, 15, 30, 60]


print("=" * 70)
print("VALIDATING GAP-AWARE MULTI-HORIZON TARGETS")
print("=" * 70)


# ============================================================
# LOAD CANONICAL TIMESTAMPS + CLOSE
# ============================================================

print("\nLoading canonical timestamps and closes...")

canonical_parts = []

for chunk in pd.read_csv(
    CANONICAL_FILE,
    usecols=["timestamp", "close"],
    chunksize=CHUNK_SIZE,
):

    canonical_parts.append(
        chunk
    )


canonical = pd.concat(
    canonical_parts,
    ignore_index=True
)

del canonical_parts


canonical_timestamps = (
    canonical["timestamp"]
    .to_numpy(dtype=np.int64)
)

canonical_close = (
    canonical["close"]
    .to_numpy(dtype=np.float64)
)


print(
    f"Canonical rows: "
    f"{len(canonical_timestamps):,}"
)


# ============================================================
# BASIC CANONICAL CHECK
# ============================================================

if not np.all(
    np.diff(canonical_timestamps) > 0
):

    raise RuntimeError(
        "Canonical timestamps are not strictly increasing."
    )


# ============================================================
# TARGET FILE CHECKS
# ============================================================

required_columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
]

for horizon in HORIZONS:
    required_columns.append(
        f"future_return_{horizon}m"
    )


total_rows = 0

duplicate_timestamps = 0

chronological = True

missing_values = 0

mismatch_counts = {
    h: 0
    for h in HORIZONS
}

valid_counts = {
    h: 0
    for h in HORIZONS
}

max_return_difference = {
    h: 0.0
    for h in HORIZONS
}

first_timestamp = None
last_timestamp = None
previous_timestamp = None


# ============================================================
# STREAM TARGET DATASET
# ============================================================

print("\nValidating target file...")


for chunk_number, chunk in enumerate(

    pd.read_csv(
        TARGET_FILE,
        usecols=required_columns,
        chunksize=CHUNK_SIZE,
    ),

    start=1,
):

    total_rows += len(chunk)


    # --------------------------------------------------------
    # Missing OHLC/timestamp
    # Targets are allowed to be NaN.
    # --------------------------------------------------------

    missing_values += int(
        chunk[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .isna()
        .sum()
        .sum()
    )


    # --------------------------------------------------------
    # Timestamp ordering
    # --------------------------------------------------------

    ts = chunk[
        "timestamp"
    ].to_numpy(dtype=np.int64)


    if len(ts):

        if first_timestamp is None:

            first_timestamp = ts[0]


        if (
            previous_timestamp is not None
            and ts[0] <= previous_timestamp
        ):

            chronological = False


        if len(ts) > 1:

            diffs = np.diff(ts)

            duplicate_timestamps += int(
                (diffs == 0).sum()
            )

            if np.any(diffs < 0):

                chronological = False


        previous_timestamp = ts[-1]

        last_timestamp = ts[-1]


    # --------------------------------------------------------
    # Validate each horizon
    # --------------------------------------------------------

    for horizon in HORIZONS:

        column = (
            f"future_return_{horizon}m"
        )

        values = chunk[
            column
        ].to_numpy(dtype=np.float64)

        valid_mask = np.isfinite(values)

        valid_counts[horizon] += int(
            valid_mask.sum()
        )

        if not valid_mask.any():

            continue


        current_timestamps = ts[
            valid_mask
        ]

        current_closes = (
            chunk["close"]
            .to_numpy(dtype=np.float64)
            [valid_mask]
        )

        target_values = values[
            valid_mask
        ]


        future_timestamps = (
            current_timestamps
            +
            horizon * 60 * 1000
        )


        # ----------------------------------------------------
        # Find exact future timestamps
        # ----------------------------------------------------

        positions = np.searchsorted(
            canonical_timestamps,
            future_timestamps
        )


        in_range = (
            positions <
            len(canonical_timestamps)
        )


        if not np.all(in_range):

            mismatch_counts[horizon] += int(
                (~in_range).sum()
            )


        if not in_range.any():

            continue


        valid_positions = positions[
            in_range
        ]

        expected_timestamps = (
            future_timestamps[
                in_range
            ]
        )


        exact_match = (
            canonical_timestamps[
                valid_positions
            ]
            ==
            expected_timestamps
        )


        mismatch_counts[horizon] += int(
            (~exact_match).sum()
        )


        if not exact_match.any():

            continue


        exact_positions = (
            valid_positions[
                exact_match
            ]
        )

        exact_current_closes = (
            current_closes[
                in_range
            ][exact_match]
        )

        actual_returns = (
            canonical_close[
                exact_positions
            ]
            /
            exact_current_closes
            - 1.0
        )

        expected_returns = (
            target_values[
                in_range
            ][exact_match]
        )


        differences = np.abs(
            actual_returns -
            expected_returns
        )


        if len(differences):

            max_difference = float(
                differences.max()
            )

            max_return_difference[horizon] = max(
                max_return_difference[horizon],
                max_difference
            )


            mismatch_counts[horizon] += int(
                (differences > 1e-12).sum()
            )


    if chunk_number % 10 == 0:

        print(
            f"Processed chunks: "
            f"{chunk_number} | "
            f"Rows: {total_rows:,}"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("TARGET VALIDATION RESULT")
print("=" * 70)

print(
    f"\nRows: {total_rows:,}"
)

print(
    f"First timestamp: "
    f"{pd.to_datetime(first_timestamp, unit='ms', utc=True)}"
)

print(
    f"Last timestamp:  "
    f"{pd.to_datetime(last_timestamp, unit='ms', utc=True)}"
)

print(
    f"Missing OHLC/timestamp values: "
    f"{missing_values:,}"
)

print(
    f"Duplicate timestamps: "
    f"{duplicate_timestamps:,}"
)

print(
    f"Chronological: "
    f"{chronological}"
)


print("\nTarget counts:")

for horizon in HORIZONS:

    print(
        f"  {horizon:>2}m: "
        f"{valid_counts[horizon]:,}"
    )


print("\nTarget validation:")

for horizon in HORIZONS:

    print(
        f"  {horizon:>2}m mismatches: "
        f"{mismatch_counts[horizon]:,}"
    )

    print(
        f"  {horizon:>2}m max difference: "
        f"{max_return_difference[horizon]:.12e}"
    )


print("\nExpected:")

print("  Missing OHLC/timestamp = 0")
print("  Duplicate timestamps   = 0")
print("  Chronological           = True")
print("  Target mismatches       = 0")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
