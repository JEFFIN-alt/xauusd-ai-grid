import os
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/processed/xauusd_m1_2019_2026_canonical.csv"
)

OUTPUT_FILE = Path(
    "data/processed/xauusd_m1_horizon_targets.csv"
)

CHUNK_SIZE = 100_000

HORIZONS = [5, 15, 30, 60]

MAX_HORIZON = max(HORIZONS)


# ============================================================
# SETUP
# ============================================================

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

print("=" * 70)
print("BUILDING GAP-AWARE MULTI-HORIZON TARGETS")
print("=" * 70)

print(f"\nInput:")
print(INPUT_FILE)

print(f"\nOutput:")
print(OUTPUT_FILE)

print(
    f"\nHorizons: "
    f"{HORIZONS}"
)

print(
    f"Maximum future lookahead: "
    f"{MAX_HORIZON} M1 bars"
)


# ============================================================
# STATE
# ============================================================

carry = pd.DataFrame()

first_output = True

last_written_timestamp = None

total_input_rows = 0

total_output_rows = 0

valid_counts = {
    h: 0
    for h in HORIZONS
}


# ============================================================
# WRITE FUNCTION
# ============================================================

def write_rows(rows):

    global first_output
    global last_written_timestamp
    global total_output_rows

    if rows.empty:
        return

    target_columns = [
        f"future_return_{h}m"
        for h in HORIZONS
    ]

    # At least one target must exist.
    rows = rows.dropna(
        subset=target_columns,
        how="all"
    )

    if rows.empty:
        return

    # Prevent duplicate timestamps.
    if last_written_timestamp is not None:

        rows = rows.loc[
            rows["timestamp"]
            > last_written_timestamp
        ]

    if rows.empty:
        return

    output_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ] + target_columns

    rows = rows[output_columns].copy()

    # Count valid targets.
    for horizon in HORIZONS:

        valid = rows[
            f"future_return_{horizon}m"
        ].notna()

        valid_counts[horizon] += int(
            valid.sum()
        )

    # Write explicitly as UTF-8.
    mode = (
        "w"
        if first_output
        else "a"
    )

    with OUTPUT_FILE.open(
        mode=mode,
        encoding="utf-8",
        newline=""
    ) as handle:

        rows.to_csv(
            handle,
            header=first_output,
            index=False,
        )

        handle.flush()
        os.fsync(
            handle.fileno()
        )

    first_output = False

    total_output_rows += len(rows)

    last_written_timestamp = int(
        rows["timestamp"].iloc[-1]
    )


# ============================================================
# PROCESS DATASET
# ============================================================

for chunk_number, chunk in enumerate(

    pd.read_csv(
        INPUT_FILE,

        usecols=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ],

        chunksize=CHUNK_SIZE,
    ),

    start=1,
):

    total_input_rows += len(chunk)

    # --------------------------------------------------------
    # Combine previous pending rows + current chunk
    # --------------------------------------------------------

    if carry.empty:

        working = chunk.copy()

        carry_length = 0

    else:

        working = pd.concat(
            [
                carry,
                chunk,
            ],
            ignore_index=True,
        )

        carry_length = len(carry)


    # --------------------------------------------------------
    # Chronological safety check
    # --------------------------------------------------------

    timestamps = (
        working["timestamp"]
        .to_numpy(
            dtype=np.int64
        )
    )

    if len(timestamps) > 1:

        diffs = np.diff(
            timestamps
        )

        if np.any(diffs <= 0):

            raise RuntimeError(
                "Canonical dataset is not "
                "strictly chronological."
            )


    # --------------------------------------------------------
    # CREATE ALL TARGETS
    # --------------------------------------------------------

    for horizon in HORIZONS:

        future_close = (
            working["close"]
            .shift(-horizon)
        )

        future_timestamp = (
            working["timestamp"]
            .shift(-horizon)
        )

        expected_diff = (
            horizon *
            60 *
            1000
        )

        actual_diff = (
            future_timestamp
            - working["timestamp"]
        )

        valid = (
            actual_diff
            ==
            expected_diff
        )

        working[
            f"future_return_{horizon}m"
        ] = np.where(

            valid,

            (
                future_close /
                working["close"]
            ) - 1.0,

            np.nan,
        )


    # --------------------------------------------------------
    # Rows that are definitely finalized
    #
    # We wait MAX_HORIZON rows so every horizon can be
    # evaluated before a row is written.
    # --------------------------------------------------------

    finalized_count = (
        len(working)
        - MAX_HORIZON
    )

    if finalized_count > 0:

        finalized = working.iloc[
            :finalized_count
        ].copy()

        # Only write rows that haven't already been written.
        if last_written_timestamp is not None:

            finalized = finalized.loc[
                finalized["timestamp"]
                > last_written_timestamp
            ]

        write_rows(
            finalized
        )


    # --------------------------------------------------------
    # Carry the last MAX_HORIZON rows
    #
    # These need future rows from the next chunk.
    # --------------------------------------------------------

    carry = working.tail(
        MAX_HORIZON
    )[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    ].copy()


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    print(
        f"Chunk {chunk_number:02d} | "
        f"Input: {total_input_rows:,} | "
        f"Output: {total_output_rows:,}"
    )


# ============================================================
# FINAL FLUSH
#
# Last MAX_HORIZON rows cannot have all horizons,
# but shorter horizons may still be valid.
# ============================================================

if not carry.empty:

    working = carry.copy()

    for horizon in HORIZONS:

        future_close = (
            working["close"]
            .shift(-horizon)
        )

        future_timestamp = (
            working["timestamp"]
            .shift(-horizon)
        )

        expected_diff = (
            horizon *
            60 *
            1000
        )

        actual_diff = (
            future_timestamp
            - working["timestamp"]
        )

        valid = (
            actual_diff
            ==
            expected_diff
        )

        working[
            f"future_return_{horizon}m"
        ] = np.where(

            valid,

            (
                future_close /
                working["close"]
            ) - 1.0,

            np.nan,
        )

    if last_written_timestamp is not None:

        working = working.loc[
            working["timestamp"]
            > last_written_timestamp
        ]

    write_rows(
        working
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MULTI-HORIZON TARGET BUILD COMPLETE")
print("=" * 70)

print(
    f"\nInput rows:  "
    f"{total_input_rows:,}"
)

print(
    f"Output rows: "
    f"{total_output_rows:,}"
)

print("\nValid targets:")

for horizon in HORIZONS:

    print(
        f"  {horizon:>2}m: "
        f"{valid_counts[horizon]:,}"
    )


print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)

print("\n" + "=" * 70)
