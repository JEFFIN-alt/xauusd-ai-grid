import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# BUILD XAUUSD 15-MINUTE TARGET DATASET
# ============================================================

INPUT_FILE = Path(
    "data/processed/xauusd_m1_2019_2026.csv"
)

OUTPUT_FILE = Path(
    "data/processed/xauusd_m1_15m_target.csv"
)

CHUNK_SIZE = 100_000
HORIZON = 15


print("=" * 70)
print("BUILDING XAUUSD 15-MINUTE TARGET DATASET")
print("=" * 70)

print("\nInput:")
print(INPUT_FILE)

print("\nOutput:")
print(OUTPUT_FILE)


# ============================================================
# We need 15 future rows.
# Carry rows between chunks so targets can cross boundaries.
# ============================================================

carry = pd.DataFrame(
    columns=["timestamp", "open", "high", "low", "close"]
)

first_chunk = True

total_input_rows = 0
total_output_rows = 0


# ============================================================
# PROCESS IN CHUNKS
# ============================================================

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=[
            "timestamp",
            "open",
            "high",
            "low",
            "close"
        ],
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    total_input_rows += len(chunk)

    # --------------------------------------------------------
    # Combine carry + current chunk
    # --------------------------------------------------------

    if not carry.empty:

        working = pd.concat(
            [carry, chunk],
            ignore_index=True
        )

        carry_length = len(carry)

    else:

        working = chunk.copy()

        carry_length = 0


    # --------------------------------------------------------
    # Ensure chronological order
    # --------------------------------------------------------

    working = working.sort_values(
        "timestamp"
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Future close
    # --------------------------------------------------------

    working["future_close_15m"] = (
        working["close"].shift(-HORIZON)
    )


    # --------------------------------------------------------
    # Future timestamp
    # --------------------------------------------------------

    working["future_timestamp_15m"] = (
        working["timestamp"].shift(-HORIZON)
    )


    # --------------------------------------------------------
    # Calculate actual elapsed time
    # --------------------------------------------------------

    working["future_time_diff_ms"] = (
        working["future_timestamp_15m"]
        - working["timestamp"]
    )


    expected_time_diff = (
        HORIZON * 60 * 1000
    )


    # --------------------------------------------------------
    # Future 15-minute return
    # --------------------------------------------------------

    working["future_return_15m"] = (
        working["future_close_15m"]
        / working["close"]
        - 1.0
    )


    # --------------------------------------------------------
    # Only keep rows belonging to the current chunk
    # --------------------------------------------------------

    start = carry_length

    end = start + len(chunk)

    current = working.iloc[start:end].copy()


    # --------------------------------------------------------
    # Valid target
    #
    # Require exactly 15 minutes between observations.
    # This prevents weekend/session gaps becoming fake
    # 15-minute targets.
    # --------------------------------------------------------

    valid_target = (
        current["future_time_diff_ms"]
        == expected_time_diff
    )


    current = current.loc[
        valid_target
    ].copy()


    # --------------------------------------------------------
    # Keep only the information we actually need
    # --------------------------------------------------------

    current = current[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "future_return_15m"
        ]
    ]


    # Remove invalid values
    current = current.dropna(
        subset=["future_return_15m"]
    )


    # --------------------------------------------------------
    # Save chunk
    # --------------------------------------------------------

    if not current.empty:

        current.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        total_output_rows += len(current)


    # --------------------------------------------------------
    # Carry last 15 rows
    # --------------------------------------------------------

    carry = working.tail(HORIZON)[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close"
        ]
    ].copy()


    print(
        f"Processed chunk {chunk_number} | "
        f"Input rows: {total_input_rows:,} | "
        f"Target rows: {total_output_rows:,}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TARGET DATASET COMPLETE")
print("=" * 70)

print(
    f"\nInput rows:  {total_input_rows:,}"
)

print(
    f"Output rows: {total_output_rows:,}"
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("\nColumns:")

print(
    "timestamp"
)

print(
    "open"
)

print(
    "high"
)

print(
    "low"
)

print(
    "close"
)

print(
    "future_return_15m"
)

print("\n" + "=" * 70)
