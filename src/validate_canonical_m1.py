import pandas as pd
import numpy as np


FILE = "data/processed/xauusd_m1_2019_2026_canonical.csv"

CHUNK_SIZE = 200_000


print("=" * 70)
print("CANONICAL XAUUSD M1 VALIDATION")
print("=" * 70)


required = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
]


total_rows = 0
duplicate_timestamps = 0
backward_timestamps = 0
non_minute_aligned = 0
invalid_ohlc = 0
non_positive = 0
missing_values = 0
flat_candles = 0
gaps_gt_1m = 0

previous_timestamp = None

first_timestamp = None
last_timestamp = None

max_gap_ms = 0


for chunk_number, chunk in enumerate(
    pd.read_csv(
        FILE,
        usecols=required,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    total_rows += len(chunk)

    missing_values += int(
        chunk.isna().sum().sum()
    )

    timestamps = chunk["timestamp"].to_numpy(
        dtype=np.int64
    )

    # --------------------------------------------------
    # Timestamp alignment
    # --------------------------------------------------

    non_minute_aligned += int(
        (timestamps % 60_000 != 0).sum()
    )

    # --------------------------------------------------
    # Timestamp ordering
    # --------------------------------------------------

    if first_timestamp is None and len(timestamps):
        first_timestamp = timestamps[0]

    if len(timestamps):

        last_timestamp = timestamps[-1]

    if (
        previous_timestamp is not None
        and len(timestamps)
    ):

        diff = (
            timestamps[0]
            - previous_timestamp
        )

        if diff == 0:
            duplicate_timestamps += 1

        elif diff < 0:
            backward_timestamps += 1

        elif diff > 60_000:

            gaps_gt_1m += 1

            max_gap_ms = max(
                max_gap_ms,
                diff
            )


    if len(timestamps) > 1:

        diffs = np.diff(timestamps)

        duplicate_timestamps += int(
            (diffs == 0).sum()
        )

        backward_timestamps += int(
            (diffs < 0).sum()
        )

        positive_diffs = (
            diffs[diffs > 0]
        )

        if len(positive_diffs):

            gaps_gt_1m += int(
                (
                    positive_diffs
                    > 60_000
                ).sum()
            )

            max_gap_ms = max(
                max_gap_ms,
                int(positive_diffs.max())
            )


    if len(timestamps):
        previous_timestamp = timestamps[-1]


    # --------------------------------------------------
    # OHLC
    # --------------------------------------------------

    valid_ohlc = (
        (chunk["high"] >= chunk["open"])
        &
        (chunk["high"] >= chunk["close"])
        &
        (chunk["low"] <= chunk["open"])
        &
        (chunk["low"] <= chunk["close"])
        &
        (chunk["high"] >= chunk["low"])
    )

    invalid_ohlc += int(
        (~valid_ohlc).sum()
    )


    positive = (
        (chunk["open"] > 0)
        &
        (chunk["high"] > 0)
        &
        (chunk["low"] > 0)
        &
        (chunk["close"] > 0)
    )

    non_positive += int(
        (~positive).sum()
    )


    flat = (
        (chunk["open"] == chunk["high"])
        &
        (chunk["high"] == chunk["low"])
        &
        (chunk["low"] == chunk["close"])
    )

    flat_candles += int(
        flat.sum()
    )


    if chunk_number % 5 == 0:

        print(
            f"Processed chunks: "
            f"{chunk_number} | "
            f"Rows: {total_rows:,}"
        )


# ============================================================
# SUMMARY
# ============================================================

first_dt = pd.to_datetime(
    first_timestamp,
    unit="ms",
    utc=True,
)

last_dt = pd.to_datetime(
    last_timestamp,
    unit="ms",
    utc=True,
)


print("\n" + "=" * 70)
print("CANONICAL VALIDATION RESULT")
print("=" * 70)

print(f"\nRows:                  {total_rows:,}")
print(f"First timestamp:       {first_dt}")
print(f"Last timestamp:        {last_dt}")
print(f"Missing values:        {missing_values:,}")
print(f"Non-minute timestamps: {non_minute_aligned:,}")
print(f"Duplicate timestamps:  {duplicate_timestamps:,}")
print(f"Backward timestamps:   {backward_timestamps:,}")
print(f"Invalid OHLC:          {invalid_ohlc:,}")
print(f"Non-positive prices:   {non_positive:,}")
print(f"Flat candles:          {flat_candles:,}")
print(f"Gaps > 1 minute:       {gaps_gt_1m:,}")
print(
    f"Maximum gap:           "
    f"{max_gap_ms / 60_000:.2f} minutes"
)

print("\nExpected:")

print("  Missing values        = 0")
print("  Non-minute timestamps = 0")
print("  Duplicate timestamps  = 0")
print("  Backward timestamps   = 0")
print("  Invalid OHLC          = 0")
print("  Non-positive prices   = 0")

print("\n" + "=" * 70)
