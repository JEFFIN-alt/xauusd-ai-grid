import pandas as pd
from pathlib import Path


FILE = Path(
    "data/processed/xauusd_m1_2019_2026.csv"
)

CHUNK_SIZE = 200_000


print("=" * 70)
print("XAUUSD M1 GAP AUDIT")
print("=" * 70)


previous_timestamp = None

total_rows = 0

exact_one_minute = 0

gaps_greater_one_minute = 0

gaps_less_one_minute = 0

largest_gaps = []

last_rows = []


for chunk_number, chunk in enumerate(

    pd.read_csv(
        FILE,
        usecols=["timestamp"],
        chunksize=CHUNK_SIZE,
    ),

    start=1,
):

    timestamps = (
        chunk["timestamp"]
        .to_numpy()
    )

    total_rows += len(timestamps)


    for timestamp in timestamps:

        timestamp = int(timestamp)

        if previous_timestamp is not None:

            diff = (
                timestamp -
                previous_timestamp
            )

            if diff == 60_000:

                exact_one_minute += 1

            elif diff > 60_000:

                gaps_greater_one_minute += 1

                largest_gaps.append(
                    (
                        diff,
                        previous_timestamp,
                        timestamp,
                    )
                )

            else:

                gaps_less_one_minute += 1


        previous_timestamp = timestamp

        last_rows.append(timestamp)

        if len(last_rows) > 20_000:
            last_rows.pop(0)


    if chunk_number % 5 == 0:

        print(
            f"Processed chunks: "
            f"{chunk_number}"
        )


# ------------------------------------------------------------
# SORT LARGEST GAPS
# ------------------------------------------------------------

largest_gaps.sort(
    reverse=True,
    key=lambda x: x[0]
)


print("\n" + "=" * 70)
print("GAP SUMMARY")
print("=" * 70)

print(
    f"\nTotal rows: "
    f"{total_rows:,}"
)

print(
    f"Exact 1-minute intervals: "
    f"{exact_one_minute:,}"
)

print(
    f"Gaps > 1 minute: "
    f"{gaps_greater_one_minute:,}"
)

print(
    f"Intervals < 1 minute: "
    f"{gaps_less_one_minute:,}"
)


# ------------------------------------------------------------
# LARGEST GAPS
# ------------------------------------------------------------

print("\nLargest gaps:")

for diff, before, after in largest_gaps[:20]:

    before_dt = pd.to_datetime(
        before,
        unit="ms",
        utc=True,
    )

    after_dt = pd.to_datetime(
        after,
        unit="ms",
        utc=True,
    )

    minutes = diff / 60_000

    print(
        f"{before_dt} -> {after_dt} | "
        f"{minutes:.2f} minutes"
    )


# ------------------------------------------------------------
# RECENT / FINAL DATA
# ------------------------------------------------------------

if previous_timestamp is not None:

    print("\n" + "=" * 70)
    print("FINAL DATASET BOUNDARY")
    print("=" * 70)

    print(
        "Last raw timestamp:",
        pd.to_datetime(
            previous_timestamp,
            unit="ms",
            utc=True,
        ),
    )


# ------------------------------------------------------------
# GAPS NEAR THE END
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GAPS NEAR DATASET END")
print("=" * 70)


end_cutoff = pd.Timestamp(
    "2026-09-21 00:00:00",
    tz="UTC",
).value // 1_000_000


recent_gaps = [
    item
    for item in largest_gaps
    if item[1] >= end_cutoff
]


if recent_gaps:

    for diff, before, after in sorted(
        recent_gaps,
        key=lambda x: x[1]
    ):

        print(
            f"{pd.to_datetime(before, unit='ms', utc=True)}"
            f" -> "
            f"{pd.to_datetime(after, unit='ms', utc=True)}"
            f" | "
            f"{diff / 60_000:.2f} minutes"
        )

else:

    print(
        "No gaps after 2026-09-21 00:00 UTC"
    )


print("\nDone.")
