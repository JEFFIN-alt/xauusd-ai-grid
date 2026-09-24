import pandas as pd

FILE = "data/processed/xauusd_m1_2019_2026.csv"

CHUNK_SIZE = 200_000

previous = None

found = []

for chunk in pd.read_csv(
    FILE,
    usecols=["timestamp"],
    chunksize=CHUNK_SIZE,
):

    timestamps = chunk["timestamp"].to_numpy()

    for ts in timestamps:

        ts = int(ts)

        if previous is not None:

            diff = ts - previous

            if diff < 60_000:

                found.append(
                    (previous, ts, diff)
                )

        previous = ts


print("=" * 70)
print("SHORT INTERVAL AUDIT")
print("=" * 70)

print(
    f"\nShort intervals found: "
    f"{len(found)}"
)

for before, after, diff in found:

    before_dt = pd.to_datetime(
        before,
        unit="ms",
        utc=True
    )

    after_dt = pd.to_datetime(
        after,
        unit="ms",
        utc=True
    )

    print(
        f"\nBefore: {before_dt}"
    )

    print(
        f"After:  {after_dt}"
    )

    print(
        f"Difference: {diff} ms"
    )

print("\nDone.")
