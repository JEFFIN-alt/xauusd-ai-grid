import pandas as pd

FILE = "data/processed/xauusd_m1_15m_features.csv"

# Split boundaries
TRAIN_START = pd.Timestamp("2019-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")

VALID_START = pd.Timestamp("2025-01-01", tz="UTC")
VALID_END = pd.Timestamp("2025-12-31 23:45:00", tz="UTC")

TEST_START = pd.Timestamp("2026-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-12-31 23:59:59", tz="UTC")

rows = {
    "train": 0,
    "validation": 0,
    "test": 0,
    "outside": 0,
}

for chunk in pd.read_csv(
    FILE,
    usecols=["timestamp"],
    chunksize=200_000,
):
    dt = pd.to_datetime(
        chunk["timestamp"],
        unit="ms",
        utc=True,
    )

    rows["train"] += (
        (dt >= TRAIN_START) &
        (dt <= TRAIN_END)
    ).sum()

    rows["validation"] += (
        (dt >= VALID_START) &
        (dt <= VALID_END)
    ).sum()

    rows["test"] += (
        (dt >= TEST_START) &
        (dt <= TEST_END)
    ).sum()

    rows["outside"] += (
        (dt < TRAIN_START) |
        ((dt > TRAIN_END) & (dt < VALID_START)) |
        ((dt > VALID_END) & (dt < TEST_START)) |
        (dt > TEST_END)
    ).sum()


print("=" * 70)
print("PURGED ML TIME SPLIT CHECK")
print("=" * 70)

print("\nRows by split:")
print(f"Train:      {rows['train']:,}")
print(f"Validation: {rows['validation']:,}")
print(f"Test:       {rows['test']:,}")
print(f"Outside:    {rows['outside']:,}")

total = sum(rows.values())

print(f"\nTotal counted: {total:,}")

print("\nSplit boundaries:")
print("Train      = 2019-01-01 → 2024-12-31 23:45")
print("Validation = 2025-01-01 → 2025-12-31 23:45")
print("Test       = 2026-01-01 → available data")

print("\nPurged:")
print("2024-12-31 23:46 → 2024-12-31 23:59")
print("2025-12-31 23:46 → 2025-12-31 23:59")

print("=" * 70)
