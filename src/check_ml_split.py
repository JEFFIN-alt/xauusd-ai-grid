import pandas as pd

FILE = "data/processed/xauusd_m1_15m_features.csv"

TRAIN_START = "2019-01-01"
TRAIN_END = "2024-12-31 23:59:59"

VALID_START = "2025-01-01"
VALID_END = "2025-12-31 23:59:59"

TEST_START = "2026-01-01"
TEST_END = "2026-12-31 23:59:59"

rows = {
    "train": 0,
    "validation": 0,
    "test": 0,
    "outside": 0,
}

first_time = None
last_time = None

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

    if first_time is None:
        first_time = dt.iloc[0]

    last_time = dt.iloc[-1]

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
print("ML TIME SPLIT CHECK")
print("=" * 70)

print(f"\nDataset first timestamp: {first_time}")
print(f"Dataset last timestamp:  {last_time}")

print("\nRows by split:")
print(f"Train:      {rows['train']:,}")
print(f"Validation: {rows['validation']:,}")
print(f"Test:       {rows['test']:,}")
print(f"Outside:    {rows['outside']:,}")

total = sum(rows.values())

print(f"\nTotal counted: {total:,}")

print("\nExpected:")
print("Train      = 2019-01-01 → 2024-12-31")
print("Validation = 2025-01-01 → 2025-12-31")
print("Test       = 2026-01-01 → 2026-09-22")

print("=" * 70)
