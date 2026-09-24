from pathlib import Path
import pandas as pd

ROOT = Path.home() / "xauusd-ai-grid" / "data" / "raw" / "usd_index"
FILES = sorted(ROOT.glob("dollaridxusd-m1-bid-*.csv"))

print("=" * 70)
print("USD INDEX RAW DATA VALIDATION")
print("=" * 70)
print(f"Files found: {len(FILES)}")

if not FILES:
    raise SystemExit("No USD Index CSV files found.")

required = {"timestamp", "open", "high", "low", "close"}

total_rows = 0
total_missing = 0
total_duplicates = 0
total_invalid_ohlc = 0
total_nonpositive = 0
total_gaps_gt_1m = 0
total_flat = 0

prev_last_ts = None

for path in FILES:
    df = pd.read_csv(path)

    if not required.issubset(df.columns):
        raise ValueError(
            f"{path.name}: missing columns "
            f"{sorted(required - set(df.columns))}"
        )

    n = len(df)
    total_rows += n

    missing = int(df[list(required)].isna().sum().sum())
    dup = int(df["timestamp"].duplicated().sum())

    invalid_ohlc = int(
        (
            (df["high"] < df[["open", "close", "low"]].max(axis=1))
            | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        ).sum()
    )

    nonpositive = int(
        (df[["open", "high", "low", "close"]] <= 0)
        .any(axis=1)
        .sum()
    )

    flat = int(
        (df[["open", "high", "low", "close"]].nunique(axis=1) == 1).sum()
    )

    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    diffs = ts.diff().dropna()
    gaps = int((diffs > pd.Timedelta(minutes=1)).sum())

    if prev_last_ts is not None:
        cross_gap = ts.iloc[0] - prev_last_ts

        if cross_gap > pd.Timedelta(minutes=1):
            gaps += 1
        elif cross_gap <= pd.Timedelta(0):
            print(
                f"WARNING: non-forward file boundary: "
                f"{path.name} ({cross_gap})"
            )

    prev_last_ts = ts.iloc[-1]

    total_missing += missing
    total_duplicates += dup
    total_invalid_ohlc += invalid_ohlc
    total_nonpositive += nonpositive
    total_gaps_gt_1m += gaps
    total_flat += flat

    print(f"\n{path.name}")
    print(f"  Rows: {n:,}")
    print(f"  Range: {ts.iloc[0]} -> {ts.iloc[-1]}")
    print(f"  Missing: {missing}")
    print(f"  Duplicate timestamps: {dup}")
    print(f"  Invalid OHLC: {invalid_ohlc}")
    print(f"  Non-positive prices: {nonpositive}")
    print(f"  Gaps > 1 minute: {gaps}")
    print(f"  Flat candles: {flat:,}")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Total rows: {total_rows:,}")
print(f"Total missing values: {total_missing:,}")
print(f"Total duplicate timestamps: {total_duplicates:,}")
print(f"Total invalid OHLC rows: {total_invalid_ohlc:,}")
print(f"Total non-positive prices: {total_nonpositive:,}")
print(f"Total gaps > 1 minute: {total_gaps_gt_1m:,}")
print(f"Total flat candles: {total_flat:,}")

status = (
    total_missing == 0
    and total_duplicates == 0
    and total_invalid_ohlc == 0
    and total_nonpositive == 0
)

print(f"\nDATASET VALIDATION: {'PASS' if status else 'FAIL'}")
