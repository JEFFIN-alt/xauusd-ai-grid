from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RAW_DIR = Path("data/raw/dukascopy")
OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = OUTPUT_DIR / "xauusd_m1_2019_2026.csv"


# ============================================================
# LOAD FILES
# ============================================================

files = sorted(RAW_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(
        f"No CSV files found in {RAW_DIR}"
    )

print("=" * 70)
print("XAUUSD DATASET BUILD")
print("=" * 70)

print(f"\nFiles found: {len(files)}")

for file in files:
    print(f"  {file.name}")


# ============================================================
# READ DATA
# ============================================================

frames = []

for file in files:

    print(f"\nLoading: {file.name}")

    df = pd.read_csv(file)

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{file.name} missing columns: {missing}"
        )

    frames.append(
        df[required].copy()
    )

    print(f"  Rows: {len(df):,}")


# ============================================================
# COMBINE
# ============================================================

print("\nCombining files...")

df = pd.concat(
    frames,
    ignore_index=True
)

print(
    f"Combined rows: {len(df):,}"
)


# ============================================================
# TIMESTAMP
# ============================================================

print("\nConverting timestamps...")

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    utc=True
)


# ============================================================
# SORT
# ============================================================

print("Sorting chronologically...")

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


# ============================================================
# REMOVE DUPLICATE TIMESTAMPS
# ============================================================

duplicates = df["datetime"].duplicated().sum()

print(
    f"Duplicate timestamps found: {duplicates:,}"
)

if duplicates > 0:

    df = df.drop_duplicates(
        subset="datetime",
        keep="first"
    ).reset_index(drop=True)


# ============================================================
# NUMERIC TYPES
# ============================================================

for column in [
    "open",
    "high",
    "low",
    "close",
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# DATA QUALITY FLAGS
# ============================================================

print("\nCreating quality flags...")


# Flat candle
df["is_flat"] = (
    (df["open"] == df["high"])
    & (df["high"] == df["low"])
    & (df["low"] == df["close"])
)


# OHLC validity
df["is_valid_ohlc"] = (
    (df["high"] >= df["open"])
    & (df["high"] >= df["close"])
    & (df["low"] <= df["open"])
    & (df["low"] <= df["close"])
    & (df["high"] >= df["low"])
)


# Positive prices
df["is_positive_price"] = (
    (df["open"] > 0)
    & (df["high"] > 0)
    & (df["low"] > 0)
    & (df["close"] > 0)
)


# Time gap from previous observation
df["time_gap"] = (
    df["datetime"].diff()
)


# Gap larger than one minute
df["has_gap"] = (
    df["time_gap"] > pd.Timedelta(minutes=1)
)


# ============================================================
# MARKET DATE INFORMATION
# ============================================================

df["date"] = df["datetime"].dt.date
df["day_of_week"] = df["datetime"].dt.dayofweek
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute


# ============================================================
# PRICE CHANGE
# ============================================================

df["return_1m"] = (
    df["close"].pct_change(fill_method=None)
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

columns = [
    "timestamp",
    "datetime",

    "open",
    "high",
    "low",
    "close",

    "return_1m",

    "time_gap",
    "has_gap",
    "is_flat",
    "is_valid_ohlc",
    "is_positive_price",

    "date",
    "day_of_week",
    "hour",
    "minute",
]

df = df[columns]


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET CHECK")
print("=" * 70)

print(
    f"Rows: {len(df):,}"
)

print(
    f"First: {df['datetime'].iloc[0]}"
)

print(
    f"Last:  {df['datetime'].iloc[-1]}"
)

print(
    f"Duplicate timestamps: "
    f"{df['datetime'].duplicated().sum():,}"
)

print(
    f"Missing values: "
    f"{df.isna().sum().sum():,}"
)

print(
    f"Flat candles: "
    f"{df['is_flat'].sum():,}"
)

print(
    f"Gaps > 1 minute: "
    f"{df['has_gap'].sum():,}"
)

print(
    f"Invalid OHLC rows: "
    f"{(~df['is_valid_ohlc']).sum():,}"
)

print(
    f"Non-positive price rows: "
    f"{(~df['is_positive_price']).sum():,}"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(
    f"\nSaving to:\n{OUTPUT_FILE}"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("BUILD COMPLETE")
print("=" * 70)

print(
    f"\nSaved rows: {len(df):,}"
)

print(
    f"Output: {OUTPUT_FILE}"
)
