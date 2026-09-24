from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("data/raw/usd_index")
OUTPUT = Path("data/processed/usd_index_m1_features.csv")

FILES = sorted(ROOT.glob("dollaridxusd-m1-bid-*.csv"))

if not FILES:
    raise SystemExit("No USD Index raw files found.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "return_1m",
    "return_5m",
    "return_15m",
    "return_30m",
    "return_60m",
    "volatility_15m",
    "volatility_60m",
    "distance_sma_60",
    "distance_sma_240",
]

first_write = True
total_rows = 0
total_feature_rows = 0


def process_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["timestamp", "open", "high", "low", "close"],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # Session segmentation
    # Reset all lookbacks when timestamps are not exactly
    # one minute apart.
    # --------------------------------------------------------
    gap = df["timestamp"].diff() != pd.Timedelta(minutes=1)
    segment = gap.cumsum()

    close = df["close"].astype("float64")

    # --------------------------------------------------------
    # Backward returns
    # --------------------------------------------------------
    for h in [1, 5, 15, 30, 60]:
        previous = close.groupby(segment, sort=False).shift(h)
        df[f"return_{h}m"] = close / previous - 1.0

    # --------------------------------------------------------
    # Volatility of 1-minute returns
    # pandas rolling std uses ddof=1, matching the XAUUSD
    # feature methodology.
    # --------------------------------------------------------
    r1 = df["return_1m"]

    df["volatility_15m"] = (
        r1.groupby(segment, sort=False)
        .rolling(15, min_periods=15)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["volatility_60m"] = (
        r1.groupby(segment, sort=False)
        .rolling(60, min_periods=60)
        .std()
        .reset_index(level=0, drop=True)
    )

    # --------------------------------------------------------
    # SMA distance
    # --------------------------------------------------------
    sma60 = (
        close.groupby(segment, sort=False)
        .rolling(60, min_periods=60)
        .mean()
        .reset_index(level=0, drop=True)
    )

    sma240 = (
        close.groupby(segment, sort=False)
        .rolling(240, min_periods=240)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["distance_sma_60"] = close / sma60 - 1.0
    df["distance_sma_240"] = close / sma240 - 1.0

    # --------------------------------------------------------
    # Keep only usable rows.
    # --------------------------------------------------------
    keep = ["timestamp", "open", "high", "low", "close"] + FEATURES

    out = df[keep].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=FEATURES).copy()

    # float32 for lower memory / storage footprint
    for col in FEATURES:
        out[col] = out[col].astype("float32")

    return out


print("=" * 70)
print("USD INDEX M1 FEATURE ENGINEERING")
print("=" * 70)
print(f"Input files: {len(FILES)}")
print(f"Output: {OUTPUT}")

for path in FILES:
    out = process_file(path)

    total_rows += len(pd.read_csv(path, usecols=["timestamp"]))
    total_feature_rows += len(out)

    out.to_csv(
        OUTPUT,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
    )

    first_write = False

    print(
        f"{path.name}: "
        f"feature rows={len(out):,}"
    )

print("\n" + "=" * 70)
print("USD FEATURE DATASET COMPLETE")
print("=" * 70)
print(f"Raw rows processed:     {total_rows:,}")
print(f"Feature rows saved:     {total_feature_rows:,}")
print(f"Feature columns:        {len(FEATURES)}")
print(f"Saved to:               {OUTPUT}")
print("=" * 70)
