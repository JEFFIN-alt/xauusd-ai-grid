from pathlib import Path
import pandas as pd


XAU_FILE = Path(
    "data/processed/xauusd_m1_ml_features_v4_session.csv"
)

USD_FILE = Path(
    "data/processed/usd_index_m1_features.csv"
)

OUTPUT = Path(
    "data/processed/xauusd_usd_combined_features_2019_2025.csv"
)

USD_FEATURES = [
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


print("=" * 70)
print("XAUUSD + USD INDEX EXACT-MATCH DATASET")
print("=" * 70)

# ------------------------------------------------------------
# Load USD features.
# 301 MB CSV but only ~1.76M rows. Keep required columns.
# ------------------------------------------------------------
usd = pd.read_csv(
    USD_FILE,
    usecols=["timestamp"] + USD_FEATURES,
)

usd["timestamp"] = pd.to_datetime(
    usd["timestamp"],
    utc=True,
)

# ------------------------------------------------------------
# Load XAU V4 features.
# ------------------------------------------------------------
xau = pd.read_csv(
    XAU_FILE,
)

# XAU timestamp is epoch milliseconds.
xau["timestamp"] = pd.to_datetime(
    xau["timestamp"],
    unit="ms",
    utc=True,
)

# ------------------------------------------------------------
# Development period only.
# 2026 deliberately excluded.
# ------------------------------------------------------------
xau = xau.loc[
    xau["timestamp"].dt.year <= 2025
].copy()

# ------------------------------------------------------------
# Exact timestamp inner join.
# ------------------------------------------------------------
combined = xau.merge(
    usd,
    on="timestamp",
    how="inner",
    suffixes=("", "_usd"),
)

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------
combined = combined.sort_values(
    "timestamp"
).reset_index(drop=True)

if combined["timestamp"].duplicated().any():
    raise RuntimeError(
        "Duplicate timestamps after merge."
    )

print(
    f"XAU development rows: {len(xau):,}"
)

print(
    f"USD feature rows:     {len(usd):,}"
)

print(
    f"Combined rows:        {len(combined):,}"
)

coverage = (
    len(combined) / len(xau) * 100
)

print(
    f"Exact-match coverage: {coverage:.3f}%"
)

print(
    f"Range: {combined['timestamp'].iloc[0]}"
    f" -> {combined['timestamp'].iloc[-1]}"
)

# ------------------------------------------------------------
# Year counts
# ------------------------------------------------------------
print("\nYearly combined rows:")

year_counts = (
    combined["timestamp"]
    .dt.year
    .value_counts()
    .sort_index()
)

for year, count in year_counts.items():
    print(f"{year}: {count:,}")

# ------------------------------------------------------------
# Save timestamp back in the same epoch-ms format as XAU V4.
# ------------------------------------------------------------
combined["timestamp"] = (
    combined["timestamp"]
    .astype("int64")
    // 1_000_000
)

combined.to_csv(
    OUTPUT,
    index=False,
)

print("\n" + "=" * 70)
print("COMBINED DATASET COMPLETE")
print("=" * 70)
print(f"Saved to: {OUTPUT}")
print(f"Rows: {len(combined):,}")
print(f"Columns: {len(combined.columns):,}")
