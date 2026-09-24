from pathlib import Path
import pandas as pd


XAU_FILE = Path(
    "data/processed/xauusd_m1_ml_features_v4_session.csv"
)

USD_FILE = Path(
    "data/processed/usd_index_m1_features.csv"
)

CHUNK_SIZE = 100_000

print("=" * 70)
print("XAUUSD ↔ USD INDEX EXACT TIMESTAMP ALIGNMENT TEST")
print("=" * 70)

if not XAU_FILE.exists():
    raise SystemExit(f"Missing XAU file: {XAU_FILE}")

if not USD_FILE.exists():
    raise SystemExit(f"Missing USD file: {USD_FILE}")


# ============================================================
# Load USD timestamps as UTC datetimes
# ============================================================

usd_parts = []

for chunk in pd.read_csv(
    USD_FILE,
    usecols=["timestamp"],
    chunksize=CHUNK_SIZE,
):
    usd_ts = pd.to_datetime(
        chunk["timestamp"],
        utc=True,
    )

    usd_parts.append(usd_ts)

usd_index = pd.DatetimeIndex(
    pd.concat(usd_parts, ignore_index=True)
).drop_duplicates().sort_values()

print(f"USD feature timestamps: {len(usd_index):,}")


# ============================================================
# Scan XAU timestamps
# ============================================================

total_xau = 0
matched = 0

year_total = {}
year_matched = {}

first_xau = None
last_xau = None

reader = pd.read_csv(
    XAU_FILE,
    usecols=["timestamp"],
    chunksize=CHUNK_SIZE,
)

for chunk in reader:

    # XAU timestamp is epoch MILLISECONDS
    xau_ts = pd.to_datetime(
        chunk["timestamp"],
        unit="ms",
        utc=True,
    )

    if first_xau is None:
        first_xau = xau_ts.iloc[0]

    last_xau = xau_ts.iloc[-1]

    exact = xau_ts.isin(usd_index)

    total_xau += len(xau_ts)
    matched += int(exact.sum())

    years = xau_ts.dt.year.to_numpy()

    for year in sorted(set(years)):

        mask = years == year

        year_total[year] = (
            year_total.get(year, 0) + int(mask.sum())
        )

        year_matched[year] = (
            year_matched.get(year, 0)
            + int(exact.to_numpy()[mask].sum())
        )


# ============================================================
# Results
# ============================================================

coverage = (
    matched / total_xau * 100
    if total_xau
    else 0.0
)

print("\n" + "=" * 70)
print("OVERALL")
print("=" * 70)

print(f"XAU rows:             {total_xau:,}")
print(f"Exact USD matches:    {matched:,}")
print(f"No exact USD match:   {total_xau - matched:,}")
print(f"Exact coverage:       {coverage:.3f}%")

print(
    "XAU range:            "
    f"{first_xau} -> {last_xau}"
)

print("\n" + "=" * 70)
print("YEAR-BY-YEAR COVERAGE")
print("=" * 70)

for year in sorted(year_total):

    total = year_total[year]
    hit = year_matched[year]
    pct = hit / total * 100

    print(
        f"{year}: "
        f"XAU={total:,} | "
        f"matched={hit:,} | "
        f"coverage={pct:.3f}%"
    )

print("=" * 70)
