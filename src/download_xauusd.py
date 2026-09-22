import subprocess
from pathlib import Path

OUTPUT_DIR = Path("data/raw/dukascopy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(2019, 2027)

for year in YEARS:
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    expected = OUTPUT_DIR / f"xauusd-m1-bid-{start}-{end}.csv"

    if expected.exists():
        print(f"[SKIP] {year}: already downloaded")
        continue

    print(f"\n[DOWNLOAD] XAUUSD {year}")

    command = [
        "npx",
        "dukascopy-node",
        "-i", "xauusd",
        "-from", start,
        "-to", end,
        "-t", "m1",
        "-f", "csv",
        "-dir", str(OUTPUT_DIR),
    ]

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"[FAILED] {year}")
    else:
        print(f"[DONE] {year}")
