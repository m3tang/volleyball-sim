"""Download pre-scraped NCAA women's volleyball CSVs from the ncaavolleyballr project.

Source: https://jeffreyrstevens.github.io/ncaavolleyballr/articles/data.html (MIT license).
Files already present are skipped, so the script is safe to re-run.

Usage:
    python pipeline/download.py                      # D1, 2020-2025, all levels
    python pipeline/download.py --years 2024 2025 --levels pbp
"""
import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://media.githubusercontent.com/media/JeffreyRStevens/ncaavolleyballr/refs/heads/main/data-csv"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
LEVELS = ["pbp", "teammatch", "teamseason", "playermatch", "playerseason"]
YEARS = list(range(2020, 2026))


def download(name: str) -> None:
    dest = RAW_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip  {name} (already downloaded)")
        return
    tmp = dest.with_suffix(".part")
    print(f"get   {name} ...", flush=True)
    with urllib.request.urlopen(f"{BASE}/{name}") as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    # Git LFS returns a tiny pointer file when the object is missing.
    if tmp.stat().st_size < 1024 and tmp.read_text(errors="ignore").startswith("version https://git-lfs"):
        tmp.unlink()
        raise RuntimeError(f"{name}: got a Git LFS pointer, not data")
    tmp.rename(dest)
    print(f"done  {name} ({dest.stat().st_size / 1e6:.0f} MB)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=YEARS)
    p.add_argument("--levels", nargs="+", choices=LEVELS, default=LEVELS)
    p.add_argument("--division", type=int, default=1, choices=[1, 2, 3])
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for level in args.levels:
        for year in args.years:
            name = f"wvb_{level}_div{args.division}_{year}.csv"
            try:
                download(name)
            except Exception as e:  # keep going; report at the end
                print(f"FAIL  {name}: {e}", file=sys.stderr)
                failed.append(name)
    if failed:
        print(f"{len(failed)} file(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
