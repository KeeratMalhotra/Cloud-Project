"""Get the data for one dataset.

Default source is the repository's `data-mirror` branch (fast, checksum-verified Parquet files):
    python scripts/download_data.py --dataset citypulse
    python scripts/download_data.py --dataset airdelhi

The original hosts are only needed to rebuild the mirror (see scripts/build_data_mirror.py):
    python scripts/download_data.py --dataset citypulse --source original
    python scripts/download_data.py --dataset airdelhi --source original [--days 2020-12-01 ...]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from edgesim.data import airdelhi, citypulse, mirror

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["citypulse", "airdelhi"], required=True)
    ap.add_argument("--source", choices=["mirror", "original"], default="mirror")
    ap.add_argument("--days", nargs="*", default=None, help="original AirDelhi only: subset of days (YYYY-MM-DD)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.source == "mirror":
        paths = mirror.fetch(args.dataset, ROOT / "data" / "mirror")
        logging.info("%s mirror ready: %s", args.dataset, ", ".join(p.name for p in paths))
        return
    raw_dir = ROOT / "data" / "raw" / args.dataset
    if args.dataset == "citypulse":
        out = citypulse.download(raw_dir)
        logging.info("CityPulse ready in %s", out)
    else:
        paths = airdelhi.download(raw_dir, days=args.days)
        logging.info("AirDelhi: %d daily files ready in %s", len(paths), raw_dir)


if __name__ == "__main__":
    main()
