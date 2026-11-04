"""Convert the raw downloads into the compact Parquet mirror files kept on the `data-mirror` branch.

Usage:
    python scripts/download_data.py --dataset citypulse --source original
    python scripts/download_data.py --dataset airdelhi --source original
    python scripts/build_data_mirror.py          # writes data/mirror/*.parquet and SHA256SUMS.txt

Only maintainers need this: they commit data/mirror/* to the `data-mirror` branch (see its README).
Everyone else gets the files with `python scripts/download_data.py --dataset <name>`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from edgesim.data import airdelhi, citypulse, mirror

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = ROOT / "data" / "mirror"
    out.mkdir(parents=True, exist_ok=True)

    cp = citypulse.load(ROOT / "data" / "raw" / "citypulse")
    for p in citypulse.POLLUTANTS:  # index values are integers
        cp[p] = cp[p].round().astype("int16")
    mirror.write_parquet(cp, out / mirror.ASSETS["citypulse"][0])
    logging.info("citypulse: %d rows", len(cp))

    ad = airdelhi.load(ROOT / "data" / "raw" / "airdelhi")
    month = ad["ts"].dt.strftime("%Y-%m")
    for name in mirror.ASSETS["airdelhi"]:
        key = name.removeprefix("airdelhi_").removesuffix(".parquet")
        part = ad[month == key]
        mirror.write_parquet(part, out / name)
        logging.info("airdelhi %s: %d rows", key, len(part))
    logging.info("airdelhi total: %d rows", len(ad))

    files = [n for names in mirror.ASSETS.values() for n in names]
    lines = [f"{mirror.sha256(out / n)}  {n}" for n in files]
    (out / mirror.CHECKSUMS).write_text("\n".join(lines) + "\n")
    for n in files:
        logging.info("%s %.1f MB", n, (out / n).stat().st_size / 1e6)


if __name__ == "__main__":
    main()
