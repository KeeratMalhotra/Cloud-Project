"""Dataset A: CityPulse pollution measurements (the data EdgePlus used).

The 449-site, Aug-Oct 2014 pollution stream on the CityPulse site is located in Aarhus, Denmark, and is
generated (a bounded random walk of index values, one step every 5 minutes).

Each file `pollutionData<id>.csv` has columns:
    ozone, particullate_matter, carbon_monoxide, sulfure_dioxide, nitrogen_dioxide, longitude, latitude, timestamp
"""
from __future__ import annotations

import logging
import tarfile
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = (
    "http://iot.ee.surrey.ac.uk:8080/datasets/pollution/"
    "citypulse_pollution_csv_data_aarhus_aug_oct_2014.tar.gz"
)
POLLUTANTS = ["ozone", "particulate_matter", "carbon_monoxide", "sulfur_dioxide", "nitrogen_dioxide"]
_RENAME = {
    "particullate_matter": "particulate_matter",
    "sulfure_dioxide": "sulfur_dioxide",
    "longitude": "lon",
    "latitude": "lat",
}

log = logging.getLogger(__name__)


def download(raw_dir: Path) -> Path:
    """Download and extract the CSV archive into raw_dir/pollution/. Returns the extracted folder."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "pollution"
    if out.exists() and len(list(out.glob("pollutionData*.csv"))) == 449:
        return out
    archive = raw_dir / "pollution_csv.tar.gz"
    if not archive.exists():
        log.info("downloading %s", ARCHIVE_URL)
        with requests.get(ARCHIVE_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            tmp = archive.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            tmp.rename(archive)
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("pollution/") and m.name.endswith(".csv")]
        tar.extractall(raw_dir, members=members, filter="data")
    n = len(list(out.glob("pollutionData*.csv")))
    if n != 449:
        raise RuntimeError(f"expected 449 sensor files, found {n}")
    return out


def load(raw_dir: Path) -> pd.DataFrame:
    """Load all sensor files into the common readings schema: sensor_id, ts, lat, lon, <pollutants>."""
    folder = raw_dir / "pollution"
    files = sorted(folder.glob("pollutionData*.csv"))
    if not files:
        raise FileNotFoundError(f"no CityPulse files in {folder}; run the download step first")
    frames = []
    for f in files:
        d = pd.read_csv(f).rename(columns=_RENAME)
        d["sensor_id"] = f.stem.replace("pollutionData", "")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df["sensor_id"] = df["sensor_id"].astype("string")
    for p in POLLUTANTS:
        df[p] = df[p].astype("float32")
    return df[["sensor_id", "ts", "lat", "lon", *POLLUTANTS]]
