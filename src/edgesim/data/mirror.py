"""Compact Parquet copies of the raw datasets, kept on this repository's `data-mirror` branch.

Why: the original hosts are slow or throttled (Google Drive limits bursts of downloads), and sandbox
files do not persist between sessions. The branch has its own history, so `main` stays small, and its
files download quickly over HTTPS from raw.githubusercontent.com.

Both datasets are licensed CC BY 4.0, which allows redistribution with attribution:
  * CityPulse pollution stream (Aarhus, generated), http://iot.ee.surrey.ac.uk:8080/datasets.html
  * AirDelhi (Chauhan, Ranu, Sen, Patel, Batra, NeurIPS 2023 D&B), http://www.cse.iitd.ac.in/pollutiondata
The copies only change the file format: same rows and values, columns renamed to the common schema
(sensor_id, ts, lat, lon, pollutants...), and AirDelhi's per-row random `uid` column dropped.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

REPO = "KeeratMalhotra/Cloud-Project"
BRANCH = "data-mirror"
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
CHECKSUMS = "SHA256SUMS.txt"

ASSETS = {
    "citypulse": ["citypulse_aarhus_2014.parquet"],
    "airdelhi": ["airdelhi_2020-11.parquet", "airdelhi_2020-12.parquet", "airdelhi_2021-01.parquet"],
}

log = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write with dictionary-encoded sensor ids and zstd compression, sorted by sensor and time."""
    df = df.sort_values(["sensor_id", "ts"], kind="mergesort").reset_index(drop=True)
    df = df.assign(sensor_id=df["sensor_id"].astype("category"), ts=df["ts"].astype("datetime64[s]"))
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd", compression_level=9, use_dictionary=True)


def _download(url: str, dst: Path, timeout_s: float = 300.0) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=(15, timeout_s)) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(dst)


def fetch(dataset: str, dest: Path) -> list[Path]:
    """Download (if needed) and checksum-verify the mirror files of one dataset. Returns their paths."""
    dest.mkdir(parents=True, exist_ok=True)
    sums_path = dest / CHECKSUMS
    _download(f"{BASE_URL}/{CHECKSUMS}", sums_path)
    expected = dict(line.split()[::-1] for line in sums_path.read_text().splitlines() if line.strip())
    paths = []
    for name in ASSETS[dataset]:
        dst = dest / name
        if not (dst.exists() and sha256(dst) == expected[name]):
            log.info("fetching %s/%s", BASE_URL, name)
            _download(f"{BASE_URL}/{name}", dst)
            if sha256(dst) != expected[name]:
                dst.unlink()
                raise OSError(f"checksum mismatch for {name}")
        paths.append(dst)
    return paths


def load(paths: list[Path]) -> pd.DataFrame:
    """Read mirror files back into the common readings schema (ts as datetime64[ns], sensor_id as string)."""
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["sensor_id"] = df["sensor_id"].astype("string")
    df["ts"] = df["ts"].astype("datetime64[ns]")
    return df
