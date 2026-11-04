"""Dataset B: AirDelhi bus-mounted PM sensors (New Delhi, Nov 2020 - Jan 2021).

Source: S. K. Chauhan, S. Ranu, R. Sen, Z. B. Patel, N. Batra, "AirDelhi: Fine-Grained Spatio-Temporal
Particulate Matter Dataset From Delhi For ML based Modeling", NeurIPS 2023 Datasets and Benchmarks.
Portal: http://www.cse.iitd.ac.in/pollutiondata (links to the public Google Drive folder below).
License: CC BY 4.0.

Each daily file `YYYY-MM-DD_all.csv` has columns:
    <index>, uid, dateTime (local time, +05:30), deviceId, lat, long, pm1_0, pm2_5, pm10
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1PSylFDkLXq9w9tZv6_y0UaHSpS14qL0U?usp=sharing"
POLLUTANTS = ["pm1_0", "pm2_5", "pm10"]
EXPECTED_HEADER = ["uid", "dateTime", "deviceId", "lat", "long", "pm1_0", "pm2_5", "pm10"]

log = logging.getLogger(__name__)


def list_remote_files() -> list[tuple[str, str]]:
    """Return (file name, Drive file id) for every daily CSV in the public folder, sorted by name."""
    import gdown  # imported lazily: only needed for downloads

    entries = gdown.download_folder(DRIVE_FOLDER_URL, skip_download=True, quiet=True)
    return sorted((e.path, e.id) for e in entries if e.path.endswith("_all.csv"))


def _header_ok(path: Path) -> bool:
    with open(path, encoding="utf-8") as fh:
        cols = [c.strip() for c in fh.readline().split(",")]
    return all(c in cols for c in EXPECTED_HEADER)


DIRECT_URL = "https://drive.usercontent.google.com/download"


def _direct_download(file_id: str, dst: Path, timeout_s: float = 120.0) -> None:
    """Fetch one public Drive file over plain HTTPS (gdown's URL lookup gets throttled; this endpoint does not)."""
    import requests

    tmp = dst.with_suffix(".part")
    params = {"id": file_id, "export": "download", "confirm": "t"}
    with requests.get(DIRECT_URL, params=params, stream=True, timeout=(15, timeout_s)) as resp:
        resp.raise_for_status()
        if "text/html" in resp.headers.get("content-type", ""):
            raise OSError("Drive returned an HTML page instead of the file (quota or permission)")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(dst)


def _fetch_one(name: str, file_id: str, raw_dir: Path, retries: int) -> bool:
    dst = raw_dir / name
    for attempt in range(1, retries + 1):
        try:
            _direct_download(file_id, dst)
            if not _header_ok(dst):
                raise OSError("unexpected CSV header")
            log.info("downloaded %s (%.1f MB)", name, dst.stat().st_size / 1e6)
            return True
        except Exception as exc:
            log.warning("attempt %d/%d failed for %s: %s", attempt, retries, name, str(exc).splitlines()[0])
            for p in (dst, dst.with_suffix(".part")):
                if p.exists():
                    p.unlink()
            time.sleep(2.0 * attempt)
    return False


def download(raw_dir: Path, days: list[str] | None = None, retries: int = 3, workers: int = 4) -> list[Path]:
    """Download daily CSVs into raw_dir in parallel; existing valid files are kept and the call is resumable.

    `days` filters by 'YYYY-MM-DD'. Returns the paths of all valid files present afterwards.
    """
    from concurrent.futures import ThreadPoolExecutor

    raw_dir.mkdir(parents=True, exist_ok=True)
    for stale in raw_dir.glob("*.part"):
        stale.unlink()
    wanted = list_remote_files()
    if days is not None:
        keep = set(days)
        wanted = [(n, i) for n, i in wanted if n[:10] in keep]
    todo = [(n, i) for n, i in wanted
            if not ((raw_dir / n).exists() and (raw_dir / n).stat().st_size > 0 and _header_ok(raw_dir / n))]
    log.info("%d of %d files already present, %d to download", len(wanted) - len(todo), len(wanted), len(todo))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ok = list(pool.map(lambda item: _fetch_one(item[0], item[1], raw_dir, retries), todo))
    failed = [n for (n, _), good in zip(todo, ok) if not good]
    if failed:
        log.warning("%d files failed; rerun to resume: %s", len(failed), ", ".join(failed))
    return [raw_dir / n for n, _ in wanted if (raw_dir / n).exists()]


def load(raw_dir: Path, days: list[str] | None = None) -> pd.DataFrame:
    """Load daily CSVs into the common readings schema: sensor_id, ts, lat, lon, <pollutants>."""
    files = sorted(raw_dir.glob("*_all.csv"))
    if days is not None:
        keep = set(days)
        files = [f for f in files if f.name[:10] in keep]
    if not files:
        raise FileNotFoundError(f"no AirDelhi files in {raw_dir}; run the download step first")
    frames = []
    for f in files:
        d = pd.read_csv(
            f,
            usecols=["dateTime", "deviceId", "lat", "long", *POLLUTANTS],
            dtype={"deviceId": "string", "lat": "float64", "long": "float64",
                   "pm1_0": "float32", "pm2_5": "float32", "pm10": "float32"},
        )
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # Timestamps are local (IST, +05:30). Keep them as naive local time: the offset is constant.
    df["ts"] = pd.to_datetime(df["dateTime"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")
    df = df.rename(columns={"deviceId": "sensor_id", "long": "lon"}).drop(columns=["dateTime"])
    return df[["sensor_id", "ts", "lat", "lon", *POLLUTANTS]]
