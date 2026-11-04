# Dataset mirror (branch `data-mirror`)

This branch only holds data. It has its own history, separate from `main`, so the code history stays small.
To get the code without this data, clone with `git clone --single-branch --branch main <url>`.

The pipeline fetches these files over HTTPS and checks them against `SHA256SUMS.txt`
(`edgesim.data.mirror.fetch`). The original hosts are slow or throttled, and sandbox files do not persist
between sessions, so this copy keeps the experiments reproducible.

| File | Rows | Content |
|---|---|---|
| `citypulse_aarhus_2014.parquet` | 7,887,974 | CityPulse pollution stream, 449 sites, 1 Aug to 1 Oct 2014, 5-minute steps (Dataset A, the data EdgePlus used) |
| `airdelhi_2020-11.parquet` | 2,288,327 | AirDelhi, November 2020 (Dataset B) |
| `airdelhi_2020-12.parquet` | 4,799,292 | AirDelhi, December 2020 |
| `airdelhi_2021-01.parquet` | 5,379,622 | AirDelhi, 1 to 30 January 2021 |

Schema: `sensor_id, ts, lat, lon` plus pollutant columns. Only the file format changed from the originals:
same rows and values, columns renamed to a common schema, and AirDelhi's per-row random `uid` column dropped.
CityPulse index values are stored as int16.

## Sources and licenses

Both datasets are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- **CityPulse Smart City Datasets**, pollution measurements, <http://iot.ee.surrey.ac.uk:8080/datasets.html>.
  The stream is a generated mock-up located in Aarhus, Denmark.
- **AirDelhi**: S. K. Chauhan, S. Ranu, R. Sen, Z. B. Patel, N. Batra, "AirDelhi: Fine-Grained Spatio-Temporal
  Particulate Matter Dataset From Delhi For ML based Modeling", NeurIPS 2023 Datasets and Benchmarks Track.
  Data portal: <http://www.cse.iitd.ac.in/pollutiondata>.
