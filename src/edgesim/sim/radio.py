"""Access-link rate model (EdgePlus Eq. 30) and expected uplink time per task."""
from __future__ import annotations

import numpy as np


def snr_db(dist_km: np.ndarray, net: dict) -> np.ndarray:
    lo, hi = net["snr_db_range"]
    d = np.maximum(np.asarray(dist_km, np.float64), net["ref_dist_km"])
    val = net["snr_ref_db"] - 10.0 * net["pathloss_exp"] * np.log10(d / net["ref_dist_km"])
    return np.clip(val, lo, hi)


def access_rate_bps(slice_idx: np.ndarray, dist_km: np.ndarray, net: dict) -> np.ndarray:
    """R = zeta_k * W_k / reuse * log2(1 + SNR) * eta_mac * eta_backhaul, in bits per second."""
    k = np.asarray(slice_idx, np.int64)
    bw = np.asarray(net["bw_mhz"], np.float64)[k] * 1e6
    zeta = np.asarray(net["zeta"], np.float64)[k]
    snr = 10.0 ** (snr_db(dist_km, net) / 10.0)
    return zeta * bw / net["reuse"] * np.log2(1.0 + snr) * net["eta_mac"] * net["eta_backhaul"]


def uplink_time_s(payload_bits: np.ndarray, slice_idx: np.ndarray, dist_km: np.ndarray, net: dict) -> np.ndarray:
    """Expected device-to-site uplink time: eta_OH * b / R."""
    return net["overhead"] * np.asarray(payload_bits, np.float64) / access_rate_bps(slice_idx, dist_km, net)
