"""Measure the cryptographic operations of EdgePlus's ledger on this machine (EdgePlus Sec. III-M, OBJ7).

Usage: python scripts/bench_crypto.py

Times (median over many repetitions, microseconds): Ed25519 sign and verify (Eq. 25), AES-GCM tag check of
one log tuple (Eqs. 18, 26a), and the SHA3-256 -> BLAKE2b block hash (Eq. 20) per tuple and per block.
The simulator's ledger model (configs/sim.yaml, ledger.consensus) uses these numbers to time edge validation
(Proof of Authority) and cloud finalization (BFT). Writes results/summary/crypto_bench.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parents[1]
TUPLE_BYTES = 96  # (i, c, s, p, d, t_start, t_end, on-time, energy, nonce, tag) packed, Eq. 18


def median_us(fn, reps: int = 2000, inner: int = 10) -> float:
    for _ in range(50):
        fn()
    samples = []
    for _ in range(reps // inner):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner * 1e6)
    return float(statistics.median(samples))


def block_hash(data: bytes) -> bytes:
    return hashlib.blake2b(hashlib.sha3_256(data).digest()).digest()


def main() -> None:
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()
    msg = os.urandom(64)
    sig = sk.sign(msg)
    key = AESGCM.generate_key(bit_length=128)
    aes = AESGCM(key)
    nonce = os.urandom(12)
    rec = os.urandom(TUPLE_BYTES - 16)
    ct = aes.encrypt(nonce, rec, None)
    out = {
        "ed25519_sign_us": median_us(lambda: sk.sign(msg)),
        "ed25519_verify_us": median_us(lambda: pk.verify(sig, msg)),
        "aesgcm_open_tuple_us": median_us(lambda: aes.decrypt(nonce, ct, None)),
        "tuple_bytes": TUPLE_BYTES,
    }
    sizes = [1, 16, 64, 128, 256]
    per_block = {b: median_us(lambda b=b: block_hash(os.urandom(0) + b"\0" * (b * TUPLE_BYTES + 32)), reps=1000)
                 for b in sizes}
    out["block_hash_us"] = {str(b): round(v, 3) for b, v in per_block.items()}
    # linear fit hash(B) = a + b * B
    xs, ys = sizes, [per_block[b] for b in sizes]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    out["block_hash_fixed_us"] = round(my - slope * mx, 3)
    out["block_hash_per_tuple_us"] = round(slope, 4)
    out["machine"] = {"cpu": platform.processor() or platform.machine(), "python": platform.python_version()}
    for k in ("ed25519_sign_us", "ed25519_verify_us", "aesgcm_open_tuple_us"):
        out[k] = round(out[k], 3)
    path = ROOT / "results" / "summary" / "crypto_bench.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
