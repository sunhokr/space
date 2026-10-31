"""Synthetic stripmap SAR raw data + Sentinel-1-like Level-0 emulation.

Used by the demo (``scripts/run_demo.py``) and the tests so that the whole
pipeline of the paper (L0 decoding -> adaptation -> compression -> focusing ->
metrics) can be exercised without downloading real data:

1. :func:`make_scene` - complex reflectivity with ocean, vegetation/bare soil
   texture, an urban area with bright scatterers, an airport runway and
   corner reflectors (speckled, i.e. circular Gaussian).
2. :func:`simulate_raw` - exact inverse Range-Doppler model (azimuth phase
   ``exp(-j 4 pi R0 D(f) / lambda)``, hyperbolic range migration, azimuth
   antenna pattern, chirp convolution), thermal noise and a 10-bit uniform ADC.
   Radar parameters are quantized exactly like the Sentinel-1 packet header
   fields so the emulated L0 file decodes to the very same geometry.
3. :func:`fdbaq_encode` / :func:`fdbaq_reconstruct` - FDBAQ-like block
   quantization with the genuine Sentinel-1 reconstruction tables (THIDX chosen
   from the block sigma, BRC from the block SNR).  This is an *emulation* of the
   on-board encoder (whose decision rules are not public in full detail).
4. :func:`write_l0_file` - genuine ISP packets (primary + secondary header,
   Huffman-coded FDBAQ user data, sub-commutated ephemeris) readable by
   :mod:`cvsarc.sentinel1_l0`.
"""
from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from . import s1_tables as T
from .focusing import (C_LIGHT, _fft, _ifft, _sinc_interp_rows, chirp_replica, doppler_frequencies,
                       next_fast_len)
from .sentinel1_l0 import effective_velocity

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

GM_EARTH = 3.986004418e14
GPS_T0 = 1.3e9  # arbitrary GPS time of the first line [s]

__all__ = ["SimParams", "radar_parameters", "make_scene", "simulate_raw", "fdbaq_encode",
           "fdbaq_reconstruct", "write_l0_file", "generate_dataset"]


@dataclass
class SimParams:
    n_az: int = 2048
    n_rg: int = 1536  # scene extent [range bins]
    rgdec: int = 8  # fs = 3/7 * 4 * F_REF = 64.34 MHz
    chirp_bandwidth: float = 42e6
    pulse_length: float = 6e-6
    prf: float = 1717.0
    near_range: float = 850e3
    orbit_radius: float = 7071e3
    doppler_centroid: float = 30.0
    doppler_bandwidth_fraction: float = 0.75  # 3-dB two-way Doppler bandwidth / PRF
    snr_db: float = 12.0
    adc_rms: float = 30.0  # per-component RMS of the 10-bit samples
    adc_max: int = 511
    rcm_margin: int = 32
    scene: str = "mixed"
    seed: int = 0


# --------------------------------------------------------------------------- #
# radar / orbit parameters (quantized as in the S-1 packet header)
# --------------------------------------------------------------------------- #
def radar_parameters(sp: SimParams) -> Dict:
    fs = T.range_decimation_to_fs(sp.rgdec)
    txpl_raw = int(round(sp.pulse_length * T.F_REF))
    Tp = txpl_raw / T.F_REF
    prr_mag = int(round(sp.chirp_bandwidth / Tp * 2 ** 21 / T.F_REF ** 2))
    Kr = prr_mag * T.F_REF ** 2 / 2 ** 21
    txprr_raw = 0x8000 | prr_mag  # positive ramp
    f0_target = -0.5 * Kr * Tp
    rest = f0_target - Kr / (4.0 * T.F_REF)
    psf_mag = int(round(abs(rest) * 2 ** 14 / T.F_REF))
    txpsf_raw = (0x8000 if rest >= 0 else 0) | psf_mag
    f0 = Kr / (4.0 * T.F_REF) + (1.0 if rest >= 0 else -1.0) * psf_mag * T.F_REF / 2 ** 14
    pri_raw = int(round(T.F_REF / sp.prf))
    pri = pri_raw / T.F_REF
    tau = 2.0 * sp.near_range / C_LIGHT
    rank = int(math.floor((tau - T.SWST_OFFSET) / pri))
    swst_raw = int(round((tau - T.SWST_OFFSET - rank * pri) * T.F_REF))
    swst = swst_raw / T.F_REF
    near = C_LIGHT / 2.0 * (rank * pri + swst + T.SWST_OFFSET)
    Np = int(round(Tp * fs))
    n_raw = sp.n_rg + Np + sp.rcm_margin
    n_raw += n_raw % 2  # even -> integer number of quads
    swl_raw = int(round(n_raw / fs * T.F_REF))
    vs = math.sqrt(GM_EARTH / sp.orbit_radius)
    return {"fs": fs, "pulse_length": Tp, "chirp_rate": Kr, "chirp_start_freq": f0, "prf": 1.0 / pri,
            "pri": pri, "rank": rank, "swst": swst, "near_range": near, "n_raw": n_raw, "Np": Np,
            "num_quads": n_raw // 2, "txpl_raw": txpl_raw, "txprr_raw": txprr_raw,
            "txpsf_raw": txpsf_raw, "pri_raw": pri_raw, "swst_raw": swst_raw, "swl_raw": swl_raw,
            "rgdec": sp.rgdec, "orbit_speed": vs, "wavelength": T.S1_WAVELENGTH}


def orbit_state(sp: SimParams, t: float):
    """Circular polar orbit in the x-z plane; t relative to GPS_T0."""
    r = sp.orbit_radius
    vs = math.sqrt(GM_EARTH / r)
    w = vs / r
    pos = np.array([r * math.cos(w * t), 0.0, r * math.sin(w * t)])
    vel = np.array([-vs * math.sin(w * t), 0.0, vs * math.cos(w * t)])
    return pos, vel


def _ephemeris_records(sp: SimParams, t_rel: np.ndarray):
    recs = []
    for t in t_rel:
        p, v = orbit_state(sp, float(t))
        recs.append({"time": GPS_T0 + float(t), "x": p[0], "y": p[1], "z": p[2],
                     "vx": v[0], "vy": v[1], "vz": v[2]})
    return recs


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
def _smooth_field(shape, corr: float, rng) -> np.ndarray:
    n = rng.standard_normal(shape)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    k = np.exp(-2.0 * (np.pi * corr) ** 2 * (fx ** 2 + fy ** 2))
    f = np.real(np.fft.ifft2(np.fft.fft2(n) * k))
    return (f - f.mean()) / (f.std() + 1e-12)


def make_scene(n_az: int, n_rg: int, kind: str = "mixed", seed: int = 0):
    """Return (complex reflectivity, sigma0) on the slant-range / azimuth grid."""
    rng = np.random.default_rng(seed)
    shape = (n_az, n_rg)
    land = np.exp(0.6 * _smooth_field(shape, 20, rng)) * (1.0 + 0.25 * _smooth_field(shape, 4, rng)).clip(0.2)
    sigma0 = land
    big = _smooth_field(shape, max(shape) / 5.0, rng)
    sea_level = {"mixed": -0.2, "coastal": 0.3, "urban": -1.5, "rural": -1.2}.get(kind, -0.2)
    sea = big < sea_level
    sigma0 = np.where(sea, 0.03 * np.exp(0.3 * _smooth_field(shape, 30, rng)), sigma0)
    pts = []
    # urban areas: enhanced backscatter + building grid of bright scatterers
    n_towns = {"mixed": 1, "coastal": 1, "urban": 3, "rural": 0}.get(kind, 1)
    for _ in range(n_towns):
        h, w = int(n_az * rng.uniform(0.15, 0.3)), int(n_rg * rng.uniform(0.15, 0.3))
        a0, r0 = int(rng.integers(0, n_az - h)), int(rng.integers(0, n_rg - w))
        sigma0[a0:a0 + h, r0:r0 + w] *= 2.5
        sea[a0:a0 + h, r0:r0 + w] = False
        sp_ = int(rng.integers(8, 14))
        for a in range(a0, a0 + h, sp_):
            for r in range(r0, r0 + w, sp_):
                if rng.random() < 0.6:
                    pts.append((a + int(rng.integers(0, 3)), r + int(rng.integers(0, 3)), rng.uniform(4, 25)))
    # airport runway (dark strip) + bright apron
    if kind in ("mixed", "coastal", "rural"):
        a0, r0 = int(n_az * 0.6), int(n_rg * 0.55)
        L = int(min(n_az, n_rg) * 0.3)
        for t in range(L):
            a, r = a0 + int(t * 0.35), r0 + t
            if a + 6 < n_az and r < n_rg:
                sigma0[a:a + 6, r] = 0.01
    # corner reflectors
    for _ in range(6):
        pts.append((int(rng.integers(64, n_az - 64)), int(rng.integers(64, n_rg - 64)), rng.uniform(60, 120)))
    refl = np.sqrt(sigma0) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2.0)
    for a, r, amp in pts:
        if 0 <= a < n_az and 0 <= r < n_rg:
            refl[a, r] += amp * np.exp(2j * np.pi * rng.random())
    return refl.astype(np.complex64), sigma0.astype(np.float32)


# --------------------------------------------------------------------------- #
# raw data simulation (inverse RDA)
# --------------------------------------------------------------------------- #
def simulate_raw(sp: SimParams, refl: Optional[np.ndarray] = None, azimuth_sign: int = -1):
    """Return (adc_raw complex64 with integer I/Q, info dict)."""
    rng = np.random.default_rng(sp.seed + 12345)
    rp = radar_parameters(sp)
    if refl is None:
        refl, _ = make_scene(sp.n_az, sp.n_rg, sp.scene, sp.seed)
    Na = refl.shape[0]
    n_raw = rp["n_raw"]
    fs, lam = rp["fs"], rp["wavelength"]
    dr = C_LIGHT / (2.0 * fs)
    near = rp["near_range"]
    t_mid = 0.5 * Na * rp["pri"]
    eph = _ephemeris_records(sp, np.array([t_mid - 1.0, t_mid, t_mid + 1.0]))
    vr = float(effective_velocity(eph, GPS_T0 + t_mid, near + 0.5 * sp.n_rg * dr))

    S = np.zeros((Na, n_raw), dtype=np.complex64)
    S[:, : refl.shape[1]] = refl
    S = _fft(S, axis=0).astype(np.complex64)
    f = doppler_frequencies(Na, rp["prf"], sp.doppler_centroid)
    b3 = sp.doppler_bandwidth_fraction * rp["prf"]
    G = (np.sinc(0.886 * (f - sp.doppler_centroid) / b3) ** 2).astype(np.float32)
    R = near + np.arange(n_raw) * dr
    kk = np.arange(n_raw, dtype=np.float64)[None, :]
    for j0 in range(0, Na, 128):
        j1 = min(Na, j0 + 128)
        D = np.sqrt(1.0 - (lam * f[j0:j1, None] / (2.0 * vr)) ** 2)
        rows = S[j0:j1] * (np.exp(1j * azimuth_sign * 4.0 * np.pi * R[None, :] * D / lam)
                           * G[j0:j1, None]).astype(np.complex64)
        src = D * kk - near * (1.0 - D) / dr  # inverse RCMC: target bin k -> R_k / D
        S[j0:j1] = _sinc_interp_rows(rows, src, 8)
    rc = _ifft(S, axis=0).astype(np.complex64)
    del S
    rep = chirp_replica(fs, rp["chirp_rate"], rp["pulse_length"], rp["chirp_start_freq"])
    nfft = next_fast_len(n_raw + rep.size)
    raw = _ifft(_fft(rc, n=nfft, axis=1) * _fft(rep, n=nfft)[None, :], axis=1)[:, :n_raw]
    del rc
    p_sig = float(np.mean(np.abs(raw) ** 2))
    sn = math.sqrt(p_sig / 10 ** (sp.snr_db / 10.0) / 2.0)
    raw = raw + sn * (rng.standard_normal(raw.shape) + 1j * rng.standard_normal(raw.shape))
    g = sp.adc_rms / math.sqrt(float(np.mean(np.abs(raw) ** 2)) / 2.0)
    re = np.clip(np.rint(raw.real * g), -sp.adc_max, sp.adc_max)
    im = np.clip(np.rint(raw.imag * g), -sp.adc_max, sp.adc_max)
    adc = (re + 1j * im).astype(np.complex64)
    info = dict(rp)
    info.update({"effective_velocity": vr, "noise_sigma_adc": sn * g, "doppler_centroid_true": sp.doppler_centroid,
                 "line_times": GPS_T0 + np.arange(Na) * rp["pri"], "sim_params": asdict(sp)})
    return adc, info


def sim_metadata(info: Dict, n_lines: int, n_samples: int, label: str) -> Dict:
    return {
        "sensor": "synthetic (Sentinel-1-like stripmap)",
        "label": label,
        "wavelength": info["wavelength"], "carrier_frequency": T.S1_CARRIER_FREQ,
        "prf": info["prf"], "pri": info["pri"], "fs": info["fs"], "rgdec": info["rgdec"],
        "chirp_rate": info["chirp_rate"], "chirp_start_freq": info["chirp_start_freq"],
        "pulse_length": info["pulse_length"], "near_range": info["near_range"],
        "effective_velocity": info["effective_velocity"], "doppler_centroid": None,
        "doppler_centroid_true": info["doppler_centroid_true"], "polarisation": "HH",
        "noise_sigma_adc": info["noise_sigma_adc"], "n_lines": n_lines, "n_samples": n_samples,
        "sim_params": info["sim_params"],
    }


# --------------------------------------------------------------------------- #
# FDBAQ-like encoder (emulation of the on-board compression)
# --------------------------------------------------------------------------- #
_LUT = T.fdbaq_lut()


def _channels(raw: np.ndarray) -> np.ndarray:
    """(n, 2nq) complex -> (n, 4, nq) real channels IE, IO, QE, QO."""
    return np.stack([raw.real[:, 0::2], raw.real[:, 1::2], raw.imag[:, 0::2], raw.imag[:, 1::2]], axis=1)


def fdbaq_encode(raw: np.ndarray, noise_sigma: float, snr_thresholds=(0.5, 2.0, 6.0, 20.0),
                 chunk_lines: int = 64):
    """Return signs/mcodes (n, 4, nq) uint8 and brc/thidx (n, nblk) uint8."""
    n, ns = raw.shape
    nq = ns // 2
    nblk = (nq + 127) // 128
    signs = np.zeros((n, 4, nq), dtype=np.uint8)
    mcodes = np.zeros((n, 4, nq), dtype=np.uint8)
    brc = np.zeros((n, nblk), dtype=np.uint8)
    thidx = np.zeros((n, nblk), dtype=np.uint8)
    sf = T.SIGMA_FACTORS[:255]  # strictly increasing part
    blk_of = np.arange(nq) // 128
    for l0 in range(0, n, chunk_lines):
        l1 = min(n, l0 + chunk_lines)
        ch = _channels(np.asarray(raw[l0:l1])).astype(np.float64)  # (h, 4, nq)
        h = ch.shape[0]
        pad = np.zeros((h, 4, nblk * 128))
        cnt = np.zeros((nblk * 128,))
        pad[:, :, :nq] = ch ** 2
        cnt[:nq] = 1
        var = pad.reshape(h, 4, nblk, 128).sum(axis=(1, 3)) / (4.0 * cnt.reshape(nblk, 128).sum(axis=1))
        sig = np.sqrt(var)
        t = np.clip(np.searchsorted(sf, sig), 1, 254)
        t = np.where(np.abs(sf[t - 1] - sig) <= np.abs(sf[t] - sig), t - 1, t)
        snr = var / max(noise_sigma, 1e-6) ** 2 - 1.0
        b = np.digitize(snr, snr_thresholds).clip(0, 4)
        brc[l0:l1], thidx[l0:l1] = b, t
        lev = _LUT[b, t]  # (h, nblk, 16)
        lev = np.where(np.isnan(lev), np.inf, lev)
        lev_s = lev[:, blk_of, :]  # (h, nq, 16)
        a = np.abs(ch)  # (h, 4, nq)
        m = np.argmin(np.abs(a[..., None] - lev_s[:, None, :, :]), axis=-1)
        mcodes[l0:l1] = m.astype(np.uint8)
        signs[l0:l1] = (ch < 0).astype(np.uint8)
    return signs, mcodes, brc, thidx


def fdbaq_reconstruct(signs, mcodes, brc, thidx) -> np.ndarray:
    """Decoder-side reconstruction (identical to the L0 decoder output)."""
    n, _, nq = mcodes.shape
    blk_of = np.arange(nq) // 128
    b = brc[:, blk_of][:, None, :].astype(np.int64)
    t = thidx[:, blk_of][:, None, :].astype(np.int64)
    v = _LUT[b, t, mcodes.astype(np.int64)]
    v = np.where(signs == 1, -v, v).astype(np.float32)
    out = np.zeros((n, 2 * nq), dtype=np.complex64)
    out.real[:, 0::2], out.real[:, 1::2] = v[:, 0], v[:, 1]
    out.imag[:, 0::2], out.imag[:, 1::2] = v[:, 2], v[:, 3]
    return out


# --------------------------------------------------------------------------- #
# ISP packet writer
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _put(out, pos, value, n):
    for k in range(n - 1, -1, -1):
        if (value >> k) & 1:
            out[pos >> 3] |= np.uint8(1 << (7 - (pos & 7)))
        pos += 1
    return pos


@njit(cache=True)
def _pack_fdbaq(signs, mcodes, brc, thidx, nq, cval, clen, out):
    nblk = (nq + 127) // 128
    pos = 0
    for ch in range(4):
        for blk in range(nblk):
            if ch == 0:
                pos = _put(out, pos, brc[blk], 3)
            elif ch == 2:
                pos = _put(out, pos, thidx[blk], 8)
            b = brc[blk]
            for i in range(blk * 128, min(blk * 128 + 128, nq)):
                pos = _put(out, pos, signs[ch, i], 1)
                m = mcodes[ch, i]
                pos = _put(out, pos, cval[b, m], clen[b, m])
        pos = ((pos + 15) // 16) * 16
    return pos // 8


def _words_from_ephemeris(rec: Dict) -> np.ndarray:
    b = struct.pack(">ddd", rec["x"], rec["y"], rec["z"]) + struct.pack(">fff", rec["vx"], rec["vy"], rec["vz"])
    w = list(np.frombuffer(b, dtype=">u2").astype(np.int64))
    t = rec["time"]
    d18 = int(t // 2 ** 24)
    r = t - d18 * 2 ** 24
    d19 = int(r // 2 ** 8)
    r -= d19 * 2 ** 8
    d20 = int(r * 2 ** 8)
    r -= d20 / 2 ** 8
    d21 = min(int(round(r * 2 ** 24)), 65535)
    w += [d18, d19, d20, d21]
    w += [0] * (64 - len(w))
    return np.array(w[:64], dtype=np.int64)


def write_l0_file(path, signs, mcodes, brc, thidx, rp: Dict, sp: SimParams, swath: int = 1,
                  pol_code: int = 1, ecc: int = 1, baq_mode: int = 12, data_take_id: int = 0x12345) -> Path:
    """Write emulated FDBAQ echo packets as a Sentinel-1 L0 measurement file."""
    n, _, nq = mcodes.shape
    cval, clen = T.huffman_encode_tables()
    t_rel = np.arange(n) * rp["pri"]
    n_cycles = n // 64 + 1
    eph = _ephemeris_records(sp, np.arange(n_cycles) * 64 * rp["pri"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = 4 * ((nq * 14 + ((nq + 127) // 128) * 8) // 8 + 8)
    with open(path, "wb") as fh:
        for i in range(n):
            out = np.zeros(cap, dtype=np.uint8)
            nb = int(_pack_fdbaq(signs[i].astype(np.int64), mcodes[i].astype(np.int64),
                                 brc[i].astype(np.int64), thidx[i].astype(np.int64), nq, cval, clen, out))
            nb += (-nb) % 4  # user data field padded to 32-bit words
            user = out[:nb].tobytes()
            cyc, widx = divmod(i, 64)
            words = _words_from_ephemeris(eph[cyc])
            t = GPS_T0 + t_rel[i]
            sec = bytearray(62)
            tcoar = int(math.floor(t))
            tfine = int(min(65535, max(0, round((t - tcoar) * 65536.0 - 0.5))))
            sec[0:4] = tcoar.to_bytes(4, "big")
            sec[4:6] = tfine.to_bytes(2, "big")
            sec[6:10] = T.SYNC_MARKER.to_bytes(4, "big")
            sec[10:14] = int(data_take_id).to_bytes(4, "big")
            sec[14] = ecc
            sec[15] = (0 << 4) | (1 if pol_code in (1, 5) else 0)
            sec[16:20] = (0).to_bytes(4, "big")
            sec[20] = widx + 1
            sec[21:23] = int(words[widx]).to_bytes(2, "big")
            sec[23:27] = i.to_bytes(4, "big")
            sec[27:31] = (1000 + i).to_bytes(4, "big")
            sec[31] = baq_mode & 0x1F
            sec[32] = 31
            sec[34] = rp["rgdec"]
            sec[35] = 10
            sec[36:38] = rp["txprr_raw"].to_bytes(2, "big")
            sec[38:40] = rp["txpsf_raw"].to_bytes(2, "big")
            sec[40:43] = rp["txpl_raw"].to_bytes(3, "big")
            sec[43] = rp["rank"] & 0x1F
            sec[44:47] = rp["pri_raw"].to_bytes(3, "big")
            sec[47:50] = rp["swst_raw"].to_bytes(3, "big")
            sec[50:53] = rp["swl_raw"].to_bytes(3, "big")
            sec[53] = (0 << 7) | (pol_code << 4)
            sec[54:56] = ((3 << 12) | 512).to_bytes(2, "big")
            sec[56] = 0
            sec[57] = 0  # signal type 0 = echo
            sec[58] = swath
            sec[59:61] = int(nq).to_bytes(2, "big")
            body = bytes(sec) + user
            w0 = (0 << 13) | (0 << 12) | (1 << 11) | (65 << 4) | 12
            w1 = (3 << 14) | (i & 0x3FFF)
            fh.write(struct.pack(">HHH", w0, w1, len(body) - 1))
            fh.write(body)
    return path


# --------------------------------------------------------------------------- #
# one-call dataset generation
# --------------------------------------------------------------------------- #
def generate_dataset(out_root, name: str, sp: SimParams, write_l0: bool = False,
                     adapt_seed: int = 0, verbose: bool = True) -> Dict[str, str]:
    """Create ``<name>_uncompressed``, ``<name>_l0decoded`` and ``<name>_adapted`` datasets."""
    from .adaptation import adapt_with_side_info
    from .utils import save_raw_dataset

    out_root = Path(out_root)
    if verbose:
        print(f"[sim] {name}: scene '{sp.scene}' {sp.n_az} x {sp.n_rg}")
    adc, info = simulate_raw(sp)
    n, ns = adc.shape
    meta = sim_metadata(info, n, ns, name)
    paths = {}
    paths["uncompressed"] = str(save_raw_dataset(out_root / f"{name}_uncompressed",
                                                 adc, dict(meta, stage="uncompressed (10-bit ADC)")))
    signs, mcodes, brc, thidx = fdbaq_encode(adc, info["noise_sigma_adc"])
    dec = fdbaq_reconstruct(signs, mcodes, brc, thidx)
    side = {"brc": brc, "thidx": thidx, "baq_mode": np.full(n, 12, dtype=np.uint8),
            "valid_line": np.ones(n, dtype=bool), "sample_start": np.int64(0), "num_quads": np.int64(ns // 2)}
    paths["l0decoded"] = str(save_raw_dataset(out_root / f"{name}_l0decoded", dec,
                                              dict(meta, stage="FDBAQ-decoded (L0-like)"), side))
    ada = adapt_with_side_info(dec, side, seed=adapt_seed)
    paths["adapted"] = str(save_raw_dataset(out_root / f"{name}_adapted", ada,
                                            dict(meta, stage="adapted (ref. [20])")))
    if write_l0:
        rp = radar_parameters(sp)
        paths["l0_file"] = str(write_l0_file(out_root / f"{name}.dat", signs, mcodes, brc, thidx, rp, sp))
    if verbose:
        bits = _fdbaq_mean_bits(mcodes, brc)
        print(f"[sim] {name}: FDBAQ-like mean rate ~ {bits:.2f} bit/real sample")
    return paths


def _fdbaq_mean_bits(mcodes, brc) -> float:
    _, clen = T.huffman_encode_tables()
    nq = mcodes.shape[2]
    b = brc[:, np.arange(nq) // 128][:, None, :].astype(np.int64)
    return float(np.mean(clen[b, mcodes.astype(np.int64)]) + 1.0)
