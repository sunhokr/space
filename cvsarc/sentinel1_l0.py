"""Sentinel-1 Level-0 (RAW) product decoder.

The paper uses Sentinel-1 StripMap (SM) L0 products whose echoes were
compressed on board with FDBAQ (paper Sec. II-A).  This module

1. indexes all ISP packets of a measurement data file (``*.dat`` inside the
   ``.SAFE`` folder) and decodes every primary / secondary header field,
2. splits the echo packets into *acquisition chunks* (constant swath, number of
   quads, SWST, SWL, PRI, chirp, ...),
3. decodes the user data (FDBAQ modes 12/13/14, BAQ 3/4/5-bit, bypass) into a
   complex matrix (azimuth lines x range samples), keeping the per-block BRC
   and THIDX as side information (needed by the adaptation of ref. [20]),
4. decodes the sub-commutated ephemeris and builds the focusing metadata.

The bit-level kernels are compiled with numba when available (strongly
recommended for whole scenes); otherwise they run in pure Python.
Optionally the open-source ``sentinel1decoder`` package (Rust backend) can be
used for the sample decoding (``backend="sentinel1decoder"``).
"""
from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import s1_tables as T
from .utils import create_raw_dataset, save_json, SIDE_FILE

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

__all__ = ["index_packets", "find_echo_chunks", "decode_chunk", "parse_ephemeris",
           "effective_velocity", "build_metadata", "decode_l0_file", "decode_user_data"]

_TREE = T.huffman_tree_array()
_FDBAQ_LUT = T.fdbaq_lut()
_BAQ_LUTS = {n: T.baq_lut(n) for n in (3, 4, 5)}


# =========================================================================== #
# packet headers
# =========================================================================== #
def _be(s: np.ndarray, a: int, b: int) -> np.ndarray:
    """Big-endian unsigned integer from columns a..b-1 of a (n, k) byte array."""
    v = np.zeros(s.shape[0], dtype=np.int64)
    for k in range(a, b):
        v = (v << 8) | s[:, k].astype(np.int64)
    return v


def _packet_offsets(mm: np.ndarray) -> np.ndarray:
    n = mm.shape[0]
    offs = []
    pos = 0
    while pos + T.PRIMARY_HEADER_LEN <= n:
        dlen = ((int(mm[pos + 4]) << 8) | int(mm[pos + 5])) + 1
        if pos + T.PRIMARY_HEADER_LEN + dlen > n:
            break  # truncated last packet
        offs.append(pos)
        pos += T.PRIMARY_HEADER_LEN + dlen
    return np.asarray(offs, dtype=np.int64)


def index_packets(path) -> Dict[str, np.ndarray]:
    """Decode all packet headers of an L0 measurement file (vectorised)."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    off = _packet_offsets(mm)
    if off.size == 0:
        raise ValueError(f"no ISP packets found in {path}")
    hl = T.PRIMARY_HEADER_LEN + T.SECONDARY_HEADER_LEN
    valid = off + hl <= mm.shape[0]
    off = off[valid]
    hdr = np.asarray(mm[off[:, None] + np.arange(hl)[None, :]])
    p = hdr[:, :6].astype(np.int64)
    s = hdr[:, 6:].astype(np.int64)
    pk: Dict[str, np.ndarray] = {"offset": off}
    pk["packet_version"] = p[:, 0] >> 5
    pk["packet_type"] = (p[:, 0] >> 4) & 1
    pk["secondary_header_flag"] = (p[:, 0] >> 3) & 1
    pk["pid"] = ((p[:, 0] & 0x07) << 4) | (p[:, 1] >> 4)
    pk["pcat"] = p[:, 1] & 0x0F
    pk["sequence_flags"] = p[:, 2] >> 6
    pk["sequence_count"] = ((p[:, 2] & 0x3F) << 8) | p[:, 3]
    pk["data_length"] = ((p[:, 4] << 8) | p[:, 5]) + 1
    # datation
    pk["tcoar"] = _be(s, 0, 4)
    pk["tfine"] = _be(s, 4, 6)
    pk["time"] = pk["tcoar"] + (pk["tfine"] + 0.5) * 2.0 ** -16
    # fixed ancillary data
    pk["sync"] = _be(s, 6, 10)
    pk["data_take_id"] = _be(s, 10, 14)
    pk["ecc"] = s[:, 14]
    pk["test_mode"] = (s[:, 15] >> 4) & 0x07
    pk["rx_channel_id"] = s[:, 15] & 0x0F
    pk["instrument_config_id"] = _be(s, 16, 20)
    # sub-commutated ancillary data
    pk["adw_index"] = s[:, 20]
    pk["adw"] = _be(s, 21, 23)
    # counters
    pk["space_packet_count"] = _be(s, 23, 27)
    pk["pri_count"] = _be(s, 27, 31)
    # radar configuration support
    pk["error_flag"] = s[:, 31] >> 7
    pk["baq_mode"] = s[:, 31] & 0x1F
    pk["baq_block_length"] = s[:, 32]
    pk["rgdec"] = s[:, 34]
    pk["rx_gain_db"] = -0.5 * s[:, 35]
    txprr_raw = _be(s, 36, 38)
    txpsf_raw = _be(s, 38, 40)
    pk["txprr_raw"], pk["txpsf_raw"] = txprr_raw, txpsf_raw
    sgn = np.where((txprr_raw >> 15) & 1, 1.0, -1.0)
    pk["txprr"] = sgn * (txprr_raw & 0x7FFF) * T.F_REF ** 2 / 2.0 ** 21  # [Hz/s]
    sgn2 = np.where((txpsf_raw >> 15) & 1, 1.0, -1.0)
    pk["txpsf"] = pk["txprr"] / (4.0 * T.F_REF) + sgn2 * (txpsf_raw & 0x7FFF) * T.F_REF / 2.0 ** 14  # [Hz]
    pk["txpl_raw"] = _be(s, 40, 43)
    pk["txpl"] = pk["txpl_raw"] / T.F_REF  # [s]
    pk["rank"] = s[:, 43] & 0x1F
    pk["pri_raw"] = _be(s, 44, 47)
    pk["swst_raw"] = _be(s, 47, 50)
    pk["swl_raw"] = _be(s, 50, 53)
    pk["pri"] = pk["pri_raw"] / T.F_REF
    pk["swst"] = pk["swst_raw"] / T.F_REF
    pk["swl"] = pk["swl_raw"] / T.F_REF
    pk["ssb_flag"] = s[:, 53] >> 7
    pk["polarisation"] = (s[:, 53] >> 4) & 0x07
    pk["temp_comp"] = (s[:, 53] >> 2) & 0x03
    t16 = _be(s, 54, 56)
    pk["elevation_beam_address"] = np.where(pk["ssb_flag"] == 0, t16 >> 12, -1)
    pk["azimuth_beam_address"] = np.where(pk["ssb_flag"] == 0, t16 & 0x3FF, -1)
    pk["cal_type"] = np.where(pk["ssb_flag"] == 1, (t16 >> 12) & 0x07, -1)
    pk["cal_mode"] = s[:, 56] >> 6
    pk["tx_pulse_number"] = s[:, 56] & 0x1F
    pk["signal_type"] = s[:, 57] >> 4
    pk["swap"] = s[:, 57] & 0x01
    pk["swath"] = s[:, 58]
    pk["num_quads"] = _be(s, 59, 61)
    # user data
    pk["user_data_offset"] = off + hl
    pk["user_data_length"] = pk["data_length"] - T.SECONDARY_HEADER_LEN
    fs = np.full(off.size, np.nan)
    for code, (L, M) in T.RGDEC_RATIOS.items():
        fs[pk["rgdec"] == code] = L / M * 4.0 * T.F_REF
    pk["fs"] = fs
    return pk


_CHUNK_KEYS = ("swath", "num_quads", "baq_mode", "swst_raw", "swl_raw", "pri_raw", "rank",
               "rgdec", "txprr_raw", "txpsf_raw", "txpl_raw", "polarisation", "rx_channel_id")


def find_echo_chunks(pk: Dict[str, np.ndarray], min_lines: int = 1) -> List[Dict]:
    """Group consecutive echo packets with identical radar configuration."""
    echo = np.flatnonzero((pk["signal_type"] == T.SIGNAL_TYPE_ECHO)
                          & (pk["sync"] == T.SYNC_MARKER)
                          & (pk["secondary_header_flag"] == 1)
                          & (pk["num_quads"] > 0))
    if echo.size == 0:
        return []
    key = np.stack([pk[k][echo] for k in _CHUNK_KEYS], axis=1)
    brk = np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1
    starts = np.concatenate([[0], brk])
    stops = np.concatenate([brk, [echo.size]])
    chunks = []
    for a, b in zip(starts, stops):
        idx = echo[a:b]
        if idx.size < min_lines:
            continue
        i0 = idx[0]
        prict = pk["pri_count"][idx]
        span = int((prict[-1] - prict[0]) % (1 << 32)) + 1
        chunks.append({
            "packets": idx,
            "n_packets": int(idx.size),
            "n_lines": span,
            "missing_lines": span - int(idx.size),
            "swath": int(pk["swath"][i0]),
            "num_quads": int(pk["num_quads"][i0]),
            "baq_mode": int(pk["baq_mode"][i0]),
            "polarisation": T.POLARISATION_NAMES.get(int(pk["polarisation"][i0]), "?"),
            "ecc": int(pk["ecc"][i0]),
            "time_first": float(pk["time"][idx[0]]),
            "time_last": float(pk["time"][idx[-1]]),
        })
    return chunks


# =========================================================================== #
# user-data kernels
# =========================================================================== #
@njit(cache=True)
def _getbits(data, pos, n):
    v = 0
    nbytes = data.shape[0]
    for i in range(n):
        p = pos + i
        byte = p >> 3
        bit = 0
        if byte < nbytes:
            bit = (data[byte] >> (7 - (p & 7))) & 1
        v = (v << 1) | bit
    return v


@njit(cache=True)
def _fdbaq_kernel(data, nq, tree, lut, re, im, brc_out, thidx_out):
    """Decode one FDBAQ packet. Returns bits consumed, or <0 on error."""
    nblk = (nq + 127) // 128
    signs = np.zeros((4, nq), dtype=np.uint8)
    mcodes = np.zeros((4, nq), dtype=np.uint8)
    nbits_total = data.shape[0] * 8
    pos = 0
    for ch in range(4):
        for blk in range(nblk):
            if ch == 0:
                b = _getbits(data, pos, 3)
                pos += 3
                if b > 4:
                    return -1
                brc_out[blk] = b
            elif ch == 2:
                thidx_out[blk] = _getbits(data, pos, 8)
                pos += 8
            b = brc_out[blk]
            i0 = blk * 128
            i1 = min(i0 + 128, nq)
            for i in range(i0, i1):
                signs[ch, i] = _getbits(data, pos, 1)
                pos += 1
                node = 0
                m = 0
                while True:
                    bit = _getbits(data, pos, 1)
                    pos += 1
                    nxt = tree[b, node, bit]
                    if nxt < 0:
                        m = -nxt - 1
                        break
                    node = nxt
                mcodes[ch, i] = m
            if pos > nbits_total + 16:
                return -2
        pos = ((pos + 15) // 16) * 16  # channel ends on a 16-bit word boundary
    for blk in range(nblk):
        b = brc_out[blk]
        t = thidx_out[blk]
        i0 = blk * 128
        i1 = min(i0 + 128, nq)
        for i in range(i0, i1):
            v0 = lut[b, t, mcodes[0, i]]
            v1 = lut[b, t, mcodes[1, i]]
            v2 = lut[b, t, mcodes[2, i]]
            v3 = lut[b, t, mcodes[3, i]]
            if signs[0, i]:
                v0 = -v0
            if signs[1, i]:
                v1 = -v1
            if signs[2, i]:
                v2 = -v2
            if signs[3, i]:
                v3 = -v3
            re[2 * i] = v0  # IE
            re[2 * i + 1] = v1  # IO
            im[2 * i] = v2  # QE
            im[2 * i + 1] = v3  # QO
    return pos


@njit(cache=True)
def _baq_kernel(data, nq, nbits, lut, re, im, thidx_out):
    nblk = (nq + 127) // 128
    nw = (nbits * nq + 15) // 16
    nw_qe = (nbits * nq + 8 * nblk + 15) // 16
    b_ie = 0
    b_io = 16 * nw
    b_qe = 32 * nw
    b_qo = 32 * nw + 16 * nw_qe
    mmask = (1 << (nbits - 1)) - 1
    for blk in range(nblk):
        t = _getbits(data, b_qe + blk * 128 * nbits + blk * 8, 8)
        thidx_out[blk] = t
        i0 = blk * 128
        i1 = min(i0 + 128, nq)
        for i in range(i0, i1):
            o = i * nbits
            c0 = _getbits(data, b_ie + o, nbits)
            c1 = _getbits(data, b_io + o, nbits)
            c2 = _getbits(data, b_qe + o + (blk + 1) * 8, nbits)
            c3 = _getbits(data, b_qo + o, nbits)
            v0 = lut[t, c0 & mmask]
            v1 = lut[t, c1 & mmask]
            v2 = lut[t, c2 & mmask]
            v3 = lut[t, c3 & mmask]
            if c0 >> (nbits - 1):
                v0 = -v0
            if c1 >> (nbits - 1):
                v1 = -v1
            if c2 >> (nbits - 1):
                v2 = -v2
            if c3 >> (nbits - 1):
                v3 = -v3
            re[2 * i] = v0
            re[2 * i + 1] = v1
            im[2 * i] = v2
            im[2 * i + 1] = v3
    return b_qo + 16 * nw


@njit(cache=True)
def _bypass_kernel(data, nq, re, im):
    nw = (10 * nq + 15) // 16
    for i in range(nq):
        for ch in range(4):
            code = _getbits(data, ch * 16 * nw + 10 * i, 10)
            v = float(code & 0x1FF)
            if code >> 9:
                v = -v
            if ch == 0:
                re[2 * i] = v
            elif ch == 1:
                re[2 * i + 1] = v
            elif ch == 2:
                im[2 * i] = v
            else:
                im[2 * i + 1] = v
    return 64 * nw


def decode_user_data(data: np.ndarray, num_quads: int, baq_mode: int):
    """Decode the user data field of one echo packet.

    Returns ``(line complex64[2*nq], brc uint8[nblk], thidx uint8[nblk])``;
    BRC is 255 for non-FDBAQ modes, THIDX is 255 for bypass mode.
    """
    data = np.ascontiguousarray(data, dtype=np.uint8)
    nq = int(num_quads)
    nblk = (nq + 127) // 128
    re = np.zeros(2 * nq, dtype=np.float32)
    im = np.zeros(2 * nq, dtype=np.float32)
    brc = np.full(nblk, 255, dtype=np.uint8)
    thidx = np.full(nblk, 255, dtype=np.uint8)
    if baq_mode in T.FDBAQ_MODES:
        brc_i = np.zeros(nblk, dtype=np.int64)
        thidx_i = np.zeros(nblk, dtype=np.int64)
        r = _fdbaq_kernel(data, nq, _TREE, _FDBAQ_LUT, re, im, brc_i, thidx_i)
        if r < 0:
            raise ValueError(f"FDBAQ decoding error (code {r})")
        brc[:] = brc_i
        thidx[:] = thidx_i
    elif baq_mode in T.BAQ_MODES:
        thidx_i = np.zeros(nblk, dtype=np.int64)
        _baq_kernel(data, nq, int(baq_mode), _BAQ_LUTS[int(baq_mode)], re, im, thidx_i)
        thidx[:] = thidx_i
    elif baq_mode == 0:
        _bypass_kernel(data, nq, re, im)
    else:
        raise ValueError(f"unsupported BAQ mode {baq_mode}")
    return (re + 1j * im).astype(np.complex64), brc, thidx


# =========================================================================== #
# ephemeris / geometry
# =========================================================================== #
def parse_ephemeris(pk: Dict[str, np.ndarray]) -> List[Dict]:
    """Decode the 64-word sub-commutated ancillary data cycles (PVT + attitude)."""
    idx = pk["adw_index"]
    words = pk["adw"]
    n = idx.size
    out, seen = [], set()
    for s in np.flatnonzero(idx == 1):
        if s + 64 > n or not np.array_equal(idx[s:s + 64], np.arange(1, 65)):
            continue
        d = words[s:s + 64].astype(np.int64)
        b = np.asarray(d, dtype=">u2").tobytes()
        x, y, z = struct.unpack(">ddd", b[0:24])
        vx, vy, vz = struct.unpack(">fff", b[24:36])
        t = d[18] * 2.0 ** 24 + d[19] * 2.0 ** 8 + d[20] * 2.0 ** -8 + d[21] * 2.0 ** -24
        q0, q1, q2, q3 = struct.unpack(">ffff", b[44:60])
        wx, wy, wz = struct.unpack(">fff", b[60:72])
        ta = d[36] * 2.0 ** 24 + d[37] * 2.0 ** 8 + d[38] * 2.0 ** -8 + d[39] * 2.0 ** -24
        if t in seen:
            continue
        seen.add(t)
        out.append({"time": float(t), "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz,
                    "q": [q0, q1, q2, q3], "omega": [wx, wy, wz], "attitude_time": float(ta)})
    out.sort(key=lambda e: e["time"])
    return out


def _interp_state(eph: Sequence[Dict], t: float):
    times = np.array([e["time"] for e in eph])
    pos = np.array([[e["x"], e["y"], e["z"]] for e in eph])
    vel = np.array([[e["vx"], e["vy"], e["vz"]] for e in eph])
    if times.size == 1:
        return pos[0] + vel[0] * (t - times[0]), vel[0]
    p = np.array([np.interp(t, times, pos[:, k]) for k in range(3)])
    v = np.array([np.interp(t, times, vel[:, k]) for k in range(3)])
    return p, v


def effective_velocity(eph: Sequence[Dict], t: float, slant_range) -> np.ndarray:
    """Effective (radar) velocity Vr = sqrt(Vs * Vg) from orbit state and slant range."""
    p, v = _interp_state(eph, t)
    vs = float(np.linalg.norm(v))
    H = float(np.linalg.norm(p))
    lat = np.arctan2(p[2], np.hypot(p[0], p[1]))
    a, b = T.WGS84_A, T.WGS84_B
    re = np.sqrt(((a * a * np.cos(lat)) ** 2 + (b * b * np.sin(lat)) ** 2)
                 / ((a * np.cos(lat)) ** 2 + (b * np.sin(lat)) ** 2))
    R = np.asarray(slant_range, dtype=np.float64)
    cos_beta = (re * re + H * H - R * R) / (2.0 * re * H)
    vg = re * (vs / H) * cos_beta
    return np.sqrt(vs * vg)


def build_metadata(pk: Dict[str, np.ndarray], chunk: Dict, packets: np.ndarray,
                   n_lines: int, n_samples: int, ephemeris: List[Dict], source: str = "") -> Dict:
    i0 = int(packets[0])
    fs = float(pk["fs"][i0])
    pri = float(pk["pri"][i0])
    rank = int(pk["rank"][i0])
    swst = float(pk["swst"][i0])
    near = T.C_LIGHT / 2.0 * (rank * pri + swst + T.SWST_OFFSET)
    far = near + (n_samples - 1) * T.C_LIGHT / (2.0 * fs)
    times = pk["time"][packets]
    tmid = float(0.5 * (times[0] + times[-1]))
    meta = {
        "sensor": "Sentinel-1",
        "source_file": str(source),
        "decoder": "cvsarc.sentinel1_l0",
        "carrier_frequency": T.S1_CARRIER_FREQ,
        "wavelength": T.S1_WAVELENGTH,
        "prf": 1.0 / pri,
        "pri": pri,
        "fs": fs,
        "rgdec": int(pk["rgdec"][i0]),
        "chirp_rate": float(pk["txprr"][i0]),
        "chirp_start_freq": float(pk["txpsf"][i0]),
        "pulse_length": float(pk["txpl"][i0]),
        "chirp_bandwidth": abs(float(pk["txprr"][i0]) * float(pk["txpl"][i0])),
        "rank": rank,
        "swst": swst,
        "swl": float(pk["swl"][i0]),
        "swst_offset": T.SWST_OFFSET,
        "near_range": near,
        "far_range": far,
        "doppler_centroid": None,
        "swath_number": int(pk["swath"][i0]),
        "polarisation": T.POLARISATION_NAMES.get(int(pk["polarisation"][i0]), "?"),
        "rx_channel": "H" if int(pk["rx_channel_id"][i0]) == 1 else "V",
        "ecc_number": int(pk["ecc"][i0]),
        "mode": T.ECC_NAMES.get(int(pk["ecc"][i0]), "unknown"),
        "baq_mode": int(pk["baq_mode"][i0]),
        "baq_mode_name": T.BAQ_MODE_NAMES.get(int(pk["baq_mode"][i0]), "?"),
        "rx_gain_db": float(pk["rx_gain_db"][i0]),
        "data_take_id": int(pk["data_take_id"][i0]),
        "num_quads": int(pk["num_quads"][i0]),
        "n_lines": int(n_lines),
        "n_samples": int(n_samples),
        "line_time_first": float(times[0]),
        "line_time_last": float(times[-1]),
        "missing_lines": int(n_lines - packets.size),
        "ephemeris": [{k: e[k] for k in ("time", "x", "y", "z", "vx", "vy", "vz")} for e in ephemeris],
    }
    if ephemeris:
        vr = effective_velocity(ephemeris, tmid, np.array([near, 0.5 * (near + far), far]))
        meta["effective_velocity_near"] = float(vr[0])
        meta["effective_velocity"] = float(vr[1])
        meta["effective_velocity_far"] = float(vr[2])
    else:
        meta["effective_velocity"] = None
    return meta


# =========================================================================== #
# high-level decoding
# =========================================================================== #
def decode_chunk(path, pk: Dict[str, np.ndarray], chunk: Dict, out_dir=None,
                 line_start: int = 0, max_lines: Optional[int] = None,
                 sample_start: int = 0, max_samples: Optional[int] = None,
                 verbose: bool = True):
    """Decode (a sub-window of) an acquisition chunk.

    Returns ``(raw, side_info, packets_used)``.  If ``out_dir`` is given,
    ``raw`` is a memory map written to ``out_dir/raw.npy``.  Lines missing in
    the PRI-count sequence are left at zero.
    """
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    pk_idx = chunk["packets"]
    prict = pk["pri_count"][pk_idx]
    line_of = ((prict - prict[0]) % (1 << 32)).astype(np.int64)
    n_total = int(line_of[-1]) + 1
    l0 = int(line_start)
    l1 = n_total if max_lines is None else min(n_total, l0 + int(max_lines))
    sel = (line_of >= l0) & (line_of < l1)
    packets = pk_idx[sel]
    rows = line_of[sel] - l0
    nq = chunk["num_quads"]
    ns_full = 2 * nq
    s0 = int(sample_start)
    s1 = ns_full if max_samples is None else min(ns_full, s0 + int(max_samples))
    n_lines, n_samples = l1 - l0, s1 - s0
    nblk = (nq + 127) // 128
    if out_dir is not None:
        raw = create_raw_dataset(out_dir, (n_lines, n_samples))
    else:
        raw = np.zeros((n_lines, n_samples), dtype=np.complex64)
    brc = np.full((n_lines, nblk), 255, dtype=np.uint8)
    thidx = np.full((n_lines, nblk), 255, dtype=np.uint8)
    baqm = np.zeros(n_lines, dtype=np.uint8)
    valid = np.zeros(n_lines, dtype=bool)
    t0 = time.time()
    for k, (pi, r) in enumerate(zip(packets, rows)):
        a = int(pk["user_data_offset"][pi])
        n = int(pk["user_data_length"][pi])
        line, b, t = decode_user_data(np.asarray(mm[a:a + n]), nq, int(pk["baq_mode"][pi]))
        raw[r] = line[s0:s1]
        brc[r], thidx[r], baqm[r], valid[r] = b, t, pk["baq_mode"][pi], True
        if verbose and (k % 2000 == 0 or k == len(packets) - 1):
            el = time.time() - t0
            print(f"  decoded {k + 1}/{len(packets)} lines  ({el:.1f} s)", flush=True)
    side = {"brc": brc, "thidx": thidx, "baq_mode": baqm, "valid_line": valid,
            "sample_start": np.int64(s0), "num_quads": np.int64(nq)}
    if out_dir is not None:
        raw.flush()
        np.savez_compressed(Path(out_dir) / SIDE_FILE, **side)
    return raw, side, packets


def decode_l0_file(path, out_dir, chunk_index: Optional[int] = None, swath: Optional[int] = None,
                   line_start: int = 0, max_lines: Optional[int] = None, sample_start: int = 0,
                   max_samples: Optional[int] = None, backend: str = "builtin",
                   verbose: bool = True) -> Dict:
    """Decode an L0 measurement file into a raw dataset directory (raw.npy + metadata)."""
    path = Path(path)
    if backend == "sentinel1decoder":
        return _decode_with_sentinel1decoder(path, out_dir, chunk_index, line_start, max_lines,
                                             sample_start, max_samples, verbose)
    if verbose and not HAVE_NUMBA:
        print("WARNING: numba not installed - the pure-Python decoder is very slow "
              "(use --max-lines or `pip install numba`).")
    pk = index_packets(path)
    chunks = find_echo_chunks(pk)
    if swath is not None:
        chunks = [c for c in chunks if c["swath"] == swath]
    if not chunks:
        raise RuntimeError("no echo packets found")
    if verbose:
        print(f"{path.name}: {pk['offset'].size} packets, {len(chunks)} echo chunk(s)")
        for i, c in enumerate(chunks):
            print(f"  chunk {i}: swath {c['swath']}, pol {c['polarisation']}, {c['n_lines']} lines "
                  f"(missing {c['missing_lines']}), {2 * c['num_quads']} samples, "
                  f"{T.BAQ_MODE_NAMES.get(c['baq_mode'], c['baq_mode'])}")
    if chunk_index is None:
        chunk_index = int(np.argmax([c["n_packets"] for c in chunks]))
    chunk = chunks[chunk_index]
    if verbose:
        print(f"decoding chunk {chunk_index}")
    raw, side, packets = decode_chunk(path, pk, chunk, out_dir, line_start, max_lines,
                                      sample_start, max_samples, verbose)
    eph = parse_ephemeris(pk)
    meta = build_metadata(pk, chunk, packets, raw.shape[0], raw.shape[1], eph, source=str(path))
    if sample_start:
        meta["near_range"] += sample_start * T.C_LIGHT / (2.0 * meta["fs"])
        if eph:
            far = meta["near_range"] + (raw.shape[1] - 1) * T.C_LIGHT / (2.0 * meta["fs"])
            tmid = 0.5 * (meta["line_time_first"] + meta["line_time_last"])
            vr = effective_velocity(eph, tmid, np.array([meta["near_range"],
                                                         0.5 * (meta["near_range"] + far), far]))
            meta["effective_velocity_near"], meta["effective_velocity"], \
                meta["effective_velocity_far"] = (float(v) for v in vr)
    meta["chunk_index"] = int(chunk_index)
    meta["line_start"] = int(line_start)
    meta["sample_start"] = int(sample_start)
    save_json(meta, Path(out_dir) / "metadata.json")
    return meta


def _decode_with_sentinel1decoder(path, out_dir, chunk_index, line_start, max_lines,
                                  sample_start, max_samples, verbose):  # pragma: no cover
    """Experimental backend using the `sentinel1decoder` package (no BRC/THIDX side info)."""
    try:
        import sentinel1decoder
    except ImportError as e:
        raise ImportError("pip install sentinel1decoder  (or use --backend builtin)") from e
    l0 = sentinel1decoder.Level0File(str(path))
    try:
        summary = l0.get_chunks_summary()
        cands = summary.get("echo", [])
    except AttributeError:
        cands = list(l0.acquisition_chunks)
    if not cands:
        raise RuntimeError("no echo chunks found by sentinel1decoder")
    if chunk_index is None:
        sizes = [len(l0.get_acquisition_chunk_metadata(c)) for c in cands]
        chunk_index = cands[int(np.argmax(sizes))]
    data = np.asarray(l0.get_acquisition_chunk_data(chunk_index), dtype=np.complex64)
    md = l0.get_acquisition_chunk_metadata(chunk_index)
    l1 = data.shape[0] if max_lines is None else min(data.shape[0], line_start + max_lines)
    s1 = data.shape[1] if max_samples is None else min(data.shape[1], sample_start + max_samples)
    data = data[line_start:l1, sample_start:s1]
    row = md.iloc[line_start]

    def val(name):
        v = row[name]
        return getattr(v, "value", v)

    rgdec = int(val("Range Decimation"))
    fs = T.range_decimation_to_fs(rgdec)
    pri = float(val("PRI"))
    near = T.C_LIGHT / 2.0 * (int(val("Rank")) * pri + float(val("SWST")) + T.SWST_OFFSET)
    near += sample_start * T.C_LIGHT / (2.0 * fs)
    meta = {"sensor": "Sentinel-1", "source_file": str(path), "decoder": "sentinel1decoder",
            "wavelength": T.S1_WAVELENGTH, "prf": 1.0 / pri, "pri": pri, "fs": fs,
            "chirp_rate": float(val("Tx Ramp Rate")),
            "chirp_start_freq": float(val("Tx Pulse Start Frequency")),
            "pulse_length": float(val("Tx Pulse Length")), "near_range": near,
            "doppler_centroid": None, "baq_mode": int(val("BAQ Mode")),
            "n_lines": int(data.shape[0]), "n_samples": int(data.shape[1])}
    try:
        eph_df = l0.ephemeris
        eph = [{"time": float(r["POD Solution Data Timestamp"]), "x": float(r["X-axis position ECEF"]),
                "y": float(r["Y-axis position ECEF"]), "z": float(r["Z-axis position ECEF"]),
                "vx": float(r["X-axis velocity ECEF"]), "vy": float(r["Y-axis velocity ECEF"]),
                "vz": float(r["Z-axis velocity ECEF"])} for _, r in eph_df.iterrows()]
        tmid = float(eph[len(eph) // 2]["time"])
        far = near + (data.shape[1] - 1) * T.C_LIGHT / (2.0 * fs)
        vr = effective_velocity(eph, tmid, np.array([near, 0.5 * (near + far), far]))
        meta.update({"effective_velocity_near": float(vr[0]), "effective_velocity": float(vr[1]),
                     "effective_velocity_far": float(vr[2]), "ephemeris": eph})
    except Exception as e:  # noqa: BLE001
        print("WARNING: could not derive the effective velocity:", e)
        meta["effective_velocity"] = None
    from .utils import save_raw_dataset

    save_raw_dataset(out_dir, data, meta)
    return meta
