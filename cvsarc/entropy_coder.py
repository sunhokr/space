"""Arithmetic coder (32-bit, Witten-Neal-Cleary) used to write the real bitstream.

The paper losslessly encodes the quantized latent symbols with an arithmetic
coder driven by the learned entropy model (Sec. II-D, Fig. 2 "AE" / "AD").

* Every symbol stream is described by an integer CDF table per channel with
  ``PRECISION`` (=16) bits; ``cdf[c, 0] = 0`` and ``cdf[c, n_c] = 2**16`` where
  ``n_c`` (``cdf_lengths[c]``) counts the in-range symbols **plus one escape
  symbol** (index ``n_c - 1``).
* In-range symbol values are ``lo[c] .. lo[c] + n_c - 2``.  A value outside that
  range is coded as the escape symbol followed by an Elias-gamma code of its
  distance, written with equiprobable binary decisions, so coding is lossless
  for *any* integer input.

The core loops are compiled with numba when it is installed (strongly
recommended); otherwise the very same code runs in pure Python (slow but
correct).
"""
from __future__ import annotations

import numpy as np

try:  # optional JIT
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - fallback path
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(f):
            return f

        return deco

PRECISION = 16
TOTAL = 1 << PRECISION
_HALF_TOTAL = TOTAL >> 1
_TOP = (1 << 32) - 1
_HALF = 1 << 31
_Q1 = 1 << 30
_Q3 = 3 << 30

__all__ = ["PRECISION", "TOTAL", "HAVE_NUMBA", "encode", "decode", "pmf_to_quantized_cdf",
           "check_cdfs"]


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _emit(st, out, bit):
    # st = [low, high, pending, nbytes, cur, nbits]
    st[4] = (st[4] << 1) | bit
    st[5] += 1
    if st[5] == 8:
        out[st[3]] = st[4] & 0xFF
        st[3] += 1
        st[4] = 0
        st[5] = 0


@njit(cache=True)
def _emit_plus_pending(st, out, bit):
    _emit(st, out, bit)
    opp = 1 - bit
    while st[2] > 0:
        _emit(st, out, opp)
        st[2] -= 1


@njit(cache=True)
def _encode_interval(st, out, cl, ch):
    low = st[0]
    high = st[1]
    rng = high - low + 1
    high = low + ((rng * ch) >> 16) - 1
    low = low + ((rng * cl) >> 16)
    while True:
        if high < _HALF:
            _emit_plus_pending(st, out, 0)
        elif low >= _HALF:
            _emit_plus_pending(st, out, 1)
            low -= _HALF
            high -= _HALF
        elif low >= _Q1 and high < _Q3:
            st[2] += 1
            low -= _Q1
            high -= _Q1
        else:
            break
        low = low << 1
        high = (high << 1) | 1
    st[0] = low
    st[1] = high


@njit(cache=True)
def _encode_uniform_bits(st, out, value, nbits):
    for i in range(nbits - 1, -1, -1):
        if (value >> i) & 1:
            _encode_interval(st, out, _HALF_TOTAL, TOTAL)
        else:
            _encode_interval(st, out, 0, _HALF_TOTAL)


@njit(cache=True)
def _bit_length(v):
    n = 0
    while v > 0:
        v >>= 1
        n += 1
    return n


@njit(cache=True)
def _encode_core(symbols, cdf_idx, cdfs, cdf_lengths, lo, out):
    st = np.zeros(6, dtype=np.int64)
    st[1] = _TOP
    for n in range(symbols.shape[0]):
        c = cdf_idx[n]
        esc = cdf_lengths[c] - 1
        v = symbols[n] - lo[c]
        if v >= 0 and v < esc:
            _encode_interval(st, out, cdfs[c, v], cdfs[c, v + 1])
        else:
            _encode_interval(st, out, cdfs[c, esc], cdfs[c, esc + 1])
            if v < 0:
                u = 2 * (-v - 1)
            else:
                u = 2 * (v - esc) + 1
            val = u + 1
            nb = _bit_length(val)
            _encode_uniform_bits(st, out, 0, nb - 1)  # nb-1 leading zeros
            _encode_uniform_bits(st, out, val, nb)  # binary of val (MSB = 1)
    # termination
    st[2] += 1
    if st[0] < _Q1:
        _emit_plus_pending(st, out, 0)
    else:
        _emit_plus_pending(st, out, 1)
    if st[5] > 0:
        out[st[3]] = (st[4] << (8 - st[5])) & 0xFF
        st[3] += 1
    return st[3]


# --------------------------------------------------------------------------- #
# decoder
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _read_bit(data, pos, nbits_total):
    if pos >= nbits_total:
        return 0
    return (data[pos >> 3] >> (7 - (pos & 7))) & 1


@njit(cache=True)
def _target(dst):
    rng = dst[1] - dst[0] + 1
    return ((dst[2] - dst[0] + 1) * TOTAL - 1) // rng


@njit(cache=True)
def _decode_update(dst, data, nbits_total, cl, ch):
    # dst = [low, high, value, bitpos]
    low = dst[0]
    high = dst[1]
    value = dst[2]
    pos = dst[3]
    rng = high - low + 1
    high = low + ((rng * ch) >> 16) - 1
    low = low + ((rng * cl) >> 16)
    while True:
        if high < _HALF:
            pass
        elif low >= _HALF:
            low -= _HALF
            high -= _HALF
            value -= _HALF
        elif low >= _Q1 and high < _Q3:
            low -= _Q1
            high -= _Q1
            value -= _Q1
        else:
            break
        low = low << 1
        high = (high << 1) | 1
        value = (value << 1) | _read_bit(data, pos, nbits_total)
        pos += 1
    dst[0] = low
    dst[1] = high
    dst[2] = value
    dst[3] = pos


@njit(cache=True)
def _decode_uniform_bit(dst, data, nbits_total):
    if _target(dst) < _HALF_TOTAL:
        _decode_update(dst, data, nbits_total, 0, _HALF_TOTAL)
        return 0
    _decode_update(dst, data, nbits_total, _HALF_TOTAL, TOTAL)
    return 1


@njit(cache=True)
def _decode_core(data, nbits_total, cdf_idx, cdfs, cdf_lengths, lo, out):
    dst = np.zeros(4, dtype=np.int64)
    dst[1] = _TOP
    v = 0
    for i in range(32):
        v = (v << 1) | _read_bit(data, i, nbits_total)
    dst[2] = v
    dst[3] = 32
    for n in range(cdf_idx.shape[0]):
        c = cdf_idx[n]
        nsym = cdf_lengths[c]
        esc = nsym - 1
        t = _target(dst)
        a = 0
        b = nsym
        while b - a > 1:
            m = (a + b) >> 1
            if cdfs[c, m] <= t:
                a = m
            else:
                b = m
        _decode_update(dst, data, nbits_total, cdfs[c, a], cdfs[c, a + 1])
        if a < esc:
            out[n] = a + lo[c]
        else:
            zeros = 0
            while _decode_uniform_bit(dst, data, nbits_total) == 0:
                zeros += 1
                if zeros > 62:
                    break
            val = 1
            for _ in range(zeros):
                val = (val << 1) | _decode_uniform_bit(dst, data, nbits_total)
            u = val - 1
            if u % 2 == 0:
                vv = -(u // 2) - 1
            else:
                vv = (u - 1) // 2 + esc
            out[n] = vv + lo[c]
    return dst[3]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def _prep_tables(cdfs, cdf_lengths, lo):
    cdfs = np.ascontiguousarray(cdfs, dtype=np.int64)
    cdf_lengths = np.ascontiguousarray(cdf_lengths, dtype=np.int64)
    lo = np.ascontiguousarray(lo, dtype=np.int64)
    if cdfs.ndim != 2 or cdfs.shape[0] != cdf_lengths.shape[0] or lo.shape[0] != cdf_lengths.shape[0]:
        raise ValueError("inconsistent CDF table shapes")
    return cdfs, cdf_lengths, lo


def encode(symbols, cdf_idx, cdfs, cdf_lengths, lo) -> bytes:
    """Arithmetic-encode integer ``symbols``; ``cdf_idx[n]`` selects the table row."""
    cdfs, cdf_lengths, lo = _prep_tables(cdfs, cdf_lengths, lo)
    symbols = np.ascontiguousarray(np.clip(symbols, -(2 ** 40), 2 ** 40), dtype=np.int64).reshape(-1)
    cdf_idx = np.ascontiguousarray(cdf_idx, dtype=np.int64).reshape(-1)
    if symbols.shape != cdf_idx.shape:
        raise ValueError("symbols and cdf_idx must have the same length")
    if symbols.size == 0:
        return b""
    v = symbols - lo[cdf_idx]
    n_esc = int(np.count_nonzero((v < 0) | (v >= cdf_lengths[cdf_idx] - 1)))
    cap = 2 * symbols.size + 24 * n_esc + 64
    out = np.zeros(cap, dtype=np.uint8)
    n = int(_encode_core(symbols, cdf_idx, cdfs, cdf_lengths, lo, out))
    return out[:n].tobytes()


def decode(data: bytes, cdf_idx, cdfs, cdf_lengths, lo) -> np.ndarray:
    """Inverse of :func:`encode` (the number of symbols is ``len(cdf_idx)``)."""
    cdfs, cdf_lengths, lo = _prep_tables(cdfs, cdf_lengths, lo)
    cdf_idx = np.ascontiguousarray(cdf_idx, dtype=np.int64).reshape(-1)
    out = np.zeros(cdf_idx.size, dtype=np.int64)
    if cdf_idx.size == 0:
        return out
    buf = np.frombuffer(data, dtype=np.uint8) if len(data) else np.zeros(1, dtype=np.uint8)
    _decode_core(buf, len(data) * 8, cdf_idx, cdfs, cdf_lengths, lo, out)
    return out


def pmf_to_quantized_cdf(pmf, precision: int = PRECISION) -> np.ndarray:
    """Convert a probability vector to an integer CDF where every symbol has freq >= 1."""
    p = np.clip(np.asarray(pmf, dtype=np.float64), 0.0, None)
    n = p.size
    total = 1 << precision
    if n == 0 or n > total:
        raise ValueError("invalid pmf length")
    s = p.sum()
    p = p / s if s > 0 else np.full(n, 1.0 / n)
    scaled = p * (total - n)
    freq = np.floor(scaled).astype(np.int64) + 1
    remainder = int(total - freq.sum())
    if remainder > 0:
        order = np.argsort(-(scaled - np.floor(scaled)), kind="stable")
        freq[order[:remainder]] += 1
    cdf = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(freq, out=cdf[1:])
    assert cdf[-1] == total
    return cdf


def check_cdfs(cdfs, cdf_lengths) -> None:
    """Raise if a CDF table is not strictly increasing from 0 to 2**PRECISION."""
    cdfs = np.asarray(cdfs)
    for c, n in enumerate(np.asarray(cdf_lengths)):
        row = cdfs[c, : n + 1]
        if row[0] != 0 or row[-1] != TOTAL or np.any(np.diff(row) <= 0):
            raise ValueError(f"invalid CDF in row {c}")
