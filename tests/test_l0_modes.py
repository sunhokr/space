"""Tests of the BAQ 3/4/5-bit and bypass user-data layouts, generic adaptation, JPEG2000."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvsarc import s1_tables as T  # noqa: E402
from cvsarc.adaptation import adapt_generic  # noqa: E402
from cvsarc.metrics import gain_tables, sqnr_db  # noqa: E402
from cvsarc.sentinel1_l0 import decode_user_data  # noqa: E402


class BitWriter:
    def __init__(self):
        self.bits = []

    def put(self, value, n):
        self.bits += [(int(value) >> k) & 1 for k in range(n - 1, -1, -1)]

    def align16(self):
        self.bits += [0] * ((-len(self.bits)) % 16)

    def bytes(self):
        b = self.bits + [0] * ((-len(self.bits)) % 8)
        return np.packbits(np.array(b, dtype=np.uint8))


@pytest.mark.parametrize("nbits", [3, 4, 5])
def test_baq_mode_layout(nbits):
    rng = np.random.default_rng(nbits)
    nq = 300  # 3 blocks, last one partial
    nblk = (nq + 127) // 128
    nlev = 1 << (nbits - 1)
    thidx = rng.integers(0, 120, nblk)
    codes = rng.integers(0, 1 << nbits, (4, nq))
    w = BitWriter()
    for ch in range(4):
        for i in range(nq):
            if ch == 2 and i % 128 == 0:
                w.put(thidx[i // 128], 8)
            w.put(codes[ch, i], nbits)
        w.align16()
    line, brc, th = decode_user_data(w.bytes(), nq, nbits)
    assert np.array_equal(th, thidx)
    exp = np.zeros((4, nq))
    for ch in range(4):
        for i in range(nq):
            c = codes[ch, i]
            v = T.baq_level(nbits, int(thidx[i // 128]), int(c & (nlev - 1)))
            exp[ch, i] = -v if c >> (nbits - 1) else v
    assert np.allclose(line.real[0::2], exp[0]) and np.allclose(line.real[1::2], exp[1])
    assert np.allclose(line.imag[0::2], exp[2]) and np.allclose(line.imag[1::2], exp[3])


def test_bypass_layout():
    rng = np.random.default_rng(0)
    nq = 77
    vals = rng.integers(-511, 512, (4, nq))
    w = BitWriter()
    for ch in range(4):
        for v in vals[ch]:
            w.put((1 << 9 if v < 0 else 0) | abs(int(v)), 10)
        w.align16()
    line, _, _ = decode_user_data(w.bytes(), nq, 0)
    assert np.array_equal(line.real[0::2], vals[0]) and np.array_equal(line.real[1::2], vals[1])
    assert np.array_equal(line.imag[0::2], vals[2]) and np.array_equal(line.imag[1::2], vals[3])


def test_generic_adaptation_fills_gaps():
    rng = np.random.default_rng(0)
    levels = np.array([0.0, 5.0, 12.0, 20.0])
    x = rng.choice(levels, (4, 512)) * rng.choice([-1, 1], (4, 512))
    y = rng.choice(levels, (4, 512)) * rng.choice([-1, 1], (4, 512))
    raw = (x + 1j * y).astype(np.complex64)
    out = adapt_generic(raw, seed=0, round_output=True)
    assert np.unique(np.abs(out.real)).size > 15
    # every value stays inside the decision cell of its level
    lo = {0.0: 0.0, 5.0: 2.5, 12.0: 8.5, 20.0: 16.0}
    hi = {0.0: 2.5, 5.0: 8.5, 12.0: 16.0, 20.0: 24.0}
    for a, b in ((raw.real, out.real), (raw.imag, out.imag)):
        for lev in levels:
            m = np.abs(a) == lev
            assert np.all(np.abs(b[m]) >= np.floor(lo[lev])) and np.all(np.abs(b[m]) <= np.ceil(hi[lev]))


def test_gain_tables_sign_convention():
    curves = {"CAE": {"bpp": [2, 4], "m": [10.0, 20.0]}, "BAQ": {"bpp": [2, 4], "m": [8.0, 16.0]}}
    rows = gain_tables(curves, "m", (3.0,), reference="CAE", higher_is_better=True)
    assert rows[0]["CAE-BAQ gain"] == pytest.approx(3.0)
    rows = gain_tables(curves, "m", (3.0,), reference="CAE", higher_is_better=False)
    assert rows[0]["CAE-BAQ gain"] == pytest.approx(-3.0)


def test_jpeg2000_rate_control():
    from cvsarc.jpeg2000_codec import JPEG2000Codec, jpeg2000_available

    if not jpeg2000_available():
        pytest.skip("no JPEG2000 encoder")
    rng = np.random.default_rng(0)
    raw = np.rint((rng.standard_normal((256, 384)) + 1j * rng.standard_normal((256, 384))) * 40).astype(np.complex64)
    rec2, info2 = JPEG2000Codec(2.0).compress_decompress(raw)
    rec4, info4 = JPEG2000Codec(4.0).compress_decompress(raw)
    assert info2["bpp"] <= 2.0 * 1.02 and info4["bpp"] <= 4.0 * 1.02
    assert info4["bpp"] > info2["bpp"]
    assert sqnr_db(raw, rec4) > sqnr_db(raw, rec2) > 3.0
