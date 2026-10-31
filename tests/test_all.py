"""Unit / integration tests:  python -m pytest -q tests"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvsarc import entropy_coder as ac  # noqa: E402
from cvsarc import s1_tables as T  # noqa: E402
from cvsarc.adaptation import adapt_with_side_info, level_cells  # noqa: E402
from cvsarc.baq import BAQCodec, lloyd_max_gaussian, lloyd_max_sqnr_db  # noqa: E402
from cvsarc.codec import CAECodec  # noqa: E402
from cvsarc.complex_layers import ComplexConv2d, ComplexConvTranspose2d, ComplexGDN  # noqa: E402
from cvsarc.focusing import FocusParams, focus  # noqa: E402
from cvsarc.metrics import coherence, evaluate_pair, interp_at, sqnr_db  # noqa: E402
from cvsarc.model import CVSARCompressor, RateDistortionLoss  # noqa: E402


# --------------------------------------------------------------------- layers
def test_complex_conv_matches_definition():
    torch.manual_seed(0)
    conv = ComplexConv2d(3, 4, 5, stride=2, padding=2)
    x = torch.randn(2, 3, 16, 16, dtype=torch.complex64)
    y = conv(x)
    wr, wi = conv.weight_real, conv.weight_imag
    br, bi = conv.bias_real, conv.bias_imag
    yr = F.conv2d(x.real, wr, br, 2, 2) - F.conv2d(x.imag, wi, None, 2, 2)
    yi = F.conv2d(x.real, wi, bi, 2, 2) + F.conv2d(x.imag, wr, None, 2, 2)
    assert torch.allclose(y.real, yr, atol=1e-5) and torch.allclose(y.imag, yi, atol=1e-5)


def test_complex_conv_transpose_matches_definition():
    torch.manual_seed(0)
    tc = ComplexConvTranspose2d(4, 3, 5, stride=2, padding=2, output_padding=1)
    x = torch.randn(2, 4, 8, 8, dtype=torch.complex64)
    y = tc(x)
    assert y.shape == (2, 3, 16, 16)
    wr, wi = tc.weight_real, tc.weight_imag
    args = (2, 2, 1)
    yr = F.conv_transpose2d(x.real, wr, tc.bias_real, *args) - F.conv_transpose2d(x.imag, wi, None, *args)
    yi = F.conv_transpose2d(x.real, wi, tc.bias_imag, *args) + F.conv_transpose2d(x.imag, wr, None, *args)
    assert torch.allclose(y.real, yr, atol=1e-5) and torch.allclose(y.imag, yi, atol=1e-5)


def test_complex_conv_is_complex_linear():
    torch.manual_seed(1)
    conv = ComplexConv2d(2, 2, 3, padding=1, bias=False)
    x = torch.randn(1, 2, 8, 8, dtype=torch.complex64)
    a = torch.tensor(0.3 - 1.2j, dtype=torch.complex64)
    assert torch.allclose(conv(a * x), a * conv(x), atol=1e-5)


def test_gdn_preserves_phase_and_igdn_gradients():
    torch.manual_seed(0)
    g = ComplexGDN(6)
    ig = ComplexGDN(6, inverse=True)
    x = torch.randn(2, 6, 8, 8, dtype=torch.complex64, requires_grad=True)
    y = g(x)
    assert torch.allclose(torch.angle(y), torch.angle(x), atol=1e-5)
    (ig(y).abs() ** 2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad.real).all()


# --------------------------------------------------------------------- entropy coding
def test_arithmetic_coder_roundtrip_with_escapes():
    rng = np.random.default_rng(0)
    C, L = 5, 9
    cdfs = np.zeros((C, L + 2), dtype=np.int64)
    lens = np.zeros(C, dtype=np.int64)
    lo = np.zeros(C, dtype=np.int64)
    for c in range(C):
        n = int(rng.integers(2, L))
        p = rng.random(n) + 0.01
        cdf = ac.pmf_to_quantized_cdf(np.concatenate([p / p.sum() * 0.99, [0.01]]))
        cdfs[c, : n + 2] = cdf
        cdfs[c, n + 2:] = ac.TOTAL
        lens[c] = n + 1
        lo[c] = -int(rng.integers(0, 4))
    ac.check_cdfs(cdfs, lens)
    idx = rng.integers(0, C, 20000)
    sym = rng.integers(-12, 12, 20000)
    sym[::97] = 1000  # far escapes
    sym[::101] = -777
    data = ac.encode(sym, idx, cdfs, lens, lo)
    dec = ac.decode(data, idx, cdfs, lens, lo)
    assert np.array_equal(dec, sym)


def test_arithmetic_coder_rate_close_to_entropy():
    rng = np.random.default_rng(1)
    p = np.array([0.5, 0.25, 0.125, 0.0625, 0.0625 - 1e-4, 1e-4])
    cdf = ac.pmf_to_quantized_cdf(p)[None, :]
    lens = np.array([p.size])
    lo = np.array([0])
    sym = rng.choice(p.size - 1, size=50000, p=p[:-1] / p[:-1].sum())
    data = ac.encode(sym, np.zeros_like(sym), cdf, lens, lo)
    ent = -np.sum(p[:-1] / p[:-1].sum() * np.log2(p[:-1] / p[:-1].sum()))
    assert len(data) * 8 / sym.size < ent * 1.01 + 0.01


# --------------------------------------------------------------------- model
def _small_model():
    torch.manual_seed(0)
    return CVSARCompressor(N=8, M=12)


def test_model_shapes_and_loss():
    m = _small_model()
    x = torch.randn(2, 1, 64, 64, dtype=torch.complex64)
    out = m(x)
    assert out["x_hat"].shape == x.shape and out["y"].shape == (2, 12, 4, 4)
    assert out["likelihoods"].shape == (2, 24, 4, 4)
    losses = RateDistortionLoss(64.0)(out, x)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])


@pytest.mark.parametrize("basis", ["identity", "haar"])
def test_polyphase_init_is_exact_transform(basis):
    torch.manual_seed(0)
    m = CVSARCompressor(N=64, M=256).init_polyphase_(latent_scale=3.0, basis=basis)
    x = torch.randn(2, 1, 48, 64, dtype=torch.complex64)
    with torch.no_grad():
        y = m.g_a(x)
        xr = m.g_s(y)
    assert torch.allclose(xr, x, atol=1e-3)
    # orthonormal analysis: latent energy = input energy * latent_scale^2
    ratio = y[:, :256].abs().pow(2).mean() / x.abs().pow(2).mean()
    assert abs(float(ratio) - 9.0) < 0.5


def test_compress_decompress_equals_quantized_forward():
    m = _small_model().eval()
    m.update(force=True)
    x = torch.randn(2, 1, 64, 48, dtype=torch.complex64)
    with torch.no_grad():
        ref = m(x)["x_hat"]
        enc = m.compress(x)
        dec = m.decompress(enc["strings"], enc["shape"])["x_hat"]
    assert torch.allclose(ref, dec, atol=1e-4)


def test_checkpoint_roundtrip_and_codec(tmp_path):
    m = _small_model().eval()
    m.update(force=True)
    ck = tmp_path / "m.pt"
    torch.save({"model_state": m.state_dict(), "model_config": m.config(), "lmbda": 1.0,
                "normalization": {"mode": "patch_rms", "global_scale": 1.0},
                "train_config": {"patch_size": 64}}, ck)
    codec = CAECodec.from_checkpoint(ck)
    rng = np.random.default_rng(0)
    raw = (rng.standard_normal((100, 90)) + 1j * rng.standard_normal((100, 90))).astype(np.complex64) * 20
    rec, info = codec.compress_decompress(raw)
    assert rec.shape == raw.shape and info["bpp"] > 0 and np.isfinite(rec).all()
    blob, _ = codec.compress(raw)
    assert np.allclose(codec.decompress(blob), rec)


# --------------------------------------------------------------------- BAQ / metrics
def test_lloyd_max_known_values():
    _, lev1 = lloyd_max_gaussian(1)
    assert abs(lev1[-1] - 0.7979) < 1e-3
    _, lev2 = lloyd_max_gaussian(2)
    assert np.allclose(np.abs(lev2[2:]), [0.4528, 1.510], atol=2e-3)
    assert abs(lloyd_max_sqnr_db(3) - 14.62) < 0.05
    assert abs(lloyd_max_sqnr_db(4) - 20.22) < 0.05


@pytest.mark.parametrize("nbits", [2, 3, 4])
def test_baq_sqnr_gaussian(nbits):
    rng = np.random.default_rng(nbits)
    sig = np.repeat(rng.uniform(5, 50, (64, 1)), 512, axis=1)
    x = ((rng.standard_normal((64, 512)) + 1j * rng.standard_normal((64, 512))) * sig).astype(np.complex64)
    rec, info = BAQCodec(nbits).compress_decompress(x)
    assert abs(sqnr_db(x, rec) - lloyd_max_sqnr_db(nbits)) < 0.6
    assert abs(info["bpp"] - nbits - 8 / 256) < 0.01


def test_metrics_identity():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))).astype(np.complex64)
    m = evaluate_pair(a, a)
    assert m["sqnr_db"] == float("inf") and m["phase_std_deg"] < 1e-6
    assert abs(m["coherence_local"] - 1.0) < 1e-9 and abs(m["coherence_global"] - 1.0) < 1e-9
    assert np.allclose(interp_at([1, 2, 3], [10, 20, 30], [2.5, 4.0]), [25.0, np.nan], equal_nan=True)


# --------------------------------------------------------------------- Sentinel-1 tables / adaptation
def test_s1_tables():
    assert T.FDBAQ_NUM_LEVELS == [4, 5, 7, 10, 16]
    assert T.fdbaq_level(0, 2, 3) == 3.16 and T.fdbaq_level(4, 8, 15) == 16.05
    assert abs(T.fdbaq_level(4, 100, 15) - 3.6623 * T.SIGMA_FACTORS[100]) < 1e-9
    tree = T.huffman_tree_array()
    for b, codes in T.HUFFMAN_CODES.items():  # decode every code word through the trie
        for m, code in enumerate(codes):
            node = 0
            for ch in code:
                nxt = tree[b, node, int(ch)]
                if nxt < 0:
                    assert -nxt - 1 == m
                node = nxt


def test_level_cells():
    lo, hi = level_cells(np.array([0.0, 1.0, 2.0, 3.5]))
    assert np.allclose(lo, [0, 0.5, 1.5, 2.75]) and np.allclose(hi, [0.5, 1.5, 2.75, 4.25])


# --------------------------------------------------------------------- simulation / L0 / focusing
@pytest.fixture(scope="module")
def sim():
    from cvsarc.simulation import SimParams, simulate_raw

    sp = SimParams(n_az=512, n_rg=256, pulse_length=3e-6, chirp_bandwidth=40e6, seed=3)
    adc, info = simulate_raw(sp)
    return sp, adc, info


def test_l0_write_decode_roundtrip(sim, tmp_path):
    from cvsarc.sentinel1_l0 import decode_l0_file
    from cvsarc.simulation import fdbaq_encode, fdbaq_reconstruct, radar_parameters, write_l0_file
    from cvsarc.utils import load_raw_dataset

    sp, adc, info = sim
    adc = adc[:200]
    signs, mcodes, brc, thidx = fdbaq_encode(adc, info["noise_sigma_adc"])
    dec = fdbaq_reconstruct(signs, mcodes, brc, thidx)
    assert sqnr_db(adc, dec) > 8.0
    path = write_l0_file(tmp_path / "s1.dat", signs, mcodes, brc, thidx, radar_parameters(sp), sp)
    meta = decode_l0_file(path, tmp_path / "decoded", verbose=False)
    raw, meta2, side = load_raw_dataset(tmp_path / "decoded")
    assert raw.shape == dec.shape
    assert np.array_equal(np.asarray(raw), dec)  # bit-exact FDBAQ decoding
    assert np.array_equal(side["brc"], brc) and np.array_equal(side["thidx"], thidx)
    assert abs(meta["prf"] - info["prf"]) < 1e-6 and abs(meta["fs"] - info["fs"]) < 1e-3
    assert abs(meta["near_range"] - info["near_range"]) < 1e-3
    assert abs(meta["chirp_rate"] - info["chirp_rate"]) < 1.0
    assert abs(meta["chirp_start_freq"] - info["chirp_start_freq"]) < 1.0
    assert abs(meta["effective_velocity"] - info["effective_velocity"]) < 0.5
    # adaptation keeps every sample inside its FDBAQ decision cell and yields integers
    ada = adapt_with_side_info(np.asarray(raw), side, seed=0, round_output=False)
    ada_r = adapt_with_side_info(np.asarray(raw), side, seed=0, round_output=True)
    assert np.all(np.round(ada_r.real) == ada_r.real)
    assert sqnr_db(adc, ada) > sqnr_db(adc, dec) - 3.0
    assert np.unique(np.abs(ada.real)).size > 10 * np.unique(np.abs(dec.real)).size


def test_focus_params_precedence():
    meta = {"wavelength": 0.0555, "prf": 1700.0, "fs": 64e6, "chirp_rate": 7e12, "pulse_length": 6e-6,
            "near_range": 850e3, "effective_velocity": 7100.0, "focusing": {"azimuth_sign": 0}}
    p = FocusParams.from_metadata(meta, defaults={"azimuth_sign": -1, "range_window_alpha": 0.6})
    assert p.azimuth_sign == 0 and p.range_window_alpha == 0.6  # metadata beats config defaults
    p = FocusParams.from_metadata(meta, defaults={"azimuth_sign": -1}, azimuth_sign=1, doppler_centroid=None)
    assert p.azimuth_sign == 1 and p.doppler_centroid is None  # explicit override beats metadata


def test_focusing_point_target():
    from cvsarc.simulation import SimParams, radar_parameters, simulate_raw

    sp = SimParams(n_az=1024, n_rg=256, pulse_length=3e-6, chirp_bandwidth=40e6, snr_db=60.0,
                   doppler_centroid=40.0, seed=1)
    refl = np.zeros((sp.n_az, sp.n_rg), dtype=np.complex64)
    refl[500, 120] = 1.0
    adc, info = simulate_raw(sp, refl=refl)
    p = FocusParams(wavelength=info["wavelength"], prf=info["prf"], fs=info["fs"], chirp_rate=info["chirp_rate"],
                    pulse_length=info["pulse_length"], chirp_start_freq=info["chirp_start_freq"],
                    near_range=info["near_range"], velocity=info["effective_velocity"], azimuth_sign=0)
    slc, finfo = focus(adc, p)
    a, r = np.unravel_index(np.argmax(np.abs(slc)), slc.shape)
    assert abs(a - 500) <= 1 and abs(r - 120) <= 1
    assert finfo["azimuth_sign"] == -1
    assert abs(finfo["doppler_centroid"] - 40.0) < 15.0
    # energy concentration: >50% of the target energy within a 5x5 neighbourhood
    e = np.abs(slc) ** 2
    assert e[a - 2:a + 3, r - 2:r + 3].sum() / e.sum() > 0.5
