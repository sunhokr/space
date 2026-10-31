"""Constants and look-up tables of the Sentinel-1 SAR space packet format.

Source: "Sentinel-1 SAR Space Packet Protocol Data Unit", S1-IF-ASD-PL-0007
(Issue 13), sections on FDBAQ / BAQ decoding (Huffman codes, simple
reconstruction parameters, normalized reconstruction levels, sigma factors).
The values were cross-checked against the open-source decoder
``sentinel1decoder`` (Rich Hall).
"""
from __future__ import annotations

import numpy as np

C_LIGHT = 299792458.0
F_REF = 37.53472224e6  # reference frequency [Hz]
S1_CARRIER_FREQ = 5.405e9  # [Hz]
S1_WAVELENGTH = C_LIGHT / S1_CARRIER_FREQ
SYNC_MARKER = 0x352EF853
SWST_OFFSET = 320.0 / (8.0 * F_REF)  # fixed delay added to SWST for the first sample [s]
WGS84_A = 6378137.0
WGS84_B = 6356752.314245

PRIMARY_HEADER_LEN = 6
SECONDARY_HEADER_LEN = 62
BAQ_BLOCK_QUADS = 128  # quads per (FD)BAQ block

# range decimation code -> (L, M); fs = L / M * 4 * F_REF
RGDEC_RATIOS = {0: (3, 4), 1: (2, 3), 3: (5, 9), 4: (4, 9), 5: (3, 8), 6: (1, 3),
                7: (1, 6), 8: (3, 7), 9: (5, 16), 10: (3, 26), 11: (4, 11)}
RGDEC_FILTER_BANDWIDTH = {0: 100e6, 1: 87.71e6, 3: 74.25e6, 4: 59.44e6, 5: 50.62e6, 6: 44.89e6,
                          7: 22.2e6, 8: 56.59e6, 9: 42.86e6, 10: 15.1e6, 11: 48.35e6}


def range_decimation_to_fs(code: int) -> float:
    L, M = RGDEC_RATIOS[int(code)]
    return L / M * 4.0 * F_REF


BAQ_MODE_NAMES = {0: "BYPASS", 3: "BAQ 3-bit", 4: "BAQ 4-bit", 5: "BAQ 5-bit",
                  12: "FDBAQ mode 0", 13: "FDBAQ mode 1", 14: "FDBAQ mode 2"}
FDBAQ_MODES = (12, 13, 14)
BAQ_MODES = (3, 4, 5)
POLARISATION_NAMES = {0: "H", 1: "HH", 2: "HV", 3: "H-VH", 4: "V", 5: "VH", 6: "VV", 7: "V-VH"}
SIGNAL_TYPE_ECHO = 0
SIGNAL_TYPE_NOISE = 1
ECC_NAMES = {1: "SM S1", 2: "SM S2", 3: "SM S3", 4: "SM S4", 5: "SM S5-N", 6: "SM S6",
             8: "IW", 9: "WV", 10: "SM S5-S", 11: "SM S1 (no int.cal)", 12: "SM S2 (no int.cal)",
             13: "SM S3 (no int.cal)", 14: "SM S4 (no int.cal)", 25: "SM S5-N (no int.cal)",
             26: "SM S5-S (no int.cal)", 27: "SM S6 (no int.cal)", 32: "EW"}

# --------------------------------------------------------------------------- #
# FDBAQ Huffman codes: magnitude code (mcode) -> bit string, sent *after* the sign bit
# --------------------------------------------------------------------------- #
HUFFMAN_CODES = {
    0: ["0", "10", "110", "111"],
    1: ["0", "10", "110", "1110", "1111"],
    2: ["0", "10", "110", "1110", "11110", "111110", "111111"],
    3: ["00", "01", "10", "110", "1110", "11110", "111110", "1111110", "11111110", "11111111"],
    4: ["00", "010", "011", "100", "101", "1100", "1101", "1110", "11110", "111110",
        "11111100", "11111101", "111111100", "111111101", "111111110", "111111111"],
}
FDBAQ_NUM_LEVELS = [len(HUFFMAN_CODES[b]) for b in range(5)]  # [4, 5, 7, 10, 16]

# simple-reconstruction thresholds: THIDX <= value -> simple method
FDBAQ_SIMPLE_THIDX_MAX = [3, 3, 5, 6, 8]
BAQ_SIMPLE_THIDX_MAX = {3: 3, 4: 5, 5: 10}

# simple reconstruction parameters (value used for the largest mcode)
SIMPLE_B = {
    0: [3.0, 3.0, 3.16, 3.53],
    1: [4.0, 4.0, 4.08, 4.37],
    2: [6.0, 6.0, 6.0, 6.15, 6.5, 6.88],
    3: [9.0, 9.0, 9.0, 9.0, 9.36, 9.50, 10.1],
    4: [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.22, 15.50, 16.05],
}
SIMPLE_A = {
    3: [3.0, 3.0, 3.12, 3.55],
    4: [7.0, 7.0, 7.0, 7.17, 7.4, 7.76],
    5: [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.44, 15.56, 16.11, 16.38, 16.65],
}

# normalized reconstruction levels
NRL_B = {
    0: [0.3637, 1.0915, 1.8208, 2.6406],
    1: [0.3042, 0.9127, 1.5216, 2.1313, 2.8426],
    2: [0.2305, 0.6916, 1.1528, 1.6140, 2.0754, 2.5369, 3.1191],
    3: [0.1702, 0.5107, 0.8511, 1.1916, 1.5321, 1.8726, 2.2131, 2.5536, 2.8942, 3.3744],
    4: [0.1130, 0.3389, 0.5649, 0.7908, 1.0167, 1.2428, 1.4687, 1.6947, 1.9206, 2.1466,
        2.3725, 2.5985, 2.8244, 3.0504, 3.2764, 3.6623],
}
NRL_A = {
    3: [0.2490, 0.7681, 1.3655, 2.1864],
    4: [0.1290, 0.3900, 0.6601, 0.9471, 1.2623, 1.6261, 2.0793, 2.7467],
    5: [0.0660, 0.1985, 0.3320, 0.4677, 0.6061, 0.7487, 0.8964, 1.0510, 1.2143, 1.3896,
        1.5800, 1.7914, 2.0329, 2.3234, 2.6971, 3.2692],
}

SIGMA_FACTORS = np.array([
    0.00, 0.63, 1.25, 1.88, 2.51, 3.13, 3.76, 4.39, 5.01, 5.64, 6.27, 6.89, 7.52, 8.15, 8.77, 9.40,
    10.03, 10.65, 11.28, 11.91, 12.53, 13.16, 13.79, 14.41, 15.04, 15.67, 16.29, 16.92, 17.55, 18.17,
    18.80, 19.43, 20.05, 20.68, 21.31, 21.93, 22.56, 23.19, 23.81, 24.44, 25.07, 25.69, 26.32, 26.95,
    27.57, 28.20, 28.83, 29.45, 30.08, 30.71, 31.33, 31.96, 32.59, 33.21, 33.84, 34.47, 35.09, 35.72,
    36.35, 36.97, 37.60, 38.23, 38.85, 39.48, 40.11, 40.73, 41.36, 41.99, 42.61, 43.24, 43.87, 44.49,
    45.12, 45.75, 46.37, 47.00, 47.63, 48.25, 48.88, 49.51, 50.13, 50.76, 51.39, 52.01, 52.64, 53.27,
    53.89, 54.52, 55.15, 55.77, 56.40, 57.03, 57.65, 58.28, 58.91, 59.53, 60.16, 60.79, 61.41, 62.04,
    62.98, 64.24, 65.49, 66.74, 68.00, 69.25, 70.50, 71.76, 73.01, 74.26, 75.52, 76.77, 78.02, 79.28,
    80.53, 81.78, 83.04, 84.29, 85.54, 86.80, 88.05, 89.30, 90.56, 91.81, 93.06, 94.32, 95.57, 96.82,
    98.08, 99.33, 100.58, 101.84, 103.09, 104.34, 105.60, 106.85, 108.10, 109.35, 110.61, 111.86,
    113.11, 114.37, 115.62, 116.87, 118.13, 119.38, 120.63, 121.89, 123.14, 124.39, 125.65, 126.90,
    128.15, 129.41, 130.66, 131.91, 133.17, 134.42, 135.67, 136.93, 138.18, 139.43, 140.69, 141.94,
    143.19, 144.45, 145.70, 146.95, 148.21, 149.46, 150.71, 151.97, 153.22, 154.47, 155.73, 156.98,
    158.23, 159.49, 160.74, 161.99, 163.25, 164.50, 165.75, 167.01, 168.26, 169.51, 170.77, 172.02,
    173.27, 174.53, 175.78, 177.03, 178.29, 179.54, 180.79, 182.05, 183.30, 184.55, 185.81, 187.06,
    188.31, 189.57, 190.82, 192.07, 193.33, 194.58, 195.83, 197.09, 198.34, 199.59, 200.85, 202.10,
    203.35, 204.61, 205.86, 207.11, 208.37, 209.62, 210.87, 212.13, 213.38, 214.63, 215.89, 217.14,
    218.39, 219.65, 220.90, 222.15, 223.41, 224.66, 225.91, 227.17, 228.42, 229.67, 230.93, 232.18,
    233.43, 234.69, 235.94, 237.19, 238.45, 239.70, 240.95, 242.21, 243.46, 244.71, 245.97, 247.22,
    248.47, 249.73, 250.98, 252.23, 253.49, 254.74, 255.99, 255.99,
], dtype=np.float64)
assert SIGMA_FACTORS.size == 256


def fdbaq_level(brc: int, thidx: int, mcode: int) -> float:
    """Unsigned reconstruction value of an FDBAQ sample (spec. 'simple'/'normal' methods)."""
    n = FDBAQ_NUM_LEVELS[brc]
    if not 0 <= mcode < n:
        raise ValueError("mcode out of range")
    if thidx <= FDBAQ_SIMPLE_THIDX_MAX[brc]:
        return float(mcode) if mcode < n - 1 else SIMPLE_B[brc][thidx]
    return NRL_B[brc][mcode] * SIGMA_FACTORS[thidx]


def baq_level(nbits: int, thidx: int, mcode: int) -> float:
    """Unsigned reconstruction value of a (fixed-rate) BAQ 3/4/5-bit sample."""
    n = 1 << (nbits - 1)
    if not 0 <= mcode < n:
        raise ValueError("mcode out of range")
    if thidx <= BAQ_SIMPLE_THIDX_MAX[nbits]:
        return float(mcode) if mcode < n - 1 else SIMPLE_A[nbits][thidx]
    return NRL_A[nbits][mcode] * SIGMA_FACTORS[thidx]


def fdbaq_lut() -> np.ndarray:
    """(5, 256, 16) float32 table of unsigned levels; unused entries are NaN."""
    lut = np.full((5, 256, 16), np.nan, dtype=np.float32)
    for b in range(5):
        for t in range(256):
            for m in range(FDBAQ_NUM_LEVELS[b]):
                lut[b, t, m] = fdbaq_level(b, t, m)
    return lut


def baq_lut(nbits: int) -> np.ndarray:
    """(256, 2**(nbits-1)) float32 table of unsigned BAQ levels."""
    n = 1 << (nbits - 1)
    lut = np.zeros((256, n), dtype=np.float32)
    for t in range(256):
        for m in range(n):
            lut[t, m] = baq_level(nbits, t, m)
    return lut


def huffman_tree_array() -> np.ndarray:
    """Binary decoding trie for the 5 BRC codes: tree[brc, node, bit] (leaf = -(mcode+1))."""
    tree = np.zeros((5, 64, 2), dtype=np.int16)
    for b in range(5):
        nodes = [[0, 0]]
        for m, code in enumerate(HUFFMAN_CODES[b]):
            node = 0
            for k, ch in enumerate(code):
                bit = int(ch)
                if k == len(code) - 1:
                    nodes[node][bit] = -(m + 1)
                else:
                    if nodes[node][bit] <= 0:
                        nodes.append([0, 0])
                        nodes[node][bit] = len(nodes) - 1
                    node = nodes[node][bit]
        tree[b, : len(nodes)] = np.array(nodes, dtype=np.int16)
    return tree


def huffman_encode_tables():
    """(code_value, code_length) arrays of shape (5, 16) for the magnitude codes."""
    val = np.zeros((5, 16), dtype=np.int64)
    length = np.zeros((5, 16), dtype=np.int64)
    for b in range(5):
        for m, code in enumerate(HUFFMAN_CODES[b]):
            val[b, m] = int(code, 2)
            length[b, m] = len(code)
    return val, length


def _check_prefix_codes() -> None:
    for b, codes in HUFFMAN_CODES.items():
        kraft = sum(2.0 ** -len(c) for c in codes)
        assert abs(kraft - 1.0) < 1e-12, f"BRC{b} Huffman code is not complete"
        for i, c in enumerate(codes):
            for j, d in enumerate(codes):
                assert i == j or not d.startswith(c), f"BRC{b}: code {c} is a prefix of {d}"


_check_prefix_codes()
