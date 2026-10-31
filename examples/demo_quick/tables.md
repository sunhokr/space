# Evaluation on the synthetic test scene (`scripts/run_demo.py`, quick preset)

Synthetic coastal scene 2048 x 1442 raw samples, written as Sentinel-1 ISP packets, decoded and adapted.
CAE: reduced width (N=64, M=320), 64x64 patches, Haar start, 600 iterations per lambda on a CPU.
These numbers verify the pipeline; they are not the paper's results (see README section 6).

valid SLC region (az0, az1, rg0, rg1): (551, 1497, 0, 1056)


## Table I - SQNR (dB)

| bpp | BAQ | JPEG2000 | CAE | CAE-BAQ gain | CAE-JPEG2000 gain |
|---|---|---|---|---|---|
| 2.00 | n/a | 12.96 | n/a | n/a | n/a |
| 3.00 | 15.79 | 18.97 | 18.00 | 2.21 | -0.98 |
| 4.00 | 21.49 | n/a | 23.06 | 1.57 | n/a |

## Table II - phase error std (deg)

| bpp | BAQ | JPEG2000 | CAE | CAE-BAQ gain | CAE-JPEG2000 gain |
|---|---|---|---|---|---|
| 2.00 | n/a | 44.03 | n/a | n/a | n/a |
| 3.00 | 29.55 | 24.73 | 24.81 | 4.75 | -0.08 |
| 4.00 | 17.56 | n/a | 15.43 | 2.13 | n/a |

## Table III - complex coherence (%)

| bpp | BAQ | JPEG2000 | CAE | CAE-BAQ gain | CAE-JPEG2000 gain |
|---|---|---|---|---|---|
| 2.00 | n/a | 86.75 | n/a | n/a | n/a |
| 3.00 | 94.83 | 96.70 | 96.33 | 1.50 | -0.37 |
| 4.00 | 98.48 | n/a | 98.79 | 0.31 | n/a |

## Bit rate needed for a target SQNR

| target SQNR (dB) | BAQ | JPEG2000 | CAE | saving vs BAQ (%) | saving vs JPEG2000 (%) |
|---|---|---|---|---|---|
| 15.00 | 2.86 | 2.34 | 2.42 | 15.41 | -3.13 |
| 20.00 | 3.74 | 3.19 | 3.39 | 9.27 | -6.52 |

## All operating points

| method | setting | bpp | bpp_est | sqnr_db | phase_std_deg | phase_mean_abs_deg | coherence_local | coherence_global | sqnr_raw_db |
|---|---|---|---|---|---|---|---|---|---|
| BAQ | 2 bit | 2.033 | None | 10.462 | 45.555 | 29.878 | 0.857 | 0.954 | 9.308 |
| BAQ | 3 bit | 3.033 | None | 15.974 | 29.002 | 16.928 | 0.951 | 0.987 | 14.668 |
| BAQ | 4 bit | 4.033 | None | 21.676 | 17.166 | 8.987 | 0.986 | 0.997 | 20.322 |
| BAQ | 5 bit | 5.033 | None | 27.478 | 9.819 | 4.664 | 0.996 | 0.999 | 26.121 |
| JPEG2000 | target 1.5 bpp | 1.500 | None | 10.175 | 56.849 | 38.910 | 0.756 | 0.952 | 10.501 |
| JPEG2000 | target 2 bpp | 2.000 | None | 12.961 | 44.039 | 28.032 | 0.867 | 0.975 | 13.049 |
| JPEG2000 | target 2.5 bpp | 2.499 | None | 15.930 | 33.249 | 19.638 | 0.932 | 0.987 | 15.792 |
| JPEG2000 | target 3 bpp | 2.999 | None | 18.969 | 24.737 | 13.818 | 0.967 | 0.994 | 18.757 |
| JPEG2000 | target 4 bpp | 4.000 | None | 24.522 | 14.285 | 7.215 | 0.991 | 0.998 | 24.446 |
| CAE | lambda=16.0 | 2.223 | 2.215 | 14.010 | 34.743 | 21.153 | 0.926 | 0.980 | 13.160 |
| CAE | lambda=64.0 | 3.270 | 3.261 | 19.383 | 21.353 | 11.704 | 0.976 | 0.994 | 18.721 |
| CAE | lambda=256.0 | 4.293 | 4.281 | 24.530 | 13.045 | 6.546 | 0.993 | 0.998 | 24.029 |
| BAQ | 3 bit (no adaptation) | 3.033 | None | 16.114 | 28.386 | 16.464 | 0.954 | 0.988 | 14.897 |
