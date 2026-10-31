"""cvsarc - Complex-Valued autoencoder-based neural data compression for SAR raw data.

Re-implementation of

    R. Mohammadi Asiyabi, M. Datcu, A. Anghel, A. Focsa, M. Martone, P. Rizzoli,
    E. Imbembo, "Complex-Valued Autoencoder-Based Neural Data Compression for SAR
    Raw Data", IEEE Journal of Selected Topics in Signal Processing, vol. 19,
    no. 3, pp. 572-582, April 2025.  DOI: 10.1109/JSTSP.2025.3558651

Package layout
--------------
complex_layers   CV convolution / transposed convolution, CV-GDN / CV-IGDN
entropy_models   factorized-prior entropy bottleneck (Balle et al.)
entropy_coder    arithmetic coder used to produce the real bitstream
model            CV autoencoder of Fig. 2 + rate-distortion loss (Eq. 1, 2, 7)
codec            tiled compression of whole raw-data matrices with a trained model
baq              Block Adaptive Quantization baseline
jpeg2000_codec   JPEG2000 baseline (applied separately to Re / Im)
sentinel1_l0     Sentinel-1 Level-0 decoder (FDBAQ / BAQ / bypass) + metadata
adaptation       adaptation of FDBAQ-decoded raw data (ref. [20] of the paper)
ers_reader       ERS-1/2 CEOS raw-data reader (generalisation experiment)
focusing         Range-Doppler algorithm (raw data -> SLC)
metrics          SQNR, phase-error std, complex coherence, interpolation, gains
datasets         PyTorch datasets of 256x256 complex raw patches
simulation       synthetic stripmap raw data + FDBAQ-like L0 emulation (demo/tests)
evaluation       end-to-end experiment of Fig. 3 (compress -> focus -> metrics)
plotting         figures equivalent to Figs. 4-9 of the paper
"""

__version__ = "1.0.0"

PAPER_REFERENCE = (
    "R. Mohammadi Asiyabi et al., 'Complex-Valued Autoencoder-Based Neural Data "
    "Compression for SAR Raw Data', IEEE JSTSP, vol. 19, no. 3, pp. 572-582, 2025."
)
