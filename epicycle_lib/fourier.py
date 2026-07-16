"""Samples -> Fourier coefficients (3b1b frequency order) + truncated reconstruction.

The reconstruction is the exact truncated-series path: the pen tip at t = i/N
equals recon[i], so the trace geometry, the ghost target and the inspect
preview all match precisely what the rotating vectors draw.
"""
from __future__ import annotations

import numpy as np


def fourier_series(samples: np.ndarray,
                   n_vectors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (coeffs, freqs, recon).

    Keeps the n_vectors largest coefficients by magnitude, ordered in the
    3b1b chain sequence (0, 1, -1, 2, -2, ...). The centroid was subtracted
    upstream, so the DC term is ~0 and never makes the cut.
    """
    n = len(samples)
    spectrum = np.fft.fft(samples) / n
    freqs = np.fft.fftfreq(n, d=1.0 / n)  # integer-valued frequencies

    selected = np.argsort(-np.abs(spectrum))[:min(n_vectors, n)]
    selected = sorted(selected, key=lambda i: (abs(freqs[i]), freqs[i] < 0))

    truncated = np.zeros_like(spectrum)
    truncated[selected] = spectrum[selected]
    recon = np.fft.ifft(truncated) * n

    return spectrum[selected], freqs[selected], recon
