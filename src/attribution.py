"""Explanation / attribution over the innovation vector.

Model-based attribution: each channel's normalized squared residual
(nu_i^2 / S_ii). The most-attributed channel is the explanation's guess at the
fault source. The random baseline picks a channel uniformly at random.
"""
import numpy as np


def model_based_argmax(NU, Sdiag):
    """Return (N,) array: index of the most-attributed channel at each step."""
    scores = (NU ** 2) / Sdiag
    return np.argmax(scores, axis=1)


def random_argmax(N, n_channels, rng):
    """Return (N,) array of uniformly random channel indices (chance baseline)."""
    return rng.integers(0, n_channels, size=N)
