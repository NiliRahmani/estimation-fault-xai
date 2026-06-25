"""Residual-based fault detection via the normalized innovation squared (NIS).

Under a correct model and no fault, NIS follows a chi-square distribution with
degrees of freedom = number of measurement channels. We flag a fault when NIS
exceeds the chi-square quantile at level `alpha`.
"""
import numpy as np
from scipy.stats import chi2


def nis_threshold(n_channels, alpha):
    return float(chi2.ppf(alpha, df=n_channels))


def detect(NIS, threshold):
    """Boolean array: True where a fault is flagged."""
    return NIS > threshold
