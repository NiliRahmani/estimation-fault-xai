"""Fault injection. MVP: a single constant sensor-bias fault on one channel."""
import numpy as np


def inject_sensor_bias(Y, channel, onset_idx, bias):
    """Add a constant bias to one measurement channel from onset_idx onward.

    Returns a copy of Y; the true fault source is (channel, onset_idx).
    """
    Y_faulted = Y.copy()
    Y_faulted[onset_idx:, channel] += bias
    return Y_faulted
