"""Evaluation metrics: detection, false alarms, latency, RMSE, faithfulness."""
import numpy as np


def false_alarm_rate(fire, onset_idx, warmup):
    """Fraction of fired steps in the healthy window [warmup, onset)."""
    healthy = fire[warmup:onset_idx]
    return float(np.mean(healthy)) if len(healthy) else np.nan


def detection_and_latency(fire, onset_idx):
    """Was the fault detected after onset, and how many steps later?"""
    fault_window = fire[onset_idx:]
    idx = np.where(fault_window)[0]
    if len(idx):
        return True, int(idx[0])  # latency in steps from onset
    return False, np.nan


def state_rmse(Xhat, X, warmup):
    """Root-mean-square state-estimation error (after the warmup transient)."""
    err = Xhat[warmup:] - X[warmup:]
    return float(np.sqrt(np.mean(err ** 2)))


def top1_faithfulness(argmax_channel, onset_idx, true_channel):
    """Fraction of post-onset steps where the top-attributed channel is the true one."""
    post = argmax_channel[onset_idx:]
    return float(np.mean(post == true_channel))
