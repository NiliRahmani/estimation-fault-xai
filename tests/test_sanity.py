"""Sanity checks: with the correct model and no fault, the Kalman filter should
track well and its NIS should sit near its expected value (= #channels)."""
import os
import numpy as np
import yaml

from src.system import make_matrices, simulate_true
from src.estimators import kalman_filter, sif_filter

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "..", "config.yaml")


def _cfg():
    with open(CFG) as f:
        return yaml.safe_load(f)


def test_kalman_tracks_when_model_correct():
    cfg = _cfg()
    mats = make_matrices(cfg)
    rng = np.random.default_rng(0)
    X, Y, U = simulate_true(cfg, mats, rng)  # no fault
    Xhat, NU, Sdiag, NIS = kalman_filter(Y, U, mats["A"], mats["B"], mats["C"],
                                         mats["Q"], mats["R"])
    warmup = cfg["detection"]["warmup"]
    rmse = np.sqrt(np.mean((Xhat[warmup:] - X[warmup:]) ** 2))
    assert rmse < 0.2, f"RMSE too high for a correct model: {rmse}"


def test_nis_near_dof_when_model_correct():
    cfg = _cfg()
    mats = make_matrices(cfg)
    rng = np.random.default_rng(1)
    X, Y, U = simulate_true(cfg, mats, rng)
    _, _, _, NIS = kalman_filter(Y, U, mats["A"], mats["B"], mats["C"],
                                 mats["Q"], mats["R"])
    warmup = cfg["detection"]["warmup"]
    mean_nis = np.mean(NIS[warmup:])
    # Expected NIS ~ number of channels (3). Allow a generous band.
    assert 1.5 < mean_nis < 6.0, f"Mean NIS {mean_nis} far from expected dof=3"


def test_sif_tracks_when_model_correct():
    cfg = _cfg()
    mats = make_matrices(cfg)
    rng = np.random.default_rng(2)
    X, Y, U = simulate_true(cfg, mats, rng)  # no fault
    delta = cfg["sif"]["boundary_layer_sigmas"] * np.array(cfg["measurement_noise_std"])
    Xhat, NU, Sdiag, NIS = sif_filter(Y, U, mats["A"], mats["B"], mats["C"],
                                      mats["Q"], mats["R"], delta)
    warmup = cfg["detection"]["warmup"]
    rmse = np.sqrt(np.mean((Xhat[warmup:] - X[warmup:]) ** 2))
    assert rmse < 0.5, f"SIF-style RMSE unexpectedly high for a correct model: {rmse}"
