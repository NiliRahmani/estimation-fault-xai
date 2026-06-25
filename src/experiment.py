"""Orchestration: run one trial (both estimators on the same data), and sweep
all conditions x seeds."""
import numpy as np
import pandas as pd

from src.system import make_matrices, simulate_true
from src.faults import inject_sensor_bias
from src.estimators import kalman_filter, sif_filter
from src.detection import nis_threshold, detect
from src.attribution import model_based_argmax, random_argmax
from src import metrics

CONDITIONS = ["nominal", "mismatch"]
ESTIMATORS = ["Kalman", "SIF-style"]


def _evaluate(estimator, condition, seed, fault_channel, X, Xhat, NU, Sdiag, NIS,
              onset_idx, warmup, thr, faith_random):
    """Turn one estimator's output into a metrics row."""
    fire = detect(NIS, thr)
    argmax_model = model_based_argmax(NU, Sdiag)
    detected, latency = metrics.detection_and_latency(fire, onset_idx)
    return {
        "estimator": estimator,
        "condition": condition,
        "seed": seed,
        "fault_channel": fault_channel,
        "detected": detected,
        "false_alarm_rate": metrics.false_alarm_rate(fire, onset_idx, warmup),
        "latency_steps": latency,
        "rmse": metrics.state_rmse(Xhat, X, warmup),
        "faith_model": metrics.top1_faithfulness(argmax_model, onset_idx, fault_channel),
        "faith_random": faith_random,
    }


def run_one(cfg, mats, condition, seed):
    """Run a single trial; return one metrics row per estimator (same data)."""
    rng = np.random.default_rng(seed)

    N = cfg["system"]["n_steps"]
    onset_idx = int(cfg["fault"]["onset_fraction"] * N)
    warmup = cfg["detection"]["warmup"]
    n_ch = cfg["n_channels"]
    thr = nis_threshold(n_ch, cfg["detection"]["alpha"])
    delta = cfg["sif"]["boundary_layer_sigmas"] * np.array(cfg["measurement_noise_std"])

    # True trajectory + fault-free measurements (true plant = nominal dynamics).
    X, Y, U = simulate_true(cfg, mats, rng)

    # Inject a sensor-bias fault on a randomly chosen channel.
    fault_channel = int(rng.integers(0, n_ch))
    bias = cfg["fault"]["bias_sigmas"] * cfg["measurement_noise_std"][fault_channel]
    Y_faulted = inject_sensor_bias(Y, fault_channel, onset_idx, bias)

    # Filter uses correct (nominal) or wrong (mismatch) dynamics.
    if condition == "mismatch":
        A, B = mats["A_mis"], mats["B_mis"]
    else:
        A, B = mats["A"], mats["B"]

    # Random-attribution baseline is estimator-independent (shared chance level).
    argmax_random = random_argmax(N, n_ch, rng)
    faith_random = metrics.top1_faithfulness(argmax_random, onset_idx, fault_channel)

    rows = []

    Xhat, NU, Sdiag, NIS = kalman_filter(Y_faulted, U, A, B, mats["C"], mats["Q"], mats["R"])
    rows.append(_evaluate("Kalman", condition, seed, fault_channel, X, Xhat, NU, Sdiag,
                          NIS, onset_idx, warmup, thr, faith_random))

    Xhat, NU, Sdiag, NIS = sif_filter(Y_faulted, U, A, B, mats["C"], mats["Q"], mats["R"], delta)
    rows.append(_evaluate("SIF-style", condition, seed, fault_channel, X, Xhat, NU, Sdiag,
                          NIS, onset_idx, warmup, thr, faith_random))

    return rows


def run_all(cfg):
    """Sweep both conditions over all seeds; one row per (estimator, condition, seed)."""
    mats = make_matrices(cfg)
    rows = []
    for condition in CONDITIONS:
        for seed in range(cfg["seed_count"]):
            rows.extend(run_one(cfg, mats, condition, seed))
    return pd.DataFrame(rows)


def summarize(raw):
    """Aggregate per-run results into a summary table (one row per estimator x condition)."""
    out = []
    for estimator in ESTIMATORS:
        for condition in CONDITIONS:
            d = raw[(raw["estimator"] == estimator) & (raw["condition"] == condition)]
            out.append({
                "estimator": estimator,
                "condition": condition,
                "detection_rate": d["detected"].mean(),
                "false_alarm_rate": d["false_alarm_rate"].mean(),
                "mean_latency_steps": d["latency_steps"].mean(skipna=True),
                "rmse": d["rmse"].mean(),
                "faith_model_mean": d["faith_model"].mean(),
                "faith_model_std": d["faith_model"].std(),
                "faith_random_mean": d["faith_random"].mean(),
            })
    return pd.DataFrame(out)
