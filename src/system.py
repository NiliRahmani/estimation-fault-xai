"""Linear mass-spring-damper plant with a 3-channel measurement model.

State x = [position, velocity]. The true plant is always simulated with the
nominal dynamics; the *filter* may be given mismatched dynamics (see estimators).
"""
import numpy as np
from scipy.signal import cont2discrete


def make_matrices(cfg):
    """Build discrete-time system matrices for the true plant and the mismatched model."""
    s = cfg["system"]
    m, k, c, dt = s["mass"], s["stiffness"], s["damping"], s["dt"]

    # Continuous-time dynamics: m*x'' + c*x' + k*x = u
    A_c = np.array([[0.0, 1.0], [-k / m, -c / m]])
    B_c = np.array([[0.0], [1.0 / m]])
    # Three measurement channels: position, velocity, position+velocity.
    C = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    D = np.zeros((3, 1))

    A_d, B_d, C_d, D_d, _ = cont2discrete((A_c, B_c, C, D), dt)

    # Mismatched model: filter assumes a different stiffness/damping.
    k2 = k * cfg["mismatch"]["stiffness_factor"]
    c2 = c * cfg["mismatch"]["damping_factor"]
    A_c2 = np.array([[0.0, 1.0], [-k2 / m, -c2 / m]])
    A_d2, B_d2, _, _, _ = cont2discrete((A_c2, B_c, C, D), dt)

    R = np.diag(np.array(cfg["measurement_noise_std"]) ** 2)
    Q = (cfg["process_noise_std"] ** 2) * np.eye(2)

    return {
        "A": A_d, "B": B_d, "C": C_d, "Q": Q, "R": R,
        "A_mis": A_d2, "B_mis": B_d2, "dt": dt,
    }


def input_signal(cfg):
    """Sinusoidal forcing so the system keeps moving (excites all channels)."""
    N = cfg["system"]["n_steps"]
    dt = cfg["system"]["dt"]
    amp = cfg["input"]["amplitude"]
    omega = cfg["input"]["omega"]
    return amp * np.sin(omega * dt * np.arange(N))


def simulate_true(cfg, mats, rng):
    """Simulate the true trajectory and noisy (fault-free) measurements.

    Returns
    -------
    X : (N, 2) true states
    Y : (N, 3) noisy measurements (no fault yet)
    U : (N,)   input signal
    """
    N = cfg["system"]["n_steps"]
    A, B, C, Q, R = mats["A"], mats["B"], mats["C"], mats["Q"], mats["R"]
    U = input_signal(cfg)

    x = np.zeros(2)
    X = np.zeros((N, 2))
    Y = np.zeros((N, 3))
    for t in range(N):
        X[t] = x
        Y[t] = C @ x + rng.multivariate_normal(np.zeros(3), R)
        w = rng.multivariate_normal(np.zeros(2), Q)
        x = A @ x + B.flatten() * U[t] + w
    return X, Y, U
