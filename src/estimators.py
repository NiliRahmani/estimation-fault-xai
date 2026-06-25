"""Kalman filter. Returns state estimates plus the innovations needed for
detection (NIS) and attribution (per-channel residuals)."""
import numpy as np


def kalman_filter(Y, U, A, B, C, Q, R):
    """Run a linear Kalman filter.

    Parameters
    ----------
    Y : (N, 3) measurements
    U : (N,)   inputs
    A, B, C, Q, R : model matrices used BY THE FILTER (may be mismatched).

    Returns
    -------
    Xhat  : (N, 2) state estimates
    NU    : (N, 3) innovations (measurement - predicted measurement)
    Sdiag : (N, 3) diagonal of the innovation covariance per step
    NIS   : (N,)   normalized innovation squared
    """
    N = len(Y)
    n_x = A.shape[0]
    n_y = C.shape[0]

    xhat = np.zeros(n_x)
    P = np.eye(n_x)

    Xhat = np.zeros((N, n_x))
    NU = np.zeros((N, n_y))
    Sdiag = np.zeros((N, n_y))
    NIS = np.zeros(N)

    for t in range(N):
        # Predict. The plant obeys x[t] = A x[t-1] + B u[t-1], so predicting the
        # current state from the previous estimate uses the PREVIOUS input.
        u_prev = U[t - 1] if t > 0 else 0.0
        xpred = A @ xhat + B.flatten() * u_prev
        Ppred = A @ P @ A.T + Q

        # Innovation
        nu = Y[t] - C @ xpred
        S = C @ Ppred @ C.T + R
        Sinv = np.linalg.inv(S)

        NU[t] = nu
        Sdiag[t] = np.diag(S)
        NIS[t] = float(nu @ Sinv @ nu)

        # Update
        K = Ppred @ C.T @ Sinv
        xhat = xpred + K @ nu
        P = (np.eye(n_x) - K @ C) @ Ppred
        Xhat[t] = xhat

    return Xhat, NU, Sdiag, NIS


def sif_filter(Y, U, A, B, C, Q, R, delta):
    """Simplified Sliding-Innovation-Filter-style estimator.

    NOTE: this is a *simplified, SIF-style* robust estimator for comparison,
    NOT a reproduction of Gadsden's full SIF/SVSF formulations. The defining
    idea is kept: the corrective gain reacts directly to the innovation through
    a saturated sliding boundary layer rather than the model-derived Kalman
    gain, i.e.  K = C^+ * diag( sat(|nu| / delta) ),  x = x_pred + K nu.
    Because the gain does not depend on the (possibly wrong) model covariance,
    it is more robust to modeling error.

    A covariance P is propagated (Joseph form) ONLY so the same NIS detector
    and the same residual attribution can be applied as for the Kalman filter;
    it does not drive the SIF gain.

    `delta` : (n_y,) sliding boundary-layer widths (one per channel).
    """
    N = len(Y)
    n_x = A.shape[0]
    n_y = C.shape[0]
    C_pinv = np.linalg.pinv(C)  # (n_x, n_y)
    I = np.eye(n_x)

    xhat = np.zeros(n_x)
    P = np.eye(n_x)

    Xhat = np.zeros((N, n_x))
    NU = np.zeros((N, n_y))
    Sdiag = np.zeros((N, n_y))
    NIS = np.zeros(N)

    for t in range(N):
        # Predict (same prediction as the Kalman filter)
        u_prev = U[t - 1] if t > 0 else 0.0
        xpred = A @ xhat + B.flatten() * u_prev
        Ppred = A @ P @ A.T + Q

        # Innovation
        nu = Y[t] - C @ xpred
        S = C @ Ppred @ C.T + R
        NU[t] = nu
        Sdiag[t] = np.diag(S)
        NIS[t] = float(nu @ np.linalg.inv(S) @ nu)

        # SIF-style gain: saturate |innovation| / boundary layer in [0, 1]
        sat = np.clip(np.abs(nu) / delta, 0.0, 1.0)
        K = C_pinv @ np.diag(sat)  # (n_x, n_y)

        # Update
        xhat = xpred + K @ nu
        # Joseph-form covariance update (for the NIS detector only)
        ImKC = I - K @ C
        P = ImKC @ Ppred @ ImKC.T + K @ R @ K.T
        Xhat[t] = xhat

    return Xhat, NU, Sdiag, NIS
