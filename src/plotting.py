"""The single figure: top-1 faithfulness, Kalman vs SIF-style, nominal vs mismatch."""
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt

from src.experiment import CONDITIONS, ESTIMATORS

COLORS = {"Kalman": "#2b6cb0", "SIF-style": "#dd6b20"}


def faithfulness_figure(raw, n_channels, out_path):
    x = np.arange(len(CONDITIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, est in enumerate(ESTIMATORS):
        means = [raw[(raw.estimator == est) & (raw.condition == c)]["faith_model"].mean()
                 for c in CONDITIONS]
        stds = [raw[(raw.estimator == est) & (raw.condition == c)]["faith_model"].std()
                for c in CONDITIONS]
        offset = (i - (len(ESTIMATORS) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               label=est, color=COLORS.get(est))

    ax.axhline(1.0 / n_channels, ls="--", lw=1, color="gray",
               label=f"Chance (1/{n_channels})")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CONDITIONS])
    ax.set_ylabel("Top-1 faithfulness (post-onset)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Does the explanation point to the true faulty sensor?", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
