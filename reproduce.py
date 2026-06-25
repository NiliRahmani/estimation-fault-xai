"""One-command reproduction of the MVP experiment.

    python reproduce.py

Writes:
    results/metrics_raw.csv      one row per run (2 conditions x 20 seeds)
    results/summary_table.csv    one row per condition
    results/figure1_faithfulness.png
"""
import os
import yaml

from src.experiment import run_all, summarize
from src.plotting import faithfulness_figure

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def main():
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(RESULTS, exist_ok=True)

    raw = run_all(cfg)
    raw.to_csv(os.path.join(RESULTS, "metrics_raw.csv"), index=False)

    summary = summarize(raw)
    summary.to_csv(os.path.join(RESULTS, "summary_table.csv"), index=False)

    fig_path = os.path.join(RESULTS, "figure1_faithfulness.png")
    faithfulness_figure(raw, cfg["n_channels"], fig_path)

    # Console report
    with_pct = summary.copy()
    print("\n=== Summary (mean over {} seeds per condition) ===".format(cfg["seed_count"]))
    print(with_pct.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nFigure written to {fig_path}")
    print("Tables written to results/metrics_raw.csv and results/summary_table.csv")


if __name__ == "__main__":
    main()
