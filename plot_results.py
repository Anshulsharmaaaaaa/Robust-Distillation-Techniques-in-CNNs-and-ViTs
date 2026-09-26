import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV with columns experiment and accuracy")
    p.add_argument("--output", default="results/accuracy_plot.png")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    required = {"experiment", "accuracy"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")

    plt.figure(figsize=(10, 5))
    plt.bar(df["experiment"], df["accuracy"])
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Experiment")
    plt.title("Robust Distillation Experiment Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
