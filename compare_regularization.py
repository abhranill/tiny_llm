
"""Compare TinyGPT regularization experiments."""

import csv
import math
from pathlib import Path

RESULTS_PATH = Path("experiments/regularization/results.csv")
REPORT_PATH = Path("experiments/regularization/comparison_report.csv")


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing experiment results: {RESULTS_PATH}"
        )

    with RESULTS_PATH.open("r", newline="", encoding="utf-8") as file:
        results = list(csv.DictReader(file))

    if not results:
        raise ValueError("No experiment results found.")

    for result in results:
        result["best_validation_loss"] = float(
            result["best_validation_loss"]
        )
        result["best_train_loss"] = float(result["best_train_loss"])
        result["dropout"] = float(result["dropout"])
        result["weight_decay"] = float(result["weight_decay"])
        result["best_epoch"] = int(result["best_epoch"])

    baselines = [
        result for result in results
        if result["experiment"] == "baseline"
    ]

    if len(baselines) != 1:
        raise ValueError("Expected exactly one baseline experiment.")

    baseline_loss = baselines[0]["best_validation_loss"]

    report_fields = [
        "experiment",
        "dropout",
        "weight_decay",
        "best_epoch",
        "best_train_loss",
        "best_validation_loss",
        "validation_perplexity",
        "improvement_vs_baseline_percent",
        "checkpoint",
    ]

    report = []

    for result in results:
        val_loss = result["best_validation_loss"]

        report.append({
            "experiment": result["experiment"],
            "dropout": result["dropout"],
            "weight_decay": result["weight_decay"],
            "best_epoch": result["best_epoch"],
            "best_train_loss": result["best_train_loss"],
            "best_validation_loss": val_loss,
            "validation_perplexity": round(math.exp(val_loss), 6),
            "improvement_vs_baseline_percent": round(
                (baseline_loss - val_loss) / baseline_loss * 100,
                4,
            ),
            "checkpoint": result["checkpoint"],
        })

    report.sort(key=lambda row: row["best_validation_loss"])

    with REPORT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(report)

    print("\n=== Regularization Experiment Report ===")

    for row in report:
        print(
            f"{row['experiment']:12s} | "
            f"dropout={row['dropout']:.1f} | "
            f"val_loss={row['best_validation_loss']:.4f} | "
            f"perplexity={row['validation_perplexity']:.3f} | "
            f"improvement={row['improvement_vs_baseline_percent']:+.2f}%"
        )

    print(f"\nReport saved to: {REPORT_PATH}")
    print("This report compares validation results; it does not prove")
    print("that the best configuration will generalize to new datasets.")


if __name__ == "__main__":
    main()
