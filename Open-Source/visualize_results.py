import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_by_length_bin(results_dir, metric="f1", length_bins=None,
                              baseline_path="/home/da530038/lang-pro/Open-Source/correction_report.csv"):
    """
    Load evaluation CSVs from a folder and include baseline, group by token length bins, and plot metrics.
    """
    if length_bins is None:
        length_bins = [0, 100, 200, 300, 400, 500, 1000]
    bin_labels = [
        f"{length_bins[i]}-{length_bins[i+1]-1}" for i in range(len(length_bins)-1)]

    all_data = []

    # Load baseline
    if os.path.exists(baseline_path):
        base_df = pd.read_csv(baseline_path)
        base_df["source"] = "baseline"
        all_data.append(base_df)
    else:
        print("⚠️ Baseline file not found. Skipping...")

    # Load other experiment CSVs
    for file in os.listdir(results_dir):
        if file.endswith(".csv") and file.startswith("correction_csv_"):
            df = pd.read_csv(os.path.join(results_dir, file))
            df["source"] = file.replace(
                "correction_csv_", "").replace(".csv", "")
            all_data.append(df)

    if not all_data:
        print("No CSV files loaded.")
        return

    # Combine and bin
    data = pd.concat(all_data, ignore_index=True)
    data["length_bin"] = pd.cut(
        data["token_count"], bins=length_bins, labels=bin_labels, right=False)

    # Group by source and length_bin
    grouped = data.groupby(["source", "length_bin"]).agg({
        "precision": "mean",
        "recall": "mean",
        "ex_precision": "mean",
        "ex_recall": "mean",
        "ex_match": "mean"
    }).reset_index()

    # Compute F1 and ex_F1
    grouped["f1"] = 2 * grouped["precision"] * grouped["recall"] / \
        (grouped["precision"] + grouped["recall"])
    grouped["f1"] = grouped["f1"].fillna(0)

    grouped["ex_f1"] = 2 * grouped["ex_precision"] * \
        grouped["ex_recall"] / (grouped["ex_precision"] + grouped["ex_recall"])
    grouped["ex_f1"] = grouped["ex_f1"].fillna(0)
    numeric_cols = grouped.select_dtypes(include='number').columns
    grouped[numeric_cols] = grouped[numeric_cols].fillna(0)
    # Plot
    plt.figure(figsize=(12, 6))
    for source, group in grouped.groupby("source"):
        plt.plot(group["length_bin"], group[metric], marker="o", label=source)

    plt.title(f"{metric.replace('_', ' ').title()} by Token Length Bin")
    plt.xlabel("Token Count Bin")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.legend(title="Setting")
    plt.tight_layout()

    output_file = f"{metric}_by_token_length.png"
    plt.savefig(output_file)
    print(f"✅ Plot saved as: {output_file}")
    plt.show()


plot_metric_by_length_bin(
    "/home/da530038/lang-pro/Open-Source/results", metric="f1")
plot_metric_by_length_bin(
    "/home/da530038/lang-pro/Open-Source/results", metric="ex_f1")
plot_metric_by_length_bin(
    "/home/da530038/lang-pro/Open-Source/results", metric="precision")
plot_metric_by_length_bin(
    "/home/da530038/lang-pro/Open-Source/results", metric="ex_precision")
plot_metric_by_length_bin(
    "/home/da530038/lang-pro/Open-Source/results", metric="recall")
plot_metric_by_length_bin(
    "/home/da530038/lang-pro/Open-Source/results", metric="ex_recall")
