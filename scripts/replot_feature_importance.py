#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Column mapping from analyze_features.py
COLUMN_MAPPING = {
    "confidence": "Raw Confidence",
    "Mass Error": "Mass Error",
    "ion_matches": "Ion Matches",
    "ion_match_intensity": "Ion Match Intensity",
    "chimeric_ion_matches": "Chimeric Ion Matches",
    "chimeric_ion_match_intensity": "Chimeric Ion Match Intensity",
    "iRT error": "iRT Error",
    "margin": "Margin",
    "median_margin": "Median Margin",
    "entropy": "Entropy",
    "z-score": "Z-Score",
}


def plot_feature_importance_from_csv(
    csv_path: Path,
    output_dir: Path,
    column_mapping: dict,
) -> None:
    """Recreate feature importance plots from saved CSV with nicer column names.

    Args:
        csv_path: Path to the feature_importance_results.csv file
        output_dir: Directory to save the updated plots
        column_mapping: Dictionary mapping original names to display names
    """
    # Read the CSV file
    results = pd.read_csv(csv_path)

    # Apply column mapping to feature names
    results["Display_Feature"] = results["Feature"].map(
        lambda x: column_mapping.get(x, x)
    )

    # Create permutation importance plot
    plt.figure(figsize=(10, 8))

    # Sort by importance
    perm_data = results.sort_values("Permutation Importance")

    plt.barh(range(len(perm_data)), perm_data["Permutation Importance"])
    plt.yticks(range(len(perm_data)), perm_data["Display_Feature"])
    plt.xlabel("Importance Score")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(
        output_dir / "permutation_importance_remapped.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "permutation_importance_remapped.pdf", bbox_inches="tight")
    plt.close()

    # Create SHAP importance plot
    plt.figure(figsize=(10, 8))

    # Sort by SHAP importance
    shap_data = results.sort_values("Mean |SHAP| (impact on P(correct))")

    plt.barh(range(len(shap_data)), shap_data["Mean |SHAP| (impact on P(correct))"])
    plt.yticks(range(len(shap_data)), shap_data["Display_Feature"])
    plt.xlabel("Mean |SHAP| Value")
    plt.title("SHAP Feature Importance for P(correct)")
    plt.tight_layout()
    plt.savefig(
        output_dir / "shap_importance_remapped.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "shap_importance_remapped.pdf", bbox_inches="tight")
    plt.close()

    # Create comparison plot
    plt.figure(figsize=(12, 8))

    # Normalize both importance measures to 0-1 for comparison
    perm_norm = (
        results["Permutation Importance"] - results["Permutation Importance"].min()
    ) / (
        results["Permutation Importance"].max()
        - results["Permutation Importance"].min()
    )
    shap_norm = (
        results["Mean |SHAP| (impact on P(correct))"]
        - results["Mean |SHAP| (impact on P(correct))"].min()
    ) / (
        results["Mean |SHAP| (impact on P(correct))"].max()
        - results["Mean |SHAP| (impact on P(correct))"].min()
    )

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(
        {
            "Feature": results["Display_Feature"],
            "Permutation Importance (normalized)": perm_norm,
            "SHAP Importance (normalized)": shap_norm,
        }
    )

    # Sort by average of both measures
    comparison_df["Average"] = (
        comparison_df["Permutation Importance (normalized)"]
        + comparison_df["SHAP Importance (normalized)"]
    ) / 2
    comparison_df = comparison_df.sort_values("Average")

    # Create grouped bar plot
    x = np.arange(len(comparison_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    _ = ax.barh(
        x - width / 2,
        comparison_df["Permutation Importance (normalized)"],
        width,
        label="Permutation Importance",
        alpha=0.8,
    )
    _ = ax.barh(
        x + width / 2,
        comparison_df["SHAP Importance (normalized)"],
        width,
        label="SHAP Importance",
        alpha=0.8,
    )

    ax.set_xlabel("Normalized Importance Score")
    ax.set_title("Feature Importance Comparison (Normalized)")
    ax.set_yticks(x)
    ax.set_yticklabels(comparison_df["Feature"])
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        output_dir / "feature_importance_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "feature_importance_comparison.pdf", bbox_inches="tight")
    plt.close()

    print(f"Updated feature importance plots saved to {output_dir}")
    print("\nTop 5 features by permutation importance:")
    top_perm = results.nlargest(5, "Permutation Importance")
    for _, row in top_perm.iterrows():
        print(f"{row['Display_Feature']}: {row['Permutation Importance']:.4f}")

    print("\nTop 5 features by SHAP importance:")
    top_shap = results.nlargest(5, "Mean |SHAP| (impact on P(correct))")
    for _, row in top_shap.iterrows():
        print(
            f"{row['Display_Feature']}: {row['Mean |SHAP| (impact on P(correct))']:.4f}"
        )


def main():
    """Main function to recreate feature importance plots with nicer names."""
    parser = argparse.ArgumentParser(
        description="Recreate feature importance plots with nicer column names"
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default="feature_importance_results.csv",
        help="Path to the feature importance results CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=".",
        help="Directory to save the updated plots",
    )

    args = parser.parse_args()

    # Check if CSV file exists
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading feature importance data from: {args.csv_path}")
    print(f"Saving updated plots to: {args.output_dir}")

    # Create plots with remapped column names
    plot_feature_importance_from_csv(
        args.csv_path,
        args.output_dir,
        COLUMN_MAPPING,
    )


if __name__ == "__main__":
    main()
