#!/usr/bin/env python3
"""Script to calculate hits and recall at specific confidence cutoffs corresponding to FDR levels.

This script takes a dataset with boolean columns (proteome_hits or correct) and confidence scores,
then calculates the number of hits and recall at specified confidence cutoffs.
"""

import pandas as pd
import argparse
import os
from typing import Tuple
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_recall_at_cutoff(
    df: pd.DataFrame, confidence_col: str, hit_col: str, cutoff: float
) -> Tuple[int, int, float]:
    """Calculate the number of hits and recall at a specific confidence cutoff.

    Args:
        df: DataFrame containing the data
        confidence_col: Name of the confidence column
        hit_col: Name of the boolean hit column (proteome_hit or correct)
        cutoff: Confidence cutoff threshold

    Returns:
        Tuple of (hits_at_cutoff, total_hits, recall)
    """
    # Filter by confidence cutoff
    above_cutoff = df[df[confidence_col] >= cutoff]

    # Count hits above cutoff
    hits_at_cutoff = above_cutoff[hit_col].sum()

    # Total true hits in dataset
    total_hits = df[hit_col].sum()

    # Calculate recall
    recall = hits_at_cutoff / total_hits if total_hits > 0 else 0.0

    return hits_at_cutoff, total_hits, recall


def calculate_fdr_at_cutoff(
    df: pd.DataFrame, confidence_col: str, hit_col: str, cutoff: float
) -> Tuple[int, int, float]:
    """Calculate the FDR at a specific confidence cutoff.

    Args:
        df: DataFrame containing the data
        confidence_col: Name of the confidence column
        hit_col: Name of the boolean hit column
        cutoff: Confidence cutoff threshold

    Returns:
        Tuple of (false_positives, total_predictions, fdr)
    """
    # Filter by confidence cutoff
    above_cutoff = df[df[confidence_col] >= cutoff]

    if len(above_cutoff) == 0:
        return 0, 0, 0.0

    # Count false positives (predictions above cutoff that are incorrect)
    false_positives = (~above_cutoff[hit_col]).sum()

    # Total predictions above cutoff
    total_predictions = len(above_cutoff)

    # Calculate FDR
    fdr = false_positives / total_predictions if total_predictions > 0 else 0.0

    return false_positives, total_predictions, fdr


def process_dataset(
    file_path: str, confidence_col: str, hit_col: str, cutoff: float
) -> pd.DataFrame:
    """Process a single dataset file and calculate metrics for specified cutoffs.

    Args:
        file_path: Path to the CSV file
        confidence_col: Name of the confidence column
        hit_col: Name of the boolean hit column
        cutoff: Confidence cutoff to evaluate

    Returns:
        DataFrame with results for the specified cutoff
    """
    logger.info(f"Processing {file_path}")

    # Read the dataset
    df = pd.read_csv(file_path)

    # Check if required columns exist
    if confidence_col not in df.columns:
        raise ValueError(
            f"Confidence column '{confidence_col}' not found in {file_path}"
        )
    if hit_col not in df.columns:
        raise ValueError(f"Hit column '{hit_col}' not found in {file_path}")

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Total entries: {len(df)}")
    logger.info(f"Total hits: {df[hit_col].sum()}")

    hits, total_hits, recall = calculate_recall_at_cutoff(
        df, confidence_col, hit_col, cutoff
    )
    fp, total_pred, fdr = calculate_fdr_at_cutoff(df, confidence_col, hit_col, cutoff)

    results = {
        "confidence_cutoff": cutoff,
        "hits_at_cutoff": hits,
        "total_hits": total_hits,
        "recall": recall,
        "false_positives": fp,
        "total_predictions": total_pred,
        "empirical_fdr": fdr,
    }

    return pd.DataFrame([results])


def main() -> None:
    """Main function to calculate hits and recall at confidence cutoffs for FDR analysis."""
    parser = argparse.ArgumentParser(
        description="Calculate hits and recall at confidence cutoffs for FDR analysis"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file to process",
    )
    parser.add_argument(
        "--confidence-col",
        type=str,
        default="calibrated_confidence",
        help="Name of confidence column (default: calibrated_confidence)",
    )
    parser.add_argument(
        "--hit-col",
        type=str,
        default="proteome_hit",
        help="Name of boolean hit column (default: proteome_hit)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        help="Specific confidence cutoff to evaluate (default: 0.1 to 0.99)",
    )
    parser.add_argument("--output", type=str, help="Output CSV file for results")

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.csv_file):
        logger.error(f"File not found: {args.csv_file}")
        return

    try:
        results = process_dataset(
            args.csv_file, args.confidence_col, args.hit_col, args.cutoff
        )

        # Display summary
        print("\n" + "=" * 80)
        print("SUMMARY RESULTS")
        print("=" * 80)
        print(f"\nDataset: {os.path.basename(args.csv_file)}")
        print(f"{'Cutoff':<10} {'Hits':<8} {'Total':<8} {'Recall':<8} {'FDR':<8}")
        print("-" * 50)

        for _, row in results.iterrows():
            print(
                f"{row['confidence_cutoff']:<10.4f} {row['hits_at_cutoff']:<8} "
                f"{row['total_hits']:<8} {row['recall']:<8.3f} {row['empirical_fdr']:<8.3f}"
            )

        # Save results if output specified
        if args.output:
            results.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Error processing {args.csv_file}: {e}")


if __name__ == "__main__":
    main()
