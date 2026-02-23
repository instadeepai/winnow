#!/usr/bin/env python3
"""Generate minimal sample data for winnow train and predict commands."""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path


def generate_sample_data():
    """Generate minimal sample IPC and CSV files for InstaNovo format."""
    output_dir = Path("examples/example_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 100
    spectrum_ids = [f"spectrum_{i}" for i in range(n_samples)]

    # Generate peptides using only valid amino acids (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
    # Note: Must avoid O, U, X, Z, B, J which are not standard amino acids
    peptides = [
        "PEPTIDEK",
        "MASSIVE",
        "PEPTIDES",
        "SEQQENCR",
        "PEPTIDE",
        "MASSIVE",
        "PEPTIDES",
        "SEQQENCR",
        "PEPTIDEK",
        "MASSIVE",
    ] * 10

    # Generate spectrum data (IPC format)
    # Calculate precursor_mass from mz and charge
    np.random.seed(42)  # For reproducibility
    precursor_mz = np.random.uniform(400, 1200, n_samples)
    precursor_charge = np.random.choice([2, 3, 4], n_samples)
    proton_mass = 1.007276
    precursor_mass = precursor_mz * precursor_charge - proton_mass * precursor_charge

    # Generate spectrum arrays (mz_array and intensity_array)
    mz_arrays = []
    intensity_arrays = []
    for _ in range(n_samples):
        n_peaks = np.random.randint(10, 50)
        mz_array = np.random.uniform(100, 1000, n_peaks).tolist()
        intensity_array = np.random.uniform(0.1, 1.0, n_peaks).tolist()
        mz_arrays.append(mz_array)
        intensity_arrays.append(intensity_array)

    # Create spectrum data DataFrame using polars
    spectrum_data = pl.DataFrame(
        {
            "spectrum_id": spectrum_ids,
            "precursor_mz": precursor_mz,
            "precursor_charge": precursor_charge.astype(int),
            "precursor_mass": precursor_mass,
            "retention_time": np.random.uniform(10, 60, n_samples),
            "sequence": peptides,  # Ground truth for training
            "mz_array": mz_arrays,
            "intensity_array": intensity_arrays,
        }
    )

    # Generate predictions (CSV format)
    predictions_data = {
        "spectrum_id": spectrum_ids,
        "predictions": peptides,
        "predictions_tokenised": [
            ", ".join(list(p))
            for p in peptides  # "P, E, P, T, I, D, E, K"
        ],
        "log_probs": np.log(np.random.uniform(0.1, 0.9, n_samples)),
        "sequence": peptides,  # Ground truth
    }

    # Add beam predictions (top 3 beams)
    # Generate valid alternative peptides for runner-up beams
    valid_aa = list("ACDEFGHIKLMNPQRSTVWY")
    np.random.seed(43)  # Different seed for beam alternatives
    for beam_idx in range(3):
        if beam_idx == 0:
            # Top beam uses the main prediction
            beam_predictions = peptides
        else:
            # Generate valid alternative peptides for runner-up beams
            beam_predictions = [
                "".join(np.random.choice(valid_aa, size=len(peptides[i])))
                for i in range(n_samples)
            ]
        predictions_data[f"predictions_beam_{beam_idx}"] = beam_predictions
        predictions_data[f"predictions_log_probability_beam_{beam_idx}"] = [
            np.log(np.random.uniform(0.1, 0.9)) for _ in range(n_samples)
        ]
        # Token log probabilities as string representation of list
        predictions_data[f"predictions_token_log_probabilities_beam_{beam_idx}"] = [
            str([float(np.log(np.random.uniform(0.5, 0.9))) for _ in range(len(p))])
            for p in peptides
        ]

    predictions_df = pd.DataFrame(predictions_data)

    # Save files
    spectrum_path = output_dir / "spectra.ipc"
    predictions_path = output_dir / "predictions.csv"

    spectrum_data.write_ipc(str(spectrum_path))
    predictions_df.to_csv(predictions_path, index=False)

    print("✓ Generated sample data:")
    print(f"  - {spectrum_path}")
    print(f"  - {predictions_path}")
    print("\n✓ You can now run:")
    print("  make train-sample")
    print("  make predict-sample")


if __name__ == "__main__":
    generate_sample_data()
