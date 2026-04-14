#!/usr/bin/env python3
"""Generate minimal sample data for winnow train and predict commands."""

import numpy as np
import pandas as pd
from pathlib import Path


def _format_mgf_spectrum(
    title: int,
    pepmass: float,
    charge: int,
    scans_f1: int,
    rt_seconds: float,
    sequence: str,
    mz_array: list[float],
    intensity_array: list[float],
) -> str:
    """Return one MGF record as a string."""
    lines = [
        "BEGIN IONS",
        f"TITLE={title}",
        f"PEPMASS={pepmass}",
        f"CHARGE={charge}+",
        f"SCANS=F1:{scans_f1}",
        f"RTINSECONDS={rt_seconds}",
        f"SEQ={sequence}",
    ]
    for mz, intensity in zip(mz_array, intensity_array, strict=True):
        lines.append(f"{mz} {intensity}")
    lines.append("END IONS")
    lines.append("")
    return "\n".join(lines)


def generate_sample_data() -> None:
    """Generate minimal sample MGF and CSV files for InstaNovo format."""
    output_dir = Path("examples/example_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 100
    # InstaNovo-style IDs: file stem + enumerate scan index (not MGF SCANS value)
    spectrum_ids = [f"spectra:{i}" for i in range(n_samples)]

    np.random.seed(42)  # For reproducibility

    # Generate peptides using only valid amino acids
    peptides = [
        "".join(
            np.random.choice(
                list("ACDEFGHIKLMNPQRSTVWY"), size=np.random.randint(5, 10)
            )
        )
        for _ in range(n_samples)
    ]

    precursor_mz = np.random.uniform(400, 1200, n_samples)
    precursor_charge = np.random.choice([2, 3, 4], n_samples)
    retention_time = np.random.uniform(10, 60, n_samples)
    # Realistic scan numbers monotonically increasing (not used for spectrum_id)
    base_scan = 2400
    scan_offsets = np.sort(np.random.randint(0, 5000, n_samples))

    mgf_parts: list[str] = []
    for i in range(n_samples):
        n_peaks = np.random.randint(10, 50)
        mz_array = np.random.uniform(100, 1000, n_peaks).tolist()
        intensity_array = np.random.uniform(0.1, 1.0, n_peaks).tolist()
        mgf_parts.append(
            _format_mgf_spectrum(
                title=i,
                pepmass=float(precursor_mz[i]),
                charge=int(precursor_charge[i]),
                scans_f1=int(base_scan + scan_offsets[i]),
                rt_seconds=float(retention_time[i]),
                sequence=peptides[i],
                mz_array=mz_array,
                intensity_array=intensity_array,
            )
        )

    # Generate predictions (CSV format)
    predictions_data: dict[str, object] = {
        "spectrum_id": spectrum_ids,
        "predictions": peptides,
        "predictions_tokenised": [", ".join(list(p)) for p in peptides],
        "log_probs": np.log(np.random.uniform(0.1, 0.9, n_samples)),
        "sequence": peptides,
    }

    valid_aa = list("ACDEFGHIKLMNPQRSTVWY")
    np.random.seed(43)  # Different seed for beam alternatives
    for beam_idx in range(3):
        if beam_idx == 0:
            beam_predictions = peptides
        else:
            beam_predictions = [
                "".join(np.random.choice(valid_aa, size=len(peptides[j])))
                for j in range(n_samples)
            ]
        predictions_data[f"predictions_beam_{beam_idx}"] = beam_predictions
        predictions_data[f"predictions_log_probability_beam_{beam_idx}"] = [
            np.log(np.random.uniform(0.1, 0.9)) for _ in range(n_samples)
        ]
        predictions_data[f"predictions_token_log_probabilities_beam_{beam_idx}"] = [
            str([float(np.log(np.random.uniform(0.5, 0.9))) for _ in range(len(p))])
            for p in peptides
        ]

    predictions_df = pd.DataFrame(predictions_data)

    spectrum_path = output_dir / "spectra.mgf"
    predictions_path = output_dir / "predictions.csv"

    spectrum_path.write_text("".join(mgf_parts), encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False)

    print("✓ Generated sample data:")
    print(f"  - {spectrum_path}")
    print(f"  - {predictions_path}")
    print("\n✓ You can now run:")
    print("  make train-sample")
    print("  make predict-sample")


if __name__ == "__main__":
    generate_sample_data()
