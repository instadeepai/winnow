#!/usr/bin/env python3
r"""Convert a spectrum Parquet dataset to MGF for Casanovo prediction.

Expects columns similar to Winnow-style exports:
  precursor_mz, precursor_charge, mz_array, intensity_array

Optional (for ``casanovo sequence --evaluate``):
  sequence (written as SEQ=...)

Example::

    uv run python scripts/parquet_to_mgf.py \\
      /path/to/dataset.parquet -o out.mgf --annotate

Then::

    casanovo sequence out.mgf -o results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Map PSI UNIMOD labels in sequences to MassIVE-KB / Casanovo tokenizer names.
# Raw ``C[UNIMOD:4]`` is otherwise parsed to unrecognized ``C[+4.000000]``.
_UNIMOD_TO_MSKB = (
    ("C[UNIMOD:4]", "C[Carbamidomethyl]"),
    ("M[UNIMOD:35]", "M[Oxidation]"),
    ("N[UNIMOD:7]", "N[Deamidated]"),
    ("Q[UNIMOD:7]", "Q[Deamidated]"),
    ("[UNIMOD:1]", "[Acetyl]"),
    ("[UNIMOD:5]", "[Carbamyl]"),
    ("[UNIMOD:385]", "[Ammonia-loss]"),
)


def normalize_sequence_for_casanovo(seq: str) -> str:
    """Rewrite UNIMOD brackets to names understood by ``MskbPeptideTokenizer``."""
    for old, new in _UNIMOD_TO_MSKB:
        seq = seq.replace(old, new)
    return seq


def _as_float_array(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=np.float64)
    a = np.asarray(x, dtype=np.float64).ravel()
    return a


def spectrum_to_mgf_block(
    *,
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    precursor_charge: int,
    title: str,
    seq: str | None,
    retention_seconds: float | None,
    scans: str | None,
) -> str:
    """Format one spectrum as an MGF ``BEGIN IONS`` ... ``END IONS`` block."""
    lines = ["BEGIN IONS", f"TITLE={title}", f"PEPMASS={precursor_mz:.8f}"]
    lines.append(f"CHARGE={int(precursor_charge)}+")
    if scans:
        lines.append(f"SCANS={scans}")
    if retention_seconds is not None:
        lines.append(f"RTINSECONDS={retention_seconds:.6f}")
    if seq:
        lines.append(f"SEQ={seq}")
    n = min(len(mz), len(intensity))
    for i in range(n):
        lines.append(f"{mz[i]} {intensity[i]}")
    lines.append("END IONS")
    lines.append("")
    return "\n".join(lines)


def _row_title(row, idx, columns, title_column: str | None) -> str:
    """TITLE from ``title_column`` if present, else the row index."""
    if title_column and title_column in columns:
        return str(row[title_column])
    return str(idx)


def _row_sequence(
    row, columns, seq_col: str, annotate: bool, normalize_seq: bool
) -> str | None:
    """Ground-truth sequence for ``SEQ=`` (optionally Casanovo-normalised)."""
    if not (annotate and seq_col in columns):
        return None
    value = row[seq_col]
    if not (pd.notna(value) and str(value).strip()):
        return None
    seq = str(value).strip()
    return normalize_sequence_for_casanovo(seq) if normalize_seq else seq


def _row_retention_seconds(
    row, columns, rt_col: str | None, rt_unit: str
) -> float | None:
    """Retention time in seconds, or ``None`` when absent."""
    if not (rt_col and rt_col in columns):
        return None
    value = row[rt_col]
    if pd.isna(value):
        return None
    return float(value) * 60.0 if rt_unit == "minutes" else float(value)


def _row_scans(row, columns, scans_col: str | None) -> str | None:
    """Scan number as a string, or ``None`` when absent."""
    if not (scans_col and scans_col in columns):
        return None
    value = row[scans_col]
    return None if pd.isna(value) else str(value)


def _row_arrays(row, idx, mz_col: str, intensity_col: str):
    """m/z and intensity arrays of a row; raises if their lengths differ."""
    mz = _as_float_array(row[mz_col])
    intensity = _as_float_array(row[intensity_col])
    if mz.size != intensity.size:
        raise ValueError(
            f"Row {idx}: mz_array length {mz.size} != intensity length {intensity.size}"
        )
    return mz, intensity


def parquet_to_mgf(
    parquet_path: Path,
    out_path: Path,
    *,
    annotate: bool,
    title_column: str | None,
    rt_unit: str,
    mz_col: str,
    intensity_col: str,
    precursor_mz_col: str,
    charge_col: str,
    seq_col: str,
    rt_col: str | None,
    scans_col: str | None,
    normalize_seq: bool,
) -> None:
    """Write every spectrum in ``parquet_path`` to ``out_path`` as MGF."""
    df = pd.read_parquet(parquet_path)

    required = {mz_col, intensity_col, precursor_mz_col, charge_col}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in Parquet: {sorted(missing)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = set(df.columns)

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, row in df.iterrows():
            mz, intensity = _row_arrays(row, idx, mz_col, intensity_col)
            fout.write(
                spectrum_to_mgf_block(
                    mz=mz,
                    intensity=intensity,
                    precursor_mz=float(row[precursor_mz_col]),
                    precursor_charge=int(row[charge_col]),
                    title=_row_title(row, idx, columns, title_column),
                    seq=_row_sequence(row, columns, seq_col, annotate, normalize_seq),
                    retention_seconds=_row_retention_seconds(
                        row, columns, rt_col, rt_unit
                    ),
                    scans=_row_scans(row, columns, scans_col),
                )
            )


def main() -> None:
    """Parse command-line arguments and convert the Parquet file to MGF."""
    p = argparse.ArgumentParser(
        description="Convert spectrum Parquet to MGF for Casanovo."
    )
    p.add_argument("parquet", type=Path, help="Input .parquet file")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .mgf path",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="Write SEQ= lines from the sequence column (for casanovo --evaluate)",
    )
    p.add_argument(
        "--title-column",
        default=None,
        help="Column for TITLE= (default: row index as string)",
    )
    p.add_argument(
        "--rt-unit",
        choices=("minutes", "seconds"),
        default="minutes",
        help="Unit for retention_time column → RTINSECONDS (default: minutes)",
    )
    p.add_argument("--mz-col", default="mz_array")
    p.add_argument("--intensity-col", default="intensity_array")
    p.add_argument("--precursor-mz-col", default="precursor_mz")
    p.add_argument("--charge-col", default="precursor_charge")
    p.add_argument("--seq-col", default="sequence")
    p.add_argument(
        "--rt-col",
        default="retention_time",
        help="Set empty string to omit RTINSECONDS",
    )
    p.add_argument("--scans-col", default="scan_number")
    p.add_argument(
        "--no-seq-normalization",
        action="store_true",
        help="Keep original UNIMOD strings in SEQ (may break casanovo --evaluate)",
    )

    args = p.parse_args()
    rt_col = args.rt_col if args.rt_col else None
    scans_col = args.scans_col if args.scans_col else None

    parquet_to_mgf(
        args.parquet,
        args.output,
        annotate=args.annotate,
        title_column=args.title_column,
        rt_unit=args.rt_unit,
        mz_col=args.mz_col,
        intensity_col=args.intensity_col,
        precursor_mz_col=args.precursor_mz_col,
        charge_col=args.charge_col,
        seq_col=args.seq_col,
        rt_col=rt_col,
        scans_col=scans_col,
        normalize_seq=not args.no_seq_normalization,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
