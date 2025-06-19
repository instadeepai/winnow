import os
import pandas as pd
from Bio import SeqIO
import argparse
from tqdm import tqdm
from typing import List
import re


def remove_unimod_mods(seq: str) -> str:
    """Remove UNIMOD modifications from a peptide sequence."""
    if pd.notnull(seq):
        return re.sub(r"\[UNIMOD:\d+\]", "", seq)
    return seq


def normalize_sequence(seq: str) -> str:
    """Normalize a peptide sequence."""
    if pd.notnull(seq):
        return seq.replace("I", "L")
    return seq


def preprocess_peptide(seq: str) -> str:
    """Preprocess a peptide sequence."""
    seq = remove_unimod_mods(seq)
    seq = normalize_sequence(seq)
    return seq


def find_peptide_in_proteome(peptide: str, fasta_file: str) -> bool:
    """Find a peptide in the specified proteome fasta file and return True if found."""
    if not os.path.exists(fasta_file):
        print(f"Warning: FASTA file not found at {fasta_file}")
        return False

    for record in SeqIO.parse(fasta_file, "fasta"):
        if peptide in normalize_sequence(str(record.seq)):
            return True

    return False


def main(metadata_csv: str, fasta_file: str, output_csv: str) -> None:
    """Map peptides to proteomes using FASTA files."""
    df = pd.read_csv(metadata_csv)

    print(f"Processing data with proteome file: {fasta_file}")

    # Prepare boolean column for results
    proteome_hit_col: List[bool] = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        raw_peptide = row["prediction_untokenised"]

        peptide = preprocess_peptide(raw_peptide)
        has_hit = find_peptide_in_proteome(peptide, fasta_file)
        proteome_hit_col.append(has_hit)

    df["proteome_hit"] = proteome_hit_col

    df.to_csv(output_csv, index=False)
    print(f"Done. Results written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map peptides to proteomes using FASTA files."
    )
    parser.add_argument("--metadata-csv", required=True, help="Input metadata CSV file")
    parser.add_argument(
        "--fasta-file", required=True, help="Specific FASTA file to search in"
    )
    parser.add_argument(
        "--output-csv", required=True, help="Output CSV file with mapping results"
    )
    args = parser.parse_args()
    main(args.metadata_csv, args.fasta_file, args.output_csv)
