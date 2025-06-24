import os
import polars as pl
from Bio import SeqIO
import argparse
import logging


def normalize_sequence(seq: str) -> str:
    """Normalize a peptide sequence."""
    if seq is not None:
        return seq.replace("I", "L")
    return seq


def load_proteome_sequences(fasta_file: str) -> list:
    """Load all sequences from the FASTA file into a list for substring matching."""
    if not os.path.exists(fasta_file):
        logging.warning(f"FASTA file not found at {fasta_file}")
        return []

    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        normalized_seq = normalize_sequence(str(record.seq))
        sequences.append(normalized_seq)

    return sequences


def main(metadata_csv: str, fasta_file: str, output_csv: str) -> None:
    """Map peptides to proteomes using FASTA files."""
    logging.info(f"Loading proteome sequences from: {fasta_file}")
    proteome_sequences = load_proteome_sequences(fasta_file)
    logging.info(f"Loaded {len(proteome_sequences)} sequences from proteome")

    logging.info(f"Processing data with proteome file: {fasta_file}")

    # Read CSV with polars
    df = pl.read_csv(metadata_csv)

    logging.info(f"Found in metadata {len(df)} rows")

    # Preprocess peptides using native polars expressions
    df_processed = df.with_columns(
        [
            pl.col("prediction_untokenised")
            .str.replace_all(
                r"\(\+\d+\.?\d*\)", ""
            )  # Matches (+123), (+123.45), (+123.456)
            .str.replace_all(r"\[UNIMOD:\d+\]", "")  # Remove UNIMOD modifications
            .str.replace_all("I", "L", literal=True)  # Normalize I to L
            .alias("processed_peptide")
        ]
    )

    logging.info(f"Processed {len(df_processed)} rows")

    # Check for substring hits using polars contains method
    # Create a condition that checks if the peptide is contained in any proteome sequence
    contains_conditions = [
        pl.lit(seq).str.contains(pl.col("processed_peptide"), literal=True)
        for seq in proteome_sequences
    ]

    df_with_hits = df_processed.with_columns(
        [
            pl.fold(pl.lit(False), lambda acc, x: acc | x, contains_conditions).alias(
                "proteome_hit"
            )
        ]
    )

    num_hits = df_with_hits.select("proteome_hit").sum().item()
    logging.info(f"Found {num_hits} hits")

    # Drop the intermediate processed_peptide column
    df_final = df_with_hits.drop("processed_peptide")

    # Write results
    df_final.write_csv(output_csv)
    logging.info(f"Done. Results written to {output_csv}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
