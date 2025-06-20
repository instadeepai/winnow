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


def load_proteome_sequences(fasta_file: str) -> set:
    """Load all sequences from the FASTA file into a set for fast lookup."""
    if not os.path.exists(fasta_file):
        logging.warning(f"FASTA file not found at {fasta_file}")
        return set()

    sequences = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        normalized_seq = normalize_sequence(str(record.seq))
        sequences.add(normalized_seq)

    return sequences


def main(metadata_csv: str, fasta_file: str, output_csv: str) -> None:
    """Map peptides to proteomes using FASTA files."""
    logging.info(f"Loading proteome sequences from: {fasta_file}")
    proteome_sequences = load_proteome_sequences(fasta_file)
    logging.info(f"Loaded {len(proteome_sequences)} unique sequences from proteome")

    logging.info(f"Processing data with proteome file: {fasta_file}")

    # Read CSV with polars
    df = pl.read_csv(metadata_csv)

    logging.info(f"Processing {len(df)} rows")

    # Preprocess peptides using native polars expressions
    df_processed = df.with_columns(
        [
            pl.col("prediction_untokenised")
            .str.replace_all(r"\[UNIMOD:\d+\]", "")  # Remove UNIMOD modifications
            .str.replace("I", "L")  # Normalize I to L
            .alias("processed_peptide")
        ]
    )

    logging.info(f"Processed {len(df_processed)} rows")

    # Check for hits using native polars expressions
    df_with_hits = df_processed.with_columns(
        [pl.col("processed_peptide").is_in(proteome_sequences).alias("proteome_hit")]
    )

    logging.info(f"Found {len(df_with_hits)} hits")

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
