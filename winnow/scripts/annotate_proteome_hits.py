"""Annotate Winnow predict outputs with proteome substring hits.

Post-process ``winnow predict`` folders: filter short peptides and add a
``proteome_hit`` column via FASTA matching. Invoked via the CLI::

    winnow annotate-proteome-hits OUTPUT_FOLDER --fasta proteome.fasta
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import ahocorasick
import polars as pl
import yaml
from Bio import SeqIO
from instanovo.utils.residues import ResidueSet

logger = logging.getLogger(__name__)

_PROTEOME_JOIN_SEP = "\x1f"
_MOD_ROUND = re.compile(r"\([^)]*\)-?")
_MOD_SQUARE = re.compile(r"\[[^\]]*\]-?")
_MOD_NUMERIC = re.compile(r"[+-]?\d+(?:\.\d+)?-?")


def normalize_sequence(sequence: str) -> str:
    """Normalise a peptide sequence by replacing I with L."""
    if sequence:
        return sequence.replace("I", "L")
    return sequence


def load_proteome_haystack(fasta_file: Path | str) -> str:
    """Load a FASTA file into a string for substring matching."""
    path = Path(fasta_file)
    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    sequences: list[str] = []
    for record in SeqIO.parse(path, "fasta"):
        sequence = normalize_sequence(str(record.seq))
        if sequence:
            sequences.append(sequence)
    return _PROTEOME_JOIN_SEP.join(sequences)


def processed_peptide_for_match(prediction: str) -> str:
    """Strip mods and normalise I/L for proteome substring matching.

    Modifications of the following forms are stripped:
    - Round brackets (e.g. "(+43.006)")
    - Square brackets (e.g. "[+43.006]")
    - Trailing dashes (e.g. "[+43.006]-")
    - Unbracketed mass deltas (e.g. "+15.99", "-15.99", "15.99")
    """
    if not prediction or not isinstance(prediction, str):
        return ""
    sequence = _MOD_ROUND.sub("", prediction)
    sequence = _MOD_SQUARE.sub("", sequence)
    sequence = _MOD_NUMERIC.sub("", sequence)
    sequence = normalize_sequence(sequence)
    return sequence


def _batch_peptide_substring_hits(peptides: list[str], haystack: str) -> list[bool]:
    hits = [False] * len(peptides)
    if not haystack:
        return hits

    row_indices_by_peptide: dict[str, list[int]] = {}
    for row_index, peptide in enumerate(peptides):
        if not peptide:
            continue
        row_indices_by_peptide.setdefault(peptide, []).append(row_index)

    if not row_indices_by_peptide:
        return hits

    automaton = ahocorasick.Automaton()
    peptide_by_id: list[str] = []
    for peptide_id, peptide in enumerate(row_indices_by_peptide):
        automaton.add_word(peptide, peptide_id)
        peptide_by_id.append(peptide)

    automaton.make_automaton()
    matched_peptide_ids: set[int] = set()
    for _end_index, peptide_id in automaton.iter(haystack):
        matched_peptide_ids.add(peptide_id)

    for peptide_id in matched_peptide_ids:
        peptide = peptide_by_id[peptide_id]
        for row_index in row_indices_by_peptide[peptide]:
            hits[row_index] = True
    return hits


def residue_token_count(
    prediction: str | list[str] | float | None, residue_set: ResidueSet
) -> int:
    """Tokeniser residue count, not raw string length."""
    if prediction is None:
        return 0
    if isinstance(prediction, float) and math.isnan(prediction):
        return 0
    if isinstance(prediction, list):
        return len(prediction)
    if not isinstance(prediction, str):
        return 0
    text = prediction.strip()
    if not text:
        return 0
    return len(residue_set.tokenize(text))


def filter_and_annotate_preds(
    preds: pl.DataFrame,
    haystack: str,
    residue_set: ResidueSet,
    min_residue_length: int,
) -> pl.DataFrame:
    """Filter short peptides and annotate ``proteome_hit`` via substring matching."""
    token_counts = preds["prediction"].map_elements(
        lambda prediction: residue_token_count(prediction, residue_set),
        return_dtype=pl.Int32,
    )
    filtered = preds.with_columns(token_counts.alias("_n_residue_tokens")).filter(
        pl.col("_n_residue_tokens") >= min_residue_length
    )

    processed_peptides = filtered["prediction"].map_elements(
        lambda prediction: (
            processed_peptide_for_match(prediction)
            if isinstance(prediction, str)
            else ""
        ),
        return_dtype=pl.Utf8,
    )
    hits = _batch_peptide_substring_hits(processed_peptides.to_list(), haystack)
    return filtered.drop("_n_residue_tokens").with_columns(
        pl.Series("proteome_hit", hits, dtype=pl.Boolean)
    )


def annotate_prediction_folder(
    output_folder: Path | str,
    fasta_path: Path | str,
    residue_set: ResidueSet,
    *,
    min_residue_length: int = 7,
) -> tuple[int, int, int]:
    """Filter preds, annotate proteome hits, and write back CSVs in *output_folder*."""
    folder = Path(output_folder)
    preds_path = folder / "preds_and_fdr_metrics.csv"
    meta_path = folder / "metadata.csv"
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    preds = pl.read_csv(preds_path)
    if "prediction" not in preds.columns:
        raise ValueError(f"'prediction' column missing in {preds_path}")
    if "spectrum_id" not in preds.columns:
        raise ValueError(f"'spectrum_id' column missing in {preds_path}")

    num_input = preds.height
    haystack = load_proteome_haystack(fasta_path)
    annotated = filter_and_annotate_preds(
        preds, haystack, residue_set, min_residue_length=min_residue_length
    )
    num_kept = annotated.height
    num_removed_short = num_input - num_kept

    kept_spectrum_ids = annotated.select("spectrum_id").unique()
    metadata = pl.read_csv(meta_path)
    if "spectrum_id" not in metadata.columns:
        raise ValueError(f"'spectrum_id' column missing in {meta_path}")
    metadata_kept = metadata.join(kept_spectrum_ids, on="spectrum_id", how="inner")

    annotated.write_csv(preds_path)
    metadata_kept.write_csv(meta_path)
    return num_input, num_removed_short, num_kept


def residue_set_from_residues_yaml(residues_path: Path) -> ResidueSet:
    """Build an InstaNovo ``ResidueSet`` from a Winnow ``residues.yaml`` file."""
    with residues_path.open() as f:
        data = yaml.safe_load(f)
    residue_masses = data["residue_masses"]
    return ResidueSet(residue_masses=residue_masses)
