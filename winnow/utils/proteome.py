"""Proteome substring matching and ``proteome_hit`` annotation helpers."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import ahocorasick
import pandas as pd
import yaml
from Bio import SeqIO
from instanovo.utils.residues import ResidueSet

from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.utils.peptide import tokens_to_proforma

logger = logging.getLogger(__name__)

_PROTEOME_JOIN_SEP = "\x1f"
_MOD_ROUND = re.compile(r"\([^)]*\)-?")
_MOD_SQUARE = re.compile(r"\[[^\]]*\]-?")
_MOD_NUMERIC = re.compile(r"[+-]?\d+(?:\.\d+)?-?")


@dataclass(frozen=True)
class ProteomeAnnotationCounts:
    """Row counts from a proteome-hit annotation pass."""

    num_input: int
    num_removed_short: int
    num_kept: int


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


def residue_set_from_residues_yaml(residues_path: Path) -> ResidueSet:
    """Build an InstaNovo ``ResidueSet`` from a Winnow ``residues.yaml`` file."""
    with residues_path.open() as f:
        data = yaml.safe_load(f)
    residue_masses = data["residue_masses"]
    return ResidueSet(residue_masses=residue_masses)


def _peptide_string_for_match(row: pd.Series) -> str:
    """Build a mod-stripped I/L-normalised string for proteome matching."""
    if "prediction_untokenised" in row.index:
        raw = row["prediction_untokenised"]
        if isinstance(raw, str) and raw.strip():
            return processed_peptide_for_match(raw)

    prediction = row["prediction"]
    if isinstance(prediction, list):
        return processed_peptide_for_match(tokens_to_proforma(prediction))
    if isinstance(prediction, str):
        return processed_peptide_for_match(prediction)
    return ""


def annotate_calibration_dataset(
    dataset: CalibrationDataset,
    fasta_path: Path | str,
    residue_set: ResidueSet,
    *,
    min_residue_length: int = 7,
) -> tuple[CalibrationDataset, ProteomeAnnotationCounts]:
    """Filter short peptides and annotate ``proteome_hit`` on a calibration dataset.

    Prefers ``prediction_untokenised`` for substring matching when present; otherwise
    converts tokenised ``prediction`` to ProForma then strips modifications.

    Args:
        dataset: Loaded calibration dataset with a ``prediction`` column.
        fasta_path: Reference proteome FASTA.
        residue_set: Residue tokeniser used for length filtering.
        min_residue_length: Drop PSMs with fewer than this many residue tokens.

    Returns:
        Annotated dataset and row-count statistics.

    Raises:
        ValueError: If ``prediction`` is missing from metadata.
    """
    if "prediction" not in dataset.metadata.columns:
        raise ValueError(
            "annotate_calibration_dataset requires a 'prediction' column in dataset metadata."
        )

    num_input = len(dataset.metadata)
    if "proteome_hit" in dataset.metadata.columns:
        logger.info("Overwriting existing 'proteome_hit' column.")

    filtered = dataset.filter_entries(
        metadata_predicate=lambda row: (
            residue_token_count(row["prediction"], residue_set) < min_residue_length
        )
    )
    num_kept = len(filtered.metadata)
    num_removed_short = num_input - num_kept

    haystack = load_proteome_haystack(fasta_path)
    processed_peptides = [
        _peptide_string_for_match(row) for _, row in filtered.metadata.iterrows()
    ]
    hits = _batch_peptide_substring_hits(processed_peptides, haystack)
    filtered.metadata = filtered.metadata.copy()
    filtered.metadata["proteome_hit"] = hits

    stats = ProteomeAnnotationCounts(
        num_input=num_input,
        num_removed_short=num_removed_short,
        num_kept=num_kept,
    )
    return filtered, stats
