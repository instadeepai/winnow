#!/usr/bin/env python3
"""Post-process ``winnow predict`` outputs: min residue length + proteome substring hit.

Mapping ``project -> FASTA`` is passed on the command line (typically from the Makefile).
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Annotated, Any, Optional

import ahocorasick
import polars as pl
import typer
import yaml
from Bio import SeqIO
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RESIDUES = _REPO_ROOT / "winnow" / "configs" / "residues.yaml"

_PRESET_OUTPUT_SUFFIX: dict[str, str] = {
    "biological_validation_raw": "{project}_raw",
    "unlabelled_external": "{project}_unlabelled",
}

_PROTEOME_JOIN_SEP = "\x1f"
_MOD_PLUS = re.compile(r"\(\+\d+\.?\d*\)")
_MOD_UNIMOD = re.compile(r"\[UNIMOD:\d+\]-?")

app = typer.Typer(add_completion=False, no_args_is_help=True)


def normalize_sequence(seq: str) -> str:
    """Normalise a peptide sequence by replacing I with L.

    Args:
        seq: Peptide sequence to normalise.
    """
    if seq:
        return seq.replace("I", "L")
    return seq


def load_proteome_haystack(fasta_file: Path | str) -> str:
    """Load a FASTA file into a string for substring matching.

    Args:
        fasta_file: Path to the FASTA file containing the proteome.
    """
    path = Path(fasta_file)
    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    parts: list[str] = []
    for record in SeqIO.parse(path, "fasta"):
        s = normalize_sequence(str(record.seq))
        if s:
            parts.append(s)
    return _PROTEOME_JOIN_SEP.join(parts)


def processed_peptide_for_match(prediction: str) -> str:
    """Process a peptide string for substring matching.

    Args:
        prediction: Peptide string to process.
    """
    if not prediction or not isinstance(prediction, str):
        return ""
    s = _MOD_PLUS.sub("", prediction)
    s = _MOD_UNIMOD.sub("", s)
    return s.replace("I", "L")


def _batch_peptide_substring_hits(peptides: list[str], haystack: str) -> list[bool]:
    n = len(peptides)
    out = [False] * n
    if not haystack:
        return out

    by_peptide: dict[str, list[int]] = {}
    for i, p in enumerate(peptides):
        if not p:
            continue
        by_peptide.setdefault(p, []).append(i)

    if not by_peptide:
        return out

    auto = ahocorasick.Automaton()
    peptide_for_pid: list[str] = []
    for pid, pep in enumerate(by_peptide):
        auto.add_word(pep, pid)
        peptide_for_pid.append(pep)

    auto.make_automaton()
    matched_pids: set[int] = set()
    for _end_idx, pid in auto.iter(haystack):
        matched_pids.add(pid)

    for pid in matched_pids:
        pep = peptide_for_pid[pid]
        for row_i in by_peptide[pep]:
            out[row_i] = True
    return out


def residue_token_count(prediction: Any, metrics: Metrics) -> int:
    """Tokenizer residue count (``_split_peptide``), not raw string length."""
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
    try:
        return len(metrics._split_peptide(text))
    except Exception:
        return 0


def filter_and_annotate_preds(
    preds: pl.DataFrame,
    haystack: str,
    metrics: Metrics,
    min_residue_length: int,
) -> pl.DataFrame:
    """Filter and annotate Winnow predictions with proteome substring hits.

    Args:
        preds: Polars DataFrame containing Winnow predictions.
        haystack: String containing the proteome.
        metrics: Metrics object for InstaNovo ``Metrics`` / ``_split_peptide``.
        min_residue_length: Drop PSMs with fewer than this many tokenizer residues.
    """
    n_tok = preds["prediction"].map_elements(
        lambda x: residue_token_count(x, metrics),
        return_dtype=pl.Int32,
    )
    filtered = preds.with_columns(n_tok.alias("_n_residue_tokens")).filter(
        pl.col("_n_residue_tokens") >= min_residue_length
    )

    processed = filtered["prediction"].map_elements(
        lambda x: processed_peptide_for_match(x) if isinstance(x, str) else "",
        return_dtype=pl.Utf8,
    )
    hits = _batch_peptide_substring_hits(processed.to_list(), haystack)
    return filtered.drop("_n_residue_tokens").with_columns(
        pl.Series("proteome_hit", hits, dtype=pl.Boolean)
    )


def annotate_prediction_folder(
    output_folder: Path | str,
    fasta_path: Path | str,
    metrics: Metrics,
    *,
    min_residue_length: int = 7,
) -> tuple[int, int, int]:
    """Annotate Winnow predictions with proteome substring hits.

    Args:
        output_folder: Path to the output folder containing predictions and metadata.
        fasta_path: Path to the FASTA file containing the proteome.
        metrics: Metrics object for InstaNovo ``Metrics`` / ``_split_peptide``.
        min_residue_length: Drop PSMs with fewer than this many tokenizer residues.
    """
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

    n_in = preds.height
    haystack = load_proteome_haystack(fasta_path)
    annotated = filter_and_annotate_preds(
        preds, haystack, metrics, min_residue_length=min_residue_length
    )
    n_kept = annotated.height
    n_short = n_in - n_kept

    keep_ids = annotated.select("spectrum_id").unique()
    meta = pl.read_csv(meta_path)
    if "spectrum_id" not in meta.columns:
        raise ValueError(f"'spectrum_id' column missing in {meta_path}")
    meta_kept = meta.join(keep_ids, on="spectrum_id", how="inner")

    annotated.write_csv(preds_path)
    meta_kept.write_csv(meta_path)
    return n_in, n_short, n_kept


def _metrics_from_residues_yaml(residues_path: Path) -> Metrics:
    with residues_path.open() as f:
        data = yaml.safe_load(f)
    residue_masses = data["residue_masses"]
    return Metrics(
        residue_set=ResidueSet(residue_masses=residue_masses),
        isotope_error_range=(0, 1),
    )


def _resolve_fasta_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


@app.command()
def main(
    preset: Annotated[
        str,
        typer.Argument(
            help=(
                "biological_validation_raw (*_raw/) or unlabelled_external (*_unlabelled/)"
            ),
        ),
    ],
    map_entry: Annotated[
        Optional[list[str]],
        typer.Option(
            "--map",
            help="project=path to FASTA (repeat); omit or leave path empty to skip a project.",
        ),
    ] = None,
    predictions_root: Annotated[
        Path,
        typer.Option(
            "--predictions-root",
            "-p",
            help="Root containing ``<project>_<suffix>/`` output folders.",
        ),
    ] = _REPO_ROOT / "predictions" / "general_model",
    residues_config: Annotated[
        Path,
        typer.Option(
            "--residues-config",
            help="``residues.yaml`` for InstaNovo ``Metrics`` / ``_split_peptide``.",
        ),
    ] = _DEFAULT_RESIDUES,
    min_residue_length: Annotated[
        int,
        typer.Option(
            "--min-residue-length",
            "-m",
            help="Drop PSMs with fewer than this many tokenizer residues.",
        ),
    ] = 7,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Log only; do not write CSVs.",
        ),
    ] = False,
) -> None:
    """Annotate Winnow predictions with proteome substring hits.

    Args:
        preset: One of "biological_validation_raw" or "unlabelled_external".
        map_entry: List of project=path pairs (repeat for multiple projects).
        predictions_root: Root containing <project>_<suffix>/ output folders.
        residues_config: ``residues.yaml`` for InstaNovo ``Metrics`` / ``_split_peptide``.
        min_residue_length: Drop PSMs with fewer than this many tokenizer residues.
        dry_run: Log only; do not write CSVs.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if preset not in _PRESET_OUTPUT_SUFFIX:
        raise typer.BadParameter(
            f"Unknown preset {preset!r}. Expected one of: "
            f"{', '.join(sorted(_PRESET_OUTPUT_SUFFIX))}"
        )

    pairs: list[tuple[str, Path]] = []
    for entry in map_entry or []:
        if "=" not in entry:
            raise typer.BadParameter(
                f"Invalid --map (expected project=path): {entry!r}"
            )
        project, path = entry.split("=", 1)
        project = project.strip()
        path = path.strip()
        if not project:
            raise typer.BadParameter(f"Invalid --map (empty project): {entry!r}")
        if not path:
            logger.info("skip %s (empty path in --map)", project)
            continue
        pairs.append((project, _resolve_fasta_path(path)))

    if not pairs:
        logger.info("No --map entries; nothing to do.")
        raise typer.Exit(code=0)

    metrics = _metrics_from_residues_yaml(residues_config)
    suffix_tpl = _PRESET_OUTPUT_SUFFIX[preset]

    for project, fasta_path in pairs:
        folder_name = suffix_tpl.format(project=project)
        out_dir = predictions_root / folder_name
        logger.info(
            "project=%s folder=%s fasta=%s min_residues=%s dry_run=%s",
            project,
            out_dir,
            fasta_path,
            min_residue_length,
            dry_run,
        )
        if dry_run:
            continue
        n_in, n_short, n_kept = annotate_prediction_folder(
            out_dir,
            fasta_path,
            metrics,
            min_residue_length=min_residue_length,
        )
        logger.info(
            "done project=%s rows_in=%d removed_short=%d kept=%d",
            project,
            n_in,
            n_short,
            n_kept,
        )


if __name__ == "__main__":
    app()
