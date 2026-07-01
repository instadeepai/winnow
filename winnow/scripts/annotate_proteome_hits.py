"""Annotate Winnow predict outputs with proteome substring hits.

Post-process ``winnow predict`` folders: filter short peptides and add a
``proteome_hit`` column via FASTA matching. From a repo checkout::

    uv run python winnow/scripts/annotate_proteome_hits.py OUTPUT_FOLDER --fasta proteome.fasta
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Annotated, Any

import ahocorasick
import polars as pl
import typer
import yaml
from Bio import SeqIO
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

from winnow.utils.config_path import get_config_dir

logger = logging.getLogger(__name__)

_DEFAULT_RESIDUES = get_config_dir() / "residues.yaml"

_PROTEOME_JOIN_SEP = "\x1f"
_MOD_ROUND = re.compile(r"\([^)]*\)-?")
_MOD_SQUARE = re.compile(r"\[[^\]]*\]-?")


app = typer.Typer(add_completion=False, no_args_is_help=True)


def normalize_sequence(seq: str) -> str:
    """Normalise a peptide sequence by replacing I with L."""
    if seq:
        return seq.replace("I", "L")
    return seq


def load_proteome_haystack(fasta_file: Path | str) -> str:
    """Load a FASTA file into a string for substring matching."""
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
    """Strip mods and normalise I/L for proteome substring matching."""
    if not prediction or not isinstance(prediction, str):
        return ""
    s = _MOD_ROUND.sub("", prediction)
    s = _MOD_SQUARE.sub("", s)
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
    """Filter short peptides and annotate ``proteome_hit`` via substring matching."""
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


def metrics_from_residues_yaml(residues_path: Path) -> Metrics:
    """Build InstaNovo ``Metrics`` from a Winnow ``residues.yaml`` file."""
    with residues_path.open() as f:
        data = yaml.safe_load(f)
    residue_masses = data["residue_masses"]
    return Metrics(
        residue_set=ResidueSet(residue_masses=residue_masses),
        isotope_error_range=(0, 1),
    )


def resolve_path(path: Path) -> Path:
    """Resolve *path* (absolute or relative to the current working directory)."""
    p = path.expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def resolve_fasta_path(raw: str) -> Path:
    """Resolve a FASTA path (absolute or relative to the current working directory)."""
    return resolve_path(Path(raw))


def resolve_output_folder(path: Path) -> Path:
    """Resolve and validate a Winnow ``predict`` output directory."""
    folder = resolve_path(path)
    if not folder.is_dir():
        raise typer.BadParameter(f"Not a directory: {folder}")
    for name in ("preds_and_fdr_metrics.csv", "metadata.csv"):
        if not (folder / name).is_file():
            raise typer.BadParameter(
                f"Missing {name} in {folder} (expected a Winnow predict output folder)"
            )
    return folder


@app.command()
def main(
    output_folders: Annotated[
        list[Path],
        typer.Argument(
            help=(
                "One or more Winnow predict output folders, each containing "
                "``preds_and_fdr_metrics.csv`` and ``metadata.csv``."
            ),
        ),
    ],
    fasta: Annotated[
        Path,
        typer.Option(
            "--fasta",
            "-f",
            help="FASTA file for proteome substring matching.",
        ),
    ],
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
) -> None:
    """Annotate Winnow predictions with proteome substring hits."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not output_folders:
        raise typer.BadParameter(
            "Provide at least one Winnow predict output folder "
            "(e.g. the ``output_folder`` from ``winnow predict``)."
        )

    fasta_path = resolve_fasta_path(str(fasta))
    metrics = metrics_from_residues_yaml(residues_config)

    for raw_folder in output_folders:
        out_dir = resolve_output_folder(raw_folder)
        logger.info(
            "folder=%s fasta=%s min_residues=%s",
            out_dir,
            fasta_path,
            min_residue_length,
        )
        n_in, n_short, n_kept = annotate_prediction_folder(
            out_dir,
            fasta_path,
            metrics,
            min_residue_length=min_residue_length,
        )
        logger.info(
            "done folder=%s rows_in=%d removed_short=%d kept=%d",
            out_dir,
            n_in,
            n_short,
            n_kept,
        )


if __name__ == "__main__":
    app()
