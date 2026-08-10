"""Shared preprocessing helpers for FDR tool comparisons.

PSM comparison and the external peptide score-mixture share:
1. Method-specific load / NovoBoard mass-delta → ProForma conversion.
2. Pair-gated NovoBoard target-decoy filters (equal twin counts).
3. :func:`filter_prediction_table` / :func:`filter_novoboard_prediction_table`.

Labelled correctness uses Novor token matching; proteome-hit proxies use
PTM-stripped I→L substring search against an I→L FASTA haystack.

Peptide score-mixture only then:
4. :func:`max_score_per_peptide` (no re-filtering).
5. NovoBoard max-target → twin-decoy peptide TDC helpers.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

from scripts.annotate_preds_proteome_hits import (
    _batch_peptide_substring_hits,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESIDUES_YAML = _REPO_ROOT / "winnow" / "configs" / "residues.yaml"

MIN_PEPTIDE_LENGTH = 8
# Labelled / reference sets use Novor agreement, so short peptides are valid.
# Keep a non-empty-key floor only. Unlabelled sets keep MIN_PEPTIDE_LENGTH
# because correctness is proteome substring membership.
LABELLED_MIN_PEPTIDE_LENGTH = 1
_UNIMOD_RE = re.compile(r"\[UNIMOD:\d+\]")
_MOD_SQUARE = re.compile(r"\[.*?\]")
_MOD_PAREN = re.compile(r"\(.*?\)")
_NOVOBOARD_TO_PROFORMA = {
    "C(+57.02)": "C[UNIMOD:4]",
    "M(+15.99)": "M[UNIMOD:35]",
    "N(+0.98)": "N[UNIMOD:7]",
    "Q(+0.98)": "Q[UNIMOD:7]",
    "S(+79.97)": "S[UNIMOD:21]",
    "T(+79.97)": "T[UNIMOD:21]",
    "Y(+79.97)": "Y[UNIMOD:21]",
}


def normalize_peptide_key(peptide: object) -> str:
    """Normalise sequence-only peptide identity (strip PTMs, I→L)."""
    if isinstance(peptide, list):
        peptide = "".join(str(token) for token in peptide)
    if pd.isna(peptide) or not isinstance(peptide, str):
        return ""
    if len(peptide) > 4 and peptide[1] == "." and peptide[-2] == ".":
        peptide = peptide[2:-2]
    seq = _MOD_SQUARE.sub("", peptide)
    seq = _MOD_PAREN.sub("", seq)
    seq = "".join(c for c in seq if c.isalpha())
    return seq.replace("I", "L")


def has_unsupported_unimod(peptide: object) -> bool:
    """Return True when *peptide* still contains an unsupported ``[UNIMOD:n]`` token."""
    if pd.isna(peptide) or not isinstance(peptide, str):
        return True
    return bool(_UNIMOD_RE.search(peptide))


def novoboard_to_proforma(peptide: object) -> object:
    """Convert NovoBoard's supported mass-delta notation to ProForma."""
    if pd.isna(peptide) or not isinstance(peptide, str):
        return peptide
    converted = peptide
    for novoboard_mod, proforma_mod in _NOVOBOARD_TO_PROFORMA.items():
        converted = converted.replace(novoboard_mod, proforma_mod)
    return converted


def sequence_only_correct_prediction(sequence: object, prediction: object) -> bool:
    """Full sequence equality after PTM stripping and I/L normalisation."""
    sequence_key = normalize_peptide_key(sequence)
    prediction_key = normalize_peptide_key(prediction)
    return bool(sequence_key) and sequence_key == prediction_key


def sequence_only_correctness_mask(
    sequences: pd.Series, predictions: pd.Series
) -> np.ndarray:
    """Vectorized strip-PTM I→L equality (not for labelled Novor eval)."""
    return np.array(
        [
            sequence_only_correct_prediction(sequence, prediction)
            for sequence, prediction in zip(sequences, predictions)
        ],
        dtype=bool,
    )


def load_residue_masses(residues_yaml: Path | None = None) -> dict[str, float]:
    """Load residue masses from Winnow's residues YAML."""
    path = residues_yaml if residues_yaml is not None else DEFAULT_RESIDUES_YAML
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["residue_masses"]


@lru_cache(maxsize=4)
def _metrics_from_residue_masses_frozen(
    residues_items: tuple[tuple[str, float], ...],
) -> Metrics:
    residue_masses = dict(residues_items)
    return Metrics(
        residue_set=ResidueSet(residue_masses=residue_masses),
        isotope_error_range=(0, 1),
    )


def metrics_from_residue_masses(residue_masses: dict[str, float]) -> Metrics:
    """Build an InstaNovo ``Metrics`` instance for Novor matching."""
    items = tuple(sorted((str(k), float(v)) for k, v in residue_masses.items()))
    return _metrics_from_residue_masses_frozen(items)


def novor_correct_prediction(
    sequence: object,
    prediction: object,
    metrics: Metrics,
) -> bool:
    """Winnow/InstaNovo Novor correctness: full residue-token match."""
    if isinstance(sequence, list):
        gt = sequence
    elif pd.isna(sequence) or not isinstance(sequence, str) or not sequence:
        return False
    else:
        gt = metrics._split_peptide(sequence)

    if isinstance(prediction, list):
        pred = prediction
    elif pd.isna(prediction) or not isinstance(prediction, str) or not prediction:
        return False
    else:
        pred = metrics._split_peptide(prediction)

    if not gt or not pred:
        return False
    num_matches = metrics._novor_match(gt, pred)
    return bool(num_matches == len(gt) == len(pred))


def novor_correctness_mask(
    sequences: pd.Series | Iterable[object],
    predictions: pd.Series | Iterable[object],
    *,
    residue_masses: dict[str, float] | None = None,
    metrics: Metrics | None = None,
) -> np.ndarray:
    """Vectorized Novor correctness (same rule as ``DatabaseGroundedFDRControl.fit``)."""
    if metrics is None:
        masses = residue_masses if residue_masses is not None else load_residue_masses()
        metrics = metrics_from_residue_masses(masses)
    return np.array(
        [
            novor_correct_prediction(sequence, prediction, metrics)
            for sequence, prediction in zip(sequences, predictions)
        ],
        dtype=bool,
    )


def dedupe_best_score_per_peptide(
    df: pd.DataFrame, peptide_col: str, score_col: str
) -> pd.DataFrame:
    """Keep the highest-scoring row per peptide key."""
    return (
        df.sort_values(score_col, ascending=False)
        .groupby(peptide_col, as_index=False)
        .first()
    )


def compute_q_values(fdr: np.ndarray) -> np.ndarray:
    """Convert ranked FDR estimates to q-values using suffix minima."""
    values = np.asarray(fdr, dtype=float)
    q_values = np.empty_like(values)
    fdr_min = np.inf
    for i in range(len(values) - 1, -1, -1):
        fdr_min = min(fdr_min, values[i])
        q_values[i] = fdr_min
    return q_values


def monotonize_q_by_confidence(
    confidence: np.ndarray, q_value: np.ndarray
) -> np.ndarray:
    """Enforce non-increasing q-values when confidence increases."""
    order = np.argsort(-np.asarray(confidence, dtype=float))
    q_sorted = np.asarray(q_value, dtype=float)[order]
    q_mono = np.empty_like(q_sorted)
    q_min = np.inf
    for i in range(len(q_sorted) - 1, -1, -1):
        if q_sorted[i] > q_min:
            q_mono[i] = q_min
        else:
            q_mono[i] = q_sorted[i]
            q_min = q_sorted[i]
    out = np.empty_like(q_mono)
    out[order] = q_mono
    return out


def add_peptide_key(
    df: pd.DataFrame,
    peptide_col: str,
    *,
    key_col: str = "peptide_key",
) -> pd.DataFrame:
    """Append normalised peptide keys."""
    work = df.copy()
    work[key_col] = work[peptide_col].map(normalize_peptide_key)
    return work


def filter_prediction_table(
    df: pd.DataFrame,
    peptide_col: str,
    *,
    min_length: int = MIN_PEPTIDE_LENGTH,
    key_col: str = "peptide_key",
    drop_unsupported_mods: bool = True,
    log: bool = True,
) -> pd.DataFrame:
    """Drop unsupported mods and peptides shorter than *min_length* (normalised)."""
    work = add_peptide_key(df, peptide_col, key_col=key_col)
    before = len(work)
    if drop_unsupported_mods:
        work = work[~work[peptide_col].map(has_unsupported_unimod)].copy()
    work = work[work[key_col].str.len() >= min_length].copy()
    dropped = before - len(work)
    if log and dropped:
        logger.info(
            "Filtered %d/%d rows (unsupported mods and/or length < %d) on %s",
            dropped,
            before,
            min_length,
            peptide_col,
        )
    return work.reset_index(drop=True)


def filter_novoboard_prediction_table(
    df: pd.DataFrame,
    *,
    peptide_col: str = "Peptide",
    min_length: int = MIN_PEPTIDE_LENGTH,
    key_col: str = "_peptide_key",
    log: bool = True,
) -> pd.DataFrame:
    """Convert NovoBoard modifications to ProForma, then apply shared filters."""
    work = df.copy()
    work[peptide_col] = work[peptide_col].map(novoboard_to_proforma)
    return filter_prediction_table(
        work,
        peptide_col,
        min_length=min_length,
        key_col=key_col,
        log=log,
    )


def filter_novoboard_target_decoy_pairs(
    target: pd.DataFrame,
    decoy: pd.DataFrame,
    *,
    peptide_col: str = "Peptide",
    min_length: int = MIN_PEPTIDE_LENGTH,
    key_col: str = "_peptide_key",
    log: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter NovoBoard target/decoy as spectrum twins so pair counts stay equal.

    Both sides are converted to ProForma and passed through the shared
    mod/length filter. Only ``_pair_key`` values present in **both** filtered
    tables are kept, so dropping an unsupported-mod decoy also drops its
    target (and vice versa).
    """
    if "_pair_key" not in target.columns or "_pair_key" not in decoy.columns:
        raise ValueError(
            "filter_novoboard_target_decoy_pairs requires '_pair_key' on both tables"
        )

    target_f = filter_novoboard_prediction_table(
        target,
        peptide_col=peptide_col,
        min_length=min_length,
        key_col=key_col,
        log=log,
    )
    decoy_f = filter_novoboard_prediction_table(
        decoy,
        peptide_col=peptide_col,
        min_length=min_length,
        key_col=key_col,
        log=log,
    )

    def _valid_pair_keys(series: pd.Series) -> set[str]:
        keys = series.astype(str)
        return {k for k in keys if k and k != "nan"}

    shared_keys = _valid_pair_keys(target_f["_pair_key"]) & _valid_pair_keys(
        decoy_f["_pair_key"]
    )
    target_out = target_f[target_f["_pair_key"].astype(str).isin(shared_keys)].copy()
    decoy_out = decoy_f[decoy_f["_pair_key"].astype(str).isin(shared_keys)].copy()
    n_target = target_out["_pair_key"].astype(str).nunique()
    n_decoy = decoy_out["_pair_key"].astype(str).nunique()
    if n_target != n_decoy:
        raise AssertionError(
            f"Pair filter left unequal twin counts: targets={n_target} decoys={n_decoy}"
        )
    if log:
        before_pairs = len(
            _valid_pair_keys(target_f["_pair_key"])
            | _valid_pair_keys(decoy_f["_pair_key"])
        )
        logger.info(
            "NovoBoard pair filter: %s → %s twin spectra "
            "(target rows %s → %s, decoy rows %s → %s)",
            before_pairs,
            n_target,
            len(target_f),
            len(target_out),
            len(decoy_f),
            len(decoy_out),
        )
    return target_out, decoy_out


def restrict_winnow_to_novoboard_spectra(
    winnow: pd.DataFrame, novoboard: pd.DataFrame
) -> pd.DataFrame:
    """Trim Winnow to NovoBoard twin-valid spectra under the subset invariant.

    After shared peptide filters and NovoBoard pair-gating, NovoBoard targets are
    expected to be a subset of Winnow-filtered spectra (same InstaNovo
    predictions; NovoBoard additionally drops pairs whose decoy fails). The
    shared pool is therefore the NovoBoard spectrum set: only Winnow is trimmed.

    Raises:
        AssertionError: If any NovoBoard spectrum is missing from Winnow.
    """
    winnow_ids = set(winnow["spectrum_id"].astype(str))
    novoboard_ids = set(novoboard["spectrum_id"].astype(str))
    only_novoboard = novoboard_ids - winnow_ids
    if only_novoboard:
        examples = sorted(only_novoboard)[:5]
        raise AssertionError(
            "NovoBoard twin-valid spectra are not a subset of Winnow-filtered "
            f"spectra ({len(only_novoboard)} missing); examples={examples}. "
            "Expected identical InstaNovo predictions after ProForma remapping."
        )
    winnow_shared = winnow[winnow["spectrum_id"].astype(str).isin(novoboard_ids)].copy()
    logger.info(
        "Shared spectrum pool: %d spectra (trimmed Winnow=%d; NovoBoard unchanged)",
        len(novoboard_ids),
        len(winnow) - len(winnow_shared),
    )
    return winnow_shared


def assert_shared_prediction_keys(
    winnow: pd.DataFrame,
    novoboard: pd.DataFrame,
    *,
    winnow_peptide_col: str = "prediction",
    novoboard_peptide_col: str = "Peptide",
) -> None:
    """Require I/L-normalised prediction identity on the shared spectrum pool."""
    w_ids = winnow["spectrum_id"].astype(str)
    nb_ids = novoboard["spectrum_id"].astype(str)
    if w_ids.nunique() != len(winnow) or nb_ids.nunique() != len(novoboard):
        raise AssertionError(
            "Shared-pool tables must have one row per spectrum_id before "
            f"prediction-key assert (winnow rows={len(winnow)} unique={w_ids.nunique()}, "
            f"novoboard rows={len(novoboard)} unique={nb_ids.nunique()})"
        )
    merged = (
        winnow[["spectrum_id", winnow_peptide_col]]
        .assign(spectrum_id=w_ids)
        .merge(
            novoboard[["spectrum_id", novoboard_peptide_col]].assign(
                spectrum_id=nb_ids
            ),
            on="spectrum_id",
            how="inner",
            validate="one_to_one",
        )
    )
    if len(merged) != len(winnow) or len(merged) != len(novoboard):
        raise AssertionError(
            "Shared-pool spectrum_id join is not 1:1 "
            f"(winnow={len(winnow)} novoboard={len(novoboard)} inner={len(merged)})"
        )
    w_keys = merged[winnow_peptide_col].map(normalize_peptide_key)
    nb_keys = merged[novoboard_peptide_col].map(normalize_peptide_key)
    mismatch = w_keys != nb_keys
    if bool(mismatch.any()):
        bad = merged.loc[
            mismatch, ["spectrum_id", winnow_peptide_col, novoboard_peptide_col]
        ]
        examples = bad.head(5).to_dict(orient="records")
        raise AssertionError(
            "Winnow and NovoBoard predictions disagree after I/L-normalised "
            f"peptide keys ({int(mismatch.sum())} spectra); examples={examples}"
        )


def label_series_by_spectrum_id(winnow: pd.DataFrame, label_col: str) -> pd.Series:
    """Map ``spectrum_id`` → boolean label from a Winnow table."""
    if label_col not in winnow.columns:
        raise KeyError(f"Missing label column {label_col!r}")
    ids = winnow["spectrum_id"].astype(str)
    if ids.duplicated().any():
        raise AssertionError(
            f"Duplicate spectrum_id values when building {label_col} label map"
        )
    return pd.Series(
        winnow[label_col].astype(bool).to_numpy(),
        index=ids,
        name=label_col,
    )


def attach_labels_by_spectrum_id(
    novoboard: pd.DataFrame,
    label_by_id: pd.Series,
    *,
    label_col: str,
) -> pd.DataFrame:
    """Attach a shared label column to NovoBoard rows by ``spectrum_id``."""
    out = novoboard.copy()
    mapped = out["spectrum_id"].astype(str).map(label_by_id)
    if mapped.isna().any():
        missing = out.loc[mapped.isna(), "spectrum_id"].astype(str).head(5).tolist()
        raise AssertionError(
            f"NovoBoard rows missing shared {label_col} labels; examples={missing}"
        )
    out[label_col] = mapped.astype(bool)
    return out


def _best_alc_per_pair_key(df: pd.DataFrame) -> pd.DataFrame:
    """One highest-ALC row per ``_pair_key``."""
    work = df.dropna(subset=["ALC (%)", "_pair_key"])
    work = work[work["_pair_key"].astype(str) != "nan"]
    return (
        work.sort_values("ALC (%)", ascending=False)
        .groupby("_pair_key", as_index=False)
        .first()
    )


def proteome_hit_mask(
    peptides: pd.Series | Iterable[str],
    haystack: str,
    *,
    min_length: int = MIN_PEPTIDE_LENGTH,
) -> np.ndarray:
    """True when normalised peptide key (length ≥ *min_length*) hits the proteome."""
    keys = [normalize_peptide_key(p) for p in peptides]
    eligible = [bool(k) and len(k) >= min_length for k in keys]
    unique_keys = sorted({k for k, ok in zip(keys, eligible) if ok})
    hit_map: dict[str, bool] = {}
    if unique_keys:
        hits = _batch_peptide_substring_hits(unique_keys, haystack)
        hit_map = dict(zip(unique_keys, hits))
    return np.array(
        [bool(eligible[i] and hit_map.get(keys[i], False)) for i in range(len(keys))],
        dtype=bool,
    )


def max_score_per_peptide(
    df: pd.DataFrame,
    key_col: str,
    score_col: str,
) -> pd.DataFrame:
    """Keep the max-scoring row per peptide key (no filtering).

    Call after :func:`filter_prediction_table` or
    :func:`filter_novoboard_prediction_table` so all methods share the same
    filter → max-dedupe sequence in the peptide score-mixture benchmark.
    """
    work = df.dropna(subset=[score_col, key_col])
    work = work[work[key_col].astype(str) != ""]
    return dedupe_best_score_per_peptide(work, key_col, score_col).reset_index(
        drop=True
    )


def confidence_to_log_prob(confidence: pd.Series | np.ndarray) -> np.ndarray:
    """Map raw InstaNovo confidence in (0, 1] to Glissade-style log probabilities."""
    conf = np.asarray(confidence, dtype=float)
    return np.log(np.clip(conf, 1e-300, 1.0))


def _load_mgf_title_to_scan(mgf_path: Path) -> dict[str, str]:
    """Parse TITLE→SCANS mapping from an MGF file."""
    mapping: dict[str, str] = {}
    title: str | None = None
    scan: str | None = None
    with open(mgf_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value.startswith("TITLE="):
                title = value.removeprefix("TITLE=")
            elif value.startswith("SCANS="):
                scan = value.removeprefix("SCANS=")
            elif value == "END IONS" and title is not None and scan is not None:
                mapping[title] = scan
                title = None
                scan = None
    return mapping


def attach_novoboard_pair_keys(
    target: pd.DataFrame,
    decoy: pd.DataFrame,
    *,
    novoboard_dir: Path,
    split_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach ``_pair_key`` using Scan identity or decoy-MGF TITLE→SCANS mapping."""
    target_out = target.copy()
    decoy_out = decoy.copy()
    if "Scan" not in target_out.columns or "Scan" not in decoy_out.columns:
        raise ValueError("NovoBoard tables require a 'Scan' column for twin pairing")

    target_key = target_out["Scan"].astype(str)
    decoy_key = decoy_out["Scan"].astype(str)
    best_decoy_key = decoy_key
    best_overlap = len(set(target_key) & set(decoy_key))

    mgf_path = novoboard_dir.parent / f"{split_prefix}.mgf"
    if mgf_path.is_file():
        title_to_scan = _load_mgf_title_to_scan(mgf_path)
        if title_to_scan:
            mapped = decoy_key.map(title_to_scan)
            mapped_overlap = len(set(target_key) & set(mapped.dropna()))
            if mapped_overlap > best_overlap:
                best_decoy_key = mapped
                best_overlap = mapped_overlap

    target_out["_pair_key"] = target_key
    decoy_out["_pair_key"] = best_decoy_key
    logger.info(
        "NovoBoard %s pair-key overlap: target=%d decoy=%d overlap=%d",
        split_prefix,
        target_key.nunique(dropna=True),
        pd.Series(best_decoy_key).nunique(dropna=True),
        best_overlap,
    )
    return target_out, decoy_out


def prepare_novoboard_decoy_by_pair(
    decoy_df: pd.DataFrame,
    *,
    min_length: int = MIN_PEPTIDE_LENGTH,
    already_filtered: bool = False,
) -> pd.DataFrame:
    """Index the best decoy row per ``_pair_key``.

    Args:
        decoy_df: Decoy table with ``_pair_key``. Prefer pair-gated output from
            :func:`filter_novoboard_target_decoy_pairs`.
        already_filtered: When True, skip ProForma/mod/length filtering (caller
            already pair-filtered).
    """
    if "_pair_key" not in decoy_df.columns:
        raise ValueError("NovoBoard twin TDC requires '_pair_key' on decoy")
    if already_filtered:
        decoy = decoy_df.copy()
        if "_peptide_key" not in decoy.columns:
            if "peptide_key" in decoy.columns:
                decoy["_peptide_key"] = decoy["peptide_key"]
            else:
                decoy = add_peptide_key(decoy, "Peptide", key_col="_peptide_key")
    elif "_peptide_key" in decoy_df.columns and decoy_df["_peptide_key"].notna().all():
        decoy = decoy_df.copy()
    else:
        decoy = filter_novoboard_prediction_table(
            decoy_df, min_length=min_length, key_col="_peptide_key"
        )
    decoy = decoy.dropna(subset=["ALC (%)", "_pair_key"])
    decoy = decoy[
        (decoy["_pair_key"].astype(str) != "nan") & (decoy["_peptide_key"] != "")
    ]
    return _best_alc_per_pair_key(decoy).set_index("_pair_key", drop=False)


def novoboard_psm_tdc(
    target_df: pd.DataFrame,
    decoy_df: pd.DataFrame,
    *,
    min_length: int = MIN_PEPTIDE_LENGTH,
) -> pd.DataFrame:
    """Recompute NovoBoard's pooled PSM TDC after pair-gated filtering.

    Target and decoy are filtered as spectrum twins so unsupported-mod drops
    remove the pair. Competition uses one best-ALC row per twin on each side,
    guaranteeing ``sum(is_target) == sum(~is_target)``.
    """
    target, decoy = filter_novoboard_target_decoy_pairs(
        target_df, decoy_df, min_length=min_length
    )
    target = _best_alc_per_pair_key(target).assign(is_target=True)
    decoy = _best_alc_per_pair_key(decoy).assign(is_target=False)
    n_target = int(target["_pair_key"].nunique())
    n_decoy = int(decoy["_pair_key"].nunique())
    if n_target != n_decoy or len(target) != len(decoy):
        raise AssertionError(
            f"PSM TDC unbalanced after pair gate: "
            f"target_rows={len(target)} decoy_rows={len(decoy)} "
            f"target_pairs={n_target} decoy_pairs={n_decoy}"
        )

    combined = pd.concat([target, decoy], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["ALC (%)", "is_target"], ascending=[False, False]
    ).reset_index(drop=True)
    return _assign_cumulative_tdc_fdr(combined)


def _prepare_targets_for_twin_tdc(
    target_df: pd.DataFrame,
    decoy_df: pd.DataFrame,
    *,
    min_length: int,
    decoy_by_pair: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair-gate or twin-filter targets and return ``(target, decoy_by_pair)``."""
    if "_pair_key" not in target_df.columns:
        raise ValueError("NovoBoard twin TDC requires '_pair_key' on target")
    if decoy_by_pair is None and "_pair_key" not in decoy_df.columns:
        raise ValueError("NovoBoard twin TDC requires '_pair_key' on decoy")

    if decoy_by_pair is None:
        target, decoy = filter_novoboard_target_decoy_pairs(
            target_df, decoy_df, min_length=min_length, log=False
        )
        decoy_by_pair = prepare_novoboard_decoy_by_pair(
            decoy, min_length=min_length, already_filtered=True
        )
        return target, decoy_by_pair

    target = target_df.copy()
    if "_peptide_key" not in target.columns:
        if "peptide_key" in target.columns:
            target["_peptide_key"] = target["peptide_key"]
        else:
            target = filter_novoboard_prediction_table(
                target,
                min_length=min_length,
                key_col="_peptide_key",
                log=False,
            )
    twin_keys = set(decoy_by_pair.index.astype(str))
    target = target[target["_pair_key"].astype(str).isin(twin_keys)]
    return target, decoy_by_pair


def _assign_cumulative_tdc_fdr(combined: pd.DataFrame) -> pd.DataFrame:
    """Add estimated FDR / q-value columns for a balanced target-decoy table."""
    out = combined.copy()
    n_target = out["is_target"].astype(int).cumsum()
    n_decoy = (~out["is_target"]).astype(int).cumsum()
    out["estimated_fdr"] = np.divide(
        n_decoy,
        n_target,
        out=np.ones(len(out), dtype=float),
        where=n_target > 0,
    )
    out["estimated_q_value"] = np.nan
    target_mask = out["is_target"].to_numpy()
    out.loc[target_mask, "estimated_q_value"] = compute_q_values(
        out.loc[target_mask, "estimated_fdr"].to_numpy()
    )
    return out


def novoboard_max_target_twin_decoy_tdc(
    target_df: pd.DataFrame,
    decoy_df: pd.DataFrame,
    *,
    min_length: int = MIN_PEPTIDE_LENGTH,
    target_peptide_keys: set[str] | None = None,
    decoy_by_pair: pd.DataFrame | None = None,
    log_missing_twins: bool = True,
) -> pd.DataFrame:
    """Peptide TDC: pair-gate, max ALC per target peptide, twin decoy by ``_pair_key``.

    Returns the combined ranked table with ``is_target``, ``estimated_fdr``, and
    ``estimated_q_value`` (targets only). Targets without a twin-valid decoy are
    dropped; the competition table is always 1:1.

    Args:
        decoy_by_pair: Optional precomputed output of
            :func:`prepare_novoboard_decoy_by_pair` from an already pair-gated
            decoy table. When omitted, target/decoy are pair-filtered together.
    """
    target, decoy_by_pair = _prepare_targets_for_twin_tdc(
        target_df,
        decoy_df,
        min_length=min_length,
        decoy_by_pair=decoy_by_pair,
    )
    target = target.dropna(subset=["ALC (%)", "_pair_key"])
    target = target[
        (target["_pair_key"].astype(str) != "nan") & (target["_peptide_key"] != "")
    ]

    if target_peptide_keys is not None:
        target = target[target["_peptide_key"].isin(target_peptide_keys)]

    # Max-score only among twin-valid targets.
    target_best = max_score_per_peptide(target, "_peptide_key", "ALC (%)")
    pair_keys = target_best["_pair_key"].astype(str)
    has_twin = pair_keys.isin(decoy_by_pair.index.astype(str))
    n_missing_twin = int((~has_twin).sum())
    if log_missing_twins and n_missing_twin:
        logger.warning(
            "NovoBoard twin-decoy TDC dropped %d/%d max-target peptides without twin",
            n_missing_twin,
            len(target_best),
        )
    target_keep = target_best.loc[has_twin].copy()
    if target_keep.empty:
        return pd.DataFrame(
            columns=[
                "spectrum_id",
                "Peptide",
                "ALC (%)",
                "_peptide_key",
                "_pair_key",
                "is_target",
                "estimated_fdr",
                "estimated_q_value",
            ]
        )

    decoy_keep = decoy_by_pair.loc[target_keep["_pair_key"].astype(str)].copy()
    decoy_keep = decoy_keep.reset_index(drop=True)
    target_keep = target_keep.assign(is_target=True).reset_index(drop=True)
    decoy_keep = decoy_keep.assign(is_target=False)
    if len(target_keep) != len(decoy_keep):
        raise AssertionError(
            f"Peptide TDC unbalanced: targets={len(target_keep)} decoys={len(decoy_keep)}"
        )
    # Preserve 1:1 balance: one decoy row per retained target (do not dedupe decoys).
    combined = pd.concat([target_keep, decoy_keep], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["ALC (%)", "is_target"], ascending=[False, False]
    ).reset_index(drop=True)
    return _assign_cumulative_tdc_fdr(combined)
