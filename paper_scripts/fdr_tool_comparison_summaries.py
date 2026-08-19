"""Summary tables for Winnow / NovoBoard / Glissade FDR tool comparisons.

Produces two long-form CSVs:

- ``*_acceptance.csv``: accepted counts and recovery at q-value thresholds.
- ``*_error_gain.csv``: observed FDP, excess over nominal FDR, optional mean
  absolute q-value deviation vs a database-grounded reference (calibrated-score
  DBG for Winnow; raw-score / ALC DBG for NovoBoard and Glissade), and relative
  gain/loss of a primary method vs each comparator.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence, cast

import numpy as np
import pandas as pd

_PAPER_SCRIPTS = Path(__file__).resolve().parent
if str(_PAPER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PAPER_SCRIPTS))

from fdr_tool_comparison_preprocess import compute_q_values  # noqa: E402

logger = logging.getLogger(__name__)

SUMMARY_THRESHOLDS: list[float] = [0.01, 0.05, 0.10]
# Match ``DatabaseGroundedFDRControl`` / PSM comparison default.
_DB_GROUNDED_DROP = 10

_KEY_COLS = ["dataset", "panel", "level", "method", "q_value_threshold"]


def _slug_comparator(name: str) -> str:
    """Map a method label to a filesystem-/column-safe slug."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def acceptance_rows_from_q(
    *,
    dataset: str,
    panel: str,
    level: str,
    method: str,
    q_value: np.ndarray,
    thresholds: Sequence[float] = SUMMARY_THRESHOLDS,
    label_mask: np.ndarray | None = None,
    recovery_denom: int | None = None,
) -> list[dict[str, object]]:
    """Build acceptance/yield rows for one method at each q-value threshold.

    Args:
        dataset: Dataset key (e.g. ``helaqc``).
        panel: Evaluation panel (e.g. ``labelled_test``, ``unlabelled``, ``external``).
        level: ``psm`` or ``peptide``.
        method: Method display label.
        q_value: Per-row estimated q-values.
        thresholds: Nominal FDR thresholds.
        label_mask: Optional boolean correctness / proteome-hit labels aligned with
            ``q_value``. When provided, ``n_correct`` is filled.
        recovery_denom: Denominator for recovery percentage. Defaults to the number
            of True labels when ``label_mask`` is given.

    Returns:
        One dict per threshold.
    """
    q = np.asarray(q_value, dtype=float)
    labels = None if label_mask is None else np.asarray(label_mask, dtype=bool)
    if labels is not None and len(labels) != len(q):
        raise ValueError(
            f"label_mask length {len(labels)} does not match q_value length {len(q)}"
        )
    if recovery_denom is None and labels is not None:
        recovery_denom = int(labels.sum())

    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        valid = ~np.isnan(q)
        accepted = valid & (q <= threshold)
        n_accepted = int(accepted.sum())
        n_correct: float | int = np.nan
        recovery_pct: float = np.nan
        if labels is not None:
            n_correct = int((accepted & labels).sum())
            if recovery_denom and recovery_denom > 0:
                recovery_pct = 100.0 * float(n_correct) / float(recovery_denom)
        rows.append(
            {
                "dataset": dataset,
                "panel": panel,
                "level": level,
                "method": method,
                "q_value_threshold": float(threshold),
                "n_accepted": n_accepted,
                "n_correct": n_correct,
                "recovery_pct": recovery_pct,
            }
        )
    return rows


def observed_fdp_at_thresholds(
    q_value: np.ndarray,
    label_mask: np.ndarray,
    thresholds: Sequence[float] = SUMMARY_THRESHOLDS,
) -> list[float]:
    """Return observed false-discovery proportion among accepted rows at each threshold.

    ``label_mask`` is True for correct (or proteome-hit) rows. Observed FDP is
    ``1 - n_correct / n_accepted`` when any rows are accepted, else NaN.
    """
    q = np.asarray(q_value, dtype=float)
    labels = np.asarray(label_mask, dtype=bool)
    if len(q) != len(labels):
        raise ValueError(
            f"label_mask length {len(labels)} does not match q_value length {len(q)}"
        )
    out: list[float] = []
    for threshold in thresholds:
        valid = ~np.isnan(q)
        accepted = valid & (q <= threshold)
        n_accepted = int(accepted.sum())
        if n_accepted == 0:
            out.append(float("nan"))
            continue
        n_correct = int((accepted & labels).sum())
        out.append(1.0 - n_correct / n_accepted)
    return out


def database_grounded_q_from_labels(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    drop: int = _DB_GROUNDED_DROP,
) -> np.ndarray:
    """In-sample database-grounded q-values from ranked scores and boolean labels.

    Builds the empirical precision curve ``1 - cumsum(correct) / rank`` on scores
    sorted descending (same construction as the proteome-hit shortcut in the PSM
    comparison), drops the first *drop* ranks from the FDR map, assigns FDR by
    score lookup, then converts to q-values.

    Args:
        scores: Ranking scores (higher = more confident).
        labels: Boolean correctness / hit labels aligned with *scores*.
        drop: Leading ranks excluded from the FDR map (default 10).

    Returns:
        q-value array aligned with *scores*.
    """
    scores_a = np.asarray(scores, dtype=float)
    labels_a = np.asarray(labels, dtype=bool)
    n = len(scores_a)
    if n == 0:
        return np.asarray([], dtype=float)
    if len(labels_a) != n:
        raise ValueError(
            f"labels length {len(labels_a)} does not match scores length {n}"
        )

    order = np.argsort(-scores_a, kind="mergesort")
    precision = np.cumsum(labels_a[order].astype(float)) / np.arange(1, n + 1)
    fdr_ranked = 1.0 - precision
    drop_eff = min(drop, max(0, n - 1))
    fit_scores = scores_a[order][drop_eff:]
    fit_fdr = fdr_ranked[drop_eff:]
    n_fit = len(fit_scores)

    idx = np.searchsorted(-fit_scores, -scores_a, side="left")
    fdr = np.empty(n, dtype=float)
    below = (idx == n_fit) & (scores_a < fit_scores[-1])
    above = (idx == 0) & (scores_a > fit_scores[0])
    normal = ~(below | above)
    fdr[below] = 1.0
    fdr[above] = float(fit_fdr[0])
    fdr[normal] = fit_fdr[np.clip(idx[normal], 0, n_fit - 1)]

    q_sorted = compute_q_values(fdr[order])
    q = np.empty(n, dtype=float)
    q[order] = q_sorted
    return q


def mean_abs_q_dev_vs_reference(
    q_method: np.ndarray,
    q_ref: np.ndarray,
    thresholds: Sequence[float] = SUMMARY_THRESHOLDS,
) -> list[float]:
    """Mean absolute q-value deviation vs a row-aligned reference at each threshold.

    For each threshold, restrict to rows accepted by either method
    (``q_method <= t`` or ``q_ref <= t``) with finite q for both, then report
    ``mean(|q_method - q_ref|)``.
    """
    q_m = np.asarray(q_method, dtype=float)
    q_r = np.asarray(q_ref, dtype=float)
    if len(q_m) != len(q_r):
        raise ValueError(
            f"q_ref length {len(q_r)} does not match q_method length {len(q_m)}"
        )
    out: list[float] = []
    for threshold in thresholds:
        both_finite = ~np.isnan(q_m) & ~np.isnan(q_r)
        either_accepted = both_finite & ((q_m <= threshold) | (q_r <= threshold))
        if not np.any(either_accepted):
            out.append(float("nan"))
            continue
        out.append(float(np.mean(np.abs(q_m[either_accepted] - q_r[either_accepted]))))
    return out


def error_rows_from_q(
    *,
    dataset: str,
    panel: str,
    level: str,
    method: str,
    q_value: np.ndarray,
    thresholds: Sequence[float] = SUMMARY_THRESHOLDS,
    label_mask: np.ndarray | None = None,
    q_ref: np.ndarray | None = None,
    observed_fdp: Sequence[float] | None = None,
) -> list[dict[str, object]]:
    """Build error-metric rows for one method (without relative-gain columns).

    Args:
        observed_fdp: Optional precomputed FDP values (e.g. from a mixture
            benchmark). When omitted, FDP is derived from ``label_mask`` if given.
    """
    if observed_fdp is not None and len(observed_fdp) != len(thresholds):
        raise ValueError("observed_fdp length must match thresholds")
    if observed_fdp is None and label_mask is not None:
        fdp_values = observed_fdp_at_thresholds(q_value, label_mask, thresholds)
    elif observed_fdp is not None:
        fdp_values = [float(x) for x in observed_fdp]
    else:
        fdp_values = [float("nan")] * len(thresholds)

    if q_ref is not None:
        q_dev = mean_abs_q_dev_vs_reference(q_value, q_ref, thresholds)
    else:
        q_dev = [float("nan")] * len(thresholds)

    rows: list[dict[str, object]] = []
    for threshold, fdp, dev in zip(thresholds, fdp_values, q_dev):
        fdp_f = float(fdp)
        excess = fdp_f - float(threshold) if np.isfinite(fdp_f) else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "panel": panel,
                "level": level,
                "method": method,
                "q_value_threshold": float(threshold),
                "observed_fdp": fdp_f,
                "fdp_excess": excess,
                "mean_abs_q_dev_vs_db": float(dev),
            }
        )
    return rows


def _relative_gain_column(value_col: str, comparator: str) -> str:
    """Return the plan-specified relative-gain column name for *value_col*."""
    slug = _slug_comparator(comparator)
    if value_col == "n_accepted":
        return f"accepted_pct_vs_{slug}"
    if value_col == "recovery_pct":
        return f"recovery_pct_vs_{slug}"
    if value_col == "observed_fdp":
        return f"fdp_delta_vs_{slug}"
    if "fdp" in value_col:
        return f"{value_col}_delta_vs_{slug}"
    return f"{value_col}_pct_vs_{slug}"


def _relative_gain_value(
    value_col: str, primary_val: object, comparator_val: object
) -> float:
    """Compute primary-vs-comparator gain for one metric cell."""
    if pd.isna(primary_val) or pd.isna(comparator_val):
        return float("nan")
    primary = float(cast("float | int | str", primary_val))
    comparator = float(cast("float | int | str", comparator_val))
    if value_col == "observed_fdp" or "fdp" in value_col:
        return primary - comparator
    if comparator == 0:
        return float("nan")
    return 100.0 * (primary - comparator) / comparator


def _fill_primary_relative_gains(
    work: pd.DataFrame,
    group: pd.DataFrame,
    *,
    primary_method: str,
    comparators: Sequence[str],
    value_cols: Sequence[str],
    group_cols: Sequence[str],
    group_keys: tuple[object, ...],
) -> None:
    """Write relative-gain columns onto the primary-method row for one group."""
    primary_rows = group[group["method"] == primary_method]
    if primary_rows.empty:
        return
    primary = primary_rows.iloc[0]
    mask = pd.Series(True, index=work.index)
    for col, val in zip(group_cols, group_keys):
        mask &= work[col] == val
    primary_idx = work.index[mask & (work["method"] == primary_method)]
    if len(primary_idx) == 0:
        return
    idx = primary_idx[0]

    for comparator in comparators:
        comp_rows = group[group["method"] == comparator]
        if comp_rows.empty:
            continue
        comp = comp_rows.iloc[0]
        for col in value_cols:
            out_col = _relative_gain_column(col, comparator)
            gain = _relative_gain_value(col, primary[col], comp[col])
            if np.isfinite(gain):
                work.at[idx, out_col] = gain


def add_relative_gain_columns(
    df: pd.DataFrame,
    *,
    primary_method: str,
    comparators: Iterable[str],
    value_cols: Sequence[str],
    group_cols: Sequence[str] = ("dataset", "panel", "level", "q_value_threshold"),
) -> pd.DataFrame:
    """Attach primary-vs-comparator relative columns onto a long-form metrics table.

    For ``n_accepted`` / ``recovery_pct``, writes
    ``100 * (primary - comparator) / comparator``.
    For ``observed_fdp``, writes the signed difference ``primary - comparator``.
    Relative columns are filled only on primary-method rows.
    """
    if df.empty:
        return df.copy()

    work = df.copy()
    comparator_list = list(comparators)
    for col in value_cols:
        for comparator in comparator_list:
            work[_relative_gain_column(col, comparator)] = np.nan

    group_list = list(group_cols)
    for keys, group in work.groupby(group_list, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        _fill_primary_relative_gains(
            work,
            group,
            primary_method=primary_method,
            comparators=comparator_list,
            value_cols=value_cols,
            group_cols=group_list,
            group_keys=keys,
        )
    return work


def merge_acceptance_and_error(
    acceptance: pd.DataFrame,
    error: pd.DataFrame,
    *,
    key_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Join acceptance counts onto error rows for relative-gain construction."""
    if acceptance.empty or error.empty:
        return error.copy()
    keys = list(key_cols) if key_cols is not None else list(_KEY_COLS)
    keys = [c for c in keys if c in acceptance.columns and c in error.columns]
    cols = [
        c
        for c in ("n_accepted", "n_correct", "recovery_pct")
        if c in acceptance.columns
    ]
    return error.merge(
        acceptance[keys + cols],
        on=keys,
        how="left",
    )


def finalise_error_gain_table(
    acceptance: pd.DataFrame,
    error: pd.DataFrame,
    *,
    primary_method: str,
    comparators: Sequence[str],
    key_cols: Sequence[str] | None = None,
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Merge counts into error rows and add primary-vs-comparator relative columns."""
    merged = merge_acceptance_and_error(acceptance, error, key_cols=key_cols)
    value_cols = [
        c for c in ("n_accepted", "recovery_pct", "observed_fdp") if c in merged.columns
    ]
    gain_groups = (
        tuple(group_cols)
        if group_cols is not None
        else ("dataset", "panel", "level", "q_value_threshold")
    )
    with_gain = add_relative_gain_columns(
        merged,
        primary_method=primary_method,
        comparators=comparators,
        value_cols=value_cols,
        group_cols=gain_groups,
    )
    # Keep error-table identity columns first; drop helper count cols that duplicate
    # the acceptance table except when used only for gain calculation.
    drop_helpers = [c for c in ("n_accepted", "n_correct") if c in with_gain.columns]
    return with_gain.drop(columns=drop_helpers, errors="ignore")


def write_summary_tables(
    acceptance_df: pd.DataFrame,
    error_df: pd.DataFrame,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    """Write acceptance and error/gain CSVs under *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    acceptance_path = output_dir / f"{stem}_acceptance.csv"
    error_path = output_dir / f"{stem}_error_gain.csv"
    acceptance_df.to_csv(acceptance_path, index=False)
    error_df.to_csv(error_path, index=False)
    logger.info("Wrote %s", acceptance_path)
    logger.info("Wrote %s", error_path)
    return acceptance_path, error_path


def summarise_holdout_results(
    raw: pd.DataFrame,
    *,
    thresholds: Sequence[float] = SUMMARY_THRESHOLDS,
    primary_method: str = "Winnow",
    comparators: Sequence[str] = ("NovoBoard", "Glissade"),
    panel: str = "score_mixture",
    level: str = "peptide",
    group_extra: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate mixture-benchmark iterations into the two summary tables.

    Args:
        raw: Per-iteration rows from ``external_peptide_holdout_results.csv``.
        thresholds: Nominal FDR thresholds to retain.
        primary_method: Method used for relative gain/loss columns.
        comparators: Comparator method labels.
        panel: Panel name written into the summary tables.
        level: Identification level written into the summary tables.
        group_extra: Extra columns to group by (e.g. ``pi0_target``).

    Returns:
        ``(acceptance_df, error_gain_df)``.
    """
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    filtered = raw[raw["q_value_threshold"].isin(thresholds)].copy()
    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    extra = [c for c in group_extra if c in filtered.columns]
    group_cols = ["dataset", *extra, "method", "q_value_threshold"]
    gain_group_cols = ("dataset", *extra, "panel", "level", "q_value_threshold")
    agg_kwargs: dict[str, tuple[str, str]] = {
        "n_accepted": ("accepted_peptides", "mean"),
        "n_correct": ("true_correct_peptides", "mean"),
        "recovery_pct": ("correct_discovery_pct", "mean"),
        "observed_fdp": ("observed_fdp", "mean"),
        "n_accepted_std": ("accepted_peptides", "std"),
        "observed_fdp_std": ("observed_fdp", "std"),
        "recovery_pct_std": ("correct_discovery_pct", "std"),
    }
    if "mean_abs_q_dev_vs_db" in filtered.columns:
        agg_kwargs["mean_abs_q_dev_vs_db"] = ("mean_abs_q_dev_vs_db", "mean")
    agg = (
        filtered.groupby(group_cols, as_index=False)
        .agg(**agg_kwargs)
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    agg["panel"] = panel
    agg["level"] = level
    agg["fdp_excess"] = agg["observed_fdp"] - agg["q_value_threshold"]
    if "mean_abs_q_dev_vs_db" not in agg.columns:
        agg["mean_abs_q_dev_vs_db"] = np.nan

    id_cols = ["dataset", *extra, "panel", "level", "method", "q_value_threshold"]
    acceptance = agg[
        id_cols
        + [
            "n_accepted",
            "n_correct",
            "recovery_pct",
            "n_accepted_std",
            "recovery_pct_std",
        ]
    ].copy()

    error = agg[
        id_cols
        + [
            "observed_fdp",
            "fdp_excess",
            "mean_abs_q_dev_vs_db",
            "observed_fdp_std",
        ]
    ].copy()

    error_gain = finalise_error_gain_table(
        acceptance.drop(
            columns=["n_accepted_std", "recovery_pct_std"], errors="ignore"
        ),
        error,
        primary_method=primary_method,
        comparators=comparators,
        key_cols=id_cols,
        group_cols=gain_group_cols,
    )
    return acceptance, error_gain
