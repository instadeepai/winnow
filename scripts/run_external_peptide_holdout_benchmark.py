#!/usr/bin/env python3
"""Benchmark external peptide FDR with a Glissade-style score mixture.

For each dataset and method, build two peptide-level score pools:

- S_m: database-matched correct peptides from the labelled split.
- S_e: external peptide candidates from the unlabelled split.

A fixed subset S_c is held out from S_m, then mixed into bootstrap resamples of
S_e. Each method estimates peptide q-values with its intended procedure on that
synthetic external set, and the observed FDP is measured because S_c is known to
be correct and S_e is treated as null/external.

This copy uses peptide-level NovoBoard TDC (dedupe then compete), not pairwise
scan TDC. With defaults (3 iterations, 5 Glissade bootstraps, seed 42) it
reproduces results/external_peptide_holdout_benchmark_old for Winnow and
NovoBoard; Glissade remains stochastic.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.annotate_preds_proteome_hits import load_proteome_haystack  # noqa: E402
from scripts.plot_eval_results import _PALETTE, _display_name, _save_fig, _style_ax  # noqa: E402
from scripts.plot_fdr_method_comparison import (  # noqa: E402
    DEFAULT_GLISSADE_ROOT,
    DEFAULT_NOVOBOARD_ROOT,
    DEFAULT_WINNOW_RESULTS,
    build_dataset_configs,
    dedupe_best_score_per_peptide,
    load_glissade,
    load_novoboard_target_decoy,
    load_winnow,
    normalize_peptide_key,
    sequence_only_correctness_mask,
    _compute_q_values,
)
from winnow.fdr.nonparametric import NonParametricFDRControl  # noqa: E402

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# Reproduces results/external_peptide_holdout_benchmark_old (Winnow/NovoBoard).
# Default write path avoids clobbering the saved CSV; pass --output-dir to overwrite.
DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT / "results/external_peptide_holdout_benchmark_copy_verify"
)
DEFAULT_GLISSADE_REPO = Path("/home/j-daniel/repos/glissade")
DEFAULT_DATASETS = ["helaqc", "celegans"]
DEFAULT_Q_THRESHOLDS = [round(float(x), 2) for x in np.linspace(0.0, 0.25, 26)]
DEFAULT_N_ITERATIONS = 3
DEFAULT_N_BOOTSTRAPS = 5
DEFAULT_HOLDOUT_FRAC = 0.5
DEFAULT_SEED = 42


@dataclass(frozen=True)
class PeptideScorePool:
    """Peptide-level matched and external score pools for one method."""

    method: str
    matched: pd.DataFrame
    external: pd.DataFrame
    estimator: Callable[[pd.DataFrame, pd.DataFrame, np.random.Generator], pd.DataFrame]


def _normalise_pool(df: pd.DataFrame, *, min_peptide_length: int = 0) -> pd.DataFrame:
    work = df.copy()
    work["peptide_key"] = work["peptide"].map(normalize_peptide_key)
    work = work[work["peptide_key"] != ""]
    if min_peptide_length > 0:
        work = work[work["peptide_key"].str.len() >= min_peptide_length]
    work = work.dropna(subset=["score"])
    return dedupe_best_score_per_peptide(work, "peptide_key", "score").reset_index(
        drop=True
    )


def _dedupe_novoboard_peptides(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["Peptide", "ALC (%)"]).copy()
    work["_peptide_key"] = work["Peptide"].map(normalize_peptide_key)
    work = work[work["_peptide_key"] != ""]
    return dedupe_best_score_per_peptide(work, "_peptide_key", "ALC (%)")


def _novoboard_peptide_tdc_table(
    target_df: pd.DataFrame,
    decoy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Peptide-dedupe then cumulative TDC (matches plot_fdr_method_comparison copy).

    Do not import the pairwise-scan TDC from plot_fdr_method_comparison.py; that
    path produces the non-copy CSV under results/external_peptide_holdout_benchmark.
    """
    target = _dedupe_novoboard_peptides(target_df)
    decoy = _dedupe_novoboard_peptides(decoy_df)
    target = target.assign(is_target=True)
    decoy = decoy.assign(is_target=False)
    combined = pd.concat([target, decoy], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["ALC (%)", "is_target"], ascending=[False, False]
    ).reset_index(drop=True)

    n_target = combined["is_target"].astype(int).cumsum()
    n_decoy = (~combined["is_target"]).astype(int).cumsum()
    estimated_fdr = np.divide(
        n_decoy,
        n_target,
        out=np.ones(len(combined), dtype=float),
        where=n_target > 0,
    )
    combined["estimated_fdr"] = estimated_fdr
    combined["estimated_q_value"] = np.nan
    target_mask = combined["is_target"]
    combined.loc[target_mask, "estimated_q_value"] = _compute_q_values(
        combined.loc[target_mask, "estimated_fdr"].to_numpy()
    )
    return combined


def _external_peptide_mask(df: pd.DataFrame, fasta: Path) -> pd.Series:
    haystack = load_proteome_haystack(fasta)
    keys = df["peptide"].map(normalize_peptide_key)
    return (keys != "") & ~keys.map(lambda pep: pep in haystack)


def load_winnow_score_pool(
    winnow_test_dir: Path, winnow_unlabelled_dir: Path, fasta: Path
) -> PeptideScorePool:
    """Build Winnow matched/external pools from labelled and unlabelled outputs."""
    labelled = load_winnow(winnow_test_dir, fasta, "labelled")
    correct_mask = sequence_only_correctness_mask(
        labelled["sequence"], labelled["prediction"]
    )
    matched = pd.DataFrame(
        {
            "peptide": labelled.loc[correct_mask, "prediction"],
            "score": labelled.loc[correct_mask, "calibrated_confidence"],
        }
    )

    unlabelled = load_winnow(winnow_unlabelled_dir, fasta, "unlabelled")
    external = pd.DataFrame(
        {
            "peptide": unlabelled.loc[
                ~unlabelled["proteome_hit"].astype(bool), "prediction"
            ],
            "score": unlabelled.loc[
                ~unlabelled["proteome_hit"].astype(bool), "calibrated_confidence"
            ],
        }
    )
    return PeptideScorePool(
        method="Winnow",
        matched=_normalise_pool(matched),
        external=_normalise_pool(external),
        estimator=_estimate_winnow_q_values,
    )


def _labelled_truth(data_root: Path, dataset: str) -> pd.DataFrame:
    path = data_root / f"{dataset}_split_parquet" / "annotated_test.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    truth = pd.read_parquet(path, columns=["spectrum_id", "sequence"])
    return truth[["spectrum_id", "sequence"]]


def load_novoboard_score_pool(
    *,
    target_test: pd.DataFrame,
    target_unlabelled: pd.DataFrame,
    decoy_df: pd.DataFrame,
    data_root: Path,
    dataset: str,
    fasta: Path,
) -> PeptideScorePool:
    """Build NovoBoard pools with correctness from the labelled parquet split."""
    truth = _labelled_truth(data_root, dataset)
    labelled = target_test.merge(truth, on="spectrum_id", how="inner")
    correct = labelled[
        sequence_only_correctness_mask(labelled["sequence"], labelled["Peptide"])
    ]
    matched = pd.DataFrame({"peptide": correct["Peptide"], "score": correct["ALC (%)"]})

    ext_mask = _external_peptide_mask(
        pd.DataFrame({"peptide": target_unlabelled["Peptide"]}), fasta
    )
    external = pd.DataFrame(
        {
            "peptide": target_unlabelled.loc[ext_mask, "Peptide"],
            "score": target_unlabelled.loc[ext_mask, "ALC (%)"],
        }
    )

    def estimator(
        mixed: pd.DataFrame, _: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        return _estimate_novoboard_q_values(mixed, decoy_df, rng)

    return PeptideScorePool(
        method="NovoBoard",
        matched=_normalise_pool(matched),
        external=_normalise_pool(external),
        estimator=estimator,
    )


def load_glissade_score_pool(
    glissade_dir: Path,
    glissade_repo: Path,
    n_bootstraps: int,
    *,
    min_peptide_length: int = 0,
) -> PeptideScorePool:
    """Build Glissade pools from prepared denovo/labelled inputs and peptide.tsv.

    Args:
        min_peptide_length: If > 0, drop peptides shorter than this (Glissade
            ``seperate_scores`` default is 8).
    """
    denovo_path = glissade_dir / "glissade_denovo.parquet"
    labelled_path = glissade_dir / "glissade_labelled.parquet"
    if not denovo_path.is_file():
        raise FileNotFoundError(denovo_path)
    if not labelled_path.is_file():
        raise FileNotFoundError(labelled_path)

    denovo = pd.read_parquet(
        denovo_path, columns=["spectrum_id", "predictions", "log_probs"]
    )
    labelled = pd.read_parquet(labelled_path, columns=["spectrum_id", "sequence"])
    labelled = labelled.merge(denovo, on="spectrum_id", how="inner")
    correct = labelled[
        sequence_only_correctness_mask(labelled["sequence"], labelled["predictions"])
    ]
    matched = pd.DataFrame(
        {"peptide": correct["predictions"], "score": correct["log_probs"]}
    )

    external_raw = load_glissade(glissade_dir)
    external = pd.DataFrame(
        {
            "peptide": external_raw["peptide"],
            "score": external_raw["score"],
        }
    )

    def estimator(
        mixed: pd.DataFrame, matched_reference: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        return _estimate_glissade_q_values(
            mixed,
            matched_reference,
            glissade_repo=glissade_repo,
            n_bootstraps=n_bootstraps,
            rng=rng,
            min_peptide_length=min_peptide_length,
        )

    return PeptideScorePool(
        method="Glissade",
        matched=_normalise_pool(matched, min_peptide_length=min_peptide_length),
        external=_normalise_pool(external, min_peptide_length=min_peptide_length),
        estimator=estimator,
    )


def _estimate_winnow_q_values(
    mixed: pd.DataFrame, _: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    del rng
    work = _normalise_pool(mixed[["peptide", "score", "source"]])
    ctrl = NonParametricFDRControl()
    ctrl.fit(work["score"])
    q_table = ctrl.add_psm_q_value(
        work.drop(columns=["psm_q_value"], errors="ignore"), "score"
    )
    q_table["q_value"] = q_table["psm_q_value"]
    return q_table[["peptide", "peptide_key", "score", "source", "q_value"]]


def _estimate_novoboard_q_values(
    mixed: pd.DataFrame, decoy_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    del rng
    target = pd.DataFrame(
        {
            "Peptide": mixed["peptide"],
            "ALC (%)": mixed["score"],
            "spectrum_id": np.arange(len(mixed), dtype=int).astype(str),
            "source": mixed["source"].to_numpy(),
        }
    )
    table = _novoboard_peptide_tdc_table(target, decoy_df)
    target_table = table[table["is_target"]].copy()
    return pd.DataFrame(
        {
            "peptide": target_table["Peptide"],
            "peptide_key": target_table["_peptide_key"],
            "score": target_table["ALC (%)"],
            "source": target_table["source"],
            "q_value": target_table["estimated_q_value"],
        }
    )


def _load_glissade_functions(glissade_repo: Path):
    repo = str(glissade_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    module = importlib.import_module("glissade.glissade")
    return module.run_bootstraps, module.annotate_results, module.compute_fdr_transform


def _estimate_glissade_q_values(
    mixed: pd.DataFrame,
    matched_reference: pd.DataFrame,
    *,
    glissade_repo: Path,
    n_bootstraps: int,
    rng: np.random.Generator,
    min_peptide_length: int = 0,
) -> pd.DataFrame:
    del rng
    work = _normalise_pool(
        mixed[["peptide", "score", "source"]], min_peptide_length=min_peptide_length
    )
    if len(work) < 10:
        work["q_value"] = np.nan
        return work[["peptide", "peptide_key", "score", "source", "q_value"]]

    fit_ref = _normalise_pool(
        matched_reference[["peptide", "score"]], min_peptide_length=min_peptide_length
    )
    fit_scores = fit_ref["score"].dropna().to_numpy(dtype=float)
    if len(fit_scores) < 10:
        work["q_value"] = np.nan
        return work[["peptide", "peptide_key", "score", "source", "q_value"]]

    run_bootstraps, annotate_results, compute_fdr_transform = _load_glissade_functions(
        glissade_repo
    )
    try:
        fdrs, grid, _ = run_bootstraps(
            fit_scores,
            work["score"].to_numpy(dtype=float),
            n_bootstraps=n_bootstraps,
        )
        peptides, peptide_fdrs, scores = annotate_results(
            work["peptide"].to_numpy(),
            work["score"].to_numpy(dtype=float),
            fdrs,
            grid,
        )
        q_values = compute_fdr_transform(peptide_fdrs)
    except Exception as exc:  # pragma: no cover - depends on Glissade numerical fit
        logger.warning("Glissade FDR fit failed for one mixture: %s", exc)
        work["q_value"] = np.nan
        return work[["peptide", "peptide_key", "score", "source", "q_value"]]

    q_table = pd.DataFrame({"peptide": peptides, "score": scores, "q_value": q_values})
    q_table["peptide_key"] = q_table["peptide"].map(normalize_peptide_key)
    source = work[["peptide_key", "source"]].drop_duplicates("peptide_key")
    return q_table.merge(source, on="peptide_key", how="left")[
        ["peptide", "peptide_key", "score", "source", "q_value"]
    ]


def _sample_correct_set(
    matched: pd.DataFrame, correct_frac: float, rng: np.random.Generator
) -> pd.DataFrame:
    if not 0 < correct_frac <= 1:
        raise ValueError("holdout/correct fraction must be in (0, 1]")
    n_correct = max(1, int(round(len(matched) * correct_frac)))
    n_correct = min(n_correct, len(matched))
    return matched.sample(
        n=n_correct, replace=False, random_state=int(rng.integers(0, 2**32 - 1))
    )


def _sequence_correct_against_any(
    peptides: pd.Series, reference_keys: set[str]
) -> np.ndarray:
    return np.array(
        [normalize_peptide_key(peptide) in reference_keys for peptide in peptides],
        dtype=bool,
    )


def evaluate_score_mixture(
    pool: PeptideScorePool,
    *,
    dataset: str,
    correct_frac: float,
    seed: int,
    n_iterations: int,
    thresholds: list[float],
) -> list[dict[str, object]]:
    """Evaluate observed FDP and recall-style yield for one score pool."""
    if pool.matched.empty:
        raise ValueError(f"{pool.method} has no matched correct peptides for {dataset}")
    if pool.external.empty:
        raise ValueError(f"{pool.method} has no external peptides for {dataset}")

    rng = np.random.default_rng(seed)
    correct = _sample_correct_set(pool.matched, correct_frac, rng).copy()
    correct["source"] = "correct"
    correct_keys = set(correct["peptide_key"])
    estimator_reference = pool.matched[~pool.matched["peptide_key"].isin(correct_keys)]
    if estimator_reference.empty:
        estimator_reference = pool.matched

    rows: list[dict[str, object]] = []
    for iteration in range(n_iterations):
        iter_rng = np.random.default_rng(seed + iteration + 1)
        external = pool.external.sample(
            n=len(pool.external),
            replace=True,
            random_state=int(iter_rng.integers(0, 2**32 - 1)),
        ).copy()
        external["source"] = "external"
        mixed = pd.concat([external, correct], ignore_index=True, sort=False)
        q_table = pool.estimator(mixed, estimator_reference, iter_rng)
        q_table["is_correct"] = _sequence_correct_against_any(
            q_table["peptide"], correct_keys
        )

        n_correct_total = len(correct_keys)
        true_pi0 = len(external) / (len(external) + n_correct_total)
        for threshold in thresholds:
            accepted = q_table[q_table["q_value"] <= threshold]
            n_accepted = len(accepted)
            n_true = int(accepted["is_correct"].sum())
            n_false = n_accepted - n_true
            rows.append(
                {
                    "dataset": dataset,
                    "correct_frac": correct_frac,
                    "true_pi0": true_pi0,
                    "seed": seed,
                    "iteration": iteration,
                    "method": pool.method,
                    "q_value_threshold": threshold,
                    "mixed_external_peptides": len(external),
                    "correct_peptides": n_correct_total,
                    "accepted_peptides": n_accepted,
                    "true_correct_peptides": n_true,
                    "false_external_peptides": n_false,
                    "observed_fdp": n_false / n_accepted if n_accepted else np.nan,
                    "correct_discovery_pct": 100 * n_true / n_correct_total
                    if n_correct_total
                    else np.nan,
                }
            )
    return rows


def load_score_pools(
    *,
    dataset: str,
    data_root: Path,
    winnow_results: Path,
    novoboard_root: Path,
    glissade_root: Path,
    glissade_repo: Path,
    n_bootstraps: int,
    glissade_min_peptide_length: int = 0,
) -> list[PeptideScorePool]:
    """Load Winnow, NovoBoard, and Glissade peptide score pools for one dataset."""
    cfg = build_dataset_configs(winnow_results, novoboard_root, glissade_root)[dataset]
    nb_test_target, _ = load_novoboard_target_decoy(
        cfg.novoboard_dir, "test", cfg.novoboard_decoy_rate
    )
    nb_unlabelled_target, nb_unlabelled_decoy = load_novoboard_target_decoy(
        cfg.novoboard_dir, "unlabelled", cfg.novoboard_decoy_rate
    )
    return [
        load_winnow_score_pool(cfg.winnow_test, cfg.winnow_unlabelled, cfg.fasta),
        load_novoboard_score_pool(
            target_test=nb_test_target,
            target_unlabelled=nb_unlabelled_target,
            decoy_df=nb_unlabelled_decoy,
            data_root=data_root,
            dataset=dataset,
            fasta=cfg.fasta,
        ),
        load_glissade_score_pool(
            cfg.glissade_dir,
            glissade_repo,
            n_bootstraps,
            min_peptide_length=glissade_min_peptide_length,
        ),
    ]


def _plot_one_benchmark_metric(
    dataset_results: pd.DataFrame,
    *,
    dataset_label: str,
    dataset_slug: str,
    metric: str,
    ylabel: str,
    base_name: str,
    percent_axis: bool,
    output_dir: Path,
) -> None:
    """Draw and save one metric panel for a dataset."""
    method_order = ["Winnow", "NovoBoard", "Glissade"]
    colors = {
        "Winnow": _PALETTE[0],
        "NovoBoard": _PALETTE[2],
        "Glissade": _PALETTE[4],
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in method_order:
        sub = dataset_results[dataset_results["method"] == method]
        if sub.empty:
            continue
        summary = (
            sub.groupby("q_value_threshold", as_index=False)[metric]
            .mean(numeric_only=True)
            .sort_values("q_value_threshold")
        )
        ax.plot(
            summary["q_value_threshold"],
            summary[metric],
            lw=1.5,
            label=method,
            color=colors[method],
        )
    if metric == "observed_fdp":
        max_threshold = float(dataset_results["q_value_threshold"].max())
        ax.plot(
            [0.0, max_threshold],
            [0.0, max_threshold],
            color="#666666",
            lw=1,
            ls="--",
            label="Nominal FDR",
        )
        ax.set_ylim(bottom=0)
    if percent_axis:
        ax.set_ylim(0, 100)
    ax.set_xlim(0, float(dataset_results["q_value_threshold"].max()))
    ax.set_xlabel("Estimated q-value threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset_label} external peptide score-mixture benchmark")
    ax.legend(loc="best")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"{base_name}_{dataset_slug}")


def plot_benchmark_results(results: pd.DataFrame, output_dir: Path) -> None:
    """Save observed FDP and correct-discovery percentage plots per dataset."""
    if results.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("observed_fdp", "Observed FDP", "external_peptide_score_mixture_fdp", False),
        (
            "correct_discovery_pct",
            "Correct peptide recovery\n(% of held-out correct peptides)",
            "external_peptide_score_mixture_correct_discovery_pct",
            True,
        ),
    ]
    for dataset, dataset_results in results.groupby("dataset", sort=False):
        dataset_label = _display_name(str(dataset))
        dataset_slug = str(dataset).replace("/", "_")
        for metric, ylabel, base_name, percent_axis in plot_specs:
            _plot_one_benchmark_metric(
                dataset_results,
                dataset_label=dataset_label,
                dataset_slug=dataset_slug,
                metric=metric,
                ylabel=ylabel,
                base_name=base_name,
                percent_axis=percent_axis,
                output_dir=output_dir,
            )


@app.command()
def main(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for benchmark outputs."),
    ] = DEFAULT_OUTPUT_DIR,
    datasets: Annotated[
        Optional[list[str]],
        typer.Option("--datasets", help="Dataset keys to benchmark."),
    ] = None,
    holdout_fracs: Annotated[
        Optional[list[float]],
        typer.Option(
            "--holdout-fracs",
            help="Fraction of matched correct peptides to hold out as S_c.",
        ),
    ] = None,
    q_thresholds: Annotated[
        Optional[list[float]],
        typer.Option(
            "--q-thresholds", help="Estimated q-value thresholds to evaluate."
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = DEFAULT_SEED,
    n_iterations: Annotated[
        int,
        typer.Option(
            "--n-iterations", help="Number of external-score resampling iterations."
        ),
    ] = DEFAULT_N_ITERATIONS,
    data_root: Annotated[
        Path,
        typer.Option(
            "--data-root", help="Root containing *_split_parquet directories."
        ),
    ] = _REPO_ROOT,
    winnow_results: Annotated[
        Path,
        typer.Option("--winnow-results", help="Winnow results directory."),
    ] = DEFAULT_WINNOW_RESULTS,
    novoboard_root: Annotated[
        Path,
        typer.Option("--novoboard-root", help="NovoBoard datasets root."),
    ] = DEFAULT_NOVOBOARD_ROOT,
    glissade_root: Annotated[
        Path,
        typer.Option("--glissade-root", help="Glissade build root."),
    ] = DEFAULT_GLISSADE_ROOT,
    glissade_repo: Annotated[
        Path,
        typer.Option("--glissade-repo", help="Glissade repository root."),
    ] = DEFAULT_GLISSADE_REPO,
    n_bootstraps: Annotated[
        int,
        typer.Option(
            "--n-bootstraps", help="Glissade bootstraps per mixture iteration."
        ),
    ] = DEFAULT_N_BOOTSTRAPS,
    glissade_min_peptide_length: Annotated[
        int,
        typer.Option(
            "--glissade-min-peptide-length",
            help=(
                "Minimum normalised peptide length for Glissade pools only "
                "(Glissade seperate_scores default: 8). 0 disables."
            ),
        ),
    ] = 0,
    plot: Annotated[bool, typer.Option(help="Create summary plots.")] = True,
) -> None:
    """Run the score-mixture external peptide benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_keys = datasets if datasets is not None else DEFAULT_DATASETS
    fracs = holdout_fracs if holdout_fracs is not None else [DEFAULT_HOLDOUT_FRAC]
    thresholds = (
        q_thresholds if q_thresholds is not None else list(DEFAULT_Q_THRESHOLDS)
    )

    rows: list[dict[str, object]] = []
    for dataset in dataset_keys:
        logger.info("Loading score pools for %s", _display_name(dataset))
        pools = load_score_pools(
            dataset=dataset,
            data_root=data_root,
            winnow_results=winnow_results,
            novoboard_root=novoboard_root,
            glissade_root=glissade_root,
            glissade_repo=glissade_repo,
            n_bootstraps=n_bootstraps,
            glissade_min_peptide_length=glissade_min_peptide_length,
        )
        for frac in fracs:
            logger.info(
                "Processing %s correct_frac=%.3g iterations=%d",
                _display_name(dataset),
                frac,
                n_iterations,
            )
            for pool in pools:
                logger.info(
                    "%s pools: matched=%d external=%d",
                    pool.method,
                    len(pool.matched),
                    len(pool.external),
                )
                rows.extend(
                    evaluate_score_mixture(
                        pool,
                        dataset=dataset,
                        correct_frac=frac,
                        seed=seed,
                        n_iterations=n_iterations,
                        thresholds=thresholds,
                    )
                )

    results = pd.DataFrame(rows)
    results_path = output_dir / "external_peptide_holdout_results.csv"
    results.to_csv(results_path, index=False)
    logger.info("Wrote %s", results_path)
    if plot:
        plot_benchmark_results(results, output_dir / "plots")


if __name__ == "__main__":
    app()
