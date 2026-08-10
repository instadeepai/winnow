#!/usr/bin/env python3
"""Glissade-style external peptide score-mixture benchmark.

Builds a **shared** matched pool S_m (labelled-test Novor-correct peptides) and
external pool S_e (unlabelled proteome-external peptides) after filter →
max-score-per-peptide (NovoBoard mass-deltas converted to ProForma; unsupported
mods dropped; NovoBoard target-decoy pairs gated so every retained key has a
twin). Unlabelled / external peptides require normalised length ≥ 8 (proteome
substring proxy); labelled matched peptides and Glissade's training-split
reference keep short peptides (Novor agreement). Novor and proteome-hit labels
are computed once on Winnow and reused for NovoBoard by ``spectrum_id``.
Method-specific scores are attached to the same peptide keys. Mixtures control
π₀ explicitly.

Mixtures are drawn **without replacement** so every peptide key is unique. All
three methods therefore score the identical mixture and realise the same π₀;
NovoBoard's max-score-per-peptide step is a no-op, which is asserted per
mixture.

Each tool draws its null/reference information from the same place, the
annotated *training* split of its own organism: Winnow through the pretrained
per-dataset calibrator, Glissade through the training-split matched score
distribution, NovoBoard through its training-tuned decoy masking rate. No tool
fits on the evaluation labels.

NovoBoard peptide FDR uses max-target → twin-decoy TDC. Winnow uses max
calibrated confidence then nonparametric FDR (PSM-calibrator proxy). Glissade
uses native bootstrap FDR with NumPy seeded from the benchmark RNG.

External tool checkouts for local results:

- ``--novoboard-root``: ``{root}/{dataset}/novoboard/`` target/decoy CSVs (the
  ``datasets`` dir of a NovoBoard checkout). Local runs used fork
  ``git@github.com:JemmaLDaniel/NovoBoard.git``, branch
  ``feat/adapt-to-instanovo`` at
  ``a9faab3ef1af06987599c2f01e6ba96072c80172``.
- ``--glissade-repo``: clone root with importable ``glissade.glissade``. Local
  runs used fork ``git@github.com:JemmaLDaniel/glissade.git``, branch
  ``winnow-benchmark`` at ``6ee11b51b5f21ba8fdc1eb5821608352b082a533``.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import typer

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.fdr_tool_comparison_preprocess import (  # noqa: E402
    LABELLED_MIN_PEPTIDE_LENGTH,
    MIN_PEPTIDE_LENGTH,
    assert_shared_prediction_keys,
    attach_labels_by_spectrum_id,
    confidence_to_log_prob,
    filter_novoboard_target_decoy_pairs,
    filter_prediction_table,
    label_series_by_spectrum_id,
    max_score_per_peptide,
    novoboard_max_target_twin_decoy_tdc,
    novor_correctness_mask,
    prepare_novoboard_decoy_by_pair,
    restrict_winnow_to_novoboard_spectra,
)
from scripts.fdr_tool_comparison_summaries import (  # noqa: E402
    SUMMARY_THRESHOLDS,
    database_grounded_q_from_labels,
    mean_abs_q_dev_vs_reference,
    summarise_holdout_results,
    write_summary_tables,
)
from scripts.plot_eval_results import _PALETTE, _display_name, _save_fig, _style_ax  # noqa: E402
from scripts.plot_fdr_method_comparison import (  # noqa: E402
    DEFAULT_MODEL_ROOT,
    DEFAULT_WINNOW_RESULTS,
    build_dataset_configs,
    load_novoboard_target_decoy,
    load_winnow,
)
from winnow.fdr.nonparametric import NonParametricFDRControl  # noqa: E402

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results/external_peptide_holdout_benchmark_v2"
DEFAULT_DATASETS = ["helaqc", "celegans"]
DEFAULT_Q_THRESHOLDS = [round(float(x), 2) for x in np.linspace(0.0, 0.25, 26)]
DEFAULT_PI0_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_N_ITERATIONS = 20
# Matches the Glissade package default for run_bootstraps.
DEFAULT_N_BOOTSTRAPS = 10
# Prefer the full matched pool; |S_c| is then capped so the highest π₀ remains
# drawable without replacement from S_e.
DEFAULT_HOLDOUT_FRAC = 1.0
DEFAULT_SEED = 42
METHODS = ("Winnow", "NovoBoard", "Glissade")


def max_correct_pool_for_pi0_grid(n_external_pool: int, pi0_grid: list[float]) -> int:
    """Largest |S_c| such that every π₀ in ``pi0_grid`` fits without replacement.

    Requires ``round(π₀ / (1 - π₀) · |S_c|) ≤ |S_e|`` for each target π₀.
    """
    if n_external_pool < 1:
        return 0
    limit = n_external_pool
    for pi0 in pi0_grid:
        if not 0.0 < pi0 < 1.0:
            continue
        ratio = pi0 / (1.0 - pi0)
        # Largest n with round(ratio * n) <= n_external_pool.
        # For ratio = k integer (e.g. 0.9 → 9), this is floor(pool / k).
        hi = int(n_external_pool / ratio) + 2
        n_ok = 0
        for n in range(1, hi + 1):
            if int(round(ratio * n)) <= n_external_pool:
                n_ok = n
            else:
                break
        limit = min(limit, n_ok)
    return max(0, limit)


def _load_glissade_functions(glissade_repo: Path):
    repo = str(glissade_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    module = importlib.import_module("glissade.glissade")
    return module.run_bootstraps, module.annotate_results, module.compute_fdr_transform


def _load_winnow_with_raw_confidence(
    predictions_dir: Path, fasta: Path, eval_type: str
) -> pd.DataFrame:
    """Load Winnow preds and ensure raw ``confidence`` is present."""
    df = load_winnow(predictions_dir, fasta, eval_type)  # type: ignore[arg-type]
    if "confidence" not in df.columns:
        meta_path = predictions_dir / "metadata.csv"
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        meta = pd.read_csv(meta_path, usecols=["spectrum_id", "confidence"])
        df = df.merge(meta, on="spectrum_id", how="inner")
    return df


def build_glissade_training_reference(
    train_metadata: Path,
    *,
    min_length: int = LABELLED_MIN_PEPTIDE_LENGTH,
) -> pd.DataFrame:
    """Matched reference score distribution for Glissade, from the training split.

    Glissade anchors its null-fraction estimate on a database-matched score
    distribution. Taking that anchor from the annotated training split puts it on
    the same data the Winnow calibrator was trained on and the NovoBoard decoy
    masking rate was tuned on, and keeps it disjoint from the evaluation spectra.
    Short peptides are retained: the reference is labelled (Novor-correct).

    Args:
        train_metadata: Calibrator training metadata parquet.
        min_length: Minimum normalised peptide length (default: labelled floor).

    Returns:
        One row per Novor-correct training peptide with ``raw_confidence`` and
        ``score_glissade``.
    """
    if not train_metadata.is_file():
        raise FileNotFoundError(train_metadata)
    train = pl.read_parquet(
        train_metadata,
        columns=["spectrum_id", "prediction", "sequence", "confidence"],
    ).to_pandas()
    train = filter_prediction_table(
        train, "prediction", min_length=min_length, key_col="peptide_key"
    )
    train["correct"] = novor_correctness_mask(train["sequence"], train["prediction"])
    matched = max_score_per_peptide(
        train.loc[train["correct"]], "peptide_key", "confidence"
    )
    reference = matched[["peptide_key", "confidence"]].rename(
        columns={"confidence": "raw_confidence"}
    )
    reference["score_glissade"] = confidence_to_log_prob(reference["raw_confidence"])
    logger.info(
        "Glissade training reference: %d matched peptides from %s",
        len(reference),
        train_metadata,
    )
    return reference


def _namespace_pair_keys(df: pd.DataFrame, namespace: str) -> pd.DataFrame:
    """Prefix ``_pair_key`` to avoid collisions across splits."""
    work = df.copy()
    if "_pair_key" not in work.columns:
        raise ValueError("Missing '_pair_key'")
    work["_pair_key"] = namespace + ":" + work["_pair_key"].astype(str)
    return work


def build_shared_score_tables(
    *,
    dataset: str,
    winnow_results: Path,
    novoboard_root: Path,
    model_root: Path = DEFAULT_MODEL_ROOT,
    unlabelled_min_length: int = MIN_PEPTIDE_LENGTH,
    labelled_min_length: int = LABELLED_MIN_PEPTIDE_LENGTH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return shared matched/external keys with per-method scores.

    All methods follow filter → :func:`max_score_per_peptide`. NovoBoard tables
    are pair-gated (equal target/decoy twins) before max-dedupe so mixture keys
    always have a twin. Labelled S_m and Glissade's training reference keep short
    peptides (``labelled_min_length``); unlabelled S_e uses
    ``unlabelled_min_length`` for the proteome-substring proxy. Labelled S_m uses
    Novor correctness computed once on Winnow and reused for NovoBoard by
    ``spectrum_id``; S_e uses proteome-hit the same way. Glissade scores are
    ``log`` raw InstaNovo confidence on the shared keys.

    Returns:
        matched_scores: one row per peptide_key in S_m with method scores.
        external_scores: same for S_e.
        nb_decoy_combined: namespaced twin-valid decoys for twin TDC.
        glissade_reference: training-split matched scores for Glissade's anchor.
    """
    cfg = build_dataset_configs(
        winnow_results, novoboard_root=novoboard_root, model_root=model_root
    )[dataset]

    winnow_test = _load_winnow_with_raw_confidence(
        cfg.winnow_test, cfg.fasta, "labelled"
    )
    winnow_unlab = _load_winnow_with_raw_confidence(
        cfg.winnow_unlabelled, cfg.fasta, "unlabelled"
    )
    # load_winnow already applied the labelled/unlabelled length floors; re-apply
    # explicitly so callers can override without depending on that path.
    winnow_test = filter_prediction_table(
        winnow_test,
        "prediction",
        min_length=labelled_min_length,
        key_col="peptide_key",
    )
    w_unlab = filter_prediction_table(
        winnow_unlab,
        "prediction",
        min_length=unlabelled_min_length,
        key_col="peptide_key",
    )
    if "proteome_hit" not in w_unlab.columns:
        raise KeyError("Expected proteome_hit on unlabelled Winnow table")

    nb_test_target, nb_test_decoy = load_novoboard_target_decoy(
        cfg.novoboard_dir, "test", cfg.novoboard_decoy_rate
    )
    nb_unlab_target, nb_unlab_decoy = load_novoboard_target_decoy(
        cfg.novoboard_dir, "unlabelled", cfg.novoboard_decoy_rate
    )
    nb_test_target = _namespace_pair_keys(nb_test_target, "test")
    nb_test_decoy = _namespace_pair_keys(nb_test_decoy, "test")
    nb_unlab_target = _namespace_pair_keys(nb_unlab_target, "unlabelled")
    nb_unlab_decoy = _namespace_pair_keys(nb_unlab_decoy, "unlabelled")

    nb_test_target, nb_test_decoy = filter_novoboard_target_decoy_pairs(
        nb_test_target,
        nb_test_decoy,
        min_length=labelled_min_length,
        key_col="peptide_key",
    )
    nb_unlab_target, nb_unlab_decoy = filter_novoboard_target_decoy_pairs(
        nb_unlab_target,
        nb_unlab_decoy,
        min_length=unlabelled_min_length,
        key_col="peptide_key",
    )
    nb_decoy_combined = pd.concat(
        [nb_test_decoy, nb_unlab_decoy], ignore_index=True, sort=False
    )

    # Twin-valid NovoBoard ⊆ Winnow; shared labels once on Winnow.
    winnow_test = restrict_winnow_to_novoboard_spectra(winnow_test, nb_test_target)
    w_unlab = restrict_winnow_to_novoboard_spectra(w_unlab, nb_unlab_target)
    assert_shared_prediction_keys(winnow_test, nb_test_target)
    assert_shared_prediction_keys(w_unlab, nb_unlab_target)

    winnow_test = winnow_test.copy()
    winnow_test["correct"] = novor_correctness_mask(
        winnow_test["sequence"], winnow_test["prediction"]
    )
    correct_by_id = label_series_by_spectrum_id(winnow_test, "correct")
    hit_by_id = label_series_by_spectrum_id(w_unlab, "proteome_hit")
    nb_test_target = attach_labels_by_spectrum_id(
        nb_test_target, correct_by_id, label_col="correct"
    )
    nb_unlab_f = attach_labels_by_spectrum_id(
        nb_unlab_target, hit_by_id, label_col="proteome_hit"
    )

    w_matched = max_score_per_peptide(
        winnow_test.loc[winnow_test["correct"]],
        "peptide_key",
        "calibrated_confidence",
    )
    w_matched_raw = max_score_per_peptide(
        winnow_test.loc[winnow_test["correct"]],
        "peptide_key",
        "confidence",
    )
    nb_correct = nb_test_target.loc[nb_test_target["correct"]].copy()
    nb_matched = max_score_per_peptide(nb_correct, "peptide_key", "ALC (%)")

    w_external = max_score_per_peptide(
        w_unlab.loc[~w_unlab["proteome_hit"].astype(bool)],
        "peptide_key",
        "calibrated_confidence",
    )
    w_external_raw = max_score_per_peptide(
        w_unlab.loc[~w_unlab["proteome_hit"].astype(bool)],
        "peptide_key",
        "confidence",
    )
    nb_external = max_score_per_peptide(
        nb_unlab_f.loc[~nb_unlab_f["proteome_hit"].astype(bool)],
        "peptide_key",
        "ALC (%)",
    )

    # Shared membership = Winnow ∩ twin-valid NovoBoard keys.
    sm_keys = (
        set(w_matched["peptide_key"])
        & set(nb_matched["peptide_key"])
        & set(w_matched_raw["peptide_key"])
    )
    se_keys = (
        set(w_external["peptide_key"])
        & set(nb_external["peptide_key"])
        & set(w_external_raw["peptide_key"])
    )
    if sm_keys != set(w_matched["peptide_key"]) or sm_keys != set(
        nb_matched["peptide_key"]
    ):
        raise AssertionError(
            f"{dataset} shared S_m keys disagree after shared Novor labels: "
            f"winnow={len(w_matched)} novoboard={len(nb_matched)} "
            f"intersection={len(sm_keys)}"
        )
    if se_keys != set(w_external["peptide_key"]) or se_keys != set(
        nb_external["peptide_key"]
    ):
        raise AssertionError(
            f"{dataset} shared S_e keys disagree after shared proteome-hit labels: "
            f"winnow={len(w_external)} novoboard={len(nb_external)} "
            f"intersection={len(se_keys)}"
        )
    if not sm_keys:
        raise ValueError(f"Empty shared matched pool for {dataset}")
    if not se_keys:
        raise ValueError(f"Empty shared external pool for {dataset}")

    matched = pd.DataFrame({"peptide_key": sorted(sm_keys)})
    matched = matched.merge(
        w_matched[["peptide_key", "calibrated_confidence"]].rename(
            columns={"calibrated_confidence": "score_winnow"}
        ),
        on="peptide_key",
        how="left",
    )
    matched = matched.merge(
        nb_matched[["peptide_key", "ALC (%)"]].rename(
            columns={"ALC (%)": "score_novoboard"}
        ),
        on="peptide_key",
        how="left",
    )
    matched = matched.merge(
        w_matched_raw[["peptide_key", "confidence"]].rename(
            columns={"confidence": "raw_confidence"}
        ),
        on="peptide_key",
        how="left",
    )
    matched["score_glissade"] = confidence_to_log_prob(matched["raw_confidence"])

    external = pd.DataFrame({"peptide_key": sorted(se_keys)})
    external = external.merge(
        w_external[["peptide_key", "calibrated_confidence"]].rename(
            columns={"calibrated_confidence": "score_winnow"}
        ),
        on="peptide_key",
        how="left",
    )
    external = external.merge(
        nb_external[["peptide_key", "ALC (%)"]].rename(
            columns={"ALC (%)": "score_novoboard"}
        ),
        on="peptide_key",
        how="left",
    )
    external = external.merge(
        w_external_raw[["peptide_key", "confidence"]].rename(
            columns={"confidence": "raw_confidence"}
        ),
        on="peptide_key",
        how="left",
    )
    external["score_glissade"] = confidence_to_log_prob(external["raw_confidence"])

    # Attach NovoBoard pair metadata for twin TDC on mixture subsets.
    nb_ext_meta = nb_unlab_f.loc[
        ~nb_unlab_f["proteome_hit"].astype(bool),
        ["peptide_key", "Peptide", "ALC (%)", "spectrum_id", "_pair_key", "Scan"],
    ].copy()
    nb_ext_meta = (
        nb_ext_meta.sort_values("ALC (%)", ascending=False)
        .groupby("peptide_key", as_index=False)
        .first()
    )
    external = external.merge(
        nb_ext_meta.rename(
            columns={
                "Peptide": "peptide_novoboard",
                "spectrum_id": "spectrum_id_novoboard",
            }
        ),
        on="peptide_key",
        how="left",
    )

    nb_match_meta = (
        nb_correct[
            ["peptide_key", "Peptide", "ALC (%)", "spectrum_id", "_pair_key", "Scan"]
        ]
        .sort_values("ALC (%)", ascending=False)
        .groupby("peptide_key", as_index=False)
        .first()
        .rename(
            columns={
                "Peptide": "peptide_novoboard",
                "spectrum_id": "spectrum_id_novoboard",
            }
        )
    )
    matched = matched.merge(nb_match_meta, on="peptide_key", how="left")

    twin_coverage_m = (
        float(matched["_pair_key"].notna().mean()) if len(matched) else 0.0
    )
    twin_coverage_e = (
        float(external["_pair_key"].notna().mean()) if len(external) else 0.0
    )
    if twin_coverage_m < 1.0 or twin_coverage_e < 1.0:
        raise AssertionError(
            f"{dataset} NovoBoard twin coverage incomplete: "
            f"Sm={twin_coverage_m:.3f} Se={twin_coverage_e:.3f}"
        )

    glissade_reference = build_glissade_training_reference(
        cfg.calibrator_train_metadata, min_length=labelled_min_length
    )

    logger.info(
        "%s shared pools: matched=%d external=%d glissade_reference=%d "
        "(shared Novor/proteome-hit labels; NB twin coverage 100%%)",
        dataset,
        len(matched),
        len(external),
        len(glissade_reference),
    )
    return matched, external, nb_decoy_combined, glissade_reference


def _estimate_winnow_q_values(mixed: pd.DataFrame) -> pd.DataFrame:
    work = mixed[["peptide_key", "score_winnow", "source"]].copy()
    work = work.rename(columns={"score_winnow": "score"})
    work = work.dropna(subset=["score"])
    ctrl = NonParametricFDRControl()
    ctrl.fit(work["score"])
    q_table = ctrl.add_psm_q_value(work.copy(), "score")
    return pd.DataFrame(
        {
            "peptide_key": q_table["peptide_key"],
            "score": q_table["score"],
            "source": q_table["source"],
            "q_value": q_table["psm_q_value"],
            "method": "Winnow",
        }
    )


def _estimate_novoboard_q_values(
    mixed: pd.DataFrame,
    decoy_by_pair: pd.DataFrame,
) -> pd.DataFrame:
    target = pd.DataFrame(
        {
            "Peptide": mixed["peptide_novoboard"].fillna(mixed["peptide_key"]),
            "ALC (%)": mixed["score_novoboard"],
            "spectrum_id": mixed.get(
                "spectrum_id_novoboard",
                pd.Series(np.arange(len(mixed)), dtype=str),
            ),
            "_pair_key": mixed["_pair_key"],
            "Scan": mixed.get("Scan", mixed["_pair_key"]),
            "source": mixed["source"],
            "peptide_key": mixed["peptide_key"],
        }
    )
    target = target.dropna(subset=["ALC (%)", "_pair_key"])
    # Restrict twin TDC to the mixture peptide keys only.
    table = novoboard_max_target_twin_decoy_tdc(
        target,
        pd.DataFrame(),
        target_peptide_keys=set(target["peptide_key"].astype(str)),
        decoy_by_pair=decoy_by_pair,
        log_missing_twins=False,
    )
    targets = table[table["is_target"]].copy()
    # Map back sources for FDP.
    source_map = mixed.set_index("peptide_key")["source"].to_dict()
    key_col = "_peptide_key" if "_peptide_key" in targets.columns else "peptide_key"
    targets["peptide_key"] = targets[key_col]
    targets["source"] = targets["peptide_key"].map(source_map)
    return pd.DataFrame(
        {
            "peptide_key": targets["peptide_key"],
            "score": targets["ALC (%)"],
            "source": targets["source"],
            "q_value": targets["estimated_q_value"],
            "method": "NovoBoard",
        }
    )


def _estimate_glissade_q_values(
    mixed: pd.DataFrame,
    matched_reference: pd.DataFrame,
    *,
    glissade_repo: Path,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    run_bootstraps, annotate_results, compute_fdr_transform = _load_glissade_functions(
        glissade_repo
    )
    # Glissade FDR is defined on the mixture scores; the reference is the
    # training-split matched distribution and is never scored itself.
    mixed_scores = mixed["score_glissade"].astype(float).to_numpy()
    matched_scores = matched_reference["score_glissade"].astype(float).to_numpy()
    if len(mixed_scores) < 10 or len(matched_scores) < 10:
        raise ValueError("Glissade FDR requires ≥10 reference and mixture scores")

    peptides = mixed["peptide_key"].astype(str).tolist()
    np.random.seed(int(rng.integers(0, 2**32 - 1)))
    fdrs, grid, _ = run_bootstraps(
        matched_scores,
        mixed_scores,
        n_bootstraps=n_bootstraps,
    )
    out_peptides, peptide_fdrs, scores = annotate_results(
        peptides, mixed_scores, fdrs, grid
    )
    peptide_fdrs = compute_fdr_transform(peptide_fdrs)
    source_map = mixed.set_index("peptide_key")["source"].to_dict()
    return pd.DataFrame(
        {
            "peptide_key": out_peptides,
            "score": scores,
            "source": [source_map.get(p) for p in out_peptides],
            "q_value": peptide_fdrs,
            "method": "Glissade",
        }
    )


def _mixture_result_rows(
    *,
    dataset: str,
    pi0: float,
    true_pi0: float,
    holdout_frac: float,
    seed: int,
    iteration: int,
    method: str,
    q_table: pd.DataFrame,
    correct_keys: set[str],
    n_external: int,
    thresholds: list[float],
    q_ref: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Build per-threshold result rows for one method on one mixture."""
    work = q_table.dropna(subset=["q_value"])
    if q_ref is not None:
        if len(q_ref) != len(q_table):
            raise ValueError(
                f"q_ref length {len(q_ref)} does not match q_table length {len(q_table)}"
            )
        q_ref_aligned = pd.Series(np.asarray(q_ref, dtype=float), index=q_table.index)
        q_ref_work = q_ref_aligned.loc[work.index].to_numpy(dtype=float)
        q_devs = mean_abs_q_dev_vs_reference(
            work["q_value"].to_numpy(dtype=float), q_ref_work, thresholds
        )
    else:
        q_devs = [float("nan")] * len(thresholds)

    rows: list[dict[str, object]] = []
    for threshold, q_dev in zip(thresholds, q_devs):
        accepted = work[work["q_value"] <= threshold]
        n_accepted = len(accepted)
        n_true = int(accepted["peptide_key"].isin(correct_keys).sum())
        n_false = n_accepted - n_true
        rows.append(
            {
                "dataset": dataset,
                "pi0_target": float(pi0),
                "true_pi0": float(true_pi0),
                "holdout_frac": float(holdout_frac),
                "seed": seed,
                "iteration": iteration,
                "method": method,
                "q_value_threshold": float(threshold),
                "mixed_external_peptides": n_external,
                "correct_peptides": len(correct_keys),
                "accepted_peptides": n_accepted,
                "true_correct_peptides": n_true,
                "false_external_peptides": n_false,
                "observed_fdp": (n_false / n_accepted if n_accepted else np.nan),
                "correct_discovery_pct": (
                    100.0 * n_true / len(correct_keys) if correct_keys else np.nan
                ),
                "mean_abs_q_dev_vs_db": float(q_dev),
            }
        )
    return rows


def _evaluate_mixture_methods(
    *,
    mixed: pd.DataFrame,
    estimator_reference: pd.DataFrame,
    decoy_by_pair: pd.DataFrame,
    glissade_repo: Path,
    n_bootstraps: int,
    iter_rng: np.random.Generator,
    dataset: str,
    pi0: float,
    true_pi0: float,
    holdout_frac: float,
    seed: int,
    iteration: int,
    correct_keys: set[str],
    n_external: int,
    thresholds: list[float],
) -> list[dict[str, object]]:
    """Run Winnow / NovoBoard / Glissade FDR on one mixture and collect rows."""
    estimators: dict[str, object] = {
        "Winnow": lambda m, _r: _estimate_winnow_q_values(m),
        "NovoBoard": lambda m, _r: _estimate_novoboard_q_values(m, decoy_by_pair),
        "Glissade": lambda m, r, rng=iter_rng: _estimate_glissade_q_values(
            m,
            r,
            glissade_repo=glissade_repo,
            n_bootstraps=n_bootstraps,
            rng=rng,
        ),
    }
    # Raw-confidence DBG on the mixture used for NovoBoard and Glissade q-deviation. Winnow keeps calibrated-score DBG unset here.
    is_correct = mixed["peptide_key"].astype(str).isin(correct_keys).to_numpy()
    q_db_raw = database_grounded_q_from_labels(
        mixed["score_novoboard"].to_numpy(dtype=float), is_correct
    )
    q_db_raw_by_key = dict(zip(mixed["peptide_key"].astype(str), q_db_raw, strict=True))

    rows: list[dict[str, object]] = []
    mixture_keys = set(mixed["peptide_key"].astype(str))
    for method, estimator in estimators.items():
        try:
            q_table = estimator(mixed, estimator_reference)  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 - boundary around external tool
            logger.warning(
                "%s FDR failed dataset=%s pi0=%.3g iter=%d: %s",
                method,
                dataset,
                pi0,
                iteration,
                exc,
            )
            continue
        scored_keys = set(q_table["peptide_key"].astype(str))
        if scored_keys != mixture_keys:
            raise AssertionError(
                f"{method} scored a different mixture on dataset={dataset} "
                f"pi0={pi0:.3g} iter={iteration}: mixture={len(mixture_keys)} "
                f"scored={len(scored_keys)} "
                f"missing={len(mixture_keys - scored_keys)} "
                f"extra={len(scored_keys - mixture_keys)}"
            )
        q_ref: np.ndarray | None = None
        if method in ("NovoBoard", "Glissade"):
            q_ref = (
                q_table["peptide_key"]
                .astype(str)
                .map(q_db_raw_by_key)
                .to_numpy(dtype=float)
            )
        rows.extend(
            _mixture_result_rows(
                dataset=dataset,
                pi0=pi0,
                true_pi0=true_pi0,
                holdout_frac=holdout_frac,
                seed=seed,
                iteration=iteration,
                method=method,
                q_table=q_table,
                correct_keys=correct_keys,
                n_external=n_external,
                thresholds=thresholds,
                q_ref=q_ref,
            )
        )
    return rows


def _sample_correct_component(
    *,
    dataset: str,
    matched: pd.DataFrame,
    external: pd.DataFrame,
    pi0_grid: list[float],
    holdout_frac: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, set[str], float]:
    """Sample S_c from matched, capped so the π₀ grid fits in S_e."""
    n_from_frac = max(1, int(round(len(matched) * holdout_frac)))
    n_from_frac = min(n_from_frac, len(matched))
    n_cap = max_correct_pool_for_pi0_grid(len(external), pi0_grid)
    if n_cap < 1:
        raise ValueError(
            f"{dataset}: external pool of {len(external)} cannot support any "
            f"π₀ in {pi0_grid} without replacement"
        )
    n_correct = min(n_from_frac, n_cap)
    if n_correct < n_from_frac:
        logger.info(
            "%s capping |S_c| from %d to %d so π₀ grid %s fits in |S_e|=%d "
            "without replacement",
            dataset,
            n_from_frac,
            n_correct,
            pi0_grid,
            len(external),
        )
    correct = matched.sample(n=n_correct, random_state=int(rng.integers(0, 2**32 - 1)))
    correct = correct.copy()
    correct["source"] = "correct"
    correct_keys = set(correct["peptide_key"])
    effective_holdout_frac = n_correct / len(matched) if len(matched) else 0.0
    overlap = correct_keys & set(external["peptide_key"])
    if overlap:
        raise AssertionError(
            f"{dataset} S_c and S_e share {len(overlap)} peptide keys; "
            "mixture sources would be ambiguous"
        )
    return correct, correct_keys, effective_holdout_frac


def _external_draw_size(
    *,
    dataset: str,
    pi0: float,
    n_correct: int,
    n_external_pool: int,
) -> int | None:
    """Return |S_e'| for π₀, or None to skip; raise if the pool is too small."""
    if not 0.0 < pi0 < 1.0:
        return None
    n_external = int(round(pi0 / (1.0 - pi0) * n_correct))
    if n_external < 1:
        logger.warning("Skipping pi0=%.3g: requested |S_e'|=%d", pi0, n_external)
        return None
    if n_external > n_external_pool:
        raise AssertionError(
            f"{dataset} pi0={pi0:.3g}: |S_e'|={n_external} exceeds pool "
            f"{n_external_pool} after |S_c| cap {n_correct}"
        )
    return n_external


def evaluate_controlled_mixtures(
    *,
    dataset: str,
    matched: pd.DataFrame,
    external: pd.DataFrame,
    nb_decoy: pd.DataFrame,
    glissade_reference: pd.DataFrame,
    pi0_grid: list[float],
    holdout_frac: float,
    seed: int,
    n_iterations: int,
    thresholds: list[float],
    glissade_repo: Path,
    n_bootstraps: int,
) -> list[dict[str, object]]:
    """Evaluate all methods on shared mixtures with controlled π₀.

    Uses as much of the matched pool as possible (up to holdout_frac), capped
    so every π₀ in pi0_grid can draw its null component from S_e without
    replacement. The null component is drawn without replacement, so mixture
    peptide keys are unique and every method realises the same π₀ on the same
    rows.
    """
    rng = np.random.default_rng(seed)
    correct, correct_keys, effective_holdout_frac = _sample_correct_component(
        dataset=dataset,
        matched=matched,
        external=external,
        pi0_grid=pi0_grid,
        holdout_frac=holdout_frac,
        rng=rng,
    )
    decoy_by_pair = prepare_novoboard_decoy_by_pair(nb_decoy, already_filtered=True)

    rows: list[dict[str, object]] = []
    for pi0 in pi0_grid:
        n_external = _external_draw_size(
            dataset=dataset,
            pi0=pi0,
            n_correct=len(correct_keys),
            n_external_pool=len(external),
        )
        if n_external is None:
            continue

        for iteration in range(n_iterations):
            iter_rng = np.random.default_rng(
                seed + 10_000 * iteration + int(1000 * pi0)
            )
            ext_sample = external.sample(
                n=n_external,
                replace=False,
                random_state=int(iter_rng.integers(0, 2**32 - 1)),
            ).copy()
            ext_sample["source"] = "external"
            mixed = pd.concat([ext_sample, correct], ignore_index=True, sort=False)
            if mixed["peptide_key"].duplicated().any():
                raise AssertionError(
                    f"{dataset} mixture has duplicate peptide keys at "
                    f"pi0={pi0:.3g} iter={iteration}"
                )
            true_pi0 = len(ext_sample) / (len(ext_sample) + len(correct_keys))
            rows.extend(
                _evaluate_mixture_methods(
                    mixed=mixed,
                    estimator_reference=glissade_reference,
                    decoy_by_pair=decoy_by_pair,
                    glissade_repo=glissade_repo,
                    n_bootstraps=n_bootstraps,
                    iter_rng=iter_rng,
                    dataset=dataset,
                    pi0=pi0,
                    true_pi0=true_pi0,
                    holdout_frac=effective_holdout_frac,
                    seed=seed,
                    iteration=iteration,
                    correct_keys=correct_keys,
                    n_external=len(ext_sample),
                    thresholds=thresholds,
                )
            )
    return rows


def _plot_metric_by_pi0(
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
    method_order = list(METHODS)
    colors = {"Winnow": _PALETTE[0], "NovoBoard": _PALETTE[2], "Glissade": _PALETTE[4]}
    pi0_values = sorted(dataset_results["pi0_target"].dropna().unique())
    n_pi0 = max(1, len(pi0_values))
    fig, axes = plt.subplots(
        1, n_pi0, figsize=(4.2 * n_pi0, 5.5), sharey=True, squeeze=False
    )
    for ax, pi0 in zip(axes[0], pi0_values):
        sub = dataset_results[dataset_results["pi0_target"] == pi0]
        for method in method_order:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            summary = (
                msub.groupby("q_value_threshold", as_index=False)[metric]
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
            max_threshold = float(sub["q_value_threshold"].max())
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
        ax.set_xlim(0, float(sub["q_value_threshold"].max()))
        ax.set_xlabel("Estimated q-value threshold")
        ax.set_title(f"π₀={pi0:g}")
        _style_ax(ax)
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].legend(loc="best", fontsize=9)
    fig.suptitle(f"{dataset_label} external peptide score-mixture benchmark", y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"{base_name}_{dataset_slug}")


def plot_benchmark_results(results: pd.DataFrame, output_dir: Path) -> None:
    """Save FDP and recovery plots faceted by π₀."""
    if results.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("observed_fdp", "Observed FDP", "external_peptide_score_mixture_fdp", False),
        (
            "correct_discovery_pct",
            "Correct peptide recovery\n(% of held-out correct peptides)",
            "external_peptide_score_mixture_correct_discovery_pct",
            True,
        ),
    ]
    for dataset, dataset_results in results.groupby("dataset", sort=False):
        for metric, ylabel, base_name, percent_axis in specs:
            _plot_metric_by_pi0(
                dataset_results,
                dataset_label=_display_name(str(dataset)),
                dataset_slug=str(dataset).replace("/", "_"),
                metric=metric,
                ylabel=ylabel,
                base_name=base_name,
                percent_axis=percent_axis,
                output_dir=output_dir,
            )


def write_holdout_summary_tables(
    results: pd.DataFrame,
    output_dir: Path,
    *,
    thresholds: list[float] | None = None,
) -> tuple[Path, Path]:
    """Aggregate raw mixture rows into acceptance and error/gain CSVs."""
    acceptance, error_gain = summarise_holdout_results(
        results,
        thresholds=thresholds if thresholds is not None else SUMMARY_THRESHOLDS,
        group_extra=("pi0_target",),
    )
    return write_summary_tables(
        acceptance, error_gain, output_dir, "external_peptide_holdout"
    )


@app.command()
def main(
    novoboard_root: Annotated[
        Path,
        typer.Option(
            "--novoboard-root",
            help=(
                "Root of NovoBoard per-dataset tables: "
                "{root}/{dataset}/novoboard/ with annotated_test*.csv and "
                "raw_unlabelled*.csv target/decoy pairs (the datasets/ dir of "
                "a NovoBoard checkout). Local runs used fork "
                "JemmaLDaniel/NovoBoard, branch feat/adapt-to-instanovo "
                "(commit a9faab3ef1af06987599c2f01e6ba96072c80172)."
            ),
        ),
    ],
    glissade_repo: Annotated[
        Path,
        typer.Option(
            "--glissade-repo",
            help=(
                "Glissade clone root (must import as glissade.glissade). Local "
                "runs used fork JemmaLDaniel/glissade, branch winnow-benchmark "
                "(commit 6ee11b51b5f21ba8fdc1eb5821608352b082a533)."
            ),
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for benchmark outputs."),
    ] = DEFAULT_OUTPUT_DIR,
    datasets: Annotated[
        Optional[list[str]],
        typer.Option("--datasets", help="Dataset keys to benchmark."),
    ] = None,
    pi0_grid: Annotated[
        Optional[list[float]],
        typer.Option("--pi0-grid", help="Target mixture null fractions."),
    ] = None,
    holdout_frac: Annotated[
        float,
        typer.Option(
            "--holdout-frac",
            help=(
                "Maximum fraction of shared matched peptides used as S_c "
                "(default 1 = prefer the full pool). |S_c| is further capped so "
                "every π₀ in --pi0-grid fits in S_e without replacement."
            ),
        ),
    ] = DEFAULT_HOLDOUT_FRAC,
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
    winnow_results: Annotated[
        Path,
        typer.Option("--winnow-results", help="Winnow results directory."),
    ] = DEFAULT_WINNOW_RESULTS,
    model_root: Annotated[
        Path,
        typer.Option(
            "--model-root",
            help="Per-dataset calibrator directories, used for Glissade's anchor.",
        ),
    ] = DEFAULT_MODEL_ROOT,
    n_bootstraps: Annotated[
        int,
        typer.Option(
            "--n-bootstraps", help="Glissade bootstraps per mixture iteration."
        ),
    ] = DEFAULT_N_BOOTSTRAPS,
    min_peptide_length: Annotated[
        int,
        typer.Option(
            "--min-peptide-length",
            help=(
                "Minimum normalised peptide length for unlabelled / external "
                "pools. Labelled matched pools and Glissade's training reference "
                "use the labelled floor (non-empty key only)."
            ),
        ),
    ] = MIN_PEPTIDE_LENGTH,
    plot: Annotated[bool, typer.Option(help="Create summary plots.")] = True,
    summarise_only: Annotated[
        Optional[Path],
        typer.Option(
            "--summarise-only",
            help="Only write summary CSVs/plots from an existing results CSV.",
        ),
    ] = None,
) -> None:
    """Run the controlled-π₀ external peptide score-mixture benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)

    if summarise_only is not None:
        results = pd.read_csv(summarise_only)
        write_holdout_summary_tables(results, output_dir)
        if plot:
            plot_benchmark_results(results, output_dir / "plots")
        return

    dataset_keys = datasets if datasets is not None else DEFAULT_DATASETS
    pi0s = pi0_grid if pi0_grid is not None else list(DEFAULT_PI0_GRID)
    thresholds = (
        q_thresholds if q_thresholds is not None else list(DEFAULT_Q_THRESHOLDS)
    )

    rows: list[dict[str, object]] = []
    for dataset in dataset_keys:
        logger.info("Building shared score tables for %s", _display_name(dataset))
        matched, external, nb_decoy, glissade_reference = build_shared_score_tables(
            dataset=dataset,
            winnow_results=winnow_results,
            novoboard_root=novoboard_root,
            model_root=model_root,
            unlabelled_min_length=min_peptide_length,
            labelled_min_length=LABELLED_MIN_PEPTIDE_LENGTH,
        )
        rows.extend(
            evaluate_controlled_mixtures(
                dataset=dataset,
                matched=matched,
                external=external,
                nb_decoy=nb_decoy,
                glissade_reference=glissade_reference,
                pi0_grid=pi0s,
                holdout_frac=holdout_frac,
                seed=seed,
                n_iterations=n_iterations,
                thresholds=thresholds,
                glissade_repo=glissade_repo,
                n_bootstraps=n_bootstraps,
            )
        )

    results = pd.DataFrame(rows)
    results_path = output_dir / "external_peptide_holdout_results.csv"
    results.to_csv(results_path, index=False)
    logger.info("Wrote %s (%d rows)", results_path, len(results))
    write_holdout_summary_tables(results, output_dir)
    if plot:
        plot_benchmark_results(results, output_dir / "plots")


if __name__ == "__main__":
    app()
