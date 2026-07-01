"""MZTab dataset loader for database search engines and Casanovo outputs.

Spectrum-prediction linking
~~~~~~~~~~~~~~~~~~~~~~~~~~~
PSM rows are joined to spectrum parquet/ipc/MGF rows on ``spectrum_id`` in
``{experiment_name}:{index}`` form. ``experiment_name`` is the spectrum file
stem; ``index`` is parsed from ``spectra_ref`` (``ms_run[k]:index=N``).
Spectrum row ``N`` in file order must correspond to ``index=N`` in the mzTab file.

Candidate ranking
~~~~~~~~~~~~~~~~~
All PSM rows sharing an ``index`` are sorted by raw engine score descending
(negatives last for Casanovo de novo). Metadata always receives the top row per
``index``. Casanovo beams (when ``load_beams=True``) include every row per
``index`` in that same order. Scores are transformed to log-probabilities only
after sorting.

Modes
~~~~~
* **Casanovo** — detected via ``search_engine`` (``MS:1003281`` / ``Casanovo``):
  recovered probabilities for metadata confidence; beams with token scores.
* **Database search** — raw engine scores in metadata; ``predictions`` is always
  ``None``. ``load_beams=True`` raises at load time.

Score assumptions (Casanovo)
~~~~~~~~~~~~~~~~~~~~~~~~~~
* **PSM score** (``search_engine_score[1]``): native Casanovo probability in
  ``[-1, 1]``. Values below zero encode a precursor mass-mismatch penalty
  (probability minus one). Values outside this range raise at load time.
* **Token aa_scores**: per-residue **probabilities** in ``[0, 1]``, not
  log-probabilities. Values outside this range raise at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import polars as pl
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet
from pyteomics import mztab

from winnow.compat.instanovo import ScoredSequence
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders import utils
from winnow.datasets.interfaces import DatasetLoader


class MZTabDatasetLoader(DatasetLoader):
    """Load calibration data from mzTab PSM tables and spectrum files."""

    _LOG_PROB_EPSILON = 1e-10
    _CASANOVO_CV = "MS:1003281"
    _CASANOVO_PSM_SCORE_MIN = -1.0
    _CASANOVO_PSM_SCORE_MAX = 1.0
    _CASANOVO_TOKEN_PROB_MIN = 0.0
    _CASANOVO_TOKEN_PROB_MAX = 1.0
    _DEFAULT_COLUMN_MAPPING: dict[str, Optional[str]] = {
        "predictions": "opt_ms_run[1]_proforma",
        "confidence": "search_engine_score[1]",
        "token_scores": "opt_ms_run[1]_aa_scores",
    }

    def __init__(
        self,
        residue_masses: dict[str, float],
        residue_remapping: dict[str, str],
        isotope_error_range: Tuple[int, int] = (0, 1),
        load_beams: bool = True,
        column_mapping: Optional[dict[str, Optional[str]]] = None,
    ) -> None:
        """Initialise the MZTabDatasetLoader.

        Args:
            residue_masses: Residue masses in ProForma notation.
            residue_remapping: Input notation to ProForma mapping.
            isotope_error_range: Isotope error range for mass matching.
            load_beams: Build ``ScoredSequence`` beam lists for Casanovo mzTab.
                Must be ``False`` for traditional database-search mzTab (raises if
                ``True``). When ``False`` on Casanovo, metadata is still loaded.
            column_mapping: Maps logical roles to mzTab column headers. See
                module docstring and ``_DEFAULT_COLUMN_MAPPING``. Missing mapped
                columns fail fast with available headers listed.
        """
        self.metrics = Metrics(
            residue_set=ResidueSet(
                residue_masses=residue_masses, residue_remapping=residue_remapping
            ),
            isotope_error_range=isotope_error_range,
        )
        self.load_beams = load_beams
        self.column_mapping = {
            **self._DEFAULT_COLUMN_MAPPING,
            **(column_mapping or {}),
        }

    @staticmethod
    def _is_casanovo_mztab(predictions: pl.DataFrame) -> bool:
        """Return True if the PSM table appears to be Casanovo output."""
        if "search_engine" not in predictions.columns:
            return False
        for value in predictions["search_engine"]:
            if value is None:
                continue
            text = str(value)
            if MZTabDatasetLoader._CASANOVO_CV in text or "Casanovo" in text:
                return True
        return False

    @staticmethod
    def _validate_casanovo_native_psm_score(score: float) -> None:
        """Raise if a native Casanovo PSM score is outside ``[-1, 1]``.

        Casanovo ``search_engine_score[1]`` values are assumed to be probabilities,
        with negatives indicating a mass-mismatch penalty (probability minus one).
        Log-probability or other encodings are not supported.
        """
        if (
            not MZTabDatasetLoader._CASANOVO_PSM_SCORE_MIN
            <= score
            <= (MZTabDatasetLoader._CASANOVO_PSM_SCORE_MAX)
        ):
            raise ValueError(
                "Casanovo PSM score must be in [-1, 1] (native probability encoding). "
                f"Got {score}. Values below 0 indicate a mass-mismatch penalty "
                "(probability minus one); log-probability scores are not supported."
            )

    @staticmethod
    def _validate_casanovo_token_probability(score: float) -> None:
        """Raise if a per-residue aa_score is outside ``[0, 1]``.

        Token ``aa_scores`` are assumed to be probabilities, not log-probabilities.
        """
        if (
            not MZTabDatasetLoader._CASANOVO_TOKEN_PROB_MIN
            <= score
            <= (MZTabDatasetLoader._CASANOVO_TOKEN_PROB_MAX)
        ):
            raise ValueError(
                "Casanovo per-residue aa_scores must be probabilities in [0, 1]. "
                f"Got {score}. Log-probability token scores are not supported."
            )

    @staticmethod
    def _validate_casanovo_prediction_scores(predictions: pl.DataFrame) -> None:
        """Validate Casanovo PSM and token score ranges on a processed predictions frame."""
        for confidence in predictions["confidence"]:
            if confidence is not None:
                MZTabDatasetLoader._validate_casanovo_native_psm_score(
                    float(confidence)
                )

        if "token_scores" not in predictions.columns:
            return

        for token_scores in predictions["token_scores"]:
            if token_scores is None:
                continue
            for score in token_scores:
                if score is not None:
                    MZTabDatasetLoader._validate_casanovo_token_probability(
                        float(score)
                    )

    @staticmethod
    def _casanovo_score_to_probability(score: float) -> float:
        """Recover a probability from a Casanovo PSM score.

        Expects a native Casanovo PSM score in ``[-1, 1]`` (see module docstring).
        Scores below zero indicate a precursor mass mismatch penalty (original
        probability minus one). Epsilon clamp avoids ``log(0)`` downstream.
        """
        MZTabDatasetLoader._validate_casanovo_native_psm_score(score)
        prob = score + 1.0 if score < 0 else score
        return max(float(prob), MZTabDatasetLoader._LOG_PROB_EPSILON)

    @staticmethod
    def _casanovo_raw_score_to_log_probability(score: float) -> float:
        """Convert a native Casanovo PSM score in ``[-1, 1]`` to a log-probability."""
        return float(np.log(MZTabDatasetLoader._casanovo_score_to_probability(score)))

    @staticmethod
    def _token_probability_to_log_probability(score: float) -> float:
        """Convert a per-residue Casanovo aa_score (probability in ``[0, 1]``) to log-probability."""
        MZTabDatasetLoader._validate_casanovo_token_probability(score)
        return float(np.log(max(float(score), MZTabDatasetLoader._LOG_PROB_EPSILON)))

    @staticmethod
    def _validate_load_beams_supported(is_casanovo: bool, load_beams: bool) -> None:
        """Raise if beam loading is requested for a non-Casanovo mzTab file."""
        if not is_casanovo and load_beams:
            raise ValueError(
                "load_beams=True is only supported for Casanovo mzTab files (search_engine "
                "contains MS:1003281 or Casanovo). Traditional database-search mzTab does not "
                "provide beam candidates; set load_beams=false in mztab.yaml."
            )

    @staticmethod
    def _load_dataset(predictions_path: Path | str) -> pl.DataFrame:
        """Load the PSM table from an mzTab file."""
        predictions_path = Path(predictions_path)
        if predictions_path.suffix != ".mztab":
            raise ValueError(
                f"Unsupported file format for MZTab predictions: {predictions_path.suffix}. "
                "Supported format is .mztab."
            )
        predictions = mztab.MzTab(str(predictions_path)).spectrum_match_table
        return pl.DataFrame(predictions)

    @staticmethod
    def _require_column(
        df: pl.DataFrame, logical_name: str, column_name: str | None
    ) -> None:
        """Require that a configured mzTab column exists in the dataframe.

        Raises ValueError naming the logical role, configured column name, and all
        available columns. There is no silent fallback to alternate column names.
        """
        if column_name is None:
            raise ValueError(
                f"Column mapping for '{logical_name}' is not configured. "
                f"Available columns: {sorted(df.columns)}"
            )
        if column_name not in df.columns:
            raise ValueError(
                f"Required mzTab column '{column_name}' (logical role: '{logical_name}') "
                f"not found. Available columns: {sorted(df.columns)}"
            )

    @staticmethod
    def _parse_spectra_ref(
        predictions: pl.DataFrame, experiment_name: str
    ) -> pl.DataFrame:
        """Parse ``spectra_ref`` into integer ``index`` and ``spectrum_id`` columns."""
        predictions = predictions.with_columns(
            pl.col("spectra_ref")
            .str.extract(r"index=(\d+)")
            .cast(pl.Int64)
            .alias("index")
        )
        invalid = predictions.filter(pl.col("index").is_null())
        if len(invalid) > 0:
            bad_refs = invalid["spectra_ref"].to_list()[:5]
            raise ValueError(
                "Could not parse spectrum index from spectra_ref (expected "
                "'ms_run[k]:index=N'). Examples of invalid values: "
                f"{bad_refs}"
            )
        return predictions.with_columns(
            (pl.lit(experiment_name) + ":" + pl.col("index").cast(pl.Utf8)).alias(
                "spectrum_id"
            )
        )

    @staticmethod
    def _validate_join_keys(
        spectrum_data: pl.DataFrame, predictions: pl.DataFrame
    ) -> None:
        """Validate normalized spectrum_id columns before joining."""
        for name, df in (
            ("spectrum data", spectrum_data),
            ("predictions", predictions),
        ):
            if "spectrum_id" not in df.columns:
                raise ValueError(f"{name} missing required 'spectrum_id' column.")

        if spectrum_data["spectrum_id"].n_unique() != len(spectrum_data):
            raise ValueError("Spectrum data 'spectrum_id' values must be unique.")

        missing = (
            predictions.select("spectrum_id")
            .unique()
            .join(
                spectrum_data.select("spectrum_id").unique(),
                on="spectrum_id",
                how="anti",
            )
        )
        if len(missing) > 0:
            examples = missing["spectrum_id"].head(5).to_list()
            raise ValueError(
                "Predictions reference spectrum_id values not present in spectrum data: "
                f"{examples}"
            )

    def load(
        self, *, data_path: Path, predictions_path: Optional[Path] = None, **kwargs: Any
    ) -> CalibrationDataset:
        """Load a calibration dataset from mzTab predictions and spectrum data."""
        if predictions_path is None:
            raise ValueError("predictions_path is required for MZTabDatasetLoader")

        experiment_name = Path(data_path).stem
        spectrum_data, has_labels = self._load_spectrum_data(data_path)
        spectrum_data = self._process_spectrum_data(spectrum_data)

        raw_predictions = self._load_dataset(predictions_path)
        is_casanovo = self._is_casanovo_mztab(raw_predictions)

        self._validate_load_beams_supported(is_casanovo, self.load_beams)

        predictions = self._process_predictions(
            raw_predictions,
            spectrum_data.columns,
            is_casanovo,
            experiment_name,
        )
        predictions = predictions.with_columns(
            pl.col("prediction_untokenised").alias("prediction")
        )
        residue_remapping = self.metrics.residue_set.residue_remapping
        predictions = utils.finalize_peptide_metadata(
            predictions,
            self.metrics,
            has_labels=False,
            residue_remapping=residue_remapping,
        )

        top_predictions = self._get_top_predictions(predictions, is_casanovo)
        metadata = self._merge_data(spectrum_data, top_predictions)
        metadata = utils.finalize_peptide_metadata(
            metadata,
            self.metrics,
            has_labels=has_labels,
            residue_remapping=residue_remapping,
        )

        metadata_pd = metadata.to_pandas()

        beam_predictions: Optional[List[Optional[List[ScoredSequence]]]] = None
        if is_casanovo and self.load_beams:
            ordered_indices = metadata.get_column("index").to_list()
            beam_predictions = self._create_casanovo_beam_predictions(
                predictions, ordered_indices
            )

        return CalibrationDataset(metadata=metadata_pd, predictions=beam_predictions)

    def _load_spectrum_data(
        self, spectrum_path: Path | str
    ) -> Tuple[pl.DataFrame, bool]:
        """Load spectrum data from a Parquet, IPC, or MGF file.

        Args:
            spectrum_path: Path to spectrum data file (.parquet, .ipc, or .mgf).

        Returns:
            Tuple of (DataFrame containing spectrum data, whether ground truth labels exist).
        """
        spectrum_path = Path(spectrum_path)
        df, has_labels = utils.load_spectrum_data(spectrum_path, add_index_cols=False)
        df = utils.add_row_order_spectrum_ids(df, spectrum_path.stem)
        return df, has_labels

    def _process_predictions(
        self,
        predictions: pl.DataFrame,
        spectrum_data_columns: List[str],
        is_casanovo: bool,
        experiment_name: str,
    ) -> pl.DataFrame:
        """Parse mzTab columns, extract spectrum index, and sort by native score.

        For Casanovo mzTab, validates that PSM scores lie in ``[-1, 1]`` and token
        aa_scores in ``[0, 1]`` (see module docstring).
        """
        self._require_column(predictions, "spectra_ref", "spectra_ref")
        predictions_col = self.column_mapping["predictions"]
        confidence_col = self.column_mapping["confidence"]
        token_scores_col = self.column_mapping.get("token_scores")

        self._require_column(predictions, "predictions", predictions_col)
        self._require_column(predictions, "confidence", confidence_col)

        predictions = self._parse_spectra_ref(predictions, experiment_name)

        columns_to_add = [
            pl.col(predictions_col).alias("prediction_untokenised"),
            pl.col(confidence_col).cast(pl.Float64).alias("confidence"),
        ]

        has_token_col = (
            token_scores_col is not None and token_scores_col in predictions.columns
        )
        if has_token_col:
            columns_to_add.append(
                pl.col(token_scores_col)
                .str.split(",")
                .cast(pl.List(pl.Float64))
                .alias("token_scores")
            )
        else:
            columns_to_add.append(
                pl.lit(None, dtype=pl.List(pl.Float64)).alias("token_scores")
            )

        drop_cols = {predictions_col, confidence_col, "spectra_ref"}
        if predictions_col != "sequence":
            drop_cols.add("sequence")
        if has_token_col and token_scores_col is not None:
            drop_cols.add(token_scores_col)

        predictions = predictions.with_columns(columns_to_add).drop(
            [c for c in drop_cols if c in predictions.columns]
        )

        if is_casanovo:
            self._validate_casanovo_prediction_scores(predictions)

        # Native score descending within each spectrum; negatives sink to the bottom.
        predictions = predictions.sort(
            ["index", "confidence"], descending=[False, True]
        )

        return predictions.drop(
            [
                col
                for col in predictions.columns
                if col in spectrum_data_columns and col not in {"index", "spectrum_id"}
            ]
        )

    def _get_top_predictions(
        self, predictions: pl.DataFrame, is_casanovo: bool
    ) -> pl.DataFrame:
        """First row per spectrum after native-score sort (metadata path)."""
        top = predictions.filter(pl.int_range(0, pl.len()).over("index") == 0)
        if is_casanovo:
            top = top.with_columns(
                pl.col("confidence")
                .map_elements(
                    self._casanovo_score_to_probability, return_dtype=pl.Float64
                )
                .alias("confidence")
            )
        return top

    def _create_casanovo_beam_predictions(
        self, predictions: pl.DataFrame, valid_spectra_indices: List[int]
    ) -> List[Optional[List[ScoredSequence]]]:
        """Build beam lists in native-score order with post-sort log-prob transform."""
        beam_predictions: List[Optional[List[ScoredSequence]]] = []
        for spectrum_index in valid_spectra_indices:
            spectrum_preds = predictions.filter(pl.col("index") == spectrum_index)
            scored_sequences: List[ScoredSequence] = []
            for row in spectrum_preds.iter_rows(named=True):
                token_scores = row["token_scores"]
                token_log_probs = None
                if token_scores is not None:
                    token_log_probs = [
                        self._token_probability_to_log_probability(float(s))
                        for s in token_scores
                    ]
                scored_sequences.append(
                    ScoredSequence(
                        sequence=row["prediction"],
                        mass_error=None,
                        sequence_log_probability=self._casanovo_raw_score_to_log_probability(
                            float(row["confidence"])
                        ),
                        token_log_probabilities=token_log_probs,
                    )
                )
            beam_predictions.append(scored_sequences or None)
        return beam_predictions

    def _process_spectrum_data(self, spectrum_data: pl.DataFrame) -> pl.DataFrame:
        """Return spectrum data unchanged; tokenization happens in finalize."""
        return spectrum_data

    def _merge_data(
        self, spectrum_data: pl.DataFrame, predictions: pl.DataFrame
    ) -> pl.DataFrame:
        """Inner-join spectrum rows to top PSMs on spectrum_id."""
        if "index" in spectrum_data.columns:
            spectrum_data = spectrum_data.drop("index")
        self._validate_join_keys(spectrum_data, predictions)
        return spectrum_data.join(predictions, on="spectrum_id", how="inner").sort(
            "index"
        )
