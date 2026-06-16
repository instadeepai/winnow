"""MZTab dataset loader for database search engines and Casanovo outputs.

Spectrum-prediction linking
~~~~~~~~~~~~~~~~~~~~~~~~~~~
PSM rows are joined to spectrum parquet/ipc rows on a numeric ``index`` extracted
from ``spectra_ref`` (``ms_run[k]:index=N``). ``PSM_ID`` is not used for grouping.
Spectrum row ``N`` is assumed to correspond to ``index=N`` in the mzTab file.

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
from winnow.datasets.interfaces import DatasetLoader


class MZTabDatasetLoader(DatasetLoader):
    """Load calibration data from mzTab PSM tables and spectrum files."""

    _LOG_PROB_EPSILON = 1e-10
    _CASANOVO_CV = "MS:1003281"
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
    def _casanovo_score_to_probability(score: float) -> float:
        """Recover a probability from a Casanovo PSM score.

        Scores below zero indicate a precursor mass mismatch penalty (original
        probability minus one). Epsilon clamp avoids ``log(0)`` downstream.
        """
        prob = score + 1.0 if score < 0 else score
        return max(float(prob), MZTabDatasetLoader._LOG_PROB_EPSILON)

    @staticmethod
    def _casanovo_raw_score_to_log_probability(score: float) -> float:
        """Convert a native Casanovo PSM score to a log-probability."""
        return float(np.log(MZTabDatasetLoader._casanovo_score_to_probability(score)))

    @staticmethod
    def _token_probability_to_log_probability(score: float) -> float:
        """Convert a per-residue Casanovo aa_score (probability) to log-probability."""
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
    def _load_spectrum_data(
        spectrum_path: Path | str,
    ) -> Tuple[pl.DataFrame, bool]:
        """Load spectrum data from Parquet or IPC."""
        spectrum_path = Path(spectrum_path)
        if spectrum_path.suffix == ".parquet":
            df = pl.read_parquet(spectrum_path)
        elif spectrum_path.suffix == ".ipc":
            df = pl.read_ipc(spectrum_path)
        else:
            raise ValueError(
                f"Unsupported file format for spectrum data: {spectrum_path.suffix}. "
                "Supported formats are .parquet and .ipc."
            )
        return df, "sequence" in df.columns

    @staticmethod
    def _validate_spectra_ref_indices(predictions: pl.DataFrame) -> pl.DataFrame:
        """Extract spectrum index from spectra_ref and raise if any row fails to parse."""
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
        return predictions

    def load(
        self, *, data_path: Path, predictions_path: Optional[Path] = None, **kwargs: Any
    ) -> CalibrationDataset:
        """Load a calibration dataset from mzTab predictions and spectrum data."""
        if predictions_path is None:
            raise ValueError("predictions_path is required for MZTabDatasetLoader")

        spectrum_data, has_labels = self._load_spectrum_data(data_path)
        spectrum_data = self._process_spectrum_data(spectrum_data, has_labels)

        raw_predictions = self._load_dataset(predictions_path)
        is_casanovo = self._is_casanovo_mztab(raw_predictions)

        self._validate_load_beams_supported(is_casanovo, self.load_beams)

        predictions = self._process_predictions(
            raw_predictions, spectrum_data.columns, is_casanovo
        )
        predictions = self._tokenize(
            predictions, "prediction_untokenised", "prediction"
        )

        top_predictions = self._get_top_predictions(predictions, is_casanovo)
        metadata = self._merge_data(spectrum_data, top_predictions)
        metadata = self._evaluate_predictions(metadata, has_labels)

        metadata_pd = metadata.to_pandas()
        metadata_pd["prediction"] = metadata_pd["prediction"].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

        beam_predictions: Optional[List[Optional[List[ScoredSequence]]]] = None
        if is_casanovo and self.load_beams:
            ordered_indices = metadata.get_column("index").to_list()
            beam_predictions = self._create_casanovo_beam_predictions(
                predictions, ordered_indices
            )

        return CalibrationDataset(metadata=metadata_pd, predictions=beam_predictions)

    def _process_predictions(
        self,
        predictions: pl.DataFrame,
        spectrum_data_columns: List[str],
        is_casanovo: bool,
    ) -> pl.DataFrame:
        """Parse mzTab columns, extract spectrum index, and sort by native score."""
        self._require_column(predictions, "spectra_ref", "spectra_ref")
        predictions_col = self.column_mapping["predictions"]
        confidence_col = self.column_mapping["confidence"]
        token_scores_col = self.column_mapping.get("token_scores")

        self._require_column(predictions, "predictions", predictions_col)
        self._require_column(predictions, "confidence", confidence_col)

        predictions = self._validate_spectra_ref_indices(predictions)

        columns_to_add = [
            pl.col(predictions_col)
            .str.replace("L", "I")
            .alias("prediction_untokenised"),
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
        if has_token_col and token_scores_col is not None:
            drop_cols.add(token_scores_col)

        predictions = predictions.with_columns(columns_to_add).drop(
            [c for c in drop_cols if c in predictions.columns]
        )

        # Native score descending within each spectrum; negatives sink to the bottom.
        predictions = predictions.sort(
            ["index", "confidence"], descending=[False, True]
        )

        return predictions.drop(
            [
                col
                for col in predictions.columns
                if col in spectrum_data_columns and col != "index"
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

    def _tokenize(
        self,
        predictions: pl.DataFrame,
        untokenised_column: str,
        tokenised_column: str,
    ) -> pl.DataFrame:
        """Tokenize peptide strings and remap modifications to ProForma."""
        return predictions.with_columns(
            pl.col(untokenised_column)
            .map_elements(self.metrics._split_peptide, return_dtype=pl.List(pl.Utf8))
            .alias(tokenised_column)
        ).with_columns(
            pl.col(tokenised_column)
            .map_elements(self._remap_tokens, return_dtype=pl.List(pl.Utf8))
            .alias(tokenised_column)
        )

    def _remap_tokens(self, tokens: list[str]) -> list[str]:
        """Remap tokens using residue_remapping."""
        return [
            self.metrics.residue_set.residue_remapping.get(token, token)
            for token in tokens
        ]

    def _process_spectrum_data(
        self, spectrum_data: pl.DataFrame, has_labels: bool
    ) -> pl.DataFrame:
        """Tokenize ground-truth sequences when present."""
        if has_labels:
            spectrum_data = spectrum_data.with_columns(
                pl.col("sequence").str.replace("L", "I").alias("sequence_untokenised")
            )
            spectrum_data = self._tokenize(
                spectrum_data, "sequence_untokenised", "sequence"
            )
        return spectrum_data

    def _merge_data(
        self, spectrum_data: pl.DataFrame, predictions: pl.DataFrame
    ) -> pl.DataFrame:
        """Inner-join spectrum rows to top PSMs on zero-based row index."""
        return (
            spectrum_data.with_row_index("index")
            .join(predictions, on="index", how="inner")
            .sort("index")
        )

    def _evaluate_predictions(
        self, metadata: pl.DataFrame, has_labels: bool
    ) -> pl.DataFrame:
        """Annotate merged metadata with prediction validity and ground-truth matches."""
        metadata = metadata.with_columns(
            pl.col("prediction")
            .map_elements(lambda x: isinstance(x, pl.Series), return_dtype=pl.Boolean)
            .alias("valid_prediction"),
        )

        if not has_labels:
            return metadata

        metadata = (
            metadata.with_columns(
                pl.col("sequence")
                .map_elements(
                    lambda x: isinstance(x, pl.Series), return_dtype=pl.Boolean
                )
                .alias("valid_sequence"),
            )
            .with_columns(
                pl.struct(["sequence", "prediction"])
                .map_elements(
                    lambda row: (
                        self.metrics._novor_match(row["sequence"], row["prediction"])
                        if isinstance(row["sequence"], list)
                        and isinstance(row["prediction"], list)
                        else 0
                    ),
                    return_dtype=pl.Int64,
                )
                .alias("num_matches"),
            )
            .with_columns(
                pl.struct(["sequence", "prediction", "num_matches"])
                .map_elements(
                    lambda row: (
                        row["num_matches"]
                        == len(row["sequence"])
                        == len(row["prediction"])
                        if isinstance(row["sequence"], list)
                        and isinstance(row["prediction"], list)
                        else False
                    ),
                    return_dtype=pl.Boolean,
                )
                .alias("correct")
            )
        )
        return metadata
