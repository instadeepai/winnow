"""PrimeNovo TSV + MGF dataset loader.

Spectrum-prediction linking
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Spectra are loaded from ``.mgf`` with matchms. Each ion's matchms ``title`` comes
from the MGF ``TITLE`` field and may be any non-empty string; PrimeNovo copies
that value into the TSV ``label`` column. Within one MGF file, ``TITLE`` values
must be unique.

Both sides receive the same ``spectrum_id`` form before joining:

* MGF: ``{experiment_name}:{title}``
* TSV: ``{experiment_name}:{label}``

where ``experiment_name`` is the spectrum path stem (``Path(data_path).stem``).
``TITLE`` values must be unique within one MGF file, and TSV ``label`` values
must be unique within the predictions file. The tables are then inner-joined on
``spectrum_id`` (InstaNovo standard): spectra without a prediction are dropped;
predictions whose ``label`` does not match any spectrum ``TITLE`` raise.

Score assumptions
~~~~~~~~~~~~~~~~~
PrimeNovo ``score`` values are probabilities in ``[0, 1]`` and are stored as
``confidence`` without transformation. Values outside this range raise at load
time.

Mid-sequence N-terminal modification filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``mid_sequence_n_terminal_mods`` lists substrings matched against the **raw**
TSV ``prediction`` string **before** tokenisation and ``residue_remapping``.
Supply tokens in PrimeNovo compact notation as written in the TSV (e.g.
``[+42.011]``), not remapped UNIMOD / ProForma forms (e.g. ``[UNIMOD:1]``).
Rows whose prediction contains any listed mod after residue position 0 are
dropped.

Beams
~~~~~
PrimeNovo does not emit beam predictions, so the returned
:class:`~winnow.datasets.calibration_dataset.CalibrationDataset` always has
``predictions=None``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import polars as pl
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders import utils as data_utils
from winnow.datasets.interfaces import DatasetLoader

logger = logging.getLogger(__name__)


class PrimeNovoDatasetLoader(DatasetLoader):
    """Loader for PrimeNovo predictions in TSV format.

    PrimeNovo writes predictions to a tab-separated file with fixed columns
    ``label``, ``prediction``, ``charge``, ``score``. Spectra (``.mgf``) are loaded
    with matchms. ``spectrum_id`` is ``{experiment_name}:{title}`` on the MGF side and
    ``{experiment_name}:{label}`` on the TSV side (PrimeNovo sets TSV ``label`` from the
    MGF ``TITLE``). Titles and labels may be any non-empty string but must each be
    unique within their file. Tables are inner-joined on ``spectrum_id``.

    Score values are already probabilities in ``[0, 1]`` and are stored as
    ``confidence`` without transformation. Beam predictions are not available from
    PrimeNovo, so the returned :class:`CalibrationDataset` always has
    ``predictions=None``.

    ``mid_sequence_n_terminal_mods`` are matched as substrings of the raw TSV
    ``prediction`` before remapping; use PrimeNovo compact notation, not UNIMOD.
    """

    # Score encoding: PrimeNovo ``score`` is a probability
    _SCORE_MIN = 0.0
    _SCORE_MAX = 1.0

    def __init__(
        self,
        residue_masses: dict[str, float],
        mid_sequence_n_terminal_mods: list[str],
        residue_remapping: dict[str, str],
        isotope_error_range: Tuple[int, int] = (0, 1),
    ) -> None:
        """Initialise the PrimeNovoDatasetLoader.

        Args:
            residue_masses: The mapping of residues to their masses (ProForma notation).
            mid_sequence_n_terminal_mods: N-terminal modification tokens that are
                invalid if they appear after residue position 0. Matched as
                substrings of the raw TSV ``prediction`` **before**
                ``residue_remapping``; use PrimeNovo compact notation (e.g.
                ``[+42.011]``), not remapped UNIMOD forms. Matching rows are dropped.
            residue_remapping: Optional ProForma mapping from PrimeNovo tokens.
            isotope_error_range: The range of isotope errors to consider when matching
                peptides.
        """
        self.metrics = Metrics(
            residue_set=ResidueSet(
                residue_masses=residue_masses, residue_remapping=residue_remapping
            ),
            isotope_error_range=isotope_error_range,
        )
        self.mid_sequence_n_terminal_mods = tuple(mid_sequence_n_terminal_mods)

    @staticmethod
    def _validate_join_keys(
        spectrum_data: pl.DataFrame, predictions: pl.DataFrame
    ) -> None:
        """Validate normalised ``spectrum_id`` columns before joining."""
        for name, df in (
            ("spectrum data", spectrum_data),
            ("predictions", predictions),
        ):
            if "spectrum_id" not in df.columns:
                raise ValueError(f"{name} missing required 'spectrum_id' column.")

        if spectrum_data["spectrum_id"].n_unique() != len(spectrum_data):
            raise ValueError("Spectrum data 'spectrum_id' values must be unique.")

        if predictions["spectrum_id"].n_unique() != len(predictions):
            duplicates = (
                predictions.group_by("spectrum_id")
                .len()
                .filter(pl.col("len") > 1)
                .get_column("spectrum_id")
                .head(5)
                .to_list()
            )
            raise ValueError(
                "Prediction 'spectrum_id' values must be unique (duplicate TSV "
                f"label values). Duplicates: {duplicates}"
            )

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

    @classmethod
    def _validate_primenovo_score(cls, score: float) -> None:
        """Raise if a PrimeNovo score is outside ``[0, 1]``."""
        if not cls._SCORE_MIN <= score <= cls._SCORE_MAX:
            raise ValueError(
                f"PrimeNovo scores must be probabilities in [0, 1]. Got {score}."
            )

    def load(
        self, *, data_path: Path, predictions_path: Optional[Path] = None, **kwargs: Any
    ) -> CalibrationDataset:
        """Load a CalibrationDataset from PrimeNovo TSV predictions and MGF spectra.

        Args:
            data_path: Path to the spectrum data file (``.mgf`` only).
            predictions_path: Path to the PrimeNovo predictions TSV file.
            **kwargs: Not used.

        Returns:
            CalibrationDataset: Dataset containing merged metadata. ``predictions`` is
                always ``None`` because PrimeNovo does not produce beams.

        Raises:
            ValueError: If ``predictions_path`` is None, required TSV columns are
                missing, scores are outside ``[0, 1]``, MGF ``TITLE`` or TSV
                ``label`` values are missing or not unique, or predictions
                reference labels with no matching spectrum title.
        """
        if predictions_path is None:
            raise ValueError("predictions_path is required for PrimeNovoDatasetLoader")

        experiment_name = Path(data_path).stem
        spectrum_data, has_labels = self._load_spectrum_data(data_path)

        raw_predictions = self._load_predictions(predictions_path)
        predictions = self._process_predictions(
            raw_predictions,
            spectrum_data.columns,
            experiment_name,
        )
        metadata = self._merge_data(spectrum_data, predictions)

        residue_remapping = self.metrics.residue_set.residue_remapping
        metadata = data_utils.finalize_peptide_metadata(
            metadata,
            self.metrics,
            has_labels=has_labels,
            residue_remapping=residue_remapping,
        )

        metadata_pd = metadata.to_pandas()
        # Polars List columns become numpy arrays under to_pandas(); CalibrationDataset
        # and downstream features expect Python token lists (InstaNovo convention).
        for column in ("prediction", "sequence"):
            if column not in metadata_pd.columns:
                continue
            metadata_pd[column] = metadata_pd[column].apply(
                lambda value: (
                    value if isinstance(value, list) or value is None else list(value)
                )
            )

        return CalibrationDataset(metadata=metadata_pd, predictions=None)

    def _add_prediction_spectrum_ids(
        self, predictions: pl.DataFrame, experiment_name: str
    ) -> pl.DataFrame:
        """Set prediction ``spectrum_id`` to ``{experiment_name}:{label}``.

        PrimeNovo sets TSV ``label`` from the MGF ``TITLE``, so ids match
        ``{experiment_name}:{title}`` on the spectrum side.
        """
        predictions = predictions.with_columns(
            pl.col("label")
            .map_elements(
                lambda x: (
                    str(x).strip()
                    if x is not None and str(x).strip() != "" and str(x) != "nan"
                    else None
                ),
                return_dtype=pl.Utf8,
            )
            .alias("label")
        )
        empty_labels = predictions.filter(pl.col("label").is_null())
        if len(empty_labels) > 0:
            raise ValueError(
                "PrimeNovo predictions contain empty label values; label must equal "
                "the corresponding MGF TITLE."
            )

        return predictions.with_columns(
            (pl.lit(experiment_name) + ":" + pl.col("label")).alias("spectrum_id")
        )

    def _has_nterm_mod_in_middle(self, sequence: object) -> bool:
        """Return True if sequence contains a configured N-terminal mod after position 0."""
        if not isinstance(sequence, str):
            return False
        for modification in self.mid_sequence_n_terminal_mods:
            if modification not in sequence:
                continue
            first_index = sequence.find(modification)
            if first_index != 0:
                return True
            if sequence.find(modification, len(modification)) != -1:
                return True
        return False

    def _load_predictions(self, predictions_path: Path | str) -> pl.DataFrame:
        """Load PrimeNovo TSV predictions and validate required columns and scores.

        Args:
            predictions_path: Path to the predictions TSV or TXT file.

        Returns:
            DataFrame with the raw TSV columns (``label``, ``prediction``, ``score``,
            and optional ``charge``).

        Raises:
            ValueError: If the file extension is not ``.tsv`` or ``.txt``, required
                columns are absent, or any score is outside ``[0, 1]``.
        """
        predictions_path = Path(predictions_path)
        if predictions_path.suffix not in {".tsv", ".txt"}:
            raise ValueError(
                f"Unsupported file format for PrimeNovo predictions: "
                f"{predictions_path.suffix}. Supported formats are .tsv and .txt."
            )
        predictions = pl.read_csv(predictions_path, separator="\t")

        required = ("label", "prediction", "score")
        missing = [col for col in required if col not in predictions.columns]
        if missing:
            raise ValueError(
                f"PrimeNovo predictions file is missing required column(s): {missing}. "
                f"Present columns: {list(predictions.columns)}."
            )

        for score in predictions["score"]:
            if score is not None:
                self._validate_primenovo_score(float(score))

        return predictions

    def _load_spectrum_data(
        self, spectrum_path: Path | str
    ) -> Tuple[pl.DataFrame, bool]:
        """Load MGF spectrum data and assign ``experiment_name`` / ``spectrum_id``.

        Args:
            spectrum_path: Path to the spectrum data file (``.mgf`` only).

        Returns:
            Tuple of (DataFrame containing spectrum data, whether ground truth labels
            exist).

        Raises:
            ValueError: If the suffix is not ``.mgf``, any ``TITLE`` is missing, or
                ``TITLE`` values are not unique within the file.
        """
        spectrum_path = Path(spectrum_path)

        if spectrum_path.suffix != ".mgf":
            raise ValueError(
                f"Unsupported file format for spectrum data: {spectrum_path.suffix}. "
                "Supported format is .mgf."
            )

        from matchms.importing import load_from_mgf

        spectra = list(load_from_mgf(str(spectrum_path)))
        df = data_utils.df_from_matchms(spectra)
        titles = [spectrum.metadata.get("title") for spectrum in spectra]
        df = df.with_columns(
            pl.Series(
                "title",
                [
                    (
                        str(title).strip()
                        if title is not None and str(title).strip() != ""
                        else None
                    )
                    for title in titles
                ],
                dtype=pl.Utf8,
            )
        )

        missing_titles = df.filter(pl.col("title").is_null())
        if len(missing_titles) > 0:
            raise ValueError(
                "PrimeNovo MGF spectra require a non-empty TITLE for every ion. "
                f"Missing TITLE on {len(missing_titles)} spectrum row(s)."
            )
        if df["title"].n_unique() != len(df):
            duplicates = (
                df.group_by("title")
                .len()
                .filter(pl.col("len") > 1)
                .get_column("title")
                .head(5)
                .to_list()
            )
            raise ValueError(
                "Spectrum TITLE values must be unique within an MGF file. "
                f"Duplicate titles: {duplicates}"
            )

        experiment_name = spectrum_path.stem
        df = df.with_columns(
            pl.lit(experiment_name).alias("experiment_name").cast(pl.Utf8),
            (pl.lit(experiment_name) + ":" + pl.col("title")).alias("spectrum_id"),
        )

        has_labels = data_utils.has_ground_truth_sequence_labels(df)
        if "sequence" in df.columns and not has_labels:
            df = df.drop("sequence")
        return df, has_labels

    def _merge_data(
        self, spectrum_data: pl.DataFrame, predictions: pl.DataFrame
    ) -> pl.DataFrame:
        """Inner-join spectrum rows to predictions on ``spectrum_id``."""
        self._validate_join_keys(spectrum_data, predictions)
        merged = spectrum_data.join(predictions, on="spectrum_id", how="inner")

        if merged.height == 0:
            raise ValueError(
                "PrimeNovo inner join on spectrum_id produced no rows. Ensure each "
                "TSV label equals the corresponding spectrum TITLE in the MGF."
            )
        if merged.height != predictions.height:
            raise ValueError(
                f"Merge conflict: Expected {predictions.height} rows, "
                f"but got {merged.height}."
            )
        return merged

    def _process_predictions(
        self,
        predictions: pl.DataFrame,
        spectrum_data_columns: list[str],
        experiment_name: str,
    ) -> pl.DataFrame:
        """Assign ``spectrum_id``, filter mid-sequence N-term mods, and normalise columns.

        Adds ``spectrum_id`` as ``{experiment_name}:{label}``, drops rows with
        configured N-terminal mods mid-sequence, renames ``score`` to ``confidence``,
        and sets ``prediction_untokenised``. Prediction columns that clash with
        spectrum columns (except ``spectrum_id``) are dropped so spectrum metadata
        wins.
        """
        predictions = self._add_prediction_spectrum_ids(predictions, experiment_name)

        bad_mask = predictions.get_column("prediction").map_elements(
            self._has_nterm_mod_in_middle,
            return_dtype=pl.Boolean,
        )
        n_bad = int(bad_mask.sum())
        if n_bad:
            logger.warning(
                "Filtered %d spectra with PrimeNovo N-terminal modifications "
                "incorrectly placed in the middle of the peptide sequence.",
                n_bad,
            )
            predictions = predictions.filter(~bad_mask)

        predictions = predictions.rename({"score": "confidence"}).with_columns(
            pl.col("prediction").alias("prediction_untokenised")
        )

        spectrum_columns = set(spectrum_data_columns)
        return predictions.drop(
            [
                column
                for column in predictions.columns
                if column in spectrum_columns and column != "spectrum_id"
            ]
        )
