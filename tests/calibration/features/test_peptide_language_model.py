"""Unit tests for peptide language model calibration features."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from winnow.calibration.features.peptide_language_model import (
    ESMC_DEFAULT_MODEL,
    PLM_EMBEDDING_DIAGNOSTIC_COLUMNS,
    PLM_EMBEDDING_COLUMNS,
    PLM_EXTENDED_EMBEDDING_COLUMNS,
    PLM_LOG_PROB_COLUMN,
    PLM_MISSING_COLUMN,
    PeptideLanguageModelFeature,
    PeptideLanguageModelResult,
    normalize_peptide_for_plm,
)
from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.datasets.calibration_dataset import CalibrationDataset


class FakeBackend:
    """Deterministic backend for feature tests."""

    def __init__(self):
        self.calls = []

    def score(self, sequences, feature_mode):
        self.calls.append((tuple(sequences), feature_mode))
        results = {}
        for sequence in sequences:
            if feature_mode == "pseudo_likelihood":
                results[sequence] = PeptideLanguageModelResult(
                    mean_log_probability=-float(len(sequence))
                )
            else:
                value = float(len(sequence))
                if feature_mode == "embedding_diagnostics":
                    results[sequence] = PeptideLanguageModelResult(
                        embedding_diagnostics=tuple(
                            value + float(offset)
                            for offset in range(len(PLM_EMBEDDING_DIAGNOSTIC_COLUMNS))
                        )
                    )
                else:
                    results[sequence] = PeptideLanguageModelResult(
                        embedding_summary=(value, value + 1.0, value + 2.0, value + 3.0)
                    )
        return results


class MissingEmbeddingBackend(FakeBackend):
    """Backend that marks selected sequences as unscored."""

    def score(self, sequences, feature_mode):
        self.calls.append((tuple(sequences), feature_mode))
        results = {}
        for sequence in sequences:
            if sequence == "AC":
                results[sequence] = PeptideLanguageModelResult()
            else:
                value = float(len(sequence))
                if feature_mode == "embedding_diagnostics":
                    results[sequence] = PeptideLanguageModelResult(
                        embedding_diagnostics=tuple(
                            value + float(offset)
                            for offset in range(len(PLM_EMBEDDING_DIAGNOSTIC_COLUMNS))
                        )
                    )
                else:
                    results[sequence] = PeptideLanguageModelResult(
                        embedding_summary=(value, value + 1.0, value + 2.0, value + 3.0)
                    )
        return results


def test_normalize_peptide_for_plm_strips_ptms_and_terminal_mods():
    assert normalize_peptide_for_plm(["A", "C[UNIMOD:4]", "[UNIMOD:1]", "M"]) == "ACM"
    assert normalize_peptide_for_plm("ACD") == "ACD"
    assert normalize_peptide_for_plm("A,C[UNIMOD:4],D") == "ACD"


def test_normalize_peptide_for_plm_rejects_invalid_or_empty_inputs():
    assert normalize_peptide_for_plm(["[UNIMOD:1]"]) is None
    assert normalize_peptide_for_plm(["A", "U"]) is None
    assert normalize_peptide_for_plm(None) is None


def test_properties_for_pseudo_likelihood_and_embedding_modes():
    pseudo = PeptideLanguageModelFeature(
        backend="precomputed",
        feature_mode="pseudo_likelihood",
        learn_from_missing=True,
    )
    assert pseudo.name == "Peptide Language Model Feature"
    assert pseudo.dependencies == []
    assert pseudo.columns == [PLM_LOG_PROB_COLUMN, PLM_MISSING_COLUMN]

    embedding = PeptideLanguageModelFeature(
        backend="pepbert",
        feature_mode="embedding_summary",
    )
    assert embedding.columns == PLM_EMBEDDING_COLUMNS

    diagnostics = PeptideLanguageModelFeature(
        backend="pepbert",
        feature_mode="embedding_diagnostics",
        learn_from_missing=True,
    )
    assert diagnostics.columns == [
        *PLM_EXTENDED_EMBEDDING_COLUMNS,
        PLM_MISSING_COLUMN,
    ]


def test_esmc_default_uses_biohub_6b_model():
    assert ESMC_DEFAULT_MODEL == "biohub/ESMC-6B"


def test_precomputed_mode_reads_scalar_column():
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "prediction": [["A"], ["C"]],
                "confidence": [0.8, 0.7],
                "external_plm": [-0.2, -0.4],
            }
        )
    )
    feature = PeptideLanguageModelFeature(
        backend="precomputed",
        feature_mode="pseudo_likelihood",
        precomputed_column="external_plm",
    )

    feature.compute(dataset)

    assert list(dataset.metadata[PLM_LOG_PROB_COLUMN]) == [-0.2, -0.4]
    assert "external_plm" in dataset.metadata.columns


def test_precomputed_mode_errors_when_column_is_absent():
    dataset = CalibrationDataset(
        metadata=pd.DataFrame({"prediction": [["A"]], "confidence": [0.8]})
    )
    feature = PeptideLanguageModelFeature(
        backend="precomputed",
        feature_mode="pseudo_likelihood",
        precomputed_column="missing_plm",
    )

    with pytest.raises(ValueError, match="not found"):
        feature.compute(dataset)


def test_missing_precomputed_values_are_filtered_by_default():
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "prediction": [["A"], ["C"], ["D"]],
                "confidence": [0.8, 0.7, 0.6],
                "external_plm": [-0.2, None, -0.4],
            }
        )
    )
    feature = PeptideLanguageModelFeature(
        backend="precomputed",
        feature_mode="pseudo_likelihood",
        precomputed_column="external_plm",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feature.compute(dataset)

    assert any("Filtering 1 PSMs" in str(warning.message) for warning in caught)
    assert len(dataset.metadata) == 2
    assert list(dataset.metadata[PLM_LOG_PROB_COLUMN]) == [-0.2, -0.4]


def test_missing_precomputed_values_can_be_learned_from():
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "prediction": [["A"], ["C"], ["D"]],
                "confidence": [0.8, 0.7, 0.6],
                "external_plm": [-0.2, None, -0.4],
            }
        )
    )
    feature = PeptideLanguageModelFeature(
        backend="precomputed",
        feature_mode="pseudo_likelihood",
        precomputed_column="external_plm",
        learn_from_missing=True,
    )

    feature.compute(dataset)

    assert list(dataset.metadata[PLM_LOG_PROB_COLUMN]) == [-0.2, 0.0, -0.4]
    assert list(dataset.metadata[PLM_MISSING_COLUMN]) == [False, True, False]


def test_mock_backend_computes_embedding_summaries_and_caches_duplicates():
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "prediction": [["A", "C"], ["A", "C"], ["D"]],
                "confidence": [0.8, 0.7, 0.6],
            }
        )
    )
    backend = FakeBackend()
    feature = PeptideLanguageModelFeature(
        backend="pepbert",
        feature_mode="embedding_summary",
        batch_size=10,
    )
    feature._backend_instance = backend

    feature.compute(dataset)

    assert backend.calls == [(("AC", "D"), "embedding_summary")]
    assert list(dataset.metadata["plm_embedding_mean"]) == [2.0, 2.0, 1.0]
    assert list(dataset.metadata["plm_embedding_max"]) == [5.0, 5.0, 4.0]


def test_mock_backend_computes_pseudo_likelihood():
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "prediction": [["A", "C"], ["D"]],
                "confidence": [0.8, 0.7],
            }
        )
    )
    feature = PeptideLanguageModelFeature(
        backend="esm2",
        feature_mode="pseudo_likelihood",
    )
    feature._backend_instance = FakeBackend()

    feature.compute(dataset)

    assert list(dataset.metadata[PLM_LOG_PROB_COLUMN]) == [-2.0, -1.0]


def test_invalid_normalized_peptides_follow_missing_strategy():
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "prediction": [["A"], ["U"], ["C"]],
                "confidence": [0.8, 0.7, 0.6],
            }
        )
    )
    feature = PeptideLanguageModelFeature(
        backend="esm2",
        feature_mode="pseudo_likelihood",
        learn_from_missing=True,
    )
    feature._backend_instance = FakeBackend()

    feature.compute(dataset)

    assert list(dataset.metadata[PLM_LOG_PROB_COLUMN]) == [-1.0, 0.0, -1.0]
    assert list(dataset.metadata[PLM_MISSING_COLUMN]) == [False, True, False]


def test_backend_missing_embedding_summaries_can_be_learned_from():
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "prediction": [["A"], ["A", "C"], ["D"]],
                "confidence": [0.8, 0.7, 0.6],
            }
        )
    )
    feature = PeptideLanguageModelFeature(
        backend="pepbert",
        feature_mode="embedding_summary",
        learn_from_missing=True,
    )
    feature._backend_instance = MissingEmbeddingBackend()

    feature.compute(dataset)

    assert list(dataset.metadata[PLM_MISSING_COLUMN]) == [False, True, False]
    assert list(dataset.metadata["plm_embedding_mean"]) == [1.0, 0.0, 1.0]
    assert list(dataset.metadata["plm_embedding_max"]) == [4.0, 0.0, 4.0]


def test_embedding_diagnostics_include_length_and_missing_interaction():
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "prediction": [["A"], ["A", "C"], ["D"]],
                "confidence": [0.8, 0.7, 0.6],
            }
        )
    )
    feature = PeptideLanguageModelFeature(
        backend="pepbert",
        feature_mode="embedding_diagnostics",
        learn_from_missing=True,
        batch_size=10,
    )
    feature._backend_instance = MissingEmbeddingBackend()

    feature.compute(dataset)

    assert feature._backend_instance.calls == [
        (("A", "AC", "D"), "embedding_diagnostics")
    ]
    assert list(dataset.metadata["plm_sequence_length"]) == [1.0, 2.0, 1.0]
    assert np.allclose(
        dataset.metadata["plm_log_sequence_length"],
        [np.log1p(1.0), np.log1p(2.0), np.log1p(1.0)],
    )
    assert list(dataset.metadata[PLM_MISSING_COLUMN]) == [False, True, False]
    assert list(dataset.metadata["plm_embedding_mean"]) == [1.0, 0.0, 1.0]
    assert list(dataset.metadata["plm_residue_norm_max"]) == [10.0, 0.0, 10.0]
    assert np.allclose(
        dataset.metadata["plm_missing_x_log_sequence_length"],
        [0.0, np.log1p(2.0), 0.0],
    )


def test_calibrator_integration_uses_plm_feature_columns():
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "prediction": [["A"], ["C"]],
                "confidence": [0.8, 0.7],
                "correct": [True, False],
                "external_plm": [-0.2, -0.4],
            }
        )
    )
    feature = PeptideLanguageModelFeature(
        backend="precomputed",
        feature_mode="pseudo_likelihood",
        precomputed_column="external_plm",
    )
    calibrator = ProbabilityCalibrator(features=[feature])

    features, labels = calibrator.compute_features(dataset, labelled=True)

    assert features.tolist() == [[0.8, -0.2], [0.7, -0.4]]
    assert labels.tolist() == [True, False]


def test_loaded_backend_is_not_pickled():
    feature = PeptideLanguageModelFeature(
        backend="esm2", feature_mode="pseudo_likelihood"
    )
    feature._backend_instance = FakeBackend()

    loaded = pickle.loads(pickle.dumps(feature))

    assert loaded._backend_instance is None
