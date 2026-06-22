"""Unit tests for winnow calibration feature base class."""

import pytest
import pandas as pd

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset


class TestCalibrationFeaturesInterface:
    """Test the abstract CalibrationFeatures interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that CalibrationFeatures cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CalibrationFeatures()

    def test_feature_dependency_cannot_instantiate(self):
        """Test that FeatureDependency cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FeatureDependency()


class ConcreteFeature(CalibrationFeatures):
    """Concrete implementation for testing the interface."""

    @property
    def dependencies(self):
        return []

    @property
    def name(self):
        return "Test Feature"

    @property
    def columns(self):
        return ["test_col"]

    def prepare(self, dataset):
        pass

    def compute(self, dataset):
        dataset.metadata["test_col"] = [1, 2, 3]


class TestConcreteFeatureImplementation:
    """Test a concrete feature implementation."""

    def test_concrete_feature_can_be_instantiated(self):
        """Test that concrete feature implementation can be instantiated."""
        feature = ConcreteFeature()
        assert feature.name == "Test Feature"
        assert feature.columns == ["test_col"]
        assert feature.dependencies == []

    def test_concrete_feature_compute(self):
        """Test that concrete feature can compute values."""
        feature = ConcreteFeature()
        metadata = pd.DataFrame({"existing_col": [1, 2, 3]})
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        feature.compute(dataset)

        assert "test_col" in dataset.metadata.columns
        assert list(dataset.metadata["test_col"]) == [1, 2, 3]
