from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Any
import inspect
import pandas as pd
from sklearn.neural_network import MLPRegressor

from winnow.datasets.calibration_dataset import CalibrationDataset


class FeatureDependency(metaclass=ABCMeta):
    """Base class for common dependencies of several features."""

    @property
    @abstractmethod
    def name(self) -> str:
        """A natural language identifier for the feature.

        This is used as the key for the feature in the `ProbabilityCalibrator`
        class and the name of the feature in the feature dataframe.

        Returns:
            str: The feature identifier
        """

    @abstractmethod
    def compute(
        self,
        dataset: CalibrationDataset,
    ) -> Dict[str, pd.Series]:
        """Compute the feature dependencies based on the given dataset.

        Args:
            dataset (CalibrationDataset): The dataset containing metadata and predictions.

        Returns:
            Dict[str, pd.Series]: A dictionary mapping feature dependency names to computed values.
        """
        pass


class CalibrationFeatures(metaclass=ABCMeta):
    """The abstract interface for features for the calibration classifier."""

    irt_predictor: Optional[MLPRegressor] = None

    @property
    @abstractmethod
    def dependencies(self) -> List[FeatureDependency]:
        """A list of dependencies that need to be computed before a feature.

        Returns:
            List[FeatureDependency]:
                The list of dependencies.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """A natural language identifier for the feature.

        This is used as the key for the feature in the `ProbabilityCalibrator` class and the name of the feature in the feature dataframe.

        Returns:
            str: The feature identifier.
        """

    @property
    @abstractmethod
    def columns(self) -> List[str]:
        """A natural language identifier for the feature.

        This is used as the key for the feature in the `ProbabilityCalibrator`
        class and the name of the feature in the feature dataframe.

        Returns:
            str: The feature identifier
        """

    @staticmethod
    def _to_plain(value: Any) -> Any:
        """Convert OmegaConf containers to plain Python types."""
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf

            if isinstance(value, (DictConfig, ListConfig)):
                return OmegaConf.to_container(value, resolve=True)
        except ImportError:
            pass
        return value

    @abstractmethod
    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepare the dataset for feature computation.

        This method is intended to perform any preprocessing required before computing the feature.

        Args:
            dataset (CalibrationDataset): The dataset to prepare for feature computation.
        """
        pass

    @abstractmethod
    def compute(self, dataset: CalibrationDataset) -> None:
        """Compute the feature for the dataset.

        This method calculates the feature values based on the dataset and its dependencies.

        Args:
            dataset (CalibrationDataset): The dataset on which to compute the feature.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return a serialisable dict of constructor parameters.

        The returned dict includes a ``_target_`` key with the fully-qualified
        class path, plus every ``__init__`` parameter and its current value.
        This is sufficient to reconstruct the feature instance.

        OmegaConf containers (``DictConfig``, ``ListConfig``) are resolved to
        plain Python types so that the result is always JSON-serialisable.

        Returns:
            Dict whose values are JSON-serialisable.
        """
        sig = inspect.signature(self.__class__.__init__)
        config: Dict[str, Any] = {
            "_target_": (f"{self.__class__.__module__}.{self.__class__.__qualname__}"),
        }
        for param_name in sig.parameters:
            if param_name == "self":
                continue
            if hasattr(self, param_name):
                config[param_name] = self._to_plain(getattr(self, param_name))
        return config
