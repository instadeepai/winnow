from abc import ABCMeta, abstractmethod
import bisect
from math import exp, isnan
from typing import Dict, List, Tuple


import pandas as pd
import numpy as np
from numpy import median
from scipy.stats import entropy
from sklearn.neural_network import MLPRegressor

import koinapy

from winnow.datasets.calibration_dataset import CalibrationDataset


def map_modification(peptide):
    return [
        'M[UNIMOD:35]' if residue == 'M(ox)' else residue
        for residue in peptide
    ]

class FeatureDependency(metaclass=ABCMeta):
    """Base class for common dependencies of several features"""
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
        self, dataset: CalibrationDataset,
    ) -> Dict[str, pd.Series]:
        pass


class CalibrationFeatures(metaclass=ABCMeta):
    """The abstract interface for features for the calibration classifier.
    """
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

        This is used as the key for the feature in the `ProbabilityCalibrator`
        class and the name of the feature in the feature dataframe.

        Returns:
            str: The feature identifier
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

    @abstractmethod
    def prepare(
        self, dataset: CalibrationDataset
    ) -> None:
        pass

    @abstractmethod
    def compute(
        self, dataset: CalibrationDataset
    ) -> Dict[str, pd.Series]:
        pass


def find_matching_ions(
    source_mz: List[float],
    target_mz: List[float],
    target_intensities: List[float],
    mz_tolerance: float
) -> Tuple[
    List[float],
    List[float]
]:
    if isinstance(source_mz, float) and isnan(source_mz):
        return 0.0, 0.0
    num_matches, match_intensity = 0, 0.0
    for ion_mz in source_mz:
        nearest = bisect.bisect_left(target_mz, ion_mz)
        if nearest < len(target_mz):
            if target_mz[nearest] - ion_mz < mz_tolerance:
                num_matches += 1
                match_intensity += target_intensities[nearest]
                continue
        if nearest > 0:
            if ion_mz - target_mz[nearest-1] < mz_tolerance:
                num_matches += 1
                match_intensity += target_intensities[nearest-1]
    return num_matches/len(source_mz), match_intensity/sum(target_intensities)


def compute_ion_identifications(dataset: pd.DataFrame, source_column: str, mz_tolerance: float):
    matches = [
        find_matching_ions(
            source_mz=row[source_column],
            target_mz=row['mz_array'],
            target_intensities=row['intensity_array'],
            mz_tolerance=mz_tolerance
        )
        for _, row in dataset.iterrows()
    ]
    return zip(*matches)


class PrositFeatures(CalibrationFeatures):
    def __init__(self, mz_tolerance: float) -> None:
        self.mz_tolerance = mz_tolerance
        self.model = koinapy.Koina("Prosit_2020_intensity_HCD", "koina.wilhelmlab.org:443")

    @property
    def dependencies(self) -> List[FeatureDependency]:
        return []

    @property
    def name(self) -> str:
        return 'Prosit Features'

    @property
    def columns(self) -> List[str]:
        return ['ion_matches', 'ion_match_intensity']

    def prepare(
        self, dataset: CalibrationDataset
    ) -> None:
        return

    def compute(self, dataset: CalibrationDataset) -> Dict[str, pd.Series]:
        inputs = pd.DataFrame()
        inputs['peptide_sequences'] = np.array([
            ''.join(peptide) for peptide in dataset.metadata['prediction'].apply(map_modification)
            ]
        )
        inputs['precursor_charges'] = np.array(dataset.metadata['precursor_charge'])
        inputs['collision_energies'] = np.array(len(dataset.metadata)*[25])

        predictions: pd.DataFrame = self.model.predict(inputs, debug=True)
        predictions['Index'] = predictions.index

        grouped_predictions = predictions.groupby(by='Index').agg(
                {'peptide_sequences': 'first',
                 'precursor_charges': 'first',
                 'collision_energies': 'first',
                 'intensities': list,
                 'mz': list,
                 'annotation': list}
        )
        grouped_predictions['intensities'] = grouped_predictions.apply(
            lambda row: np.array(row['intensities'])[np.argsort(row['mz'])].tolist(),
            axis=1
        )
        grouped_predictions['annotation'] = grouped_predictions.apply(
            lambda row: np.array(row['annotation'])[np.argsort(row['mz'])].tolist(),
            axis=1
        )
        grouped_predictions['mz'] = grouped_predictions['mz'].apply(np.sort)
        dataset.metadata['prosit_mz'] = grouped_predictions['mz']
        dataset.metadata['prosit_intensity'] = grouped_predictions['intensities']
        ion_matches, match_intensity = compute_ion_identifications(
            dataset=dataset.metadata,
            source_column='prosit_mz',
            mz_tolerance=self.mz_tolerance
        )
        dataset.metadata['ion_matches'] = ion_matches
        dataset.metadata['ion_match_intensity'] = match_intensity


class ChimericFeatures(CalibrationFeatures):
    def __init__(self, mz_tolerance: float) -> None:
        self.mz_tolerance = mz_tolerance

    @property
    def dependencies(self) -> List[FeatureDependency]:
        return []

    @property
    def name(self) -> str:
        return 'Chimeric Features'

    @property
    def columns(self) -> List[str]:
        return ['chimeric_ion_matches', 'chimeric_ion_match_intensity']

    def prepare(
        self, dataset: CalibrationDataset
    ) -> None:
        return

    def compute(self, dataset: CalibrationDataset) -> Dict[str, pd.Series]:
        inputs = pd.DataFrame()
        inputs['peptide_sequences'] = np.array([
                ''.join(map_modification(items[1].sequence))
                for items in dataset.predictions
            ]
        )
        inputs['precursor_charges'] = np.array(dataset.metadata['precursor_charge'])
        inputs['collision_energies'] = np.array(len(dataset.metadata)*[25])
        model = koinapy.Koina("Prosit_2020_intensity_HCD", "koina.wilhelmlab.org:443")
        predictions: pd.DataFrame = model.predict(inputs)
        predictions['Index'] = predictions.index

        grouped_predictions = predictions.groupby(by='Index').agg(
                {'peptide_sequences': 'first',
                 'precursor_charges': 'first',
                 'collision_energies': 'first',
                 'intensities': list,
                 'mz': list,
                 'annotation': list}
        )
        grouped_predictions['intensities'] = grouped_predictions.apply(
            lambda row: np.array(row['intensities'])[np.argsort(row['mz'])].tolist(),
            axis=1
        )
        grouped_predictions['annotation'] = grouped_predictions.apply(
            lambda row: np.array(row['annotation'])[np.argsort(row['mz'])].tolist(),
            axis=1
        )
        grouped_predictions['mz'] = grouped_predictions['mz'].apply(np.sort)
        dataset.metadata['runner_up_prosit_mz'] = grouped_predictions['mz']
        dataset.metadata['runner_up_prosit_intensity'] = grouped_predictions['intensities']

        ion_matches, match_intensity = compute_ion_identifications(
            dataset=dataset.metadata,
            source_column='runner_up_prosit_mz',
            mz_tolerance=self.mz_tolerance
        )
        dataset.metadata['chimeric_ion_matches'] = ion_matches
        dataset.metadata['chimeric_ion_match_intensity'] = match_intensity


class MassErrorFeature(CalibrationFeatures):
    """Calculates the difference between the precursor and theoretical mass
    """
    h2o_mass: float = 18.0106
    proton_mass: float = 1.007276
    def __init__(self, residue_masses: Dict[str, float]) -> None:
        super().__init__()
        self.residue_masses = residue_masses

    @property
    def dependencies(self) -> List[FeatureDependency]:
        return []

    @property
    def columns(self) -> List[str]:
        return [self.name]

    @property
    def name(self) -> str:
        return 'Mass Error'

    def prepare(
        self, dataset: CalibrationDataset
    ) -> None:
        return

    def compute(
        self, dataset: CalibrationDataset,
    ) -> Dict[str, pd.Series]:
        theoretical_mass = dataset.metadata['prediction'].apply(
            lambda peptide: sum(
                self.residue_masses[residue] for residue in peptide
            ) if isinstance(peptide, list) else float("-inf")
        )
        dataset.metadata[self.name] = (
            dataset.metadata['Mass'] - (theoretical_mass + self.h2o_mass + self.proton_mass)
        )


class BeamFeatures(CalibrationFeatures):
    """Calculates the margin, median margin and entropy of beam runners-up"""
    @property
    def dependencies(self) -> List[FeatureDependency]:
        return []

    @property
    def name(self) -> str:
        return 'Beam Features'

    @property
    def columns(self) -> List[str]:
        return ['margin', 'median_margin', 'entropy']

    def prepare(
        self, dataset: CalibrationDataset
    ) -> None:
        return

    def compute(self, dataset: CalibrationDataset) -> Dict[str, pd.Series]:
        top_probs = [
            exp(prediction[0].sequence_log_probability) for prediction in dataset.predictions
        ]
        second_probs = [
            exp(prediction[1].sequence_log_probability) for prediction in dataset.predictions
        ]
        second_margin = [
            top_prob - second_prob for top_prob, second_prob in zip(top_probs, second_probs)
        ]
        runner_up_probs = [
            [exp(item.sequence_log_probability) for item in prediction[1:]]
            for prediction in dataset.predictions
        ]
        normalised_runner_up_probs = [
            [probability/sum(probabilities) for probability in probabilities]
            for probabilities in runner_up_probs
        ]
        runner_up_entropy = [
            entropy(probs) for probs in normalised_runner_up_probs
        ]
        runner_up_median = [
            median(probs) for probs in runner_up_probs
        ]
        median_margin = [
            top_prob - median_prob for top_prob, median_prob in zip(top_probs, runner_up_median)
        ]
        # dataset.metadata['confidence'] = top_probs
        dataset.metadata['margin'] = second_margin
        dataset.metadata['median_margin'] = median_margin
        dataset.metadata['entropy'] = runner_up_entropy
        return super().compute(dataset)


class RetentionTimeFeature(CalibrationFeatures):
    """"""
    def __init__(self, hidden_dim: int, train_fraction: float) -> None:
        self.train_fraction = train_fraction
        self.hidden_dim = hidden_dim
        self.prosit_model = koinapy.Koina("Prosit_2019_irt", "koina.wilhelmlab.org:443")
        self.irt_predictor = MLPRegressor(hidden_layer_sizes=[hidden_dim])

    @property
    def dependencies(self) -> List[FeatureDependency]:
        return []

    @property
    def name(self) -> str:
        return 'Prosit iRT Features'

    @property
    def columns(self) -> List[str]:
        return ['iRT error']

    def prepare(
        self, dataset: CalibrationDataset
    ) -> None:
        # -- Make calibration dataset
        train_data = dataset.metadata.copy(deep=True)
        train_data = train_data.sort_values(by='confidence', ascending=False)
        train_data = train_data.iloc[:int(self.train_fraction*len(train_data))]

        # -- Get predictions
        inputs = pd.DataFrame()
        inputs['peptide_sequences'] = np.array([
            ''.join(peptide) for peptide in train_data['prediction'].apply(map_modification)
            ]
        )
        inputs = inputs.set_index(train_data.index)
        predictions = self.prosit_model.predict(inputs)
        train_data['iRT'] = predictions['irt']

        # -- Fit model
        X, y = train_data['Retention time'].values, train_data['iRT'].values
        self.irt_predictor.fit(X.reshape(-1, 1), y)

    def compute(self, dataset: CalibrationDataset) -> Dict[str, pd.Series]:
        # -- Get predictions
        inputs = pd.DataFrame()
        inputs['peptide_sequences'] = np.array([
            ''.join(peptide) for peptide in dataset.metadata['prediction'].apply(map_modification)
            ]
        )
        predictions = self.prosit_model.predict(inputs)
        dataset.metadata['iRT'] = predictions['irt']

        # - Predict iRT
        dataset.metadata['predicted iRT'] = self.irt_predictor.predict(
            dataset.metadata['Retention time'].values.reshape(-1, 1)
        )
        dataset.metadata['iRT error'] = np.abs(dataset.metadata['predicted iRT'] - dataset.metadata['iRT'])
        