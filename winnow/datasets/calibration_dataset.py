from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from pyteomics import mztab, mgf

from instanovo.utils.metrics import Metrics
from instanovo.inference.beam_search import ScoredSequence


RESIDUE_MASSES: dict[str, float] = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    #   "C(+57.02)": 160.030649, # 103.009185 + 57.021464
    "C": 160.030649,  # C+57.021 V1
    "L": 113.084064,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
    #   "M(+15.99)": 147.035400, # Met oxidation:   131.040485 + 15.994915
    "M(ox)": 147.035400,  # Met oxidation:   131.040485 + 15.994915 V1
    "N(+.98)": 115.026943, # Asn deamidation: 114.042927 +  0.984016
    "Q(+.98)": 129.042594, # Gln deamidation: 128.058578 +  0.984016
}


metrics = Metrics(
    residues=RESIDUE_MASSES,
    isotope_error_range=[0, 1]
)


@dataclass
class CalibrationDataset:
    metadata: pd.DataFrame
    predictions: List[Optional[List[ScoredSequence]]]

    @classmethod
    def from_predictions_csv(
        cls,
        beam_predictions_path: Path,
        spectrum_path: Path,
        predictions_path: Path
    ) -> "CalibrationDataset":
        #  -- Load predictions
        with open(beam_predictions_path / 'predictions_list.pkl', 'rb') as predictions_file:
            predictions = pickle.load(predictions_file)

        # -- Load predictions metadata
        dataset = pd.read_csv(predictions_path)
        dataset.rename(
            {'targets': 'peptide', 'preds': 'prediction', 'log_probs': 'confidence'},
            axis=1, inplace=True
        )
        dataset.loc[dataset['confidence'] == -1., 'confidence'] = float("-inf")
        dataset['confidence'] = dataset['confidence'].apply(np.exp)
        dataset['peptide'] = dataset['peptide'].apply(
            lambda peptide: peptide.replace('L', 'I') if isinstance(peptide, str) else peptide
        )
        dataset['prediction'] = dataset['prediction'].apply(
            lambda peptide: peptide.replace('L', 'I') if isinstance(peptide, str) else peptide
        )

        # -- Load spectra
        spectrum_dataset = pl.read_ipc(spectrum_path).to_pandas()
        dataset = pd.merge(
            dataset, spectrum_dataset, on=['spectrum_index', 'global_index'],
            how='left'
        )

        # -- Evaluate identifications
        dataset['peptide'] = dataset['peptide'].apply(metrics._split_peptide)
        dataset['prediction'] = dataset['prediction'].apply(metrics._split_peptide)

        dataset['valid_peptide'] = dataset['peptide'].apply(lambda peptide: isinstance(peptide, list))
        dataset['valid_prediction'] = dataset['prediction'].apply(lambda prediction: isinstance(prediction, list))

        dataset['num_matches'] = dataset.apply(
            lambda row: (
                metrics._novor_match(row['peptide'], row['prediction'])
                if not (isinstance(row['peptide'], float) or isinstance(row['prediction'], float))
                else 0
            ),
            axis=1
        )
        dataset['correct'] = dataset.apply(
            lambda row: (
                row['num_matches'] == len(row['peptide']) == len(row['prediction'])
                if not (isinstance(row['peptide'], float) or isinstance(row['prediction'], float))
                else False
            ),
            axis=1
        )
        return cls(
            metadata=dataset,
            predictions=predictions
        )

    @classmethod
    def from_predictions_mztab(
        cls,
        labelled_path: Path,
        mgf_path: Path,
        predictions_path: Path
    ) -> "CalibrationDataset":
        # -- Load labelled data
        labelled = pl.read_ipc(labelled_path).to_pandas()
        labelled.rename({'spectrum_index': 'scan'}, axis=1, inplace=True)

        # -- Load MGF
        raw_mgf = list(mgf.read(open(mgf_path)))
        for spectrum in raw_mgf:
            spectrum['scan'] = int(spectrum['params']['title'].replace('"', '').split('scan=')[-1])

        # -- Load predictions
        predictions = mztab.MzTab(predictions_path).spectrum_match_table
        predictions['index'] = predictions['spectra_ref'].apply(lambda index: int(index.split('=')[-1]))
        predictions = predictions.set_index('index', drop=False)
        predictions['scan'] = predictions['index'].apply(lambda index: raw_mgf[index]['scan'])
        predictions['sequence'] = predictions['sequence'].apply(
            lambda sequence: sequence.replace(
                    'M+15.995', 'M(ox)'
                ).replace(
                    'C+57.021', 'C'
                ).replace(
                    'N+0.984', 'N(+.98)'
                ).replace(
                    'Q+0.984', 'Q(+.98)'
                )
        )
        predictions = pd.merge(labelled, predictions, on='scan')
        columns = [
            'scan', 'mz_array', 'intensity_array', 'charge', 'Retention time',
            'Mass', 'Sequence', 'modified_sequence', 'sequence', 'search_engine_score[1]'
        ]
        predictions = predictions[columns]
        predictions.rename(
            {
                'Retention time': 'retention_time', 'Sequence': 'peptide',
                'sequence': 'prediction', 'search_engine_score[1]': 'confidence',
                'charge': 'precursor_charge'
            },
            axis=1, inplace=True
        )

        predictions = predictions[predictions['prediction'].apply(
            lambda peptide: not (peptide.startswith('+') or peptide.startswith('-'))
        )]
        predictions['peptide'] = predictions['peptide'].apply(
            lambda peptide: peptide.replace('L', 'I') if isinstance(peptide, str) else peptide
        )
        predictions['prediction'] = predictions['prediction'].apply(
            lambda peptide: peptide.replace('L', 'I') if isinstance(peptide, str) else peptide
        )
        predictions['peptide'] = predictions['peptide'].apply(lambda peptide: re.split(r"(?<=.)(?=[A-Z])", peptide)) 
        predictions['prediction'] = predictions['prediction'].apply(lambda peptide: re.split(r"(?<=.)(?=[A-Z])", peptide)) 
        predictions['num_matches'] = predictions.apply(
            lambda row: (
                metrics._novor_match(row['peptide'], row['prediction'])
                if not (isinstance(row['peptide'], float) or isinstance(row['prediction'], float))
                else 0
            ),
            axis=1
        )
        predictions['correct'] = predictions.apply(
            lambda row: (
                row['num_matches'] == len(row['peptide']) == len(row['prediction'])
                if not (isinstance(row['peptide'], float) or isinstance(row['prediction'], float))
                else False
            ),
            axis=1
        )
        return cls(
            metadata=predictions,
            predictions=len(predictions) * [None]
        )

    @classmethod
    def from_pointnovo_predictions(
        cls,
        mgf_path: Path,
        predictions_path: Path
    ) -> "CalibrationDataset":
        # -- Load MGF file
        data_dict = defaultdict(list)
        for spectrum in mgf.read(open(mgf_path  )):
            data_dict['scan'].append(spectrum['params']['scans'])
            data_dict['peptide'].append(spectrum['params']['seq'])
            data_dict['precursor_charge'].append(float(spectrum['params']['charge'][0]))
            data_dict['Mass'].append(float(spectrum['params']['pepmass'][0]))
            data_dict['retention_time'].append(float(spectrum['params']['rtinseconds']))
            data_dict['mz_array'].append(spectrum['m/z array'])
            data_dict['intensity_array'].append(spectrum['intensity array'])
        spectra = pd.DataFrame(data=data_dict)

        predictions = pd.read_csv(predictions_path, sep='\t')


        predictions = predictions[['feature_id', 'predicted_sequence', 'predicted_score']]
        predictions.rename(
            {'feature_id': 'scan', 'predicted_sequence': 'prediction', 'predicted_score': 'confidence'},
            axis=1, inplace=True
        )
        dataset = pd.merge(predictions, spectra, how='left', on='scan')
        dataset['peptide'] = dataset['peptide'].apply(
            lambda peptide: peptide.replace(
                'L', 'I'
            ).replace(
                'M(+15.99)', 'M(ox)'
            ).replace(
                'C(+57.02)', 'C'
            ) if isinstance(peptide, str) else peptide
        )
        dataset['peptide'] = dataset['peptide'].apply(lambda peptide: re.split(r"(?<=.)(?=[A-Z])", peptide)) 
        dataset['prediction'] = dataset['prediction'].apply(
            lambda peptide: peptide.replace(
                'L', 'I'
            ).replace(
                'C(Carbamidomethylation)', 'C'
            ).replace(
                'N(Deamidation)', 'N(+.98)'
            ).replace(
                'Q(Deamidation)', 'Q(+.98)'
            ) if isinstance(peptide, str) else peptide
        )
        dataset['prediction'] = dataset['prediction'].map(
            lambda sequence: sequence.split(',') if isinstance(sequence, str) else []
        )
        dataset['num_matches'] = dataset.apply(
            lambda row: (
                metrics._novor_match(row['peptide'], row['prediction'])
                if not (isinstance(row['peptide'], float) or isinstance(row['prediction'], float))
                else 0
            ),
            axis=1
        )
        dataset['correct'] = dataset.apply(
            lambda row: (
                row['num_matches'] == len(row['peptide']) == len(row['prediction'])
                if not (isinstance(row['peptide'], float) or isinstance(row['prediction'], float))
                else False
            ),
            axis=1
        )
        return cls(
            metadata=dataset,
            predictions=len(dataset) * [None]
        )

    def filter(
        self,
        metadata_predicate: Callable[[Any], bool] = lambda row: False,
        predictions_predicate: Callable[[Any], bool] = lambda beam: False,
    ) -> "CalibrationDataset":
        filter_idxs = []

        # -- Get filter indices for metadata condition
        metadata_filter_idxs, = np.where(self.metadata.apply(metadata_predicate, axis=1).values)
        filter_idxs.extend(metadata_filter_idxs.tolist())

        # -- Get filter indices for predictions condition
        predictions_filter_idxs = [
            idx for idx, beam in enumerate(self.predictions) if predictions_predicate(beam)
        ]
        filter_idxs.extend(predictions_filter_idxs)

        filter_idxs = set(filter_idxs)

        # -- Gather predictions
        predictions = [
            prediction for idx, prediction in enumerate(self.predictions)
            if idx not in filter_idxs
        ]

        # -- Gather metadata
        selection_idxs = [
            idx for idx in range(len(self.metadata))
            if idx not in filter_idxs
        ]
        metadata = self.metadata.iloc[selection_idxs].copy(deep=True)
        metadata = metadata.reset_index(drop=True)

        return CalibrationDataset(
            predictions=predictions,
            metadata=metadata
        )

    @property
    def confidence_column(self) -> str:
        return 'confidence'

    def __getitem__(self, index) -> Tuple[pd.Series, List[ScoredSequence]]:
        return self.metadata.iloc[index], self.predictions[index]

    def __len__(self) -> int:
        assert self.metadata.shape[0] == len(self.predictions)
        return len(self.predictions)
