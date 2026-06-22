"""Shared fixtures for calibration feature tests."""

import pytest
import pandas as pd


class MockScoredSequence:
    """Mock class for ScoredSequence used in beam search results."""

    def __init__(self, sequence, log_prob, token_log_probs=None):
        self.sequence = sequence
        self.sequence_log_probability = log_prob
        self.token_log_probabilities = token_log_probs


@pytest.fixture()
def mock_scored_sequence_class():
    """Return the MockScoredSequence class for use in tests."""
    return MockScoredSequence


def make_intensity_mock(spectrum_id_to_ions: dict):
    """Return a callable that mimics koinapy.Koina.predict for intensity models.

    Args:
        spectrum_id_to_ions: mapping from spectrum_id to a list of
            (mz, intensity, annotation) tuples to return.
    """

    def _predict(inputs_df):
        rows, idx = [], []
        for sid in inputs_df.index:
            for mz, intensity, ann in spectrum_id_to_ions.get(sid, []):
                rows.append(
                    {
                        "peptide_sequences": inputs_df.loc[sid, "peptide_sequences"],
                        "precursor_charges": inputs_df.loc[sid, "precursor_charges"],
                        "collision_energies": inputs_df.loc[sid, "collision_energies"],
                        "intensities": intensity,
                        "mz": mz,
                        "annotation": ann,
                    }
                )
                idx.append(sid)
        return pd.DataFrame(rows, index=idx)

    return _predict
