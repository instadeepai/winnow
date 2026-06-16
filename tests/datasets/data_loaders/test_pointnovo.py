"""Tests for PointNovoDatasetLoader."""

import pytest

from winnow.datasets.data_loaders import PointNovoDatasetLoader


class TestPointNovoDatasetLoader:
    """Tests for PointNovoDatasetLoader (not yet implemented)."""

    @pytest.fixture()
    def loader(self, full_residue_masses):
        return PointNovoDatasetLoader(residue_masses=full_residue_masses)

    def test_load_raises_not_implemented(self, loader, tmp_path):
        """load() must raise NotImplementedError unconditionally."""
        with pytest.raises(NotImplementedError):
            loader.load(data_path=tmp_path)
