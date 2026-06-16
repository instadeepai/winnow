"""Tests for PointNovoDatasetLoader."""

import pytest

from winnow.datasets.data_loaders import PointNovoDatasetLoader
from tests.datasets.data_loaders.conftest import _FULL_RESIDUE_MASSES


class TestPointNovoDatasetLoader:
    """Tests for PointNovoDatasetLoader (not yet implemented)."""

    @pytest.fixture()
    def loader(self):
        return PointNovoDatasetLoader(residue_masses=_FULL_RESIDUE_MASSES)

    def test_load_raises_not_implemented(self, loader, tmp_path):
        """load() must raise NotImplementedError unconditionally."""
        with pytest.raises(NotImplementedError):
            loader.load(data_path=tmp_path)
