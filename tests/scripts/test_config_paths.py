"""Tests for config path resolution utilities."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from winnow.utils.config_path import (
    get_config_dir,
    get_config_search_path,
    get_primary_config_dir,
)


class TestGetConfigDir:
    """Tests for get_config_dir() function."""

    def test_package_mode(self, tmp_path):
        """Test config dir resolution in package mode."""
        # Mock importlib.resources to simulate installed package
        package_configs = tmp_path / "package_configs"
        package_configs.mkdir()

        # Create a mock that simulates files("winnow").joinpath("configs")
        mock_winnow_files = MagicMock()
        mock_configs = MagicMock()
        mock_configs.is_dir.return_value = True
        # Make str() return the path
        type(mock_configs).__str__ = lambda self: str(package_configs)
        mock_winnow_files.joinpath.return_value = mock_configs

        # Mock files("winnow") to return mock_winnow_files
        mock_files = MagicMock(return_value=mock_winnow_files)

        # Patch importlib.resources.files (where it's imported from)
        with patch("importlib.resources.files", mock_files):
            config_dir = get_config_dir()
            assert config_dir == package_configs

    def test_dev_mode(self, tmp_path):
        """Test config dir resolution in dev mode."""
        # Mock importlib.resources to fail (simulating dev mode)
        with patch("importlib.resources.files", side_effect=ModuleNotFoundError()):
            # Create a mock repo structure
            repo_root = tmp_path / "repo"
            winnow_dir = repo_root / "winnow"
            configs_dir = winnow_dir / "configs"
            configs_dir.mkdir(parents=True)

            # Mock __file__ to point to winnow/scripts/config_paths.py
            with patch(
                "winnow.scripts.config_path.__file__",
                str(winnow_dir / "scripts" / "config_paths.py"),
            ):
                config_dir = get_config_dir()
                assert config_dir == configs_dir

    def test_dev_mode_alt_location(self, tmp_path):
        """Test config dir resolution in dev mode with configs at repo root."""
        # Mock importlib.resources to fail
        with patch("importlib.resources.files", side_effect=ModuleNotFoundError()):
            # Create a mock repo structure with configs at root
            repo_root = tmp_path / "repo"
            configs_dir = repo_root / "configs"
            configs_dir.mkdir(parents=True)

            # Mock __file__ to point to winnow/scripts/config_paths.py
            with patch(
                "winnow.scripts.config_path.__file__",
                str(repo_root / "winnow" / "scripts" / "config_paths.py"),
            ):
                config_dir = get_config_dir()
                assert config_dir == configs_dir

    def test_not_found(self):
        """Test error when config dir cannot be found."""
        with patch("importlib.resources.files", side_effect=ModuleNotFoundError()):
            with patch("winnow.scripts.config_path.__file__", "/nonexistent/path"):
                with pytest.raises(FileNotFoundError):
                    get_config_dir()


class TestGetConfigSearchPath:
    """Tests for get_config_search_path() function."""

    def test_custom_dir_only(self, tmp_path):
        """Test search path with custom directory."""
        custom_dir = tmp_path / "custom_configs"
        custom_dir.mkdir()

        with patch(
            "winnow.scripts.config_path.get_config_dir",
            return_value=Path("/package/configs"),
        ):
            search_path = get_config_search_path(str(custom_dir))
            assert len(search_path) == 2
            assert search_path[0] == custom_dir.resolve()
            assert search_path[1] == Path("/package/configs").resolve()

    def test_no_custom_dir(self, tmp_path):
        """Test search path without custom directory."""
        package_dir = tmp_path / "package_configs"
        package_dir.mkdir()

        with patch(
            "winnow.scripts.config_path.get_config_dir", return_value=package_dir
        ):
            search_path = get_config_search_path()
            assert len(search_path) == 1
            assert search_path[0] == package_dir.resolve()

    def test_custom_dir_not_exists(self):
        """Test error when custom directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            get_config_search_path("/nonexistent/path")

    def test_custom_dir_not_directory(self, tmp_path):
        """Test error when custom path is not a directory."""
        file_path = tmp_path / "not_a_dir"
        file_path.touch()

        with pytest.raises(ValueError, match="not a directory"):
            get_config_search_path(str(file_path))


class TestGetPrimaryConfigDir:
    """Tests for get_primary_config_dir() function."""

    def test_no_custom_dir(self, tmp_path):
        """Test primary config dir without custom directory."""
        package_dir = tmp_path / "package_configs"
        package_dir.mkdir()

        with patch(
            "winnow.scripts.config_path.get_config_dir", return_value=package_dir
        ):
            primary_dir = get_primary_config_dir()
            assert primary_dir == package_dir.resolve()

    def test_with_custom_dir(self, tmp_path):
        """Test primary config dir with custom directory (merged)."""
        custom_dir = tmp_path / "custom_configs"
        custom_dir.mkdir()
        (custom_dir / "residues.yaml").write_text("custom: true")

        package_dir = tmp_path / "package_configs"
        package_dir.mkdir()
        (package_dir / "train.yaml").write_text("package: true")
        (package_dir / "residues.yaml").write_text("package: true")

        with patch(
            "winnow.scripts.config_path.get_config_dir", return_value=package_dir
        ):
            primary_dir = get_primary_config_dir(str(custom_dir))

            # Should be a temporary merged directory
            assert primary_dir.exists()
            assert primary_dir.is_dir()

            # Custom config should override package config
            residues_content = (primary_dir / "residues.yaml").read_text()
            assert "custom: true" in residues_content

            # Package config should be available for files not in custom dir
            assert (primary_dir / "train.yaml").exists()
            train_content = (primary_dir / "train.yaml").read_text()
            assert "package: true" in train_content

    def test_partial_configs(self, tmp_path):
        """Test that partial configs work (only custom residues.yaml)."""
        custom_dir = tmp_path / "custom_configs"
        custom_dir.mkdir()
        (custom_dir / "residues.yaml").write_text("custom_residues: true")

        package_dir = tmp_path / "package_configs"
        package_dir.mkdir()
        (package_dir / "train.yaml").write_text("train_config: true")
        (package_dir / "residues.yaml").write_text("package_residues: true")
        (package_dir / "calibrator.yaml").write_text("calibrator_config: true")

        with patch(
            "winnow.scripts.config_path.get_config_dir", return_value=package_dir
        ):
            primary_dir = get_primary_config_dir(str(custom_dir))

            # Custom residues should override
            residues_content = (primary_dir / "residues.yaml").read_text()
            assert "custom_residues" in residues_content
            assert "package_residues" not in residues_content

            # Package files not in custom dir should be available
            assert (primary_dir / "train.yaml").exists()
            assert (primary_dir / "calibrator.yaml").exists()

            train_content = (primary_dir / "train.yaml").read_text()
            assert "train_config" in train_content
