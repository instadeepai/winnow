"""Unit tests for winnow.utils.paths."""

import logging
from unittest.mock import patch

import pytest

from winnow.utils.paths import resolve_data_path


class TestResolveDataPath:
    """Test the resolve_data_path function."""

    def test_local_file(self, tmp_path):
        """Test resolution of an existing local file."""
        f = tmp_path / "data.parquet"
        f.touch()
        result = resolve_data_path(str(f))
        assert result == f.resolve()

    def test_local_directory(self, tmp_path):
        """Test resolution of an existing local directory."""
        d = tmp_path / "model_dir"
        d.mkdir()
        result = resolve_data_path(str(d))
        assert result == d.resolve()

    def test_nonexistent_local_raises(self):
        """Test that a nonexistent local path that is not an HF ID raises."""
        with pytest.raises(FileNotFoundError, match="Could not resolve"):
            resolve_data_path("/definitely/does/not/exist/anywhere")

    def test_relative_local_path(self, tmp_path, monkeypatch):
        """Test resolution of a relative local path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "local_data").mkdir()
        result = resolve_data_path("local_data")
        assert result == (tmp_path / "local_data").resolve()

    def test_hf_download_fallback(self, tmp_path):
        """Test that HF snapshot_download is called when path doesn't exist locally."""
        fake_download_dir = tmp_path / "hf_cache" / "models--org--repo"
        fake_download_dir.mkdir(parents=True)

        with patch(
            "winnow.utils.paths.snapshot_download",
            return_value=str(fake_download_dir),
        ) as mock_dl:
            result = resolve_data_path(
                "org/repo",
                repo_type="model",
                cache_dir=tmp_path / "hf_cache",
            )

        mock_dl.assert_called_once_with(
            repo_id="org/repo",
            repo_type="model",
            cache_dir=str(tmp_path / "hf_cache"),
        )
        assert result == fake_download_dir

    def test_hf_download_failure_raises_with_context(self):
        """Test that HF download failure wraps the original error."""
        with patch(
            "winnow.utils.paths.snapshot_download",
            side_effect=Exception("401 Unauthorized"),
        ):
            with pytest.raises(FileNotFoundError, match="401 Unauthorized"):
                resolve_data_path("org/nonexistent-model")

    def test_local_path_resembling_hf_warns(self, tmp_path, caplog):
        """Test warning when a local path looks like an HF repo ID."""
        hf_like = tmp_path / "org" / "model"
        hf_like.mkdir(parents=True)

        with caplog.at_level(logging.WARNING):
            result = resolve_data_path(str(hf_like))

        assert result == hf_like.resolve()

    def test_local_relative_hf_like_path_warns(self, tmp_path, monkeypatch, caplog):
        """Test warning when a relative path like 'org/repo' exists locally."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "myorg" / "mymodel").mkdir(parents=True)

        with caplog.at_level(logging.WARNING):
            result = resolve_data_path("myorg/mymodel")

        assert result == (tmp_path / "myorg" / "mymodel").resolve()
        assert any("HF repo ID" in rec.message for rec in caplog.records)

    def test_cache_dir_none_passes_none(self, tmp_path):
        """Test that cache_dir=None passes None to snapshot_download."""
        with patch(
            "winnow.utils.paths.snapshot_download",
            return_value=str(tmp_path),
        ) as mock_dl:
            resolve_data_path("org/repo", cache_dir=None)

        mock_dl.assert_called_once_with(
            repo_id="org/repo",
            repo_type="model",
            cache_dir=None,
        )
