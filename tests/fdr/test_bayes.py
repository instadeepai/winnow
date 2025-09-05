"""Unit tests for winnow EmpiricalBayesFDRControl."""

import pytest
from unittest.mock import patch, Mock
import jax.numpy as jnp
import pandas as pd
from winnow.fdr.bayes import EmpiricalBayesFDRControl, BetaMixtureParameters


class TestEmpiricalBayesFDRControl:
    """Test the EmpiricalBayesFDRControl class."""

    @pytest.fixture()
    def bayes_fdr_control(self):
        """Create an EmpiricalBayesFDRControl instance for testing."""
        return EmpiricalBayesFDRControl()

    @pytest.fixture()
    def sample_confidence_data(self):
        """Create sample confidence data for testing."""
        return jnp.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    def test_initialization(self, bayes_fdr_control):
        """Test EmpiricalBayesFDRControl initialization."""
        assert isinstance(bayes_fdr_control.mixture_parameters, BetaMixtureParameters)
        assert bayes_fdr_control.mixture_parameters.proportion == 0.0
        assert bayes_fdr_control.mixture_parameters.correct_alpha == 1.0
        assert bayes_fdr_control.mixture_parameters.correct_beta == 1.0

    def test_beta_mixture_parameters(self):
        """Test BetaMixtureParameters dataclass."""
        params = BetaMixtureParameters(
            proportion=jnp.array(0.7),
            correct_alpha=jnp.array(2.0),
            correct_beta=jnp.array(1.5),
            incorrect_alpha=jnp.array(1.0),
            incorrect_beta=jnp.array(3.0),
        )

        assert params.proportion == 0.7
        assert params.correct_alpha == 2.0
        assert params.correct_beta == 1.5
        assert params.incorrect_alpha == 1.0
        assert params.incorrect_beta == 3.0

    @patch("winnow.fdr.bayes.numpyro")
    def test_model_with_mocked_numpyro(self, mock_numpyro, sample_confidence_data):
        """Test model function with mocked numpyro."""
        # Mock numpyro functions
        mock_numpyro.param.return_value = jnp.array(0.5)
        mock_numpyro.distributions.constraints.unit_interval = "unit_interval"
        mock_numpyro.distributions.constraints.open_interval.return_value = (
            "open_interval"
        )
        mock_numpyro.distributions.Beta.return_value = Mock()
        mock_numpyro.distributions.MixtureSameFamily.return_value = Mock()
        mock_numpyro.sample.return_value = None

        # Should not raise an exception
        EmpiricalBayesFDRControl.model(sample_confidence_data)

        # Check that numpyro.param was called for all parameters
        assert mock_numpyro.param.call_count >= 4  # At least 4 parameters

    def test_fit_basic(self, bayes_fdr_control):
        """Test basic fitting functionality."""
        # Create sample data
        sample_data = pd.DataFrame(
            {"confidence": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}
        )

        bayes_fdr_control.fit(sample_data)

        # Check that the model has been fitted
        assert bayes_fdr_control._fitted

    def test_fit_with_parameters(self, bayes_fdr_control):
        """Test fit with custom parameters."""
        sample_data = pd.DataFrame(
            {"confidence": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}
        )

        bayes_fdr_control.fit(sample_data, lr=0.01, n_steps=1000)

        # Check that the model has been fitted
        assert bayes_fdr_control._fitted

    def test_fit_with_empty_data(self, bayes_fdr_control):
        """Test that fit method handles empty data."""
        empty_data = jnp.array([])
        with pytest.raises(AssertionError, match="Fit method requires non-empty data"):
            bayes_fdr_control.fit(empty_data)

    def test_get_confidence_cutoff_requires_fitting(self, bayes_fdr_control):
        """Test that get_confidence_cutoff requires fitting first."""
        # Should raise AttributeError if not fitted
        with pytest.raises(
            AttributeError, match=r"FDR method not fitted, please call `fit\(\)` first"
        ):
            bayes_fdr_control.get_confidence_cutoff(0.05)

    def test_compute_fdr_requires_fitting(self, bayes_fdr_control):
        """Test that compute_fdr requires fitting first."""
        # Should raise AttributeError if not fitted
        with pytest.raises(
            AttributeError, match=r"FDR method not fitted, please call `fit\(\)` first"
        ):
            bayes_fdr_control.compute_fdr(0.8)

    def test_mixture_parameters_modification(self, bayes_fdr_control):
        """Test that mixture parameters can be modified."""
        new_params = BetaMixtureParameters(
            proportion=jnp.array(0.6),
            correct_alpha=jnp.array(3.0),
            correct_beta=jnp.array(2.0),
            incorrect_alpha=jnp.array(1.5),
            incorrect_beta=jnp.array(4.0),
        )

        bayes_fdr_control.mixture_parameters = new_params

        assert bayes_fdr_control.mixture_parameters.proportion == 0.6
        assert bayes_fdr_control.mixture_parameters.correct_alpha == 3.0
