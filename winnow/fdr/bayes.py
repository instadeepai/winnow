from dataclasses import dataclass

from scipy import optimize

import jax
import jax.numpy as jnp
import numpyro
from jaxtyping import Float
from winnow.fdr.base import FDRControl


EPS = 1e-7


@dataclass
class BetaMixtureParameters:
    """Holds the parameters for a beta mixture model of PSM confidence probabilities."""

    proportion: Float[jax.Array, ""]
    correct_alpha: Float[jax.Array, ""]
    correct_beta: Float[jax.Array, ""]
    incorrect_alpha: Float[jax.Array, ""]
    incorrect_beta: Float[jax.Array, ""]


class EmpiricalBayesFDRControl(FDRControl):
    """An FDR control method that fits an empirical Bayesian model to confidence probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.mixture_parameters = BetaMixtureParameters(
            proportion=jnp.array(0.0),
            correct_alpha=jnp.array(1.0),
            correct_beta=jnp.array(1.0),
            incorrect_alpha=jnp.array(1.0),
            incorrect_beta=jnp.array(1.0),
        )

    @staticmethod
    def model(data: Float[jax.Array, "batch"]) -> None:  # noqa: F821
        """Define the probabilistic model for the beta mixture distribution.

        The model consists of a mixture of two Beta distributions:
        - One representing correctly identified PSMs.
        - One representing incorrectly identified PSMs.

        The proportion of correct PSMs is treated as a parameter to be estimated.

        Args:
            data (Float[jax.Array, "batch"]): An array of confidence scores from the dataset.
        """
        proportion = numpyro.param(
            "proportion",
            init_value=0.5,
            constraints=numpyro.distributions.constraints.unit_interval,
        )
        correct_alpha = numpyro.param(
            "correct_alpha",
            init_value=0.7,
            constraints=numpyro.distributions.constraints.open_interval(0.0, 1.0),
        )
        correct_beta = numpyro.param(
            "correct_beta",
            init_value=0.5,
            constraints=numpyro.distributions.constraints.open_interval(0.0, 1.0),
        )
        incorrect_alpha = numpyro.param(
            "incorrect_alpha",
            init_value=0.4,
            constraints=numpyro.distributions.constraints.open_interval(0.0, 1.0),
        )
        incorrect_beta = numpyro.param(
            "incorrect_beta",
            init_value=0.6,
            constraints=numpyro.distributions.constraints.open_interval(0.0, 1.0),
        )

        mixture_distributions = [
            numpyro.distributions.Beta(
                concentration0=correct_alpha, concentration1=correct_beta
            ),
            numpyro.distributions.Beta(
                concentration0=incorrect_alpha, concentration1=incorrect_beta
            ),
        ]

        with numpyro.plate("N", data.shape[0]):
            numpyro.sample(
                "y",
                numpyro.distributions.MixtureGeneral(
                    mixing_distribution=numpyro.distributions.Categorical(
                        probs=jnp.array([proportion, 1 - proportion])
                    ),
                    component_distributions=mixture_distributions,
                ),
                obs=data,
            )

    @staticmethod
    def guide(data: Float[jax.Array, "batch"]) -> None:  # noqa: F821
        """Define the variational guide for approximating the posterior distribution.

        The guide defines a variational family of distributions that approximate the true
        posterior for stochastic variational inference (SVI).
        """
        pass

    def fit(
        self,
        dataset: Float[jax.Array, "batch"],  # noqa: F821
        lr: float = 0.005,
        n_steps: int = 5000,
    ) -> None:
        """Estimate the beta mixture model parameters using stochastic variational inference.

        Args:
            dataset (Float[jax.Array, "batch"]):
                An array of confidence scores from the dataset.

            lr (float, optional):
                Learning rate for the optimiser. Defaults to 0.005.

            n_steps (int, optional):
                Number of optimisation steps for training. Defaults to 5000.
        """
        self.mixture_parameters = self.train(
            examples=jnp.clip(jnp.array(dataset), EPS, 1 - EPS), lr=lr, n_steps=n_steps
        )

    def train(
        self,
        examples: Float[jax.Array, "batch"],  # noqa: F821
        lr: float = 0.005,
        n_steps: int = 5000,
    ) -> BetaMixtureParameters:
        """Perform maximum likelihood estimation (MLE) to fit the beta mixture model.

        This function optimizes the parameters of the beta mixture model by minimising the Evidence Lower Bound (ELBO) using stochastic variational inference (SVI).

        Args:
            examples (Float[jax.Array, "batch"]):
                The confidence probabilities to fit the mixture model to.

            lr (float, optional):
                Optimiser learning. Defaults to 0.005.

            n_steps (int, optional):
                Number of optimisation steps. Defaults to 5000.

        Returns:
            BetaMixtureParameters:
                Estimated mixture model parameters.
        """
        adam = numpyro.optim.Adam(step_size=lr)
        svi = numpyro.infer.SVI(
            self.model, self.guide, adam, loss=numpyro.infer.Trace_ELBO()
        )
        rng_key = jax.random.PRNGKey(0)
        svi_state = svi.run(rng_key, n_steps, examples)
        return BetaMixtureParameters(
            proportion=svi_state.params["proportion"],
            correct_alpha=svi_state.params["correct_alpha"],
            correct_beta=svi_state.params["correct_beta"],
            incorrect_alpha=svi_state.params["incorrect_alpha"],
            incorrect_beta=svi_state.params["incorrect_beta"],
        )

    def get_confidence_cutoff(self, threshold: float) -> float:
        """Compute the confidence score cutoff for a given FDR threshold.

        This function determines the confidence score above which PSMs should be retained
        to maintain the desired FDR level.

        Args:
            threshold (float):
                The target FDR threshold, where 0 < threshold < 1.

        Returns:
            float:
                The confidence score cutoff corresponding to the specified FDR level.
        """
        return optimize.bisect(lambda cutoff: self.compute_fdr(cutoff) - threshold, 0., 1.) 
    
    def compute_fdr(self, score: float) -> float:
        # P(S >= score | incorrect) = 1 - F_incorrect(s)
        P_score_given_incorrect = 1 - jax.scipy.stats.beta.cdf(
            a=self.mixture_parameters.incorrect_alpha,
            b=self.mixture_parameters.incorrect_beta,
            x=score
            )
        # P(S >= score | correct) = 1 - F_correct(s)
        P_score_given_correct = 1 - jax.scipy.stats.beta.cdf(
            a=self.mixture_parameters.correct_alpha,
            b=self.mixture_parameters.correct_beta,
            x=score
            )

        # Mixture tail probability P(S >= s)
        P_mixture_tail = (
            self.mixture_parameters.proportion * P_score_given_correct +
            (1 - self.mixture_parameters.proportion) * P_score_given_incorrect
        )

        # P(incorrect | S >= s)
        P_incorrect_given_score = (1 - self.mixture_parameters.proportion) * P_score_given_incorrect / P_mixture_tail

        return P_incorrect_given_score.item()

    def compute_posterior_probability(self, score: float) -> float:
        """Compute posterior error probability, or local FDR, for a given confidence score."""
        # P(incorrect | S = s) = [P(incorrect) * P(S = s | incorrect)] / P(S = s)

        # f(S = s | incorrect)
        f_score_given_incorrect = jax.scipy.stats.beta.pdf(
            a=self.mixture_parameters.incorrect_alpha,
            b=self.mixture_parameters.incorrect_beta,
            x=score
        )
        # f(S = s | correct)
        f_score_given_correct = jax.scipy.stats.beta.pdf(
            a=self.mixture_parameters.correct_alpha,
            b=self.mixture_parameters.correct_beta,
            x=score
        )

        # Mixture probability f(S = s)
        f_score = (
            self.mixture_parameters.proportion * f_score_given_correct +
            (1 - self.mixture_parameters.proportion) * f_score_given_incorrect
        )
        
        return ((1 - self.mixture_parameters.proportion) * f_score_given_incorrect / f_score).item()

    def compute_p_value(self, score: float) -> float:
        """Compute the p-value for a given confidence score."""
        # P(S >= score | incorrect) = 1 - F_incorrect(s)
        P_score_given_incorrect = 1 - jax.scipy.stats.beta.cdf(
            a=self.mixture_parameters.incorrect_alpha,
            b=self.mixture_parameters.incorrect_beta,
            x=score
            )
        return P_score_given_incorrect.item()

    def compute_expect_score(self, score: float, total_matches: int) -> float:
        """Compute the expected number of false discoveries for a given score."""
        p_value = self.compute_p_value(score)
        return (total_matches * p_value).item()
