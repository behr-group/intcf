import numpy as np


def data_models(s):
    def confounding_model(
        num_samples: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Heterogenous treatment effect depending on single covariate."""
        # constant parameters of the model
        p = 5 * s
        SNR = 1.0
        gamma0 = 0.5
        # covariates
        covariates = rng.uniform(0, 1, (num_samples, p))
        # model functions
        propensity = 0.2 + 0.6/s * sum((covariates[:, s+i] <= gamma0).astype(float) for i in range(s))
        main_effect = sum((covariates[:, s+i] <= gamma0).astype(float) for i in range(s))
        treatment_effect = sum((covariates[:, i] <= gamma0).astype(float) for i in range(s))
        # treatment and outcomes according to model
        treatment = rng.binomial(1, propensity, num_samples).astype(bool)
        outcomes = (
            main_effect + (treatment - 1 / 2) * treatment_effect
            + rng.normal(0, SNR * np.std(treatment_effect), num_samples)
        )
        return covariates, treatment, outcomes, treatment_effect
    confounding_model.name = "LSS confounding"
    confounding_model.heterogeneity_features = list(range(s))
    confounding_model.confounder = list(range(s, 2*s))

    def interaction_model(
        num_samples: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Heterogenous treatment effect depending on single covariate."""
        # constant parameters of the model
        p = 5 * s
        SNR = 1.0
        gamma0 = 0.7
        # covariates
        covariates = rng.uniform(0, 1, (num_samples, p))
        # model functions
        propensity = 0.5 * np.ones(num_samples)
        main_effect = 0 * np.ones(num_samples)
        treatment_effect = sum((
            (covariates[:, 2*i] <= gamma0)*(covariates[:, 2*i+1] <= gamma0)
        ).astype(float) for i in range(s))
        # treatment and outcomes according to model
        treatment = rng.binomial(1, propensity, num_samples).astype(bool)
        outcomes = (
            main_effect + (treatment - 1 / 2) * treatment_effect
            + rng.normal(0, SNR * np.std(treatment_effect), num_samples)
        )
        return covariates, treatment, outcomes, treatment_effect
    interaction_model.name = "Interaction"
    interaction_model.heterogeneity_features = list(range(2*s))
    interaction_model.confounder = []

    def linear_model(
        num_samples: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Heterogenous treatment effect depending on single covariate."""
        # constant parameters of the model
        p = 5 * s
        SNR = 1.0
        # covariates
        covariates = rng.uniform(0, 1, (num_samples, p))
        # model functions
        propensity = 0.5 * np.sum(covariates[:, 0:s], axis=1)/s + 0.25
        main_effect = np.sum(covariates[:, 0:s], axis=1)
        treatment_effect = np.sum(covariates[:, s:2*s], axis=1)
        # treatment and outcomes according to model
        treatment = rng.binomial(1, propensity, num_samples).astype(bool)
        outcomes = (
            main_effect + (treatment - 1 / 2) * treatment_effect
            + rng.normal(0, SNR * np.std(treatment_effect), num_samples)
        )
        return covariates, treatment, outcomes, treatment_effect
    linear_model.name = "Linear"
    linear_model.heterogeneity_features = list(range(s, 2*s))
    linear_model.confounder = list(range(s))

    def mixed_model(
        num_samples: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Heterogenous treatment effect depending on single covariate."""
        # constant parameters of the model
        # constant parameters of the model
        p = 5 * s
        SNR = 1.0
        gamma0 = 0.7
        # covariates
        covariates = rng.uniform(0, 1, (num_samples, p))
        # model functions
        propensity = 0.2 + 0.6 * sum((covariates[:, 2*s+i] <= gamma0).astype(float) for i in range(s))/s
        main_effect = np.sum(covariates[:, 2*s:3*s], axis=1)
        treatment_effect = (
            np.sum(covariates[:, s:2*s], axis=1)/s
            + sum((
                (covariates[:, 2*i] <= gamma0)*(covariates[:, 2*i+1] <= gamma0)
            ).astype(float) for i in range(s))
        )
        # treatment and outcomes according to model
        treatment = rng.binomial(1, propensity, num_samples).astype(bool)
        outcomes = (
            main_effect + (treatment - 1 / 2) * treatment_effect
            + rng.normal(0, SNR * np.std(treatment_effect), num_samples)
        )
        return covariates, treatment, outcomes, treatment_effect
    mixed_model.name = "Mixed"
    mixed_model.heterogeneity_features = list(range(2*s))
    mixed_model.confounder = list(range(2*s, 3*s))

    return [interaction_model, confounding_model, linear_model, mixed_model]
