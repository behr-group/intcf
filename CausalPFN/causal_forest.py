from contextlib import redirect_stdout

import numpy as np
import pandas as pd

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
except ImportError:
    print("rpy2 not installed, skipping CausalForest baseline.")

from .base import BaselineModel


class CausalForestBaseline(BaselineModel):
    def __init__(self, hpo: bool = True):
        super().__init__(hpo)

        # Import necessary R packages
        self.base = importr("base")
        self.causalTree = importr("causalTree")
        self.stats = importr("stats")

    def estimate_cate(
        self,
        X_train: np.ndarray,
        t_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ):
        """
        Estimate CATE using R's grf package directly from Python using rpy2.

        Args:
            X_train: Training features
            t_train: Binary treatment indicators for training data
            y_train: Outcome values for training data
            X_test: Test features for which to predict CATE
        Returns:
            cate_estimates: Estimated CATE values
        """
        r_ncov = X_train.shape[1]
        column_names = [f"X{i+1}" for i in range(r_ncov)]
        # Convert numpy arrays to pandas DataFrames
        X_train = pd.DataFrame(X_train, columns=column_names, dtype=float)
        X_test = pd.DataFrame(X_test, columns=column_names, dtype=float)

        # standardize the features
        X_mean, X_std = X_train.mean(), X_train.std() + 1e-8
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        # standardize the outcomes
        y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
        y_train = (y_train - y_mean) / y_std

        # attach outcome to training data
        X_train['y'] = y_train

        # Convert pandas DataFrames to R dataframes
        with (ro.default_converter + pandas2ri.converter).context():
            r_data_train = ro.conversion.get_conversion().py2rpy(X_train)
            r_X_test = ro.conversion.get_conversion().py2rpy(X_test)

        # Convert treatment and outcome to R vectors
        r_W_train = ro.IntVector(t_train)
        # r_Y_train = ro.FloatVector(y_train)

        # Set parameters for CausalForest
        r_formula = ro.Formula("y ~ " + " + ".join(column_names))
        r_num_trees = 100
        tree_sample_size = int(X_train.shape[0] * 0.5)

        # Call R's causal_forest
        with redirect_stdout(None):
            r_cf = self.causalTree.causalForest(
                formula=r_formula,
                data=r_data_train,
                treatment=r_W_train,
                num_trees=r_num_trees,
                ncolx=r_ncov,  # Number of covariates
                ncov_sample=r_ncov,  # Number of covariates sampled for each tree
                split_Rule="CT",
                split_Honest=True,
                cv_option="CT",
                cv_Honest=True,
                sample_size_total=tree_sample_size,
            )

        # Predict CATE on test set
        with redirect_stdout(None):
            r_predictions = self.stats.predict(r_cf, r_X_test)

        # Convert R predictions to numpy arrays
        cate_estimates = np.array(r_predictions)

        return cate_estimates * y_std

    def estimate_ate(
        self,
        X: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
    ):
        """
        Estimate ATE using R's grf package directly from Python using rpy2.
        Args:
            X: Features
            t: Binary treatment indicators
            y: Outcome values
        Returns:
            ate_estimate: Estimated ATE value
        """
        r_ncov = X.shape[1]
        column_names = [f"X{i+1}" for i in range(r_ncov)]
        # Convert numpy arrays to pandas DataFrames
        X = pd.DataFrame(X, columns=column_names)

        # standardize the features
        X_mean, X_std = X.mean(), X.std() + 1e-8
        X = (X - X_mean) / X_std

        # standardize the outcomes
        y_mean, y_std = y.mean(), y.std() + 1e-8
        y = (y - y_mean) / y_std

        # attach outcome to training data
        X_train = X.copy()
        X_train['y'] = y

        # Convert pandas DataFrames to R dataframes
        with (ro.default_converter + pandas2ri.converter).context():
            r_data_train = ro.conversion.get_conversion().py2rpy(X_train)
            r_X_predict = ro.conversion.get_conversion().py2rpy(X)

        # Convert treatment and outcome to R vectors
        r_W_train = ro.IntVector(t)

        # Set parameters for CausalForest
        r_formula = ro.Formula("y ~ " + " + ".join(column_names))
        r_num_trees = 100
        tree_sample_size = int(X.shape[0] * 0.5)

        # Call R's causal_forest
        with redirect_stdout(None):
            r_cf = self.causalTree.causalForest(
                formula=r_formula,
                data=r_data_train,
                treatment=r_W_train,
                num_trees=r_num_trees,
                ncolx=r_ncov,  # Number of covariates
                ncov_sample=r_ncov,  # Number of covariates sampled for each tree
                split_Rule="CT",
                split_Honest=True,
                cv_option="CT",
                cv_Honest=True,
                sample_size_total=tree_sample_size,
            )

        # Predict CATE on test set
        with redirect_stdout(None):
            r_predictions = self.stats.predict(r_cf, r_X_predict)

        # Convert R predictions to numpy arrays
        cate_estimates = np.array(r_predictions)

        return np.mean(cate_estimates) * y_std
