from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from econml.dml import CausalForestDML
from intcf import CausalForest as OwnCF
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


class IntCF:
    name = "IntCF"

    def __init__(self, n_estimators=100, random_state=None):
        self.cf = OwnCF(
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def fit(self, X, T, y):
        self.cf.fit(X=X, T=T, y=y)
    
    def predict(self, X):
        return self.cf.predict(covariates=X)

    def feature_importances(self, heterogeneity=True, max_depth=None):
        if heterogeneity:
            importance = self.cf.feature_importances(method='heterogeneity', max_depth=max_depth)
        else:
            importance = self.cf.feature_importances(method='bias', max_depth=max_depth)
        return importance / importance[importance>=0].sum()

class IntCFunvalidated:
    name = "IntCF (without validation)"

    def __init__(self, n_estimators=100, random_state=None):
        self.cf = OwnCF(
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def fit(self, X, T, y):
        self.cf.fit(X=X, T=T, y=y, oob_reeval=False)
    
    def predict(self, X):
        return self.cf.predict(covariates=X)

    def feature_importances(self, heterogeneity=True, max_depth=None):
        if heterogeneity:
            importance = self.cf.feature_importances(method='heterogeneity', max_depth=max_depth)
        else:
            importance = self.cf.feature_importances(method='bias', max_depth=max_depth)
        return importance / importance[importance>=0].sum()

# class IntCFsplits:
#     name = "IntCF (splitwise distinction)"

#     def __init__(self, n_estimators=100, random_state=None):
#         self.cf = OwnCF(
#             n_estimators=n_estimators,
#             random_state=random_state,
#         )

#     def fit(self, X, T, y):
#         self.cf.fit(X=X, T=T, y=y)
    
#     def predict(self, X):
#         return self.cf.predict(covariates=X)

#     def feature_importances(self, heterogeneity=True, max_depth=None):
#         if heterogeneity:
#             importance = self.cf.feature_importances(method='heterogeneity splits', max_depth=max_depth)
#         else:
#             importance = self.cf.feature_importances(method='bias splits', max_depth=max_depth)
#         return importance / importance[importance>=0].sum()


# class EconmlGRF:
#     name = "GRF (econML)"

#     def __init__(self, n_estimators=100, random_state=None):
#         self.cf = PythonGRF(
#             n_estimators=n_estimators,
#             random_state=random_state,
#             n_jobs=1
#         )

#     def fit(self, X, T, y):
#         self.cf.fit(X=X, T=T, y=y)
    
#     def predict(self, X):
#         return self.cf.predict(X=X)[:, 0]

#     def feature_importances(self, heterogeneity=True, max_depth=None):
#         importance = self.cf.feature_importances(max_depth=max_depth)
#         return importance / importance.sum()

class EconmlForestDML:
    name = "Forest DML (econML)"

    def __init__(self, n_estimators=100, random_state=None):
        self.cf = CausalForestDML(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1
        )

    def fit(self, X, T, y):
        self.cf.fit(X=X, T=T, Y=y)
    
    def predict(self, X):
        return self.cf.effect(X=X)

    def feature_importances(self, heterogeneity=True, max_depth=None):
        # econML accepts max_depth=None
        importance = self.cf.feature_importances(
            max_depth=max_depth,
            depth_decay_exponent=0
        )
        return importance / importance.sum()


class GRF:
    name = "GRF"

    def __init__(self, n_estimators=100, random_state=None):
        self.num_trees = n_estimators
        self.seed = random_state if isinstance(random_state, int) else 42

        # Import necessary R packages
        self.base = importr("base")
        self.grf = importr("grf")
        self.stats = importr("stats")

    def fit(self, X, T, y):
        # Convert numpy arrays to pandas DataFrames
        X_train = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])

        # Convert pandas DataFrames to R dataframes
        with (ro.default_converter + pandas2ri.converter).context():
            r_X_train = ro.conversion.get_conversion().py2rpy(X_train)

        # Convert treatment and outcome to R vectors
        r_W_train = ro.FloatVector(T)
        r_Y_train = ro.FloatVector(y)

        # Set tuning parameters
        r_tune_params = ro.StrVector(["all"])
        r_num_trees = ro.IntVector([self.num_trees])
        r_honesty = ro.BoolVector([True])
        r_seed = ro.IntVector([self.seed])

        self.r_cf = self.grf.causal_forest(
            X=r_X_train,
            Y=r_Y_train,
            W=r_W_train,
            num_trees=r_num_trees,
            tune_parameters=r_tune_params,
            honesty=r_honesty,
            seed=r_seed,
        )
    
    def predict(self, X):
        X_test = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
        with (ro.default_converter + pandas2ri.converter).context():
            r_X_test = ro.conversion.get_conversion().py2rpy(X_test)
        # Predict CATE on test set
        r_predictions = self.grf.predict_causal_forest(self.r_cf, r_X_test)

        # Convert R predictions to numpy arrays
        predictions = np.array(r_predictions)
        return predictions

    def feature_importances(self, heterogeneity=True, max_depth=None):
        r_variable_importance = self.grf.variable_importance(
            self.r_cf,
            max_depth=1000 if max_depth is None else max_depth,
            decay_exponent=0
        )
        importance = np.array(r_variable_importance)
        return importance / importance.sum()


# class GRFzeroOrtho:
#     name = "GRF (zero orthogonalization)"

#     def __init__(self, n_estimators=100, random_state=None):
#         self.num_trees = n_estimators
#         self.seed = random_state if isinstance(random_state, int) else 42

#         # Import necessary R packages
#         self.base = importr("base")
#         self.grf = importr("grf")
#         self.stats = importr("stats")

#     def fit(self, X, T, y):
#         # Convert numpy arrays to pandas DataFrames
#         X_train = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])

#         # Convert pandas DataFrames to R dataframes
#         with (ro.default_converter + pandas2ri.converter).context():
#             r_X_train = ro.conversion.get_conversion().py2rpy(X_train)

#         # Convert treatment and outcome to R vectors
#         r_W_train = ro.FloatVector(T)
#         r_Y_train = ro.FloatVector(y)

#         # constant estimates to disable orthogonalization
#         r_W_hat = ro.FloatVector(np.zeros_like(T))
#         r_Y_hat = ro.FloatVector(np.zeros_like(y))

#         # Set tuning parameters
#         r_tune_params = ro.StrVector(["all"])
#         r_num_trees = ro.IntVector([self.num_trees])
#         r_honesty = ro.BoolVector([True])
#         r_seed = ro.IntVector([self.seed])

#         self.r_cf = self.grf.causal_forest(
#             X=r_X_train,
#             Y=r_Y_train,
#             W=r_W_train,
#             Y_hat=r_Y_hat,
#             W_hat=r_W_hat,
#             num_trees=r_num_trees,
#             tune_parameters=r_tune_params,
#             honesty=r_honesty,
#             seed=r_seed,
#         )
    
#     def predict(self, X):
#         X_test = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
#         with (ro.default_converter + pandas2ri.converter).context():
#             r_X_test = ro.conversion.get_conversion().py2rpy(X_test)
#         # Predict CATE on test set
#         r_predictions = self.grf.predict_causal_forest(self.r_cf, r_X_test)

#         # Convert R predictions to numpy arrays
#         predictions = np.array(r_predictions)
#         return predictions

#     def feature_importances(self, heterogeneity=True, max_depth=None):
#         r_variable_importance = self.grf.variable_importance(
#             self.r_cf,
#             max_depth=1000 if max_depth is None else max_depth
#         )
#         importance = np.array(r_variable_importance)
#         return importance / importance.sum()

class GRFconstOrtho:
    name = "GRF (constant orthogonalization)"

    def __init__(self, n_estimators=100, random_state=None):
        self.num_trees = n_estimators
        self.seed = random_state if isinstance(random_state, int) else 42

        # Import necessary R packages
        self.base = importr("base")
        self.grf = importr("grf")
        self.stats = importr("stats")

    def fit(self, X, T, y):
        # Convert numpy arrays to pandas DataFrames
        X_train = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])

        # Convert pandas DataFrames to R dataframes
        with (ro.default_converter + pandas2ri.converter).context():
            r_X_train = ro.conversion.get_conversion().py2rpy(X_train)

        # Convert treatment and outcome to R vectors
        r_W_train = ro.FloatVector(T)
        r_Y_train = ro.FloatVector(y)

        # constant estimates to disable orthogonalization
        r_W_hat = ro.FloatVector(np.zeros_like(T))
        r_Y_hat = ro.FloatVector(np.zeros_like(y))
        # r_W_hat = ro.FloatVector(np.mean(T) * np.ones_like(T))
        # r_Y_hat = ro.FloatVector(np.mean(y) * np.ones_like(y))

        # Set tuning parameters
        r_tune_params = ro.StrVector(["all"])
        r_num_trees = ro.IntVector([self.num_trees])
        r_honesty = ro.BoolVector([True])
        r_seed = ro.IntVector([self.seed])

        self.r_cf = self.grf.causal_forest(
            X=r_X_train,
            Y=r_Y_train,
            W=r_W_train,
            Y_hat=r_Y_hat,
            W_hat=r_W_hat,
            num_trees=r_num_trees,
            tune_parameters=r_tune_params,
            honesty=r_honesty,
            seed=r_seed,
        )
    
    def predict(self, X):
        X_test = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
        with (ro.default_converter + pandas2ri.converter).context():
            r_X_test = ro.conversion.get_conversion().py2rpy(X_test)
        # Predict CATE on test set
        r_predictions = self.grf.predict_causal_forest(self.r_cf, r_X_test)

        # Convert R predictions to numpy arrays
        predictions = np.array(r_predictions)
        return predictions

    def feature_importances(self, heterogeneity=True, max_depth=None):
        r_variable_importance = self.grf.variable_importance(
            self.r_cf,
            max_depth=1000 if max_depth is None else max_depth,
            decay_exponent=0
        )
        importance = np.array(r_variable_importance)
        return importance / importance.sum()


class CausalForest:
    name = "Causal Forest"

    def __init__(self, n_estimators=100, random_state=None):
        if random_state is not None:
            ro.r("set.seed")(random_state)
        self.num_trees = n_estimators
        self.causalTree = importr("causalTree")
        self.stats = importr("stats")

    def fit(self, X, T, y):
        _, p = X.shape
        self.covars_names = ["x" + str(i) for i in range(p)]
        data = pd.DataFrame(X, columns=self.covars_names)
        data["y"] = y
        formula = ro.Formula("y ~ " + " + ".join(self.covars_names))
        tree_sample_size = int(X.shape[0] * 0.5)
        with (ro.default_converter + pandas2ri.converter).context():
            r_data = ro.conversion.get_conversion().py2rpy(data)
        with redirect_stdout(None):  # surpress output of fitting process
            self.cf = self.causalTree.causalForest(
                formula=formula,
                data=r_data,
                treatment=ro.IntVector(T),
                num_trees=self.num_trees,
                ncolx=p,  # Number of covariates
                ncov_sample=p,  # Number of covariates sampled for each tree
                split_Rule="CT",
                split_Honest=True,
                cv_option="CT",
                cv_Honest=True,
                sample_size_total=tree_sample_size,
            )
    
    def predict(self, eval_covariates):
        with (ro.default_converter + pandas2ri.converter).context():
            r_eval = ro.conversion.get_conversion().py2rpy(pd.DataFrame(eval_covariates, columns=self.covars_names))
        with redirect_stdout(None):  # surpress output of predicting process
            r_predictions = self.stats.predict(self.cf, r_eval)
        predictions = np.array(r_predictions)
        return predictions

    def feature_importances(self, heterogeneity=True, max_depth=None):
        importance = pd.Series(0., index=self.covars_names)
        for ct in self.cf.rx2("trees"):
            tree_importance = ct.rx2('variable.importance')
            if ro.r("is.null")(tree_importance)[0]:
                continue
            importance[tree_importance.names] += np.array(tree_importance) / sum(tree_importance)
        importance = importance.to_numpy()
        return importance / importance.sum()
