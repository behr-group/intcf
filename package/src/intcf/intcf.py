"""Prototypes for Causal Tree and Causal Forest with two splitting criteria."""

import numpy as np


class _Node:
    def __init__(
        self, covariates: np.ndarray, treatment: np.ndarray, outcomes: np.ndarray
    ) -> None:
        self.is_leaf = True
        self.covariates = covariates
        self.num_samples = covariates.shape[0]
        self.treatment = treatment
        self.outcomes = outcomes
        self.treatment_effect: float = outcomes[treatment].mean() - outcomes[~treatment].mean()

    def create_split(
        self,
        total_num_samples: int,
        min_impurity_decrease: float = 1e-10,
        max_features: float | None = None,
        random_state: int
        | np.random.SeedSequence
        | np.random.BitGenerator
        | np.random.Generator
        | None = None,
        splitting_method: str = "greater",
    ) -> bool:
        """Split the node with the best split found.

        Parameters
        ----------
        total_num_samples : int
            Number of training samples in root node.
        min_impurity_decrease : float, (optional, default=1e-10)
            Smallest value for the splitting rules to be considered for a split. If only
            splits with a smaller value are found, no split will be created.
        max_features : int | float | None, (optional, default=None)
            Number of features to consider for a split. If None, all features will be
            used. If int, the given number of features will be considered. If float,
            `int(max_features * num_features)` will be tested.
        random_state : int | np.random.SeedSequence | np.random.BitGenerator
        | np.random.Generator | None, (optional, default=None)
            Random state to use for selecting and ordering covariates for consideration
            when splitting. Will be used by `np.random.default_rng`.
        splitting_method: str, (optional, default="greater")
            Either "greater" or "confounding". If "greater", the split with the greatest
            value will be used. If "confounding", all covariates will be checked (so
            `max_features` will be ignored), if their best split is based on bias
            criterion with a value of at least `min_impurity_decrease`. If there is any,
            the best of these will be used. Only otherwise the best heterogenity split
            will be used.

        Returns
        -------
        bool
            Whether a split was created.

        Raises
        ------
        ValueError
            Raised if the node is already split.
        """
        if not self.is_leaf:
            raise ValueError("Tried to split a node which is already split")
        if (splitting_method == "confounding") or (max_features is None):
            max_features = self.covariates.shape[1]
        elif isinstance(max_features, float):
            max_features = max(1, int(max_features * self.covariates.shape[1]))
        rng = np.random.default_rng(random_state)
        permutated_covars = rng.choice(self.covariates.shape[1], self.covariates.shape[1], replace=False)
        best_split_var = (None, None)  # pair of variable and split value
        criteria = (0.0, 0.0)  # heterogenity and confounding criterion at split
        best_split_type_var = True  # Whether best split found splits on variance
        for i, split_var in enumerate(permutated_covars):
            value_het, value_conf, splitting_value = self._find_best_split(split_var)
            if splitting_method == "confounding":
                if best_split_type_var and (value_het <= value_conf) and value_conf >= min_impurity_decrease:
                    # first bias reducing split was found and has sufficient value
                    best_split_var = (split_var, splitting_value)
                    criteria = (value_het, value_conf)
                    best_split_type_var = False
                if best_split_type_var and (value_het > value_conf) and value_het > criteria[0]:
                    # both splits are for heterogenity
                    best_split_var = (split_var, splitting_value)
                    criteria = (value_het, value_conf)
                    best_split_type_var = True
                if not best_split_type_var and (value_het <= value_conf) and value_conf > criteria[1]:
                    # both splits are for confounding
                    best_split_var = (split_var, splitting_value)
                    criteria = (value_het, value_conf)
                    best_split_type_var = False
            elif max(value_het, value_conf) > max(criteria):
                # using the best split regardless of type
                best_split_var = (split_var, splitting_value)
                criteria = (value_het, value_conf)
                best_split_type_var = (value_het > value_conf)
            if i >= max_features and max(criteria) >= min_impurity_decrease * total_num_samples:
                break
        # cannot use else here, because if max_features is too high, break would not be
        # reached, but a sufficient split could have been found
        if max(criteria) < min_impurity_decrease * total_num_samples:
            return False
        self.split_type_var = best_split_type_var
        self.split_variable, self.split_value = best_split_var
        self.criteria = (criteria[0] / total_num_samples, criteria[1] / total_num_samples)
        left_samples = self.covariates[:, self.split_variable] <= self.split_value
        # TODO: The following part creates copies of the arrays. That is not necessary
        # and probably memory and time extensive. Can it be optimized, for example by
        # storing only the indices in the nodes and the data-array only once in tree?
        self.left_child = _Node(
            self.covariates[left_samples],
            self.treatment[left_samples],
            self.outcomes[left_samples],
        )
        self.right_child = _Node(
            self.covariates[~left_samples],
            self.treatment[~left_samples],
            self.outcomes[~left_samples],
        )
        self.is_leaf = False
        return True

    def _find_best_split(
        self, split_var: int, min_samples: int = 2, min_group_samples: int = 1
    ) -> tuple[bool, float, float | None]:
        """Find the split along the current variable.

        Parameters
        ----------
        split_var : int
            Variable to find the best split value for.
        min_samples : int, (optional, default=2)
            Minimal number of samples in each child node.
        min_group_samples : int, (optional, default=1)
            Minimal number of samples for each treatment group in both child nodes.
            Has to be at least 1 to function correctly.

        Returns
        -------
        float
            Value of the heterogenity criterion for the found split.
        float
            Value of the confounding criterion for the found split.
        float | None
            Value of the splitting location. If None, no split was found and the return
            tuple is `(0.0, 0.0, None)`.
        """
        args_sorted = np.argsort(self.covariates[:, split_var])
        num_treated_left = self.treatment[args_sorted].cumsum()
        sum_outcome_treated_left = (self.outcomes * self.treatment)[args_sorted].cumsum()
        sum_outcome_untreated_left = (self.outcomes * (1 - self.treatment))[args_sorted].cumsum()
        num_samples_left = np.arange(1, self.num_samples + 1)
        num_samples_right = self.num_samples - num_samples_left

        # TODO: np.maximum(..., 1) to prevent division by 0. Affected entries will later
        # be overwritten, so they don't really matter. Is there a nicer solution?
        left_te = sum_outcome_treated_left / np.maximum(
            num_treated_left, 1
        ) - sum_outcome_untreated_left / np.maximum(num_samples_left - num_treated_left, 1)
        right_te: np.ndarray = (
            sum_outcome_treated_left[-1] - sum_outcome_treated_left
        ) / np.maximum(num_treated_left[-1] - num_treated_left, 1) - (
            sum_outcome_untreated_left[-1] - sum_outcome_untreated_left
        ) / np.maximum(num_samples_right - (num_treated_left[-1] - num_treated_left), 1)
        avg_te = (num_samples_left * left_te + num_samples_right * right_te) / self.num_samples
        var_te = (
            (num_samples_left * num_samples_right)
            / (self.num_samples)
            * (left_te - right_te) ** 2
        )
        sq_diff_avg = self.num_samples * (self.treatment_effect - avg_te) ** 2

        # Check that the child nodes would have sufficiently many samples (in each group)
        allowed = (
            (min_samples <= num_samples_left)
            & (min_samples <= num_samples_right)
            & (min_group_samples <= num_treated_left[-1] - num_treated_left)
            & (num_treated_left[-1] - num_treated_left <= num_samples_right - min_group_samples)
            & (min_group_samples <= num_treated_left)
            & (num_treated_left <= num_samples_left - min_group_samples)
        )
        # Further check that value differs from next one, otherwise wrong split considered
        allowed[:-1] &= (
            self.covariates[args_sorted[:-1], split_var]
            != self.covariates[args_sorted[1:], split_var]
        )
        sq_diff_avg[~allowed] = 0.0
        var_te[~allowed] = 0.0

        sq_diff_max = sq_diff_avg.argmax()
        var_max = var_te.argmax()
        if sq_diff_avg[sq_diff_max] >= var_te[var_max]:
            split_val: float = (
                self.covariates[args_sorted[sq_diff_max], split_var]
                + self.covariates[args_sorted[sq_diff_max + 1], split_var]
            ) / 2
            return var_te[sq_diff_max], sq_diff_avg[sq_diff_max], split_val
        split_val: float = (
            self.covariates[args_sorted[var_max], split_var]
            + self.covariates[args_sorted[var_max + 1], split_var]
        ) / 2
        return var_te[var_max], sq_diff_avg[var_max], split_val

    def __str__(self) -> str:
        samples_str = f"Number of samples: {self.num_samples}, "
        if self.is_leaf:
            return samples_str + f"Treatment effect: {self.treatment_effect}"
        split_str = f", Split on {self.split_variable} at {self.split_value}"
        if self.split_type_var:
            return samples_str + f"Split type: Variance ({self.criteria[0]})" + split_str
        return samples_str + f"Split type: Change of average ({self.criteria[1]})" + split_str

    def predict(self, covariates: np.ndarray) -> float:
        """Predict the treatment effect for one sample with given covariates.

        Parameters
        ----------
        covariates : np.ndarray of shape (num_covariates,)
            Covariates of the sample.

        Returns
        -------
        float
            Predicted treatment effect.
        """
        if self.is_leaf:
            return self.treatment_effect
        if covariates[self.split_variable] <= self.split_value:
            return self.left_child.predict(covariates)
        return self.right_child.predict(covariates)

    def reevaluate(
        self,
        covariates: np.ndarray | None = None,
        treatment: np.ndarray | None = None,
        outcomes: np.ndarray | None = None,
        total_num_samples: int | None = None,
    ) -> None:
        """Reevaluate the splits after tree building is finished.
        
        Either use samples which were used for fitting or new samples. If new samples are
        given, the predicted treatment effects will also be updated to the new values
        """
        if covariates is not None:
            self.covariates = covariates
            self.num_samples = covariates.shape[0]
            treatment = treatment.astype(bool)
            self.treatment = treatment
            self.outcomes = outcomes
            if treatment.any() and not treatment.all():
                # update predicted TE only if there is new data for both treated and untreated
                self.treatment_effect = outcomes[treatment].mean() - outcomes[~treatment].mean()
        if total_num_samples is None:
            total_num_samples = self.num_samples
        if self.is_leaf:
            self.mean_predict = self.treatment_effect
            return
        if covariates is None:
            self.left_child.reevaluate(total_num_samples=total_num_samples)
            self.right_child.reevaluate(total_num_samples=total_num_samples)
        else:
            left_samples = (covariates[:, self.split_variable] <= self.split_value)
            self.left_child.reevaluate(
                self.covariates[left_samples],
                self.treatment[left_samples],
                self.outcomes[left_samples],
                total_num_samples=total_num_samples
            )
            self.right_child.reevaluate(
                self.covariates[~left_samples],
                self.treatment[~left_samples],
                self.outcomes[~left_samples],
                total_num_samples=total_num_samples
            )
        if self.num_samples:
            self.mean_predict = (
                (self.left_child.num_samples * self.left_child.mean_predict
                + self.right_child.num_samples * self.right_child.mean_predict)
                / self.num_samples
            )
        else:
            self.mean_predict = self.treatment_effect
            # variance = 0
            # bias = 0
        variance = (
            (self.left_child.num_samples * self.right_child.num_samples)
            / max(self.num_samples, 1)
            * (self.left_child.mean_predict - self.right_child.mean_predict) ** 2
        )
        # Bias might be negative (if the split added bias on the validation set)
        bias = (
            self.num_samples * (self.mean_predict - self.treatment_effect) ** 2
            - self.left_child.num_samples * (self.left_child.mean_predict - self.left_child.treatment_effect) ** 2
            - self.right_child.num_samples * (self.right_child.mean_predict - self.right_child.treatment_effect) ** 2
        )
        self.split_type_var = (variance > bias)
        self.criteria = (variance / total_num_samples, bias / total_num_samples)
        # self.value_split = self.num_samples * (self.mean_predict - self.treatment_effect) ** 2
    
    def _get_total_impurity_decrease(self, method: str = 'heterogeneity', depth_limit=None) -> np.ndarray:
        """Sum the value of the splits for each covariate.

        method : str, (optional, default='heterogeneity')
            Which criterion to use for weighting splits. Possible values are:
            'heterogeneity': only use heterogeneity criterion
            'bias: only use bias criterion
            'sum': use sum of both criteria
        
        depth_limit : int or None, (optional, default=None)
            Splits for nodes up to which depth should be considered.
            If 0, no splits are included and all features will have impurity decrease 0.
            If None, no limit is imposed.
        """
        if self.is_leaf or depth_limit == 0:
            return np.zeros(self.covariates.shape[1])
        remaining_depth = None if depth_limit is None else depth_limit - 1
        total_impur_decrease = self.left_child._get_total_impurity_decrease(method, remaining_depth)
        total_impur_decrease += self.right_child._get_total_impurity_decrease(method, remaining_depth)
        # The bias criterion (self.criteria[1]) might be negative.
        # Within a tree, this should be accounted for, so only adress this when outputting
        # the feature importance for the entire tree
        match method:
            case 'heterogeneity':
                total_impur_decrease[self.split_variable] += self.criteria[0]
            case 'bias':
                total_impur_decrease[self.split_variable] += self.criteria[1]
            case 'sum':
                total_impur_decrease[self.split_variable] += self.criteria[0] + self.criteria[1]
            # case 'heterogeneity splits':
            #     if self.criteria[0] >= self.criteria[1]:
            #         total_impur_decrease[self.split_variable] += self.criteria[0]
            # case 'bias splits':
            #     if self.criteria[0] <= self.criteria[1]:
            #         total_impur_decrease[self.split_variable] += self.criteria[1]
            # case 'sum splits':
            #     total_impur_decrease[self.split_variable] += max(self.criteria[0], self.criteria[1])
            case _:
                raise ValueError(f"Unknown method {method} for weighing splits. Use 'heterogeneity', 'bias' or 'sum'.")
        return total_impur_decrease

class CausalTree:
    """A Causal Tree."""

    def fit(
        self,
        covariates: np.ndarray,
        treatment: np.ndarray,
        outcomes: np.ndarray,
        min_impurity_decrease: float = 1e-10,
        max_features: int | None = None,
        random_state: int
        | np.random.SeedSequence
        | np.random.BitGenerator
        | np.random.Generator
        | None = None,
        max_depth: int | None = None,
        splitting_method: str = "greater",
    ) -> None:
        """Train the Causal Tree with the given samples.

        Parameters
        ----------
        covariates : np.ndarray
            Two-dimensional array with shape (num_samples, num_covariates), containing
            the covariates of the samples.
        treatment : np.ndarray
            One-dimensional boolean array indicating if a sample was treated or not.
            TODO: Allow None as default with either `covariates[:, 0]` or `outcomes[:, 0]`
            as treatment. (probably benefical for compatability with sklearn)
            TODO: (far future) Consider continuous treatment values.
        outcomes : np.ndarray
            One-dimensional array with outcomes for the samples.
        min_impurity_decrease : float, (optional, default=1e-10)
            Smallest value for the splitting rules to be considered for a split. If only
            splits with a smaller value are found, no split will be created.
        max_features : int | None, (optional, default=None)
            Number of features to consider for a split. If None, all features will be
            used. If int, the given number of features will be considered. If float,
            `int(max_features * num_features)` will be tested.
        random_state : int | np.random.SeedSequence | np.random.BitGenerator
        | np.random.Generator | None, (optional, default=None)
            Random state to use for selecting and ordering covariates for consideration
            when splitting. Will be used by `np.random.default_rng`.
        max_depth : int | None, (optional, default=None)
            Maximal depth of the tree. If None, there will be no limit on
            the depth of the tree.
        splitting_method: str, (optional, default="greater")
            Either "greater" or "confounding". If "greater", the split with the greatest
            value will be used. If "confounding", all covariates will be checked (so
            `max_features` will be ignored), if their best split is based on bias
            criterion with a value of at least `min_impurity_decrease`. If there is any,
            the best of these will be used. Only otherwise the best heterogenity split
            will be used.
        """
        self.root = _Node(covariates.astype(float), treatment.astype(bool), outcomes.astype(float))
        self._build_tree(min_impurity_decrease, max_features, random_state, max_depth, splitting_method)

    def _build_tree(
        self,
        min_impurity_decrease: float = 1e-10,
        max_features: int | None = None,
        random_state: int
        | np.random.SeedSequence
        | np.random.BitGenerator
        | np.random.Generator
        | None = None,
        max_depth: int | None = None,
        splitting_method: str = "greater",
    ) -> None:
        num_samples = self.root.covariates.shape[0]
        rng = np.random.default_rng(random_state)
        nodes = [(self.root, 0)]
        while nodes:
            node, depth = nodes.pop(0)
            if (max_depth is None or max_depth > depth) and node.create_split(
                num_samples, min_impurity_decrease, max_features, rng, splitting_method
            ):
                nodes.append((node.left_child, depth + 1))
                nodes.append((node.right_child, depth + 1))

    def predict(self, covariates: np.ndarray) -> float | np.ndarray:
        """Predict the treatment effect for samples with given covariates.

        Parameters
        ----------
        covariates : np.ndarray of shape (num_samples, num_covariates) or (num_covariates,)
            If one-dimensional, this should contain the covariates of one sample.
            Otherwise, it should contain the covariates of the samples as rows.

        Returns
        -------
        float | np.ndarray of shape (num_samples,)
            If `covariates` is one-dimensional, the predicted treatment effect
            of this sample is returned. Otherwise the predicted treatment effects are
            returned as one-dimensional array.
        """
        if len(covariates.shape) == 2:
            return np.array([self.predict(cov) for cov in covariates])
        return self.root.predict(covariates)

    def __str__(self) -> str:
        """Output the tree structure."""
        return self._print_tree_from_node(self.root)
    
    def feature_importances(self, method: str = 'heterogeneity', max_depth=0) -> np.ndarray:
        """Calculate the feature importance for the tree.

        Feature importance is here defined as the scaled sum of the impurity decreases of
        splits on the respective covariate.

        Parameters
        ----------
        method : str, (optional, default='heterogeneity')
            Which criterion to use for weighting splits. Possible values are:
            'heterogeneity': only use heterogeneity criterion
            'bias: only use bias criterion
            'sum': use sum of both criteria
        max_depth : int or None, (optional, default=None)
            Maximum depth of nodes to be considered when computing impurity decrease.
            If None, no limit is imposed.

        Returns
        -------
        np.ndarray of shape (num_covariates,)
            Array with feature importances. The non-negative entries will sum to either 0
            (for a tree with a single node) or 1. For method 'heterogeneity', all entries
            will be non-negative.
        """
        impur_decrease = self.root._get_total_impurity_decrease(method, depth_limit=max_depth)
        # there can be nodes, and in extension features, with negative bias criterion
        # we want that the sum of non-negative feature importances is 1
        # for heterogeneity this cannot happen, but it also does no harm
        if total_impur_decrease := impur_decrease[impur_decrease>=0].sum():  # normailze exept if sum is 0
            impur_decrease /= total_impur_decrease
        return impur_decrease

    def ate(self) -> float:
        return self.root.mean_predict

    @classmethod
    def _print_tree_from_node(cls, node: _Node, depth: int = 0) -> str:
        out = depth * "\t" + node.__str__() + "\n"
        if node.is_leaf:
            return out
        return (
            out
            + cls._print_tree_from_node(node.left_child, depth + 1)
            + cls._print_tree_from_node(node.right_child, depth + 1)
        )


class CausalForest:
    """A Regression Forest of Causal Trees."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int
        | np.random.SeedSequence
        | np.random.BitGenerator
        | np.random.Generator
        | None = None,
    ) -> None:
        self.estimators = [CausalTree() for _ in range(n_estimators)]
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        y: np.ndarray,
        min_impurity_decrease: float = 1e-10,
        max_features: int = 1/2,
        max_depth: int | None = None,
        bootstrap: bool = False,
        max_samples: float | None = 1/2,
        oob_reeval: bool = True,
    ) -> None:
        """Train the forest with given samples.

        Parameters
        ----------
        bootstrap: bool, (optional, Default=True)
            Whether to use bootstrapping, i.e. sampling with replacement, on the samples
            for the trees.
        max_samples: int | float | None, (optional, Default=1/2)
            Number of samples to use for each tree.
        oob_reeval: bool, (optional, Default=True)
            Wheter to update the leaf predictions and reevaluate the splits
            with the OOB-samples of each tree.

        For explanation of other parameters see `CausalTree.fit`.
        """
        rng = np.random.default_rng(self.random_state)
        if max_samples is None:
            max_samples = X.shape[0]
        elif isinstance(max_samples, float):
            max_samples = max(1, int(max_samples * X.shape[0]))
        for tree in self.estimators:
            samples = rng.choice(X.shape[0], max_samples, replace=bootstrap)
            tree.fit(
                X[samples],
                T[samples],
                y[samples],
                min_impurity_decrease,
                max_features,
                rng,  # same seed would eleminate randomness, use rng instead
                max_depth,
            )
            tree._oob_samples = list(set(range(X.shape[0])) - set(samples))
            if oob_reeval and tree._oob_samples:
                tree.root.reevaluate(
                X[tree._oob_samples],
                T[tree._oob_samples],
                y[tree._oob_samples],
            )

    def predict(self, covariates: np.ndarray) -> float | np.ndarray:
        """Predict the treatment effect for sample(s) with given covariates.

        See also `CausalTree.predict`.
        """
        return np.mean([tree.predict(covariates) for tree in self.estimators], axis=0)

    def feature_importances(self, method: str = 'heterogeneity', max_depth=None) -> np.ndarray:
        """Calculate the feature importance for the forest.

        Feature importance is here defined as mean of the feature importances of the trees.

        Parameters
        ----------
        method : str, (optional, default='heterogeneity')
            Which criterion to use for weighting splits. Possible values are:
            'heterogeneity': only use heterogeneity criterion
            'bias: only use bias criterion
            'sum': use sum of both criteria
        max_depth : int or None, (optional, default=None)
            Maximum depth of nodes to be considered when computing impurity decrease.
            If None, no limit is imposed.

        Returns
        -------
        np.ndarray of shape (num_covariates,)
            Array with feature importances. The non-negative entries will sum to either 0
            (if there are no features with positive importance) or 1. For method
            'heterogeneity', all entries will be non-negative.
        """
        impur_decrease = [tree.feature_importances(method, max_depth) for tree in self.estimators]
        if not impur_decrease:  # this should not happen
            return np.zeros(self.estimators[0].root.covariates.shape[1])
        impur_decrease = np.sum(impur_decrease, axis=0)
        # impur_decrease[impur_decrease>=0].sum() == len(impur_decrease) should hold
        if total_impur_decrease := impur_decrease[impur_decrease>=0].sum():  # normailze exept if sum is 0
            impur_decrease /= total_impur_decrease
        return impur_decrease

    def reevaluate(
        self,
        covariates: np.ndarray | None = None,
        treatment: np.ndarray | None = None,
        outcomes: np.ndarray | None = None,
    ) -> None:
        """Reevaluate the splits of trees in Causal Forest after building is finished.
        
        Either use samples which were used for fitting or new samples. If new samples are
        given, the predicted treatment effects will also be updated to the new values
        """
        for tree in self.estimators:
            tree.root.reevaluate(covariates, treatment, outcomes)
