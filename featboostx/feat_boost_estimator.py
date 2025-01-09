# Author: Ahmad Alsahaf <a.m.j.a.alsahaf@rug.nl>
# Vikram Shenoy <shenoy.vi@husky.neu.edu>
# Ömer Tarik Özyilmaz <o.t.ozyilmaz@umcg.nl>

import itertools
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
import shap
from shap.plots._force import AdditiveExplanation, convert_to_link, ensure_not_numpy
from shap.utils._legacy import DenseData, Instance, Model
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, KFold

logging.basicConfig(level=logging.INFO)


class FeatBoostEstimator(BaseEstimator, ABC):
    """Base class for FeatBoostClassifier and FeatBoostRegressor."""

    def __init__(
        self,
        estimators: BaseEstimator | list[BaseEstimator],
        loss: str,
        metric: str,
        number_of_folds: int = 5,
        epsilon: float = 1e-3,
        max_number_of_features: int = 3000,
        siso_ranking_size: int = 200,
        siso_order: int = 1,
        reset: bool = True,
        fast_mode: bool = False,
        xgb_importance: str = "gain",
        learning_rate: float = 1.0,
        num_resets: int = 3,
        fold_random_state: int = 275,
        verbose: int = 0,
        stratification: bool = False,
    ) -> None:
        """
        Create FeatBoost estimator.

        :param estimators: estimator or list of estimators to use.
            List can be of length 1 or 2, first estimator is used for ranking
            and second for evaluation.
        :param loss: loss function to use, supported function depends on estimator type.
        :param metric: metric to use, supported metric depends on estimator type. Supported metrics:
            - Classification: ["acc", "f1"]
            - Regression: ["mae"]
            - Survival: ["c_index"]
        :param number_of_folds: number of k folds for cross-validation.
        :param epsilon: threshold to stop adding features.
        :param max_number_of_features: max number of features it will find.
        :param siso_ranking_size: number of features to rank and evaluate.
        :param siso_order: order of single feature selection.
        :param reset: allow reset of weights.
        :param fast_mode: allow increase in speed by not appending selected features
            for evaluation.
        :param xgb_importance: XGB importance type.
        :param learning_rate: learning rate of softmax (applies to classification).
        :param num_resets: number of resets to perform at most.
        :param fold_random_state: random state for stratified k-fold.
        :param verbose: whether to enable logging.
        :param stratification: whether to enable stratification (only for classification or survival).
        """
        self.estimator = (
            estimators if isinstance(estimators, list) else [estimators, estimators]
        )
        self.number_of_folds = number_of_folds
        self.epsilon = epsilon
        self.max_number_of_features = max_number_of_features
        self.siso_ranking_size = siso_ranking_size
        self.siso_order = siso_order
        self.loss = loss
        self.reset = reset
        self.fast_mode = fast_mode
        self.metric = metric
        self.xgb_importance = xgb_importance
        self.learning_rate = learning_rate
        self._all_selected_variables = []
        self.metric_ = []
        self.logger = logging.getLogger("FeatBoost")
        level = [logging.WARNING, logging.INFO, logging.DEBUG][verbose]
        self.logger.setLevel(level)
        self.i = 1
        self.num_resets = num_resets
        self.fold_random_state = fold_random_state
        self.stratification = stratification

        siso_size = (
            self.siso_ranking_size
            if not isinstance(self.siso_ranking_size, list)
            else self.siso_ranking_size[0]
        )
        assert (
            siso_size > self.siso_order
        ), "SISO order cannot be greater than the SISO ranking size.\n \
            Read the documentation for more details"
        assert (
            len(self.estimator) == 2
        ), "Length of list of estimators should always be equal to 2.\n \
            Read the documentation for more details"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the FeatBoost method with the estimator as provided by the user.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        Y : array-like, shape = [n_samples]
            The target values.

                Returns
        -------
        self : object

        """
        return self._fit(X, Y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
                Reduces the columns of input X to the features selected by FeatBoost.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features selected by
                        FeatBoost.
        """
        return self._transform(X)

    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fits FeatBoost and then reduces the input X to the features selected.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        Y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features selected by
                        FeatBoost.
        """
        return self._fit_transform(X, Y)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce the columns of input X to the features selected by FeatBoost.

        :param X: input data.
        :raises ValueError: if fit(X, Y) has not been called.
        :return: reduced input data.
        """
        try:
            self.selected_subset_
        except AttributeError:
            raise ValueError("fit(X, Y) needs to be called before using transform(X).")
        return X[:, self.selected_subset_]

    def _fit_transform(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit FeatBoost and then reduce the input X to the features selected.

        :param X: input data.
        :param Y: input labels.
        :return: reduced input data.
        """
        self._fit(X, Y)
        return self._transform(X)

    def _fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        feature_names: List[str] | None = None,
        global_sample_weights: np.ndarray | None = None,
    ) -> None:
        """
        Perform feature selection.

        Performs the initial ranking, SISO and MISO over multiple iterations
            based on the maximum number of features required by the user in a single
            subset.
        :param X: training data.
        :param Y: training labels.
        :param feature_names: (optional) feature names.
        :param global_sample_weights: (optional) global sample weights.
        """
        self.feature_importances_array_ = np.empty((0, X.shape[1]))
        self._feature_names = feature_names or [
            "x_%03d" % (i + 1) for i in range(len(X[0]))
        ]
        if global_sample_weights is not None:
            self._global_sample_weights = global_sample_weights
        else:
            self.__init_weights(Y)

        stop_epsilon = 10e6
        self.i = 1
        repeated_variable = False

        self._init_alpha(Y)

        reset_count = 0
        while (
            stop_epsilon > self.epsilon
            and self.i <= self.max_number_of_features
            and repeated_variable is False
        ):
            if self.i == 1:
                self.__first_iteration(X, Y)
                self.i += 1
                (
                    stop_epsilon,
                    repeated_variable,
                    reset_count,
                ) = self._check_stop_conditions(
                    stop_epsilon, repeated_variable, Y, reset_count
                )
                continue

            if reset_count >= self.num_resets:
                self.reset = False
                self.logger.debug("Infinite loop: No more resets this time!")
                reset_count = 0
            self.logger.info(
                "\n\n\n\n\nselected variable thus far:\n%s"
                % "\n".join(
                    [self._feature_names[i] for i in self._all_selected_variables]
                )
            )
            self.logger.debug("Ranking features iteration %02d" % self.i)

            # Perform Single Input Single Output (SISO) for subsequent iterations.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                selected_variable, _ = self._siso(X, Y)
            self.logger.debug("Evaluating MISO after iteration %02d" % self.i)
            new_variables = [
                x for x in selected_variable if x not in self._all_selected_variables
            ]
            repeated_variable = new_variables == []

            if repeated_variable:
                (
                    stop_epsilon,
                    repeated_variable,
                    reset_count,
                ) = self._check_stop_conditions(
                    stop_epsilon, repeated_variable, Y, reset_count
                )
                continue

            self._all_selected_variables.extend(new_variables)

            # Perform Multiple Input Single Output (MISO) for subsequent iterations.
            metric_t_miso = self._miso(X[:, self._all_selected_variables], Y)
            self._update_weights(X[:, self._all_selected_variables], Y)
            self.metric_.append(metric_t_miso)

            # stop_epsilon makes sure the accuracy doesn't fall below the threshold
            #  i.e stopping condition 2 as mentioned above.
            stop_epsilon = abs(self.metric_[self.i - 1] - self.metric_[self.i - 2])
            self.logger.info(
                "::::::::::::::::::::accuracy of MISO after iteration %02d is %05f"
                % (self.i, metric_t_miso)
            )
            self.i += 1

            stop_epsilon, repeated_variable, reset_count = self._check_stop_conditions(
                stop_epsilon, repeated_variable, Y, reset_count
            )
        if self.selected_subset_:
            unique_features = set()
            seen_add = unique_features.add
            self.selected_subset_ = [
                x
                for x in self.selected_subset_
                if not (x in unique_features or seen_add(x))
            ]

    def _check_stop_conditions(
        self,
        stop_epsilon: float,
        repeated_variable: bool,
        Y: np.ndarray,
        reset_count: int,
    ) -> Tuple[float, bool, int]:
        """
        Check the stopping conditions for the FeatBoost algorithm.

        :param stop_epsilon: epsilon value to check for threshold.
        :param repeated_variable: whether a variable has been selected twice.
        :param Y: labels for the data.
        :param reset_count: number of resets performed.
        :return: updated epsilon, whether a variable has been selected twice and
            number of resets performed.
        """
        # Condition 1 -> Maximum number of features reached.
        if self.i >= self.max_number_of_features:
            self.logger.debug(
                "Selection stopped: Maximum number of iteration %02d has been reached."
                % self.max_number_of_features
            )
            self.stopping_condition_ = "max_number_of_features_reached"
            self.selected_subset_ = self._all_selected_variables

        # Condition 2 -> epsilon value falls below the threshold.
        if stop_epsilon <= self.epsilon and not self.reset:
            self.logger.debug("Selection stopped: Tolerance has been reached.")
            self.logger.info(
                "Stopping Condition triggered at iteration number: %d" % (self.i - 1)
            )
            self.stopping_condition_ = "tolerance_reached"
            self.selected_subset_ = self._all_selected_variables[:-1]
            self.metric_ = self.metric_[:-1]

            self.logger.info("Selected variables so far:")
            self.logger.info(self.selected_subset_)

        # Condition 3 -> a specific feature has been already selected previously.
        if repeated_variable and not self.reset:
            self.logger.debug("Selection stopped: A variable has been selected twice.")
            self.logger.info(
                "Stopping Condition triggered at iteration number: %d" % (self.i - 1)
            )
            self.stopping_condition_ = "variable_selected_twice"
            self.selected_subset_ = self._all_selected_variables[:]
        if (stop_epsilon <= self.epsilon or repeated_variable) and self.reset:
            if stop_epsilon <= self.epsilon:
                self.logger.info(
                    "\n\nATTENTION: Reset occured because of tolerance reached!"
                )
                stop_epsilon = self.epsilon + 1
                self._all_selected_variables = self._all_selected_variables[:-1]
                self.metric_ = self.metric_[:-1]
            elif repeated_variable:
                # re-set the sample weights and epsilon
                self.logger.info(
                    "\n\nATTENTION: Reset occured because of selected twice!"
                )
                repeated_variable = False
                self._all_selected_variables = self._all_selected_variables[:]
                self.metric_ = self.metric_[:]
            self.__reset_weights(Y, reset_count)
            reset_count += 1
            self.logger.info("Reset count = %d" % reset_count)
            self.i -= 1

        return stop_epsilon, repeated_variable, reset_count

    def _siso(self, X: np.ndarray, Y: np.ndarray) -> Tuple[List[int], Any]:
        """
        Determine which feature to select.

        It does this based on classification accuracy of
            the 'siso_ranking_size' ranked features from _input_ranking.

        :param X: training data.
        :param Y: training labels.
        :return: selected feature and accuracy of the selected feature.
        """
        # Get a ranking of features based on the estimator.
        ranking, self.all_ranking_ = self._input_ranking(X, Y)
        self.siso_ranking_[(self.i - 1), :] = ranking
        if self.stratification:
            kf_func = StratifiedKFold
            if self.metric == "c_index":
                stratification = Y[:, 1] == np.inf
            elif self.metric in ["acc", "f1"]:
                stratification = Y
            else:
                raise NotImplementedError(
                    "Stratification is not supported for this metric."
                )
        else:
            kf_func = KFold

        kf = kf_func(
            n_splits=self.number_of_folds,
            shuffle=True,
            random_state=self.fold_random_state,
        )
        # Combination of features from the ranking up to siso_order size
        combs = [
            list(x)
            for i in range(self.siso_order)
            for x in itertools.combinations(ranking, i + 1)
        ]

        metric = None
        metric_t_all = np.zeros((len(combs), 1))
        std_t_all = np.zeros((len(combs), 1))
        for idx_1, i in enumerate(combs):
            self.logger.debug(
                "...Evaluating SISO combination %02d which is %s" % (idx_1 + 1, str(i))
            )
            X_subset = X[:, i]
            n = len(X_subset)
            X_subset = X_subset.reshape(n, len(i))

            if self.fast_mode is False:
                X_subset = np.concatenate(
                    (X_subset, X[:, self._all_selected_variables]), axis=1
                )

            count = 1
            metric_t_folds = np.zeros((self.number_of_folds, 1))
            # Compute accuracy for each SISO input.
            kf_splits = (
                kf.split(X_subset, stratification)
                if self.stratification
                else kf.split(X_subset)
            )
            for train_index, test_index in kf_splits:
                X_train, X_test = X_subset[train_index], X_subset[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                # fit model according to mode
                if self.fast_mode is False:
                    self._fit_estimator(X_train, y_train, estimator_idx=1)
                else:
                    self._fit_estimator(
                        X_train,
                        y_train,
                        sample_weight=self._global_sample_weights[train_index],
                        estimator_idx=1,
                    )

                metric = self._score(y_test, self.estimator[1].predict(X_test))  # type: ignore # noqa

                self.logger.debug("Fold %02d %s = %05f" % (count, self.metric, metric))
                metric_t_folds[count - 1, :] = metric
                count = count + 1
            metric_t_all[idx_1, :] = np.mean(metric_t_folds)
            std_t_all[idx_1, :] = np.std(metric_t_folds)
            self.logger.debug(
                "%s for combination %02d is = %05f"
                % (self.metric, idx_1 + 1, np.mean(metric_t_folds))
            )

        # regular
        if self.metric == "mae":
            best_metric_t = np.amin(metric_t_all)
            selected_variable = combs[np.argmin(metric_t_all)]
        else:
            best_metric_t = np.amax(metric_t_all)
            selected_variable = combs[np.argmax(metric_t_all)]

        self.logger.debug(
            "Selected variable is %s with %s %05f"
            % (str(selected_variable), self.metric, best_metric_t)
        )
        return selected_variable, best_metric_t

    def _miso(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate the accuracy of selected features per additional feature.

        :param X: training data.
        :param Y: training labels.
        :return: accuracy of selected features.
        """
        warnings.filterwarnings("ignore")
        if self.stratification:
            kf_func = StratifiedKFold
            if self.metric == "c_index":
                stratification = Y[:, 1] == np.inf
            elif self.metric in ["acc", "f1"]:
                stratification = Y
            else:
                raise NotImplementedError(
                    "Stratification is not supported for this metric."
                )
        else:
            kf_func = KFold

        kf = kf_func(
            n_splits=self.number_of_folds,
            shuffle=True,
            random_state=self.fold_random_state,
        )
        metric_t_folds = np.zeros(self.number_of_folds)
        # Compute the accuracy of the selected features one addition at a time.
        kf_splits = kf.split(X, stratification) if self.stratification else kf.split(X)
        for i, (train_index, test_index) in enumerate(kf_splits):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            self._fit_estimator(X_train, y_train, estimator_idx=1)
            yHat_test = self.estimator[1].predict(X_test)  # type: ignore
            metric = self._score(y_test, yHat_test)
            self.logger.debug("Fold %02d %s = %05f" % (i + 1, self.metric, metric))
            metric_t_folds[i] = metric

        return float(np.mean(metric_t_folds))

    def _input_ranking(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        Create an initial ranking of features.

        It is using the provided estimator for SISO evaluation.

        :param X: training data.
        :param Y: training labels.
        :return: ranking of features.
        """
        # Perform an initial ranking of features using the given estimator.
        check_estimator = str(self.estimator[0])
        if "XGB" in check_estimator:
            self._fit_estimator(
                X, Y, sample_weight=self._global_sample_weights, estimator_idx=0
            )
            if "Bagging" in check_estimator:
                fscore = self.estimator[0].get_feature_importances(  # type: ignore
                    importance_type=self.xgb_importance
                )
            else:
                fscore = (
                    self.estimator[0]
                    .get_booster()  # type: ignore
                    .get_score(importance_type=self.xgb_importance)
                )
            feature_importance = np.zeros(X.shape[1])
            self.feature_importances_array_ = np.vstack(
                (self.feature_importances_array_, feature_importance)
            )
            # feature_importance = self._get_shap_importance(X, Y)  # type: ignore
            for k, v in fscore.items():
                feature_importance[int(k[1:])] = v
        else:
            self._fit_estimator(
                np.nan_to_num(X),
                np.nan_to_num(np.ravel(Y)),
                sample_weight=np.nan_to_num(self._global_sample_weights),
                estimator_idx=0,
            )
            feature_importance = self.estimator[0].feature_importances_  # type: ignore
        return self.__return_ranking(feature_importance)

    def _get_shap_importance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Get SHAP importance of the features.

        :param X: training data.
        :param Y: training labels.
        """
        explainer = shap.TreeExplainer(self.estimator[0])
        shap_values = explainer.shap_values(X)
        feature_names = [str(i) for i in range(1, X.shape[1] + 1)]
        exps = []
        for k in range(shap_values.shape[0]):
            if feature_names is None:
                feature_names = [
                    "Feature %s" % str(i) for i in range(shap_values.shape[1])
                ]
            display_features = X[k, :]

            instance = Instance(np.ones((1, len(feature_names))), display_features)
            base_value = explainer.expected_value
            e = AdditiveExplanation(
                base_value,
                np.sum(shap_values[k, :]) + base_value,
                shap_values[k, :],
                None,
                instance,
                convert_to_link("identity"),
                Model(None, None),
                DenseData(np.ones((1, len(feature_names))), list(feature_names)),
            )
            exps.append(e)
        feature_values = np.zeros((len(exps), len(exps[0].effects)))
        for e_idx, e in enumerate(exps):
            features = []
            for i in range(len(e.data.group_names)):
                if e.effects[i] == 0:
                    features.append(0)
                    continue
                features.append(ensure_not_numpy(e.instance.group_display_values[i]))
            feature_values[e_idx, :] = features
        feature_values = np.mean(np.abs(feature_values), axis=0)

        feature_importance = np.zeros(X.shape[1])
        for i, v in enumerate(feature_values):
            feature_importance[i] = v
        return feature_importance

    def __init_weights(self, Y: np.ndarray) -> None:
        """
        Initialize the weights of the samples.

        :param Y: labels for the data.
        """
        shape = Y.shape[0] if len(Y.shape) > 1 else Y.shape
        self._global_sample_weights = np.ones(shape)
        self.residual_weights_ = np.zeros((self.max_number_of_features, len(Y)))
        if isinstance(self.siso_ranking_size, int):
            self.siso_ranking_ = 99 * np.ones(
                (self.max_number_of_features, self.siso_ranking_size)
            )
        elif isinstance(self.siso_ranking_size, list):
            assert (
                len(self.siso_ranking_size) == 2
            ), "siso_ranking_size of list type is of incompatible format.\
                  Please enter a list of the following type: \n \
                      siso_ranking_size=[5, 10] \n Read documentation for more details."
            self.siso_ranking_ = 99 * np.ones(
                (self.max_number_of_features, self.siso_ranking_size[0])
            )

    def __first_iteration(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Perform the first iteration of the FeatBoost algorithm by ranking.

        :param X: training data.
        :param Y: training labels.
        """
        self.logger.debug("\n\n\n\n\n\nRanking features iteration %02d" % self.i)
        # Perform Single Input Single Output (SISO) for iteration 1.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected_variable, _ = self._siso(X, Y)
        self.logger.debug("Evaluating MISO after iteration %02d" % self.i)
        # The selected feature is stored inside self._all_selected_variables.
        self._all_selected_variables.extend(selected_variable)
        # Perform Multiple Input Single Output (MISO) for iteration 1.
        metric_t_miso = self._miso(X[:, self._all_selected_variables], Y)
        self._update_weights(X[:, self._all_selected_variables], Y)
        # Accuracy of selected feature is stored in accuracy_.
        self.metric_.append(metric_t_miso)
        self.logger.debug(
            "::::::::::::::::::::accuracy of MISO after iteration %02d is %05f"
            % (self.i, metric_t_miso)
        )

    def __reset_weights(self, Y: np.ndarray, reset_count: int) -> None:
        """
        Reset the weights of the samples if necessary.

        :param Y: labels for the data.
        :param reset_count: number of resets performed.
        """
        shape = Y.shape[0] if len(Y.shape) > 1 else Y.shape
        if reset_count < self.num_resets:
            self._global_sample_weights = np.ones(shape)
        elif reset_count == self.num_resets:
            self._global_sample_weights = np.random.randn(shape)  # type: ignore
            self._global_sample_weights = (
                self._global_sample_weights
                / np.sum(self._global_sample_weights)
                * len(Y)
            )

    def __return_ranking(self, feature_importance: np.ndarray) -> Tuple[Any, Any]:
        feature_rank = np.argsort(feature_importance)
        all_ranking = feature_rank[::-1]
        self.logger.debug("feature importances of all available features:")
        if isinstance(self.siso_ranking_size, int):
            for i in range(-1, -1 * self.siso_ranking_size - 1, -1):
                self.logger.debug(
                    "%s   %05f"
                    % (
                        self._feature_names[feature_rank[i]],
                        feature_importance[feature_rank[i]],
                    )
                )
            # Return the 'siso_ranking_size' ranked features to perform SISO.
            return (
                feature_rank[: -1 * self.siso_ranking_size - 1 : -1],  # noqa
                all_ranking,
            )

        assert (
            isinstance(self.siso_ranking_size, list)
            and len(self.siso_ranking_size) == 2
        ), "siso_ranking_size of list type is of incompatible format.\
                Please enter a list of the following type: \n\
                siso_ranking_size=[5, 10] \n Read documentation for more details."
        for i in range(-1, -1 * self.siso_ranking_size[1] - 1, -1):
            self.logger.debug(
                "%s   %05f"
                % (
                    self._feature_names[feature_rank[i]],
                    feature_importance[feature_rank[i]],
                )
            )
        # Return the 'siso_ranking_size' ranked features to perform SISO.
        feature_rank = feature_rank[: -1 * self.siso_ranking_size[1] - 1 : -1]  # noqa
        return (
            np.random.choice(feature_rank, self.siso_ranking_size[0], replace=False),
            all_ranking,
        )

    @abstractmethod
    def _init_alpha(self, Y: np.ndarray) -> None:
        """
        Alpha initialization for normalization later.

        :param Y: labels for the data shape.
        """
        pass

    @abstractmethod
    def _score(self, y_test: np.ndarray, y_pred: np.ndarray) -> Any:
        """
        Calculate the metric score of the model.

        :param y_test: true labels.
        :param y_pred: predicted labels.
        :return: metric score.
        """
        pass

    @abstractmethod
    def _update_weights(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Update the weights of the samples based on the loss.

        :param X: training data.
        :param Y: training labels.
        """
        pass

    @abstractmethod
    def _fit_estimator(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        estimator_idx: int = 0,
    ) -> None:
        """
        Fit one of the estimators.

        :param X_train: training data.
        :param y_train: training labels.
        :param sample_weight: (optional) sample weights for the estimator.
        :param estimator_idx: (optional) index of the estimator to fit.
        """
        pass
