from collections import Counter
from typing import Any, List, Tuple

import numpy as np
from lifelines.utils import concordance_index
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

from featboostx import FeatBoostEstimator


class FeatBoostRegressor(FeatBoostEstimator):
    """Implementation of FeatBoost for regression."""

    def __init__(
        self,
        base_estimator: BaseEstimator,
        loss: str = "adaptive",
        metric: str = "c_index",
        **kwargs,
    ) -> None:
        """
        Create a new FeatBoostRegressor.

        :param base_estimator: base estimator to use for regression.
        :param loss: supported -> ["adaptive"].
        :param metric: supported -> ["c_index", "mae"].
        """
        super().__init__(base_estimator, loss=loss, metric=metric, **kwargs)

    def _init_alpha(self, Y: np.ndarray) -> None:
        """
        Alpha initialization for normalization later.

        :param Y: labels for the data shape.
        """
        if self.loss == "adaptive":
            self._alpha = np.ones((len(Y), self.max_number_of_features + 1))
            self._alpha_abs = np.ones((len(Y), self.max_number_of_features + 1))

    def _score(self, y_test: np.ndarray, y_pred: np.ndarray) -> Any:
        """
        Calculate the metric score of the model.

        :param y_test: true labels.
        :param y_pred: predicted labels.
        :raises NotImplementedError: when the metric is not supported.
        :return: metric score.
        """
        if self.metric == "mae":
            return mean_absolute_error(y_test, y_pred)
        if self.metric == "c_index":
            return self.c_index(y_test, y_pred)
        raise NotImplementedError

    def _update_weights(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Update the weights of the samples based on the loss.

        :param X: training data.
        :param Y: training labels.
        """
        if self.metric == "mae":
            self._update_weights_mae(X, Y)
        elif self.metric == "c_index":
            self._update_weights_c_index(X, Y)
        else:
            raise NotImplementedError

    def _update_weights_mae(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Update weights proportional to mean absolute error per sample.

        :param X: training data.
        :param Y: training labels
        """
        # Calculate the residual weights from fitting on the entire dataset.
        self._fit_estimator(X, Y)
        y_pred = self.estimator[0].predict(X)  # type: ignore
        shape = False
        if y_pred.shape != Y.shape:
            y_pred = y_pred.reshape(-1, 1)
            shape = True
        abs_errors = np.abs(Y - y_pred)

        # minmax the target
        min_val = np.quantile(Y, 0.01)
        max_val = np.quantile(Y, 0.99)
        Y = (Y - min_val) / (max_val - min_val)

        # minmax the predictions
        y_pred = (y_pred - min_val) / (max_val - min_val)

        # calculate the absolute errors
        abs_errors = np.abs(Y - y_pred)

        # sigmoid
        def sigmoid(x):
            return 2 * (1 / (1 + np.exp(-x)))

        abs_errors_with_index = {
            k: sigmoid(v[0]) if shape else sigmoid(v)
            for k, v in sorted(
                enumerate(abs_errors), key=lambda item: item[1], reverse=True
            )
        }

        self._alpha_abs[:, self.i] = [abs_errors_with_index[i] for i in range(len(Y))]
        self._alpha[:, self.i] = (
            self._alpha_abs[:, self.i] / self._alpha_abs[:, self.i - 1]
        )
        self._global_sample_weights *= self._alpha[:, self.i]

        # Re-normalize instance weights.
        self._global_sample_weights /= np.sum(self._global_sample_weights)
        self._global_sample_weights *= len(Y)

        self._alpha[:, self.i] = (
            self._alpha_abs[:, self.i] / self._alpha_abs[:, self.i - 1]
        )

    def _update_weights_c_index(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Update weights proportional to how often a sample is in the discordant pairs.

        :param X: training data.
        :param Y: training labels
        """
        # Calculate the residual weights from fitting on the entire dataset.
        self._fit_estimator(X, Y)
        y_pred = self.estimator[0].predict(X)  # type: ignore

        # Determine the missclassified samples.
        if self.estimator[0].get_params()["objective"] == "survival:cox":
            y_pred = -y_pred
        _, _, discordant_pairs = self._c_index_concordant(Y, y_pred)
        # count number of times sample is in discordant pairs
        discordant_pairs_counter = dict(
            sorted(
                Counter([i for pair in discordant_pairs for i in pair]).items(),
                key=lambda item: item[0],
                reverse=True,
            )
        )

        # scale the error
        discordant_pairs_counter = {
            k: np.log(v) + 1 for k, v in discordant_pairs_counter.items()
        }

        self._alpha_abs[:, self.i] = [
            discordant_pairs_counter[i] if i in discordant_pairs_counter else 1
            for i in range(len(Y))
        ]

        self._alpha[:, self.i] = (
            self._alpha_abs[:, self.i] / self._alpha_abs[:, self.i - 1]
        )

        self._global_sample_weights *= self._alpha[:, self.i]

        # Re-normalize instance weights.
        self._global_sample_weights /= np.sum(self._global_sample_weights)
        self._global_sample_weights *= len(Y)

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
        self.estimator[estimator_idx].fit(  # type: ignore
            X_train,
            y_train,
            sample_weight=sample_weight,
        )

    def c_index(self, y_test: np.ndarray, y_pred: np.ndarray) -> Any:
        """
        Calculate C-index using lifelines package.

        :param y_test: true labels.
        :param y_pred: predicted labels.
        :return: C-index of predicted data.
        """
        event_times = y_test[:, 0]
        event_observed = (y_test[:, 0] == y_test[:, 1]).astype(int)
        if self.estimator[0].get_params()["objective"] == "survival:cox":
            y_pred = -y_pred
        return concordance_index(event_times, y_pred, event_observed)

    def _c_index_concordant(
        self, y_test: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, List[Tuple], List[Tuple]]:
        """
        Calculate C-index using same method as lifelines but return pairs as well.

        :param y_test: true labels.
        :param y_pred: predicted labels.
        :return: the C-index, concordant pairs, and discordant pairs.
        """
        event_times = y_test[:, 0]
        event_observed = (y_test[:, 0] == y_test[:, 1]).astype(int)
        concordant = 0
        discordant = 0
        tied = 0
        concordant_pairs = []
        discordant_pairs = []
        for i in range(len(event_times)):
            for j in range(len(event_times)):
                if i == j or (event_observed[i] == 0 and event_observed[j] == 0):
                    continue
                if event_observed[i] == 1 and event_observed[j] == 1:
                    if event_times[i] < event_times[j]:
                        if y_pred[i] < y_pred[j]:
                            concordant += 1
                            concordant_pairs.append((i, j))
                        elif y_pred[i] > y_pred[j]:
                            discordant += 1
                            discordant_pairs.append((i, j))
                        else:
                            tied += 1
                    elif event_times[i] > event_times[j]:
                        if y_pred[i] > y_pred[j]:
                            concordant += 1
                            concordant_pairs.append((i, j))
                        elif y_pred[i] < y_pred[j]:
                            discordant += 1
                            discordant_pairs.append((i, j))
                        else:
                            tied += 1
                elif event_observed[i] == 1 and event_observed[j] == 0:
                    if event_times[i] < event_times[j]:
                        if y_pred[i] < y_pred[j]:
                            concordant += 1
                            concordant_pairs.append((i, j))
                        elif y_pred[i] > y_pred[j]:
                            discordant += 1
                            discordant_pairs.append((i, j))
                        elif y_pred[i] == y_pred[j]:
                            tied += 1
                    elif event_times[i] > event_times[j]:
                        # we dont know if this is concordant or discordant
                        continue
                elif event_observed[i] == 0 and event_observed[j] == 1:
                    if event_times[j] < event_times[i]:
                        if y_pred[j] < y_pred[i]:
                            concordant += 1
                            concordant_pairs.append((i, j))
                        elif y_pred[j] > y_pred[i]:
                            discordant += 1
                            discordant_pairs.append((i, j))
                        elif y_pred[i] == y_pred[j]:
                            tied += 1
                    elif event_times[j] > event_times[i]:
                        # we dont know if this is concordant or discordant
                        continue

        return (
            (concordant + 0.5 * tied) / (concordant + tied + discordant),
            concordant_pairs,
            discordant_pairs,
        )
