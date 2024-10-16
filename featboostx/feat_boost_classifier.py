from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score

from featboostx import FeatBoostEstimator


class FeatBoostClassifier(FeatBoostEstimator):
    """Implementation of FeatBoost for classification."""

    def __init__(
        self,
        base_estimator: BaseEstimator | list[BaseEstimator],
        loss: str = "softmax",
        metric: str = "acc",
        **kwargs
    ) -> None:
        """
        Create a new FeatBoostClassifier.

        :param base_estimator: base estimator to use for classification.
        :param loss: supported -> ["softmax", "adaboost"].
        :param metric: supported -> ["acc", "f1"].
        """
        super().__init__(base_estimator, loss=loss, metric=metric, **kwargs)

    def _fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        feature_names: list[str] | None = None,
        global_sample_weights: np.ndarray | None = None,
    ) -> None:
        """
        Fit the FeatBoostClassifier.

        :param X: training data.
        :param Y: training labels.
        :param feature_names: (optional) names of the features.
        :param global_sample_weights: (optional) global sample weights.
        :return: the fitted FeatBoostClassifier.
        """
        assert len(Y.shape) == 1, "FeatBoostClassifier only supports single label"
        return super()._fit(X, Y, feature_names, global_sample_weights)

    def _init_alpha(self, Y: np.ndarray) -> None:
        """
        Alpha initialization for normalization later.

        :param Y: labels for the data shape.
        """
        if self.loss == "adaboost":
            self._alpha = np.ones(self.max_number_of_features + 1)
            self._alpha_abs = np.ones(self.max_number_of_features + 1)
        elif self.loss == "softmax":
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
        if self.metric == "acc":
            return accuracy_score(y_test, y_pred)
        if self.metric == "f1":
            return f1_score(y_test, y_pred, average="weighted")
        raise NotImplementedError

    def _update_weights(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Update the weights of the samples based on the loss.

        :param X: training data.
        :param Y: training labels.
        """
        # Calculate the residual weights from fitting on the entire dataset.
        self._fit_estimator(X, Y, self._global_sample_weights, estimator_idx=0)
        y_pred = self.estimator[0].predict(X)  # type: ignore
        if self.loss == "adaboost":
            self.__ada_boost_update(Y, y_pred)
        elif self.loss == "softmax":
            self.__softmax_update(X, Y)

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
            np.ravel(y_train),
            sample_weight=sample_weight,
        )

    def __ada_boost_update(self, Y: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Update the weights of the samples based on the AdaBoost loss.

        :param Y: true labels.
        :param y_pred: predicted labels.
        """
        # Determine the missclassified samples.
        err = 1 - accuracy_score(Y, y_pred)
        self._alpha_abs[self.i] = np.log((1 - err) / err) + np.log(
            len(np.unique(Y)) - 1
        )
        self._alpha[self.i] = self._alpha_abs[self.i] / self._alpha_abs[self.i - 1]
        misclass = Y.reshape(len(Y), 1) - y_pred.reshape(len(Y), 1)
        misclass_idx = np.nonzero(misclass)[0]
        self._global_sample_weights[misclass_idx] *= np.exp(self._alpha[self.i])

        # re-normalize
        self._global_sample_weights /= np.sum(self._global_sample_weights)
        self._global_sample_weights *= len(Y)

    def __softmax_update(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Update the weights of the samples based on the Softmax loss.

        :param X: training data.
        :param Y: training labels.
        """
        # Gets all the labels.
        labels = np.unique(np.ravel(Y))
        y_class = np.zeros((len(Y), len(labels)))
        prediction_probability = self.estimator[0].predict_proba(X)  # type: ignore
        probability_weight = np.zeros(np.shape(Y))

        # Generates One-Hot encodings for Multi-Class Problems
        for i, j in zip(range(0, len(X)), range(0, len(labels))):
            if Y[i] == labels[j]:
                if len(labels) == 2:
                    Y[i] = j
                    probability_weight[i] = prediction_probability[i][j]
                y_class[i][j] = 1
        log_bias = np.finfo(np.float64).eps
        # Loss function
        self._alpha_abs[:, self.i] = -self.learning_rate * np.sum(
            y_class * np.log(prediction_probability + log_bias), axis=1
        )
        self._alpha[:, self.i] = (
            self._alpha_abs[:, self.i] / self._alpha_abs[:, self.i - 1]
        )
        self._global_sample_weights *= self._alpha[:, self.i]

        # re-normalize
        self._global_sample_weights /= np.sum(self._global_sample_weights)
        self._global_sample_weights *= len(Y)

        self._alpha[:, self.i] = np.divide(
            self._alpha_abs[:, self.i], self._alpha_abs[:, self.i - 1]
        )
