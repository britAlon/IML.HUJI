from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    indices_range = np.arange(X.shape[0])
    # np.random.shuffle(indices_range)
    train_errors, validation_errors = np.zeros(cv), np.zeros(cv)
    indices = np.mod(indices_range, cv)  # (0,1,2,3,4,0,1,2,3,4 ....)
    for i in range(cv):
        train_folds_X, train_folds_y = X[indices != i], y[indices != i]
        fitted_model = estimator.fit(train_folds_X, train_folds_y)
        train_errors[i] = scoring(fitted_model.predict(train_folds_X), train_folds_y)
        validation_folds_X, validation_folds_y = X[indices == i], y[indices == i]
        validation_errors[i] = scoring(fitted_model.predict(validation_folds_X), validation_folds_y)
    return train_errors.mean(), validation_errors.mean()
