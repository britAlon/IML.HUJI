from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import loss_functions


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        self.pi_ = np.array([np.count_nonzero(y == k) / n_samples for k in self.classes_])
        self.mu_ = np.array([X[y == k].mean(axis=0) for k in self.classes_])
        self.vars_ = np.array([X[y == k].var(axis=0) for k in self.classes_])


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        n_samples, n_features = X.shape[0], X.shape[1]
        class_probabilities = np.zeros((n_samples, len(self.classes_)))
        for idx in range(len(self.classes_)):
            class_probabilities[:, idx] = self.pdf(X, self.mu_[idx], self.vars_[idx]) + np.log(self.pi_[idx])
        return class_probabilities

    def pdf(self, X, mu, sigma):
        n_samples, n_features = X.shape[0], X.shape[1]
        const = - n_samples / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma))
        probs = 0.5 * np.sum(np.power(X - mu, 2) / sigma, 1)
        return const - probs

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return loss_functions.misclassification_error(y, self._predict(X))
