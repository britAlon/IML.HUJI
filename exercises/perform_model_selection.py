from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from IMLearn.base import BaseEstimator
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.random.uniform(low=-1.2, high=2, size=n_samples)
    y = f(X) + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X, columns=['x']), pd.Series(y, name='y'), 0.66)
    train_X, train_y = np.asarray(train_X).ravel(), np.asarray(train_y).ravel()
    test_X, test_y = np.asarray(test_X).ravel(), np.asarray(test_y).ravel()
    data_fig = go.Figure([go.Scatter(name='True Model', x=X, y=f(X), mode='markers'),
                          go.Scatter(name='Train Set', x=train_X, y=train_y, mode='markers'),
                          go.Scatter(name='Test Set', x=test_X, y=test_y, mode='markers')])
    data_fig.update_layout(title=r"Dataset for Polynomial Fitting - True Model, Test Set, Train Set",
                           xaxis_title=r"x", yaxis_title=r"y")
    data_fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors, validation_errors = np.zeros(11), np.zeros(11)
    k_range = np.linspace(0, 10, 11)
    for k in k_range:
        k = int(k)
        train_errors[k], validation_errors[k] = cross_validate(PolynomialFitting(k), train_X, train_y,
                                                               mean_square_error, 5)
    cv_fig = go.Figure([go.Scatter(name='Train Error', x=k_range, y=train_errors, mode='markers+lines'),
                        go.Scatter(name='Validation Error', x=k_range, y=validation_errors, mode='markers+lines')])
    cv_fig.update_layout(title=r"Train and Validation Errors for Different Polynomial Degree",
                  xaxis_title=r"degree", yaxis_title=r"MSE")
    cv_fig.show()
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(validation_errors)
    k_star_validation_error = np.min(validation_errors)
    best_model = PolynomialFitting(k_star).fit(train_X, train_y)
    test_error = np.around(best_model.loss(test_X, test_y), 2)
    print("best polynomial degree =", k_star, "with error = ", test_error, "validation error was ", k_star_validation_error)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    diabetes = datasets.load_diabetes()
    train_X, train_y, test_X, test_y = diabetes.data[:n_samples + 1], diabetes.target[:n_samples + 1], \
                                       diabetes.data[n_samples:], diabetes.target[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    reg_parameters = np.linspace(0.0001, 2, n_evaluations)
    ridge_train_errors, ridge_validation_errors, lasso_train_errors, lasso_validation_errors = np.zeros(
        n_evaluations), np.zeros(n_evaluations), np.zeros(n_evaluations), np.zeros(n_evaluations)
    for ind, param in enumerate(reg_parameters):
        ridge_train_errors[ind], ridge_validation_errors[ind] = cross_validate(RidgeRegression(param), train_X, train_y,
                                                                               mean_square_error)
        lasso_train_errors[ind], lasso_validation_errors[ind] = cross_validate(Lasso(alpha=param), train_X, train_y,
                                                                               mean_square_error)
    cv_fig = go.Figure(
        [go.Scatter(name='Ridge Train Error', x=reg_parameters, y=ridge_train_errors, mode='lines'),
         go.Scatter(name='Ridge Validation Error', x=reg_parameters, y=ridge_validation_errors, mode='lines'),
         go.Scatter(name='Lasso Train Error', x=reg_parameters, y=lasso_train_errors, mode='lines'),
         go.Scatter(name='Lasso Validation Error', x=reg_parameters, y=lasso_validation_errors, mode='lines')])
    cv_fig.update_layout(
        title="Ridge and Lasso Models Train and Validation Errors for Different Regularization Parameters",
        xaxis_title="regularization parameter", yaxis_title="MSE")
    cv_fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_reg_parameter = reg_parameters[np.argmin(ridge_validation_errors)]
    rigde_fitted = RidgeRegression(best_reg_parameter).fit(train_X, train_y)
    ridge_loss = rigde_fitted.loss(test_X, test_y)
    print("Ridge model with lambda = ", best_reg_parameter, " test error = ", ridge_loss)

    best_reg_parameter = reg_parameters[np.argmin(lasso_validation_errors)]
    lasso_fitted = Lasso(alpha=best_reg_parameter).fit(train_X, train_y)
    lasso_loss = mean_square_error(lasso_fitted.predict(test_X), test_y)
    print("Lasso model with lambda = ", best_reg_parameter, " test error = ", lasso_loss)

    linear_fitted = LinearRegression().fit(train_X, train_y)
    linear_loss = linear_fitted.loss(test_X, test_y)
    print("Linear model test error = ", linear_loss)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
