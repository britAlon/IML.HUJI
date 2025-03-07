import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import loss_functions


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    fitted_adaboost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_error, test_error = [], []
    for t in range(1, n_learners):
        train_error.append(fitted_adaboost.partial_loss(train_X, train_y, t))
        test_error.append(fitted_adaboost.partial_loss(test_X, test_y, t))

    fig = go.Figure(
        [go.Scatter(x=np.linspace(0, n_learners, n_learners), y=train_error, mode="lines", name="train error"),
         go.Scatter(x=np.linspace(0, n_learners, n_learners), y=test_error, mode="lines", name="test error")],
        layout=go.Layout(title="Train and test errors as a function of fitted learners",
                         xaxis={"title": r"$Number of Learners$"},
                         yaxis={"title": r"$MSE$"}))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.01, .01])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t}}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(fitted_adaboost.partial_predict, lims[0], lims[1], t, showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries Obtained By Ensembles With Different Number Of Learners}}$",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_ensemble_num = np.argmin(test_error)
    fitted_ensemble = AdaBoost(DecisionStump, best_ensemble_num).fit(train_X, train_y)
    ensemble_accuracy = loss_functions.accuracy(test_y, fitted_ensemble.predict(test_X))
    fig = make_subplots(rows=1, cols=1, subplot_titles=[rf"$\textbf{{Decision Boundaries Of Ensemble With "
                                                        rf"{best_ensemble_num} Learners, Accuracy = {ensemble_accuracy}}}$"],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces([decision_surface(fitted_adaboost.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y))], rows=1, cols=1)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 4: Decision surface with weighted samples
    sample_D = fitted_adaboost.D_
    sample_D = sample_D / np.max(sample_D) * 5
    fig = make_subplots(rows=1, cols=1, subplot_titles=[
        rf"$\textbf{{Decision Boundaries Of Ensemble With {n_learners} Learners, and Sample Size Obtained by the Sample Weights}}$"],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces([decision_surface(fitted_adaboost.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=train_y, size=sample_D))], rows=1, cols=1)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
