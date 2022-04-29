from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IMLearn.metrics import loss_functions
from math import atan2, pi

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :-1], data[:, -1]


def my_callback(fit: Perceptron, x: np.ndarray, y: int):
    fit.training_loss.append(fit.loss(fit.current_features, fit.current_labels))


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = np.load("C:\\Users\\brita\\IML.HUJI\\datasets\\" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(callback=my_callback)
        perceptron.fit(data[:, :-1], data[:, -1])
        losses = perceptron.training_loss

        # Plot figure
        go.Figure([go.Scatter(x=np.arange(perceptron.max_iter_), y=losses, mode="lines")],
                  layout=go.Layout(title=f"Perceptron Losses Over {n} Data",
                                   xaxis={"title": "$Perceptron Iterations$"},
                                   yaxis={"title": r"$Training Losses$"})).show()


def get_cov_ellipse(cov, center):
    """
    Return x,y values of an ellipse representing the covariance matrix cov centred at center.
    """
    v, w = np.linalg.eig(cov)
    n_std = 1
    t = np.linspace(0, 2 * np.pi, 50)
    x_t = n_std * np.sqrt(v[0]) * np.cos(t) * w[0][0] + n_std * np.sqrt(v[1]) * w[0][1] * np.sin(t) + center[0]
    y_t = n_std * np.sqrt(v[0]) * np.cos(t) * w[1][0] + n_std * np.sqrt(v[1]) * w[1][1] * np.sin(t) + center[1]
    return np.array(x_t), np.array(y_t)


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("C:\\Users\\brita\\IML.HUJI\\datasets\\" + f)
        # Fit models and predict over training set
        models = np.array([LDA(), GaussianNaiveBayes()])
        model_names = np.array(["LDA", "GNB"])
        predictions = np.array([m.fit(X, y).predict(X) for m in models])
        accuracies = np.array([loss_functions.accuracy(y, y_pred) for y_pred in predictions])

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[rf"$\textbf{{{m}, accuracy = {np.around(a, 3)}}}$" for m, a in
                                            zip(model_names, accuracies)])
        for i in range(len(models)):
            fig.add_trace(
                go.Scatter(x=X[:, 0], y=X[:, 1], marker=dict(color=np.uint32(predictions[i]), symbol=np.uint32(y)),
                           mode="markers", showlegend=False,
                           text=["True Class {0}, Predicted Class {1}".format(str(y[i]), str(predictions[0][i])) for i
                                 in range(X.shape[0])], hovertemplate='%{text}'), row=1, col=i + 1)
            fig.add_trace(go.Scatter(x=models[i].mu_[:, 0], y=models[i].mu_[:, 1],
                                     marker=dict(symbol='x', color='black', size=15),
                                     mode="markers", showlegend=False), row=1, col=i + 1)

        # plot ellipsis that matches the covariance matrices
        for model_idx in range(len(models)):
            for class_idx in range(len(models[model_idx].classes_)):
                if model_idx == 0:
                    fig.add_trace(get_ellipse(models[0].mu_[class_idx], models[0].cov_), row=1, col=1)
                else:
                    fig.add_trace(get_ellipse(models[1].mu_[class_idx], np.diag(models[1].vars_[class_idx])), row=1, col=2)


        fig.update_layout(title_text=f"True and Predicted Classes of LDA and GNB Models, {f.split('.')[0]} dataset")
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
