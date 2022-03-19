from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pio.templates.default = "simple_white"
LIKELIHOOD_NUM = 200


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariate_estimator = UnivariateGaussian()
    observations = np.random.normal(10, 1, 1000)
    univariate_estimator.fit(observations)
    print("(" + str(univariate_estimator.mu_) + "," + str(univariate_estimator.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    number_of_samples = np.linspace(10, 1000, 100).astype(int)
    absolute_distance = []
    for num in number_of_samples:
        partial_observations = np.random.choice(observations, num)
        absolute_distance.append(abs(np.mean(partial_observations) - univariate_estimator.mu_))
    go.Figure(go.Scatter(x=number_of_samples, y=absolute_distance, mode='markers+lines', name=r'$\widehat\mu$'),
              layout=go.Layout(
                  title=r"$\text{Absolute Distance Between Estimated and True Expectations As Function Of Number Of Samples}$",
                  xaxis_title="$\\text{number of samples}$",
                  yaxis_title=r"$ |\mu - \hat\mu|$",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_observations = np.sort(observations)
    theoretical_distribution = univariate_estimator.pdf(sorted_observations)
    go.Figure(go.Scatter(x=sorted_observations, y=theoretical_distribution, mode='lines',
                         name=r"$Empirical PDF of samples$"),
              layout=go.Layout(title=r"$\text{Empirical PDF of samples}$",
                               xaxis_title="$\\text{sample}$",
                               yaxis_title=r"$PDF$",
                               height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multivariate_estimator = MultivariateGaussian()
    cov_matrix = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    observations = np.random.multivariate_normal([0, 0, 4, 0], cov_matrix, 1000)
    multivariate_estimator.fit(observations)
    print(multivariate_estimator.mu_)
    print(multivariate_estimator.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, LIKELIHOOD_NUM)
    f3 = np.linspace(-10, 10, LIKELIHOOD_NUM)
    log_likelihood_arr = []
    expectations = []
    for i in f1:
        for j in f3:
            expectations.append([i, 0, j, 0])
            log_likelihood_arr.append(
                multivariate_estimator.log_likelihood([i, 0, j, 0], cov_matrix, observations))

    # create a data frame of f1, f3 and the log-likelihood result
    data = pd.DataFrame(
        {'f1': np.repeat(f1, LIKELIHOOD_NUM), 'f3': np.tile(f3, LIKELIHOOD_NUM), 'likelihood': log_likelihood_arr})
    data_pivoted = data.pivot('f1', 'f3', 'likelihood')
    # plot the data frame
    fig, ax = plt.subplots(1, 1)
    hm = sns.heatmap(data_pivoted, ax=ax)
    ax.set_title("The Log-Likelihood for Models with Expectations [f1, 0, f3, 0]")
    ax.set_ylabel("f1")
    ax.set_xlabel("f3")
    labels = [np.around(float(item.get_text()), 1) for item in hm.get_xticklabels()]
    hm.set_yticklabels(labels)
    hm.set_xticklabels(labels)
    plt.show()

    # Question 6 - Maximum likelihood
    print(np.around(expectations[np.argmax(log_likelihood_arr)], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
