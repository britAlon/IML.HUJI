from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


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
    observations = np.random.multivariate_normal([0, 0, 4, 0],
                                                 [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]], 1000)
    multivariate_estimator.fit(observations)
    print(multivariate_estimator.mu_)
    print(multivariate_estimator.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    log_likelihood_arr = []
    for i in f1:
        for j in f3:
            log_likelihood_arr.append(
                multivariate_estimator.log_likelihood([i, 0, j, 0], multivariate_estimator.cov_, observations))
    go.Figure() \
        .add_trace(go.Histogram2dContour(x=f1, y=f3,
                                         colorscale='Blues', reversescale=True, xaxis='x', yaxis='y')) \
        .add_trace(go.Scatter(x=f1, y=f3, xaxis='x', yaxis='y', mode='markers',
                              marker=dict(color='rgba(0,0,0,0.3)', size=3))).update_layout(
        xaxis=dict(zeroline=False, domain=[0, 0.85], showgrid=False),
        yaxis=dict(zeroline=False, domain=[0, 0.85], showgrid=False),
        xaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
        yaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
        hovermode='closest', showlegend=False,
        title=r"$\text{(4) 2D scatter and marginal distributions}$"
    ) \
        .show()
    # Question 6 - Maximum likelihood


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
