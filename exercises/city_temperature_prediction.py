import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    process the data given in the data frame.
    :param data:
    :return: a new data frame  with processed data.
    """
    data.dropna(inplace=True)
    for column in ['Year', 'Month', 'Day']:
        data = data[data[column] > 0]
    data = data[data['Temp'] > -5]  # lowest average temperature in The Netherlands was -3
    data['DayOfYear'] = data['Date'].dt.dayofyear
    return data


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date'])
    data = process_data(data)
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('C:\\Users\\brita\\IML.HUJI\\datasets\\City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    df_Israel = df.loc[df['Country'] == 'Israel']
    df_Israel["Year"] = df_Israel["Year"].astype(str)
    fig_scatter_Q2 = px.scatter(df_Israel, x="DayOfYear", y="Temp", color='Year',
                                title=f"Average Temperature in Israel as Function of Day of Year",
                                labels={"DayOfYear": 'Day of Year', 'Temp': 'Temperature'})
    fig_scatter_Q2.show()
    df_months_temps = df_Israel.groupby("Month")['Temp'].agg(np.std).reset_index()
    fig_bar_Q2 = px.bar(df_months_temps, x='Month', y='Temp',
                        title='Standard Deviation of Avg Daily Temperature as Function of Month',
                        labels={'Temp': 'Std of Temperature'})
    fig_bar_Q2.show()

    # Question 3 - Exploring differences between countries
    df_all_countries = df.groupby(['Country', 'Month'])['Temp'].agg([np.mean, np.std]).reset_index()
    fig_line_Q3 = px.line(df_all_countries, x='Month', y='mean', color='Country', error_y='std',
                          title='Average Monthly Temperature in Different Countries',
                          labels={'mean': 'Average Monthly Temperature'})
    fig_line_Q3.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(df_Israel["DayOfYear"].to_frame(), df_Israel["Temp"], 0.75)
    losses = []
    for k in range(1, 11):
        fitted_model = PolynomialFitting(k).fit(np.asarray(train_X.values.flatten()), train_y.to_numpy())
        k_loss = fitted_model.loss(np.asarray(test_X.values.flatten()), test_y.to_numpy())
        print(f"The test error for a model with k = {k} is", np.around(k_loss, 2))
        losses.append(np.around(k_loss, 2))
    fig_bar_Q4 = px.bar(x=np.arange(1, 11), y=losses,
                        title="Test Loss as a Function of the Degree of the Polynomial Fitting",
                        labels={"x": "polynomial degree", "y": "loss"})
    fig_bar_Q4.show()

    # Question 5 - Evaluating fitted model on different countries
    chosen_k = losses.index(min(losses)) + 1
    israel_fitted_model = PolynomialFitting(chosen_k).fit(np.asarray(df_Israel["DayOfYear"]),
                                                          np.asarray(df_Israel["Temp"]))
    country_loss = []
    for country in ['Jordan', 'South Africa', 'The Netherlands']:
        df_country = df.loc[df['Country'] == country]
        loss = israel_fitted_model.loss(np.asarray(df_country['DayOfYear']), np.asarray(df_country['Temp']))
        country_loss.append(loss)
    fig_bar_Q5 = px.bar(pd.DataFrame({"country": ['Jordan', 'South Africa', 'The Netherlands'], "model error": country_loss}), x='country',
                            y='model error', title='Error of Israel Fitted Model Over Other Countries')
    fig_bar_Q5.show()
