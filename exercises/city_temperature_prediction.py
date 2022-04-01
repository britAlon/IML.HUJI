import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import datetime

pio.templates.default = "simple_white"


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    data.dropna(inplace=True)
    for column in ['Year', 'Month', 'Day', 'Temp']:
        data = data[data[column] > 0]
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
    fig_scatter = px.scatter(df_Israel, x="DayOfYear", y="Temp", color='Year',
                             title=f"Average Temperature in Israel as Function of Day of Year",
                             labels={"DayOfYear": 'Day of Year', 'Temp': 'Temperature'})
    # fig_scatter.show()
    df_months_temps = df_Israel.groupby("Month")['Temp'].agg(np.std).reset_index()
    fig_bar = px.bar(df_months_temps, x='Month', y='Temp',
                     title='Standard Deviation of Avg Daily Temperature as Function of Month',
                     labels={'Temp': 'Std of Temperature'})
    # fig_bar.show()

    # Question 3 - Exploring differences between countries
    df_all_countries = df.groupby(['Country', 'Month'])['Temp'].agg([np.mean, np.std]).reset_index()
    line_fig = px.line(df_all_countries, x='Month', y='mean', color='Country', error_y='std',
                       title='Average Monthly Temperature in Different Countries',
                       labels={'mean': 'Average Monthly Temperature'})
    # line_fig.show()
    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(df_Israel["DayOfYear"].to_frame(), df_Israel["Temp"], 0.75)
    arr_train = np.asarray(train_X.values.flatten())
    arr_test = np.asarray(test_X.values.flatten())

    losses = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        fitted = model.fit(arr_train, train_y.to_numpy())
        k_loss = fitted.loss(arr_test, test_y.to_numpy())
        print(np.around(k_loss, 3))
        losses.append(np.around(k_loss, 3))
    px.bar(x=np.arange(1, 11), y=losses, title="test loss as a function of the degree of the polynomial fitting").show()

    # Question 5 - Evaluating fitted model on different countries
