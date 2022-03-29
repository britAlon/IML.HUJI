from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'

pio.templates.default = "simple_white"

THIS_YEAR = 2022


# functions added #

def erase_invalid_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    erase rows with blanks or rows that contains zero in columns that can not be zero (sqft of living room and lot, number of floors, year the house was built in,
    :param data:
    :return:
    """
    data.dropna(inplace=True)
    return data[
        (data.sqft_living > 0) & (data.sqft_lot > 0) & (data.price > 0) & (data.floors > 0) & (data.yr_built > 0) & (
                data.sqft_living15 > 0) & (data.sqft_lot15 > 0)]


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    data = erase_invalid_rows(data)
    # modify the yr_built column to express the time passed from the house building.
    data.loc[:, 'yr_built'] = data['yr_built'].apply(lambda x: THIS_YEAR - x)
    # modify the yr_renovated column to express the time passed from last renovation.
    data.loc[:, 'yr_renovated'] = data['yr_renovated'].apply(lambda x: (1 / (THIS_YEAR - x)) if (x > 0) else x)
    # modify the longitude and latitude into geohash codes that will be dummy values
    geo_data = data.apply(lambda x: encode(x.lat, x.long, precision=4), axis=1)
    dummy = pd.get_dummies(geo_data)
    data = data.merge(dummy, left_index=True, right_index=True)
    for column in ['zipcode', 'lat', 'long', 'id']:
        data = data.drop(column, axis=1)
    print(data.info())
    return data


def encode(latitude, longitude, precision=12):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    geohash = []
    bits = [ 16, 8, 4, 2, 1 ]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += __base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)

# original exercise functions #

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)
    data = process_data(data)
    y_true = data['price']
    data = data.drop('price', axis=1)
    return data, y_true


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    load_data('C:\\Users\\brita\\IML.HUJI\\datasets\\house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    # raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
