"""
Contains utility functions I used for exploring the time series data. It provides tools
to plot monthly and yearly aggregates, perform the Dickey-Fuller test to check for
stationarity, and visualize predictions against training data. These utilities are used
to facilitate initial data analysis and help in understanding underlying patterns before
the modeling phase.
"""

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def plot_monthly_aggregates(df):
    monthly_avg = df.groupby(df.index.month).mean()
    monthly_avg.plot(kind='bar')


def plot_yearly_aggregates(df):
    yearly_avg = df.groupby(df.index.year).mean()
    yearly_avg.plot(kind='bar')


def adf_test(df):
    print('Results of Dickey-Fuller Test:')
    adf_result = adfuller(df)
    adf_output = pd.Series(
        adf_result[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    )

    for key, value in adf_result[4].items():
        adf_output['Critical Value (%s)' % key] = value

    print(adf_output)


def plot_preds(train_data, preds):
    plt.figure(figsize=(16, 8))

    plt.plot(train_data, "b.-")
    plt.plot(preds, "ro-")

    plt.legend(('Train Data', 'Forecast'), fontsize=16)

    plt.title('Results', fontsize=20)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Time', fontsize=16)
