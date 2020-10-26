
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
plt.rcParams.update({'figure.figsize':(9,7)})

if __name__== "__main__":
    # Import data
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv', names=['value'], header=0)
    print(df)
    # Original Series

    """
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
    plot_acf(df.value, ax=axes[0, 1])
    result = adfuller(df.value.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    # 1st Differencing
    axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.value.diff().dropna(), ax=axes[1, 1])
    result = adfuller(df.value.diff().dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    # 2nd Differencing
    axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])
    result = adfuller(df.value.diff().diff().dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    plt.show()

    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0,5))
    plot_pacf(df.value.diff().dropna(), ax=axes[1])

    plt.show()
    """

    model = ARIMA(df.value, order=(1,1,1))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    model_fit.plot_predict(dynamic=False)
    plt.show()




