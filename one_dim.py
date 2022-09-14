import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def one_dim_analysis(df: pd.DataFrame) -> str:
    """
    Func to fill statistical analysis multiline for 1 dimensional set
    :param df:
    :return string:
    """
    return 'N: ' + f'{len(df)}' + '\nМакс: ' + f'{df.max()[0]:.4f}' + '\nМін: ' + f'{df.min()[0]:.4f}' + '\nРозмах: ' + f'{df.max()[0] - df.min()[0]:.4f}' + '\nСереднє: ' + f'{df.mean()[0]:.4f}' + '\nМедіана: ' + f'{df.median()[0]:.4f}' + '\nДисперсія: ' + f'{df.var()[0]:.4f}' + '\nСер.кв.в: ' + f'{df.std()[0]:.4f}' + '\nАсиметрія: ' + asymmetry(
        df) + '\nЕксц: ' + kurtosis(df) + '\nКонтр-ексц: ' + c_kurtosis(df) + '\nПірсон.В: ' + pearson(
        df) + '\nMED Уолша: ' + walsh_med(df)


def asymmetry(df: pd.DataFrame) -> str:
    """
    Describes assymetry of Cumulative distribution function (CDF) - функції щільності розподілу,
    If result > 0 we can say that our CDF have assymetry to the left,
    res < 0 - to the right
    :param df:
    :return string, with assymetry value .4f:
    """
    buf = 0
    s = df.std()[0]  # s = standart deviation of dataframe
    m = df.mean()[0]  # m = mean of dataframe
    for i in range(len(df)):
        buf += (df[0][i] - m) ** 3  # buf is SUM of ((value in dataframe - mean) in pow 3)
    asymmetry = (buf / len(df)) / s ** 3
    return f'{asymmetry:.4f}'


def kurtosis(df: pd.DataFrame) -> str:
    """
    Function to find coef. of kurtosis (ексцес).
    Value describes how 'sharp' is our CDF in comparision with normal distribution,
    :param df:
    :return string, with value of kurtosis :.4f:
    """
    buf = 0
    s = df.std()[0]  # s = standart deviation of dataframe
    m = df.mean()[0]  # m = mean of dataframe
    for i in range(len(df)):
        buf = (df[0][i] - m) ** 4
    kurt = (buf / len(df)) / s ** 4
    return f'{kurt:.4f}'


def c_kurtosis(df: pd.DataFrame) -> str:
    """
    Контр-ексцес.
    Describes form of our distribution in comparision with Normal Distribution,
    where coef. < 0.515 - sharp form; coef. > 0.63 - 'chapiteau' form
    :param df:
    :return string, with c_kurt coef.:.4f:
    """

    # we need absolute value, use abs.
    return f'{1 / abs(float(kurtosis(df))) ** 0.5:.4f}'


def pearson(df: pd.DataFrame) -> str:
    """
    Pearson's variation coefficient, W  = std/mean, mean != 0
    If < 1 we can say that our set is appropriate
    Don't apply it when mean is around zero
    :param df:
    :return string with pearson coef, or 'None', or Mean = 0:
    """
    if df.mean()[0] != 0:  # we can't divide by zero
        pearson_var = df.std()[0] / df.mean()[0]
        if 0 <= pearson_var <= 1:
            return f'{pearson_var:.4f}'
        return 'None'
    else:
        return 'Mean=0'


def walsh_med(df: pd.DataFrame) -> str:
    """
    Uolsch Median. Just median that is more sustainable to anomalies.
    MED_uo = 1/2(Xi + Xj), 1<=i<=j<=N, N - length of set
    :param df:
    :return string, with walsh_med value:.4f:
    """
    N = len(df)
    return f'{(N / (N - 1) / 2) * 1 / 2:.4f}'


def one_dimens_graph(df: pd.DataFrame):
    """
    Function to make and show plots for 1 dimensional sets,
    KDE plot and histogram
    ecdf plot and confidence intervals

    :param df:
    :return nothing:
    """
    plt.style.context('seaborn')
    f = plt.figure(tight_layout=True)  # make figure
    plt.xlim([df.min()[0] - (df.min()[0] * 0.1), df.max()[0] + (df.min()[0] * 0.1)])  # make normal x limit

    # Below is some figure setting that we can live without
    f.set_figwidth(8)
    f.set_figheight(8)
    plt.subplot(2, 1, 1)
    plt.grid(visible=True, color='grey', linewidth=1,
             alpha=0.8)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Figure setting is done, now creating histogram with kde,
    # The formula for N<100 and N>100 is
    if len(df) <= 100:
        x = np.linspace(df.min()[0], df.max()[0], int(len(df) ** (1 / 2)))
        bins = int(len(df) ** (1 / 2))
    else:
        x = np.linspace(df.min()[0], df.max()[0], int(len(df) ** (1 / 3)))
        bins = int(len(df) ** (1 / 3) - 1)
    sns.histplot(df, bins=bins, kde=True)

    plt.subplot(2, 1, 2)
    sns.ecdfplot(x)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(visible=True, linewidth=1, alpha=0.8)
    plt.show()
