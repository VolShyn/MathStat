import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def two_dim_analysis(df: pd.DataFrame) -> str:
    """
    Function to fill statistical analysis multiline for 2 dimensional set

    :param df:
    :return string:
    """
    return 'N: ' + f'{len(df)}' + '\nСер(0): ' + f'{df.mean()[0]:.4f}' + '\nСер(1): ' + f'{df.mean()[1]:.4f}' + '\nСер.кв.в(0): ' + f'{df.std()[0]:.4f}' + '\nСер.кв.в(1): ' + f'{df.std()[1]:.4f}' + '\nКор.Пірс: ' + pearson_corr(
        df) + '\nКор.Спірм: ' + f'{df.corr("spearman")[0][1]:.4f}' + '\nКор.Кендал: ' + f'{df.corr("kendall")[0][1]:.4f}' + '\nQ: ' + q(
        df) + '\nY: ' + y(df) + '\nI: ' + i(df)


def pearson_corr(df: pd.DataFrame) -> str:
    """
    Function to estimate pearson correlation coef. through covariance
    CovXY / (stdX * stdY)
    С = 0 sets are independent, C = 1 - functional dependence
    :param df:
    :return string with corr coef:.4f :
    """
    corr = (df.cov()[0][1]) / (df[0].std() * df[1].std())
    return f'{corr:.4f}'


def table_create(df: pd.DataFrame) -> np.array:
    """
    Function to create table for the future estimations (Q,I,Y)
    :param df:
    :return np.array (table):
    """
    table = np.empty((2, 2))
    m0 = df.mean()[0]  # mean for 0 column
    m1 = df.mean()[1]  # mean for 1 column
    for i in range(len(df)):
        if df[0][i] <= m0 and df[1][i] <= m1: table[0][0] += 1
        if df[0][i] > m0 and df[1][i] <= m1: table[0][1] += 1
        if df[0][i] <= m0 and df[1][i] > m1: table[1][0] += 1
        if df[0][i] > m0 and df[1][i] > m1: table[1][1] += 1
    return table


def q(df: pd.DataFrame) -> str:
    """
    Коефіцієнт Q зв’язку Юла
    varies from −1 to +1.
    Yule's Q, related to Y as: Q = (2Y) / (1+Y)^2
    :param df:
    :return string with value of Q:
    """
    table = table_create(df)
    q = (table[0, 0] * table[1, 1] - table[0, 1] * table[1, 0]) / (
            table[0, 0] * table[1, 1] + table[0, 1] * table[1, 0])
    return f'{q:.4f}'


def y(df: pd.DataFrame) -> str:
    """
    Коефіцієнт Y зв’язку Юла
    Yule's Y, coefficient of colligation, is a measure of association between two binary variables.
    varies from −1 to +1. −1 reflects total negative correlation, +1 reflects perfect positive association
    0 - no association
    related to Q as: Y = (1-sqrt(1-Q^2))/ Q
    :param df:
    :return string with value of Y:
    """
    table = table_create(df)
    y = ((table[0, 0] * table[1, 1]) ** 1 / 2 - (table[0, 1] * table[1, 0]) ** 1 / 2) / (
            (table[0, 0] * table[1, 1]) ** 1 / 2 + (table[0, 1] * table[1, 0]) ** 1 / 2)
    return f'{y:.4f}'


def i(df: pd.DataFrame) -> str:
    """
    Індекс Фехнера
    Fehner index
    |I| <= 1. I > 0 - positive corr, I < 0 - negative. I ~= 0 - no corr, so
    sets is independent.
    Works not only on binary data, but on discrete data too
    :param df:
    :return string with index:
    """
    table = table_create(df)
    i = (table[0, 0] + table[1, 1] - table[1, 0] - table[0, 1]) / (
            table[0, 0] + table[1, 1] + table[1, 0] + table[0, 1])
    return f'{i:.4f}'


def two_dimens_graph(df: pd.DataFrame):
    """
    Function to plot scatter plot with linear_reg,
    and kde plot
    :param df:
    :return nothing:
    """
    f = plt.figure(tight_layout=True)
    f.set_figwidth(10)
    f.set_figheight(10)
    plt.subplot(2, 1, 1)
    sns.regplot(x=df[0], y=df[1], ci=95, fit_reg=True, scatter_kws={"s": 30, "color": 'r'}, line_kws={"color": "black"})
    plt.subplot(2, 1, 2)
    sns.histplot(x=df[0], y=df[1],alpha = 0.65, cmap='coolwarm', cbar = True)
    sns.kdeplot(x=df[0], y=df[1], color='black', linewidths=1)
    plt.show()
