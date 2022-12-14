import pandas as pd
import numpy as np


def find_anomalies(df: pd.DataFrame) -> list:
    """
    Function to find anomalies (outliers)
    using 3 std rule
    :param df:
    :return LIST of anomalies:
    """
    anomalies = []
    s = df.std()[0]  # s = standart deviation of dataframe
    m = df.mean()[0]  # m = mean of dataframe
    if len(df) <= 100:  # if set length is less than 100 we can use 3 std
        anomaly_cut_off = s * 3
    else:  # else only 2, in order to save information
        anomaly_cut_off = s * 2
    l_lim = m - anomaly_cut_off  # lower limit
    up_lim = m + anomaly_cut_off  # upper limit
    for i in range(len(df)):
        if df[0][i] > up_lim or df[0][i] < l_lim:
            anomalies.append(df[0][i])

    return anomalies


def describing(df: pd.DataFrame, desc='') -> str:
    col_n = int(df.shape[1])
    for i in range(col_n):
        desc += f'max {str(df.describe()[i][7])}' + '\n' + str(df.describe()[i][1:4])
        desc += '\n'
    words = desc.split()
    for i in range(col_n):
        words.remove('dtype:')
        words.remove('float64')
    desc = ''
    for i in range(0, len(words), 2):
        desc += str(words[i]) + ' ' + str(words[i + 1] + '\n')
    return desc


def calc_cov(x: np.array, y: np.array) -> float:
    mean_x, mean_y = x.mean(), y.mean()
    n = len(x)

    return sum((x - mean_x) * (y - mean_y)) / n


def cov_mat(data: np.array) -> np.array:
    # get the rows and cols
    rows, cols = data.shape

    # the covariance matrix has a shape of n_features x n_features
    # n_featurs  = cols - 1 (not including the target column)
    matrix = np.zeros((cols, cols))

    for i in range(cols):
        for j in range(cols):
            # store the value in the matrix
            matrix[i][j] = calc_cov(data[:, i], data[:, j])

    return matrix


def cumulative_coef(eigen_val: np.array) -> list:
    summed = np.sum(eigen_val)
    cumulative_arr = []
    for item in eigen_val:
        cumulative_arr.append((item / summed)*100)
    return np.round(cumulative_arr[::-1], decimals=3)