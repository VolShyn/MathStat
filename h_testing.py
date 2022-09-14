import statistics
import pandas as pd
import scipy
from scipy.stats import uniform


def kolmogorov_1dim(X):
    statistic, pvalue = scipy.stats.kstest(X, 'norm')
    return f'{statistic:.6f}\npvalue: {pvalue:.6f}'


def kolmogorov_2dim(X, Y):
    statistic, pvalue = scipy.stats.ks_2samp(X, Y)
    return f'{statistic:.6f}\n'


def bartletts(X, Y):
    statistic, pvalue = scipy.stats.bartlett(X, Y)
    return f'Value: {statistic:.4f}\npvalue: {pvalue:.6f}'


def sign(data):
    N_plus = 0
    N_minus = 0
    m = statistics.mean(data)
    for i in range(len(data)):
        if data[i] > m:
            N_plus += 1
        elif data[i] < m:
            N_minus += 1
    try:
        p = scipy.stats.binomtest(min(N_plus, N_minus), N_plus + N_minus, 0.5).pvalue
    except AttributeError:
        p = scipy.stats.binom_test(min(N_plus, N_minus), N_plus + N_minus, 0.5)
    M = (N_plus - N_minus) / 2
    return f'M: {M}\npvalue: {p:.4f}'


def chi_squared(X):
    # x = np.linspace(X)
    statistic, pvalue = scipy.stats.chisquare(X)
    return f'{statistic:.6f}\npvalue: {pvalue:.6f}'