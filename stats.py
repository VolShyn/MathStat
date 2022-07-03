import numpy as np
import pandas as pd
import PySimpleGUI as sg
import statistics
import seaborn as sns
import scipy
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from scipy.stats import uniform, expon
from sklearn.metrics import mean_squared_error
import math

'''
Below is some functions to calculate 1 dimension
'''


def one_dim_analysis(df):
    """
    Func to fill statistical analysis multiline for 1 dimensional set
    :param df:
    :return string:
    """
    return 'N: ' + f'{len(df)}' + '\nМакс: ' + f'{df.max()[0]:.4f}' + '\nМін: ' + f'{df.min()[0]:.4f}' + '\nРозмах: ' + f'{df.max()[0] - df.min()[0]:.4f}' + '\nСереднє: ' + f'{df.mean()[0]:.4f}' + '\nМедіана: ' + f'{df.median()[0]:.4f}' + '\nДисперсія: ' + f'{df.var()[0]:.4f}' + '\nСер.кв.в: ' + f'{df.std()[0]:.4f}' + '\nАсиметрія: ' + asymmetry(
        df) + '\nЕксц: ' + kurtosis(df) + '\nКонтр-ексц: ' + c_kurtosis(df) + '\nПірсон.В: ' + pearson(
        df) + '\nMED Уолша: ' + walsh_med(df)


def asymmetry(df):
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


def kurtosis(df):
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


def c_kurtosis(df):
    """
    Контр-ексцес.
    Describes form of our distribution in comparision with Normal Distribution,
    where coef. < 0.515 - sharp form; coef. > 0.63 - 'chapiteau' form
    :param df:
    :return string, with c_kurt coef.:.4f:
    """

    # we need absolute value, use abs.
    return f'{1 / abs(float(kurtosis(df))) ** 0.5:.4f}'


def pearson(df):
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


def walsh_med(df):
    """
    Uolsch Median. Just median that is more sustainable to anomalies.
    MED_uo = 1/2(Xi + Xj), 1<=i<=j<=N, N - length of set
    :param df:
    :return string, with walsh_med value:.4f:
    """
    N = len(df)
    return f'{(N / (N - 1) / 2) * 1 / 2:.4f}'


def find_anomalies(df):
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


def one_dimens_graph(df):
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
    plt.grid(b=True, color='grey', linewidth=1,
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
    sns.distplot(df, bins=bins, kde=True,
                 kde_kws={'alpha': 0.85, 'color': 'black'})

    plt.subplot(2, 1, 2)

    ci1 = x * 0.85  # confidence interval from below
    ci2 = x * 1.15  # confidence interval from up
    sns.ecdfplot(x)
    sns.kdeplot(ci1, cumulative=True)
    sns.kdeplot(x, x=0, cumulative=True)
    sns.kdeplot(ci2, x=0, cumulative=True)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(b=True, color='grey', linewidth=1,
             alpha=0.8)
    plt.show()


'''
Below is functions to calculate 2 dimensions
'''


def two_dim_analysis(df):
    """
    Function to fill statistical analysis multiline for 2 dimensional set

    :param df:
    :return string:
    """
    return 'N: ' + f'{len(df)}' + '\nСер(0): ' + f'{df.mean()[0]:.4f}' + '\nСер(1): ' + f'{df.mean()[1]:.4f}' + '\nСер.кв.в(0): ' + f'{df.std()[0]:.4f}' + '\nСер.кв.в(1): ' + f'{df.std()[1]:.4f}' + '\nКор.Пірс: ' + pearson_corr(
        df) + '\nКор.Спірм: ' + f'{df.corr("spearman")[0][1]:.4f}' + '\nКор.Кендал: ' + f'{df.corr("kendall")[0][1]:.4f}' + '\nQ: ' + q(
        df) + '\nY: ' + y(df) + '\nI: ' + i(df)


def pearson_corr(df):
    """
    Function to estimate pearson correlation coef. through covariance
    CovXY / (stdX * stdY)
    С = 0 sets are independent, C = 1 - functional dependence
    :param df:
    :return string with corr coef:.4f :
    """
    corr = (df.cov()[0][1]) / (df[0].std() * df[1].std())
    return f'{corr:.4f}'


def table_create(df):
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


def q(df):
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


def y(df):
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


def i(df):
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


def two_dimens_graph(df):
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
    sns.regplot(x=df[0], y=df[1], marker='+', ci=70)
    plt.subplot(2, 1, 2)
    sns.histplot(x=df[0], y=df[1])
    sns.kdeplot(x=df[0], y=df[1], color='black', linewidths=2)
    plt.show()


"""
Hypothesis testing functions
"""


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


"""
Multidimensional funcs
"""


def describing(df, desc=''):
    colN = int(df.shape[1])
    for i in range(colN):
        desc += f'max {str(df.describe()[i][7])}' + '\n' + str(df.describe()[i][1:4])
        desc += '\n'
    words = desc.split()
    for i in range(colN):
        words.remove('dtype:')
        words.remove('float64')
    desc = ''
    for i in range(0, len(words), 2):
        desc += str(words[i]) + ' ' + str(words[i + 1] + '\n')
    return desc


"""
HELP_INFO
"""


def help_info(event):
    if event == 'Anomalies':
        sg.Popup('Anomalies is auto-detected\n'
                 '*only for 1-dimension', title='Anomalies', icon='math.ico')
    if event == 'Log()':
        sg.popup('You can logarithm in figure settings.\n', title='Log()', icon='math.ico')
    if event == 'About...':
        sg.popup('This program was build by V.Shynkarov', title='About', icon='math.ico')
    if event == 'General average':
        sg.popup_no_buttons('Check hypothesis about equality of average in population',
                            title='General average', icon='math.ico', image='Pearson.png')
    if event == 'Pearson':
        sg.popup_no_buttons(
            "The chi-square test for independence, also called Pearson's chi-square test or the chi-square test of association, is used to discover if there is a relationship between two categorical variables.",
            title='Pearson', icon='math.ico', image='Pearson.png')
    if event == 'Kolmogorov':
        sg.popup_no_buttons(
            "Kolmogorov–Smirnov test (K–S test or KS test): is used to decide if a sample comes from a population with a specific distribution.one-dimensional probability distributions that can be used to compare a sample with a reference probability distribution (one-sample K–S test), or to compare two samples (two-sample K–S test). In essence, the test answers the question 'What is the probability that this collection of samples could have been drawn from that probability distribution?' or, in the second case, 'What is the probability that these two sets of samples were drawn from the same (but unknown) probability distribution?'\n"
            "\n*Func for 1-sample and 2-sample test",
            title='Kolmogorov', icon='math.ico', image='kolmogorov.png')
    if event == 'F-test':
        sg.popup_no_buttons(
            "A Statistical F Test uses an F Statistic to compare two variances, s1 and s2, by dividing them. The result is always a positive number (because variances are always positive). It is most often used when comparing statistical models that have been fitted to a data set, in order to identify the model that best fits the population from which the data were sampled. ",
            title='F-test', icon='math.ico', image='F-test.png')
    if event == 'Bartletts test':
        sg.popup_no_buttons(
            "Bartlett's test is used to test the null hypothesis, H0 that all k population variances are equal against the alternative that at least two are different.",
            title='Bartletts', icon='math.ico', image='Bartletts.png')
    if event == 'Wilcoxon':
        sg.popup_no_buttons(
            "The Wilcoxon signed-rank test is a non-parametric statistical hypothesis test used either to test the location of a set of samples or to compare the locations of two populations using a set of matched samples.[1] When applied to test the location of a set of samples, it serves the same purpose as the one-sample Student's t-test.[",
            title='Wilcoxon', icon='math.ico', image='Wilcoxon.png')
    if event == 'Sign test':
        sg.popup_no_buttons(
            "The sign test is a statistical method to test for consistent differences between pairs of observations, such as the weight of subjects before and after treatment. Given pairs of observations (such as weight pre- and post-treatment) for each subject, the sign test determines if one member of the pair (such as pre-treatment) tends to be greater than (or less than) the other member of the pair (such as post-treatment).\n"
            "Procedure:\n"
            "Calculate the + and – sign for the given distribution.  Put a + sign for a value greater than the mean value, and put a – sign for a value less than the mean value.  Put 0 as the value is equal to the mean value; pairs with 0 as the mean value are considered ties.\n"
            "Denote the total number of signs by ‘n’ (ignore the zero sign) and the number of less frequent signs by ‘S.’\n"
            "Obtain the critical value (K) at .05 of the significance level by using the following formula in case of small samples:",
            title='Sign', icon='math.ico', image='Sign.png')
    if event == 'Single factor':
        sg.popup_no_buttons(
            "The one-way ANOVA compares the means between the groups you are interested in and determines whether any of those means are statistically significantly different from each other. Specifically, it tests the null hypothesis:\n"
            "where µ = group mean and k = number of groups. If, however, the one-way ANOVA returns a statistically significant result, we accept the alternative hypothesis (HA), which is that there are at least two group means that are statistically significantly different from each other.",
            title='Single Factor', icon='math.ico', image='SingleFactor.png')
    if event == 'H-test':
        sg.popup_no_buttons(
            "Kruskal–Wallis H test is a non-parametric method for testing whether samples originate from the same distribution It is used for comparing two or more independent samples of equal or different sample sizes. It extends the Mann–Whitney U test, which is used for comparing only two groups. The parametric equivalent of the Kruskal–Wallis test is the Single Factor analysis of variance",
            title='H-test', icon='math.ico', image='H-test.png')
    if event == 'T-test':
        sg.popup_no_buttons(
            "Essentially, a t-test allows us to compare the average values of the two data sets and determine if they came from the same population. In the above examples, if we were to take a sample of students from class A and another sample of students from class B, we would not expect them to have exactly the same mean and standard deviation. Similarly, samples taken from the placebo-fed control group and those taken from the drug prescribed group should have a slightly different mean and standard deviation.Mathematically, the t-test takes a sample from each of the two sets and establishes the problem statement by assuming a null hypothesis that the two means are equal. Based on the applicable formulas, certain values are calculated and compared against the standard values, and the assumed null hypothesis is accepted or rejected accordingly. \n"
            "\n"
            "**Exists 1-sample and 2-sample t-tests, check formula",
            title='t-test', icon='math.ico', image='T-test.png')
    if event == 'Laplace table':
        sg.popup_no_buttons('',
                            title='Laplace table', icon='math.ico', image='Laplas.png')
    if event == 'Inverse transform sampling':
        sg.popup_no_buttons(
            'Is a basic method for pseudo-random number sampling, i.e., for generating sample numbers at random from any probability distribution given its cumulative distribution function.\n'
            '*In programm only normal and exponential variants are available',
            title='Inversion sampling', icon='math.ico', image='')
