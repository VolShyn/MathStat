import numpy as np
import PySimpleGUI as sg
import statistics
import seaborn as sns
import scipy
import phik
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef
from mlxtend.evaluate import cochrans_q
from scipy.stats import uniform, expon

'''
Below is some functions to calculate one dimension
'''


def one_dim_analysis(data, text=''):
    text += 'Length: ' + f'{len(data)}' + '\nSumm: ' + f'{abs(sum(data)):.2f}' + '\nMax: ' + f'{max(data):.4f}' + '\nMin: ' + f'{min(data):.4f}' + '\nScale: ' + f'{max(data) - min(data):.4f}' + '\nMean: ' + f'{abs(statistics.mean(data)):.4f}' + '\nMed: ' + f'{statistics.median(data):.4f}' + '\nVar: ' + f'{statistics.variance(data):.4f}' + '\nStDev: ' + f'{statistics.stdev(data):.4f}' + '\nAsym: '
    return text + asymmetry(data) + '\nEcs: ' + f'{kurtosis(data):.4f}' + '\nCEcs: ' + c_kurtosis(
        data) + '\nPirsCor: ' + pearson(data) + '\nMedU: ' + f'{med_oul(data):.4f}' + '\nQuantiles: ' + quant(data)


def asymmetry(data):
    buf = 0
    k = 0
    s = statistics.stdev(data)
    m = statistics.mean(data)
    while (k < len(data)):
        buf += (data[k] - m) ** 3
        k += 1
    asymmetry = (buf / len(data)) / s ** 3
    return f'{asymmetry:.4f}'


def kurtosis(data):
    buf = 0
    k = 0
    m = statistics.mean(data)
    s = statistics.stdev(data)
    while (k < len(data)):
        buf += (data[k] - m) ** 4
        k += 1
    kurtosis = (buf / len(data)) / s ** 4
    return round(kurtosis, 4)


def c_kurtosis(data):
    return f'{1 / abs(kurtosis(data)) ** 0.5:.4f}'


def pearson(data):
    if np.mean(data) != 0:
        res = np.std(data) / np.mean(data)
        if 0 <= res <= 1:
            return f'{res:.4f}'
        return 'None'
    else:
        return 'Mean=0'


def med_oul(data):
    return round(data[round(len(data) / 2)], 4)


def quant(data):
    try:
        # temp = [round(x, 2) for x in statistics.quantiles(data)]
        return str(list(map(lambda x: f'{x:.2f}', statistics.quantiles(data))))
    except:
        pass


def find_anomalies(data):
    anomalies = []
    s = statistics.stdev(data)
    m = statistics.mean(data)
    if len(data) <= 100:
        anomaly_cut_off = s * 3
    else:
        anomaly_cut_off = s * 2
    lower_limit = m - anomaly_cut_off
    upper_limit = m + anomaly_cut_off
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies


import math


def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def exponential_inverse_trans(data_size, mean=1):
    U = uniform.rvs(size=data_size)
    actual = expon.rvs(size=data_size, scale=mean)
    return actual


def one_dimens_graph(arr):
    plt.style.context('seaborn')
    f = plt.figure(tight_layout=True)
    f.set_figwidth(8)
    f.set_figheight(8)
    sns.set_theme(style="whitegrid")
    plt.subplot(2, 1, 1)
    plt.grid(b=True, color='grey',
             linestyle='-.', linewidth=0.5,
             alpha=0.6)
    plt.xlim([min(arr) - (min(arr) * 0.1), max(arr) + (min(arr) * 0.1)])
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=14)

    x = np.linspace(min(arr), max(arr), len(arr))  # linspace(0,1,5) == array(0,0.25,0.5,0.75,1)

    if len(arr) <= 100:
        sns.distplot(arr, bins=(int(len(arr) ** (1 / 2))), kde=True,
                     hist_kws={'alpha': 0.6, 'color': 'g'},
                     kde_kws={'alpha': 0.8, 'color': 'black'})
        x = np.linspace(min(arr), max(arr), int(len(arr) ** (1 / 2)))
    else:
        sns.distplot(arr, bins=(int(len(arr) ** (1 / 3)) - 1), kde=True,
                     hist_kws={'alpha': 0.6, 'color': 'g'},
                     kde_kws={'alpha': 0.8, 'color': 'black'})
        x = np.linspace(min(arr), max(arr), int(len(arr) ** (1 / 3)))

    # making emperical graph

    plt.subplot(2, 1, 2)
    ci1 = [x * 0.95 for x in arr]  # confidence interval 0.95 from down
    ci2 = [x * 1.05 for x in arr]  # confidence interval from up
    sns.ecdfplot(x)
    sns.kdeplot(ci1, cumulative=True)
    sns.kdeplot(arr, cumulative=True)
    sns.kdeplot(ci2, cumulative=True)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()


'''
Below is functions to calculate 2D
'''


def two_dim_analysis(X, Y, text=''):
    # cov(X, Y) = (sum (x - mean(X)) * (y - mean(Y)) ) * 1/(n-1)
    return 'Length: ' + f'{len(X)}' + '\nMean(X): ' + f'{np.mean(X):.4f}' + '\nMean(Y): ' + f'{np.mean(Y):.4f}' + '\nSt.Dev(X): ' + f'{np.std(X):.4f}' + '\nSt.Dev(Y): ' + f'{np.std(Y):.4f}' + '\nCorr: ' + pearson_corr(
        X, Y) + '\nSpearman: ' + spearman_correlation(X, Y) + '\nKendall: ' + kendall_correlation(X,
                                                                                                  Y) + '\nQ: ' + f'{q(X, Y):.4f}' + '\nY: ' + f'{y(X, Y):.4f}' + '\n(I)Fehn: ' + f'{i(X, Y):.4f}'
    # return text


def covariation(X, Y):
    # cov = (sum(x - statistics.mean(X)) * (y - statistics.mean(Y))) * 1 / (n - 1)
    cov1, cov2 = np.cov(X, Y)
    return f'{cov1[1]}'


def pearson_corr(X, Y):
    corr1 = np.corrcoef(X, Y)
    return f'{corr1[0][1]:.4f}'


def spearman_correlation(X, Y):
    # return f'{np.cov(np.rank(X), np.rank(Y)) / (np.std(np.rank(X)) * np.std(np.rank(Y)))}:.4f'
    coeff, p = scipy.stats.spearmanr(X, Y)
    return f'{coeff:.4f}'


def kendall_correlation(X, Y):
    coeff, p = scipy.stats.kendalltau(X, Y)
    return f'{coeff:.4f}'


def table_create(X, Y):
    table = np.empty((2, 2))
    for i in range(len(X)):
        if X[i] <= np.mean(X) and Y[i] <= np.mean(Y): table[0][0] += 1
        if X[i] > np.mean(X) and Y[i] <= np.mean(Y): table[0][1] += 1
        if X[i] <= np.mean(X) and Y[i] > np.mean(Y): table[1][0] += 1
        if X[i] > np.mean(X) and Y[i] > np.mean(Y): table[1][1] += 1
    return table


def q(X, Y):
    table = table_create(X, Y)
    q = (table[0, 0] * table[1, 1] - table[0, 1] * table[1, 0]) / (
            table[0, 0] * table[1, 1] + table[0, 1] * table[1, 0])
    return q


def y(X, Y):
    table = table_create(X, Y)
    y = (math.sqrt(table[0, 0] * table[1, 1]) - math.sqrt(table[0, 1] * table[1, 0])) / (
            math.sqrt(table[0, 0] * table[1, 1]) + math.sqrt(table[0, 1] * table[1, 0]))
    return y


def i(X, Y):
    table = table_create(X, Y)
    i = (table[0, 0] + table[1, 1] - table[1, 0] - table[0, 1]) / (
            table[0, 0] + table[1, 1] + table[1, 0] + table[0, 1])
    return i


def phi_coef(X, Y):
    pass


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


def two_dimens_graph(X, Y):
    sns.set_theme(style="dark")
    f = plt.figure(tight_layout=True)
    f.set_figwidth(8)
    f.set_figheight(8)
    plt.subplot(2, 1, 1)
    sns.regplot(X, Y)
    plt.subplot(2, 1, 2)
    sns.set_theme(style="dark")
    sns.histplot(x=X, y=Y)
    sns.kdeplot(x=X, y=Y, color="black", linewidths=1)
    plt.show()


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


def check_for_columns(txt):
    try:
        test2 = []
        test1 = []
        try:
            test, test1, test2 = np.loadtxt(txt, unpack=True)
        except ValueError:
            pass
        try:
            test, test1 = np.loadtxt(txt, unpack=True)
        except ValueError:
            pass

        if type(test2) != list:
            return 3
        elif type(test1) != list:
            return 2
        else:
            return 1
    except:
        return 0
