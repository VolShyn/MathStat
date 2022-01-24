import numpy as np
import PySimpleGUI as sg
import statistics


def scale(data):
    return round(max(data) - min(data), 4)


def med(data):
    return round(statistics.median(data), 4)


def var(data):
    return round(statistics.variance(data), 4)


def mean(data):
    return round(statistics.mean(data), 4)


def standart_d(data):
    return round(statistics.stdev(data), 4)


def asymmetry(data):
    buf = 0
    k = 0
    while (k < len(data)):
        buf += (data[k] - mean(data)) ** 3
        k += 1
    asymmetry = (buf / len(data)) / standart_d(data) ** 3
    return round(asymmetry, 4)


def kurtosis(data):
    buf = 0
    k = 0
    while (k < len(data)):
        buf += (data[k] - mean(data)) ** 4
        k += 1
    kurtosis = (buf / len(data)) / standart_d(data) ** 4
    return round(kurtosis, 4)


def c_kurtosis(data):
    return round(1 / abs(kurtosis(data)) ** 0.5, 4)


def pirson(data):
    if mean(data) != 0:
        return round(standart_d(data) / mean(data), 4)
    else:
        return 'Mean=0'


def med_oul(data):
    return round(data[round(len(data) / 2)], 4)


def quant(data):
    try:
        temp = [round(x, 2) for x in statistics.quantiles(data)]
        return temp
    except:
        pass


def find_anomalies(data):
    anomalies = []
    anomaly_cut_off = standart_d(data) * 3
    lower_limit = mean(data) - anomaly_cut_off
    upper_limit = mean(data) + anomaly_cut_off
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


def help_info(event):
    if event == 'Anomalies':
        sg.Popup('Anomalies is auto-detected', title='Anomalies', icon='math.ico')

    if event == 'Log()':
        sg.popup('You can logarithm in figure settings.\n', title='Log()', icon='math.ico')

    if event == 'About...':
        sg.popup('This program was build by V.Shynkarov', title='About', icon='math.ico')

    if event == 'Pearson':
        sg.popup_no_buttons(
            "The chi-square test for independence, also called Pearson's chi-square test or the chi-square test of association, is used to discover if there is a relationship between two categorical variables.",
            title='Pearson', icon='math.ico', image='Pearson.png')

    if event == 'Kolmogorov':
        sg.popup_no_buttons(
            "Kolmogorov–Smirnov test (K–S test or KS test): is used to decide if a sample comes from a population with a specific distribution.one-dimensional probability distributions that can be used to compare a sample with a reference probability distribution (one-sample K–S test), or to compare two samples (two-sample K–S test). In essence, the test answers the question 'What is the probability that this collection of samples could have been drawn from that probability distribution?' or, in the second case, 'What is the probability that these two sets of samples were drawn from the same (but unknown) probability distribution?'",
            title='Kolmogorov', icon='math.ico')
        # if event == 'Pearson':
        # if event == 'Pearson':
        # if event == 'Pearson':
        # if event == 'Pearson':
        # if event == 'Pearson':
        # if event == 'Pearson':
        # if event == 'Pearson':
        # if event == 'Pearson':
        #     if event == '':
        #         sg.popup_no_buttons(
        #             "",
        #             title='', icon='math.ico')
