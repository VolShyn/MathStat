import numpy as np
import statistics


def mid(data):
    return round(sum(data) / len(data), 4)


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
    return statistics.quantiles(data)


def find_anomalies(data):
    # define a list to accumlate anomalies
    anomalies = []

    # Set upper and lower limit to 3 standard deviation
    random_data_std = standart_d(data)
    random_data_mean = mean(data)
    anomaly_cut_off = random_data_std * 3

    lower_limit = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies

