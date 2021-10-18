import numpy as np
import statistics
def mid(mas):
    return round(sum(mas) / len(mas), 4)


def scale(mas):
    return round(max(mas) - min(mas), 4)


def med(mas):
    return round(statistics.median(mas), 4)


def var(mas):
    return round(statistics.variance(mas), 4)


def me(mas):
    return round(statistics.mean(mas), 4)


def standart_d(mas):
    return round(statistics.stdev(mas), 4)


def asymmetry(mas):
    buf = 0
    k = 0
    while (k < len(mas)):
        buf += (mas[k] - me(mas)) ** 3
        k += 1
    asymmetry = (buf / len(mas)) / standart_d(mas) ** 3
    return round(asymmetry, 4)


def kurtosis(mas):
    buf = 0
    k = 0
    while (k < len(mas)):
        buf += (mas[k] - me(mas)) ** 4
        k += 1
    kurtosis = (buf / len(mas)) / standart_d(mas) ** 4
    return round(kurtosis, 4)


def c_kurtosis(mas):
    return round(1 / abs(kurtosis(mas)) ** 0.5, 4)


def pirson(mas):
    return round(standart_d(mas) / me(mas), 4)


def med_oul(mas):
    return round(mas[round(len(mas) / 2)], 4)


def quant(mas):
    return statistics.quantiles(mas)
