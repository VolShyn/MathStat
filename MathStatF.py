import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# seaborn gives 'FutureWarning', importing warnings to get rid of it

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from scipy import stats
import PySimpleGUI as sg
from stats import *

arr = []
temp = []
dimension = 0


def statistic_analysis(data, text=''):
    text += 'Length: ' + f'{len(data)}' + '\nSumm: ' + f'{sum(data):.2f}' + '\nMax: ' + f'{max(data):.4f}' + '\nMin: ' + f'{min(data):.4f}' + '\nScale: ' + f'{max(data) - min(data):.4f}' + '\nMean: ' + f'{statistics.mean(data):.4f}' + '\nMed: ' + f'{statistics.median(data)}' + '\nVar: ' + f'{statistics.variance(data):.4f}' + '\nStDev: ' + f'{statistics.stdev(data):.4f}' + '\nAsym: '
    return text + asymmetry(data) + '\nEcs: ' + f'{kurtosis(data):.4f}' + '\nCEcs: ' + c_kurtosis(
        data) + '\nVarPirs: ' + pirson(data) + '\nMedU: ' + f'{med_oul(data):.4f}' + '\nQuantiles: ' + quant(data)


sg.theme("Default1")
sg.SetOptions(element_padding=(3, 3))

menu_def = [['File', ['Open', 'Save', 'Exit']],
            ['Show', ['Graph']],
            ['Edit', ['Standartize', 'Shift', 'Clear', ['Normal'], 'Undo'], ],
            ['Tests', ['Pearson Chi-squared', 'Kolmogorov-Smirnov', 'F-test for 2 variances', 'Bartletts',
                       'Wilcoxon signed-rank', 'Sign', 'Single factor analysis of variance', 'Kruskal-Wallis (H)',
                       'T-test', ['1-sample', '2-sample']]],
            ['Help', ['Log()', 'Tests',
                      ['Pearson', 'Kolmogorov', 'F-test', 'Bartletts test', 'Wilcoxon', 'Sign test', 'Single factor',
                       'H-test',
                       'T-test'], 'Anomalies', 'About...']]]

layout = [
    [sg.Menu(menu_def)],
    [sg.Multiline(tooltip='Statistical analysis', size=(15, 40), key='-out-', disabled=True, no_scrollbar=True,
                  font='Courier 12'), sg.Image('mathstat.png', pad=(105, 15), key='-Image-')],
]

window = sg.Window('Mathematical Statistics', layout, resizable=False, finalize=True, font="Courier 10",
                   icon='math.ico',
                   default_element_size=(6, 1),
                   default_button_element_size=(10, 1), size=(600, 400))

while True:
    event, values = window.read()
    help_info(event)

    if event == 'Pearson Chi-squared':
        try:
            if dimension == 3:
                pass
            elif dimension == 2:
                pass
            else:
                sg.popup('You need atleast 2-dimensional array', icon='math.ico')
        except:
            sg.Popup('NO DATA!', icon='math.ico')

    if event == 'Undo':
        try:
            last_event = event
            arr = temp
            window['-out-'].update(statistic_analysis(arr))
        except:
            sg.popup_quick('Nothing to undo!', icon='math.ico')

    if event == 'Shift':
        try:
            if arr[0] is not None:
                if dimension == 1:
                    step = sg.PopupGetText('Shift step:', title='Shift', icon='math.ico')
                    arr = arr + int(step)
                    window['-out-'].update(statistic_analysis(arr))
                else:
                    sg.popup('Only for 1-dimension')
        except:
            sg.popup_quick('Some error occured.\nNO DATA!', icon='math.ico')
            pass

    if event == 'Standartize':
        try:
            if dimension == 1:
                last_event = event
                arr = (arr - statistics.mean(arr)) / statistics.stdev(arr)
                window['-out-'].update(statistic_analysis(arr))
        except:
            sg.popup('Some error occured.', icon='math.ico')
            pass

    if event == 'Normal':
        try:
            if arr[0] is not None:
                window['-out-'].update('')
                arr = []
                try:
                    arr1 = []
                except:
                    pass
                try:
                    arr2 = []
                except:
                    pass
        except:
            sg.popup_quick('Some error occured.\nNO DATA!')
            pass

    if event == 'Open':
        try:
            path = sg.popup_get_file('Open...', icon='math.ico')
            dimension = check_for_columns(path)
            if dimension == 3:
                arr, arr1, arr2 = np.sort(np.loadtxt(path, unpack=True))
                sg.popup_ok('3-dimensional', icon='math.ico')
            elif dimension == 2:
                arr, arr1 = np.sort(np.loadtxt(path, unpack=True))
                sg.popup_ok('2-dimensional', icon='math.ico')
            else:
                arr = np.sort(np.loadtxt(path))
                sg.popup_ok('1-dimensional', icon='math.ico')

            if dimension == 1:
                anom = find_anomalies(arr)
                # check for anomalies by 3-sigma rule
                if anom:
                    for anomaly in anom:
                        for item in arr:
                            if float(anomaly) == float(item):
                                arr = np.delete(arr, np.where(arr == item))
                    sg.popup('Anomalies where found and deleted!\n'
                             f'{anom}')

            # temporary arr to undo standartize
            temp = arr
            window['-out-'].update(statistic_analysis(arr))
        except:
            if event == 'Cancel':
                pass

    if event == 'Save':
        try:
            if arr[0] is not None:
                with open('sortedarray.txt', 'w') as f:
                    for item in arr:
                        f.write(str(item) + '\n')
                sg.popup('File saved as:\n'
                         'sortedarray.txt', icon='math.ico')
        except:
            sg.popup_quick('Nothing to save', icon='math.ico')

    if event == 'Graph':
        """
        x = np.linspace(min(arr), max(arr), len(arr))
        plt.plot(x, norm.pdf(x, mean(arr), statistics_stdev(arr)), color='red', linewidth=2)
        plt.hist(arr, bins=(int(len(arr) ** (1 / 2))), weights=np.zeros_like(arr) + 1 / len(arr),
                edgecolor='#E6E6E6')
        pdf = normpdf(x,mean(arr), statistics_stdev(arr))
        plt.step(arr, np.arange(len(arr)) / float(len(arr)), linewidth=3)
        plt.plot(arr, np.arange(len(arr)) / float(len(arr)), color='red', linewidth=1)
        
        plt.plot(x, norm.pdf(x, mean(arr), statistics_stdev(arr)), color='red', linewidth=2)
                    plt.hist(arr, bins=(int(len(arr) ** (1 / 2))), weights=np.zeros_like(arr) + 1 / len(arr),
                             edgecolor='#E6E6E6')
        """
        try:
            if arr[0] is not None:
                try:
                    if last_event == 'Standartize':
                        sg.popup_ok('Standartized data', icon='math.ico')
                except:
                    pass

                plt.style.context('seaborn')
                f = plt.figure(tight_layout=True)
                f.set_figwidth(8)
                f.set_figheight(8)
                # fig1, ax = plt.subplots()
                # sns.set_theme(style="ticks")
                sns.set_theme(style="whitegrid")
                plt.subplot(2, 1, 1)
                plt.grid(b=True, color='grey',
                         linestyle='-.', linewidth=0.5,
                         alpha=0.6)
                plt.xlim([min(arr) - (min(arr) * 0.1), max(arr) + (min(arr) * 0.1)])
                plt.xlabel('Value', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.tick_params(axis='both', which='major', labelsize=14)
                '''
                bins = sqrt(N) if N > 100
                bins = N ** 1/3
                '''
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
        except:
            sg.popup_quick('Some error occured.\nNO DATA!', icon='math.ico')

    if event == sg.WIN_CLOSED or event == 'Exit':
        break
