import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import PySimpleGUI as sg
from stats import *


def statistic_analysis(data):
    text = ''
    text += 'Summ: ' + str(round(sum(data), 4)) + '\nMax: ' + str(round(max(data), 5)) + '\nMin: ' + str(
        round(min(data), 5)) + '\nMid: ' + str(mid(data)) + '\nScale: ' + str(
        scale(data)) + '\nMean: ' + str(mean(data)) + '\nMed: ' + str(med(data)) + '\nVar: ' + str(
        var(data)) + '\nStDev: ' + str(standart_d(data)) + '\nAsym: '
    return text + str(asymmetry(data)) + '\nEcs: ' + str(
        kurtosis(data)) + '\nCEcs: ' + str(
        c_kurtosis(data)) + '\nPirson: ' + str(pirson(data)) + '\nMedU: ' + str(med_oul(data)) + '\nQuantiles: ' + str(
        quant(data))


sg.theme("Default1")
sg.SetOptions(element_padding=(3, 3))

menu_def = [['File', ['Open', 'Save', 'Exit']],
            ['Show', ['Graph']],
            ['Edit', ['Standartize', 'Shift', 'Clear', ['Normal'], 'Undo'], ],
            ['Help', ['Log()', 'Anomalies', 'About...']]]

layout = [
    [sg.Menu(menu_def)],
    [sg.Multiline(tooltip='Statistical analysis', size=(15, 40), key='-out-', disabled=True, no_scrollbar=True,
                  font='Courier 12')],
]

window = sg.Window('MathStat', layout, resizable=False, finalize=True, font="Courier 10", default_element_size=(6, 1),
                   default_button_element_size=(10, 1), size=(600, 400))
arr = []
while True:
    temp = []
    event, values = window.read()

    if event == 'Shift':
        try:
            if arr[0] is not None:
                step = sg.PopupGetText('Shift step:')
                arr = arr + int(step)
                window['-out-'].update(statistic_analysis(arr))
        except:
            sg.popup('Some error occured.')
            pass

    if event == 'Anomalies':
        sg.Popup('Anomalies is auto-detected', title='Anomalies')

    if event == 'Log()':
        sg.popup('You can logarithm in figure settings.\n', title='Log()')

    if event == 'Standartize':
        try:
            if arr[0] is not None:
                st_arr = (arr - mean(arr)) / standart_d(arr)
                window['-out-'].update(statistic_analysis(st_arr))
        except:
            sg.popup('Some error occured.\nMaybe already standartized?')
            pass

    if event == 'Normal':
        try:
            if arr[0] is not None:
                window['-out-'].update('')
                arr = []
        except:
            sg.popup('Some error occured.\nNO DATA!')
            pass

    if event == 'About...':
        sg.Popup('This program was build by V.Shynkarov', title='About', )

    if event == 'Open':
        try:
            path = sg.popup_get_file('Open...')
            arr = np.sort(np.loadtxt(path))
            anom = find_anomalies(arr)
            if anom:
                for anomaly in anom:
                    for item in arr:
                        if float(anomaly) == float(item):
                            arr = np.delete(arr, np.where(arr == item))
                sg.popup('Anomalies where found and deleted!\n'
                         f'{anom}')
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
        except:
            sg.popup('Nothing to save')

    if event == 'Graph':
        try:
            if arr[0] is not None:
                plt.style.context('seaborn')
                f = plt.figure(tight_layout=True)
                f.set_figwidth(8)
                f.set_figheight(8)
                # fig1, ax = plt.subplots()
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
                if len(arr) <= 100:
                    plt.hist(arr, bins=(round(len(arr) ** (1 / 2))), weights=np.zeros_like(arr) + 1 / len(arr),
                             edgecolor='#E6E6E6')
                else:
                    plt.hist(arr, bins=(round(len(arr) ** (1 / 3))), weights=np.zeros_like(arr) + 1 / len(arr),
                             edgecolor='#E6E6E6')
                x = np.linspace(0, max(arr))
                plt.plot(x, norm.pdf(x, mean(arr), standart_d(arr)), color='red', linewidth=2)
                # making emperical graph
                plt.subplot(2, 1, 2)
                plt.step(arr, np.arange(len(arr)) / float(len(arr)), linewidth=3)
                plt.plot(arr, np.arange(len(arr)) / float(len(arr)), color='red', linewidth=2)
                plt.ylim(0, 1)
                plt.xlabel('Value', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.tick_params(axis='both', which='major', labelsize=14)
                plt.show()
        except:
            sg.popup('Some error occured.\nNO DATA!')

    if event == sg.WIN_CLOSED or event == 'Exit':
        break
