import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
from stats import *


def statistic_analysis(mas):
    text = ''
    text += 'Summ: ' + str(round(sum(mas), 4)) + '\nMax: ' + str(max(mas)) + '\nMin: ' + str(
        min(mas)) + '\nMid: ' + str(mid(mas)) + '\nScale: ' + str(
        scale(mas)) + '\nMe: ' + str(me(mas)) + '\nMed: ' + str(med(mas)) + '\nVar: ' + str(
        var(mas)) + '\nStDev: ' + str(standart_d(mas)) + '\nAsym: '
    return text + str(asymmetry(mas)) + '\nEcs: ' + str(
        kurtosis(mas)) + '\nCEcs: ' + str(
        c_kurtosis(mas)) + '\nPirson: ' + str(pirson(mas)) + '\nMedU: ' + str(med_oul(mas)) + '\nQuantiles: ' + str(
        quant(mas))



sg.theme("Purple")
sg.SetOptions(element_padding=(3, 3))

menu_def = [['File', ['Open', 'Save', 'Exit']],
            ['Edit', ['Clear', ['Normal'], 'Undo'], ],
            ['Help', 'About...'], ]

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
    if event == 'Normal':
        window['-out-'].update('')
    if event == 'About...':
        sg.Popup('This program was build by V.Shynkarov', title='About', )
    if event == 'Open':
        try:
            path = sg.popup_get_file('Open...')
            arr = sorted(np.loadtxt(path))
            window['-out-'].update(statistic_analysis(arr))
        except:
            if event == 'Cancel':
                pass
    if event == 'Save':
        with open('sortedarray.txt', 'w') as f:
            for item in arr:
                f.write(str(item) + '\n')
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
