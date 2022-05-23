import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import PySimpleGUI as sg
from stats import *

# seaborn gives 'FutureWarning', importing warnings to get rid of it
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

arr = []
temp = []
dimension = 0

sg.theme("Default1")
sg.SetOptions(element_padding=(3, 3))

menu_def = [['File', ['Open', 'Save', 'Exit']],
            ['Show', ['Graph', 'For multi-dim',
                      ['Scatter matrix', 'Linear Regression', 'Corr.Matr', 'Parallel Coords', 'Heatmap',
                       'Diagnostic diagram', 'Bubble plot']]],
            ['Edit', ['Standartize', 'Shift', 'Clear', 'Undo'], ],
            ['Tests',
             ['General Average', 'Pearson Chi-squared', 'Kolmogorov-Smirnov', 'F-test for 2 variances', 'Bartletts',
              'Wilcoxon signed-rank', 'Sign', 'Single factor analysis of variance', 'Kruskal-Wallis (H)',
              'T-test', ['1-sample', '2-sample',
                         ['2-dimensions', '1-dimension (Inverse method)', ['Exp', 'PDF(Normal)']]]]],
            ['Help', ['Log()', 'Tests',
                      ['General average', 'Pearson', 'Kolmogorov', 'F-test', 'Bartletts test', 'Wilcoxon', 'Sign test',
                       'Single factor',
                       'H-test',
                       'T-test'], 'Anomalies', 'Inverse transform sampling', 'Laplace table', 'About...']]]

layout = [
    [sg.Menu(menu_def)],
    [sg.Multiline(tooltip='Statistical analysis', size=(18, 40), key='-out-', disabled=True, no_scrollbar=True,
                  font='Courier 12'),
     sg.Image('mathstat.png', pad=(105, 50), key='-Image-', visible=True),
     sg.Multiline(size=(80, 30), visible=False, key='-FILE-')]
]

window = sg.Window('Mathematical Statistics', layout, resizable=False, finalize=True, font="Courier 10",
                   icon='math.ico',
                   default_element_size=(6, 1),
                   default_button_element_size=(10, 1), size=(600, 400))

# x = [random.random() for i in range(1, 25)]
# y = [random.random() for i in range(1, 25)]

while True:
    event, values = window.read()
    help_info(event)

    if event == 'General Average':
        try:
            if dimension == 3:
                sg.popup('You need 2-dimensional array (T-test)', title='Oops!', icon='math.ico')
            elif dimension == 2:
                Z_t, pvalue = stats.ttest_ind(arr, arr1)
                alpha = sg.popup_get_text('Statistical significance(alpha): ', title='Statistical significance',
                                          icon='math.ico')
                Z_phi = (1 - (2 * float(alpha))) / 2
                sg.popup_ok(f'T-test(Z): {abs(Z_t):.4f} \nZ(Phi): {Z_phi:.4f}\n'
                            '*Check laplass table to check if Z(t) > Z(Phi)', title='Answer', icon='math.ico')
            else:
                sg.popup('You need 2-dimensional array', title='Oops!', icon='math.ico')
        except:
            pass

    if event == 'Pearson Chi-squared':
        try:
            if dimension == 1:
                sg.popup(f'Chi-Squared: {chi_squared(arr)}', title='Answer', icon='math.ico')
            elif dimension == 2:
                sg.popup('Only 1-dimension avalaible')
            else:
                sg.popup('You need 1-dimensional array', title='Oops!', icon='math.ico')
        except:
            pass

    if event == 'Kolmogorov-Smirnov':
        try:
            if dimension == 1:
                sg.popup(f'Kolmogorov-Smirno(1-dim): {kolmogorov_1dim(arr)}', title='Answer', icon='math.ico')
            elif dimension == 2:
                sg.popup(f'Kolmogorov-Smirnov(2-dim): {kolmogorov_2dim(arr, arr1)}', title='Answer',
                         icon='math.ico')
            else:
                sg.popup('You need 1 or 2 dimens. array', title='Oops!', icon='math.ico')
        except:
            pass

    if event == 'F-test for 2 variances':
        try:
            if dimension == 3:
                pass
            elif dimension == 2:
                pvar = statistics.pvariance(arr)
                pvar1 = statistics.pvariance(arr1)
                if pvar > pvar1:
                    sg.popup_ok(f'F-test: {pvar / pvar1:.4f}', title='Answer', icon='math.ico')
                else:
                    sg.popup_ok(f'F-test: {pvar1 / pvar:.4f}', title='Answer', icon='math.ico')
            else:
                sg.popup('You need atleast 2-dimensional array', title='Oops!', icon='math.ico')
        except:
            pass

    if event == 'Bartletts':
        try:
            if dimension == 3:
                statistic, pvalue = scipy.stats.bartlett(arr, arr1, arr2)
                sg.popup(f'Stat B: {statistic:.4f}\npvalue: {pvalue:.6f}', title='Answer', icon='math.ico')
            if dimension == 2:
                sg.popup(f'{bartletts(arr, arr1)}', title='Answer', icon='math.ico')
            else:
                sg.popup('You need atleast 2-dimensional array', title='Oops!', icon='math.ico')
        except:
            pass

    if event == 'Wilcoxon signed-rank':
        try:
            if dimension == 2:
                w, p = stats.wilcoxon(arr, arr1)
                sg.popup(f'W: {w:.4f}\npvalue: {p:.6f}', title='Answer', icon='math.ico')
            elif dimension == 1:
                w, p = stats.wilcoxon(arr)
                sg.popup(f'W: {w:.4f}\npvalue: {p:.6f}', title='Answer', icon='math.ico')
            else:
                sg.popup('Only 1-2 dimensional arrays', title='Oops!', icon='math.ico')
        except:
            pass

    if event == 'Sign':
        try:
            if dimension == 1:
                sg.popup_ok(f'{sign(arr)}', title='Answer', icon='math.ico')
            else:
                sg.popup_ok('You need 1-dimensional array', title='Oops!', icon='math.ico')
        except:
            pass

    if event == 'Single factor analysis of variance':
        try:
            if dimension == 3:
                fvalue, pvalue = stats.f_oneway(arr, arr1, arr2)
                sg.popup(f'fvalue: {fvalue:.6f}\npvalue: {pvalue:.6f}', title='Answer', icon='math.ico')
            elif dimension == 2:
                fvalue, pvalue = stats.f_oneway(arr, arr1)
                sg.popup(f'fvalue: {fvalue:.6f}\npvalue: {pvalue:.6f}', title='Answer', icon='math.ico')
            else:
                sg.popup('You need atleast 2-dimensional array', title='Oops!', icon='math.ico')
        except:
            pass

    if event == 'Kruskal-Wallis (H)':
        try:
            if dimension == 3:
                st, pvalue = scipy.stats.kruskal(arr, arr1, arr2)
                sg.popup(f'H: {st:.6f}\npvalue: {pvalue:.6f}', title='Answer', icon='math.ico')
            elif dimension == 2:
                st, pvalue = scipy.stats.kruskal(arr, arr1)
                sg.popup(f'H: {st:.6f}\npvalue: {pvalue:.6f}', title='Answer', icon='math.ico')
            else:
                sg.popup('You need atleast 2-dimensional array', title='Oops!', icon='math.ico')
        except:
            pass

    if event == 'Undo':
        try:
            last_event = event
            arr = temp
            window['-out-'].update(one_dim_analysis(arr))
        except:
            sg.popup_quick('Nothing to undo!', title='Undo', icon='math.ico')

    if event == 'PDF(Normal)':
        sg.popup('No realization for now, sorry :-(')

    if event == 'Exp':
        try:
            if arr[0] is not None:
                if dimension == 1:
                    try:
                        t, pvalue = stats.ttest_ind(arr, exponential_inverse_trans(len(arr)))
                        sg.popup_ok(f't: {t:.4f}\npvalue: {pvalue:.4f}', title='2-sample', icon='math.ico')
                    finally:
                        pass
                else:
                    sg.popup_quick('Only for 1 dimensional arrays!', title='Error', icon='math.ico')
        except:
            sg.popup_quick('Some error occured.\n', title='Error', icon='math.ico')
            pass

    if event == '2-dimensions':
        try:
            if arr[0] is not None:
                if dimension == 3:
                    sg.popup_ok('Oops! You have 3-dimensional array.', title='Oops!', icon='math.ico')
                if dimension == 2:
                    try:
                        t, pvalue = stats.ttest_ind(arr, arr1)
                        sg.popup_ok(f't: {t:.4f}\npvalue: {pvalue:.7f}', title='2-sample', icon='math.ico')
                    finally:
                        pass
                if dimension == 1:
                    sg.popup_ok('Oops! You have 1-dimension.\n'
                                'Try inverse method', title='Oops!', icon='math.ico')
        except:
            sg.popup_quick('Some error occured.\n', icon='math.ico')
            pass

    if event == '1-sample':
        try:
            if arr[0] is not None:
                if dimension == 1:
                    value = sg.PopupGetText('Value: ', default_text=f'{np.mean(arr):.4f}', title='Expected value',
                                            icon='math.ico')
                    diff = np.mean(arr) - float(value)
                    SE = statistics.stdev(arr) / pow(len(arr), 1 / 2)
                    t = diff / SE
                    try:
                        tscore, pvalue = stats.ttest_1samp(arr, float(value))
                        sg.popup_ok(f't: {t:.4f}\nP-value: {pvalue:.4f}', title='Answer', icon='math.ico')
                    finally:
                        pass
                else:
                    sg.popup('Only for 1-dimension', icon='math.ico')
        except:
            sg.popup_quick('Some error occured.\n', icon='math.ico')
            pass

    if event == 'Shift':
        try:
            if arr[0] is not None:
                if dimension == 1:
                    step = sg.PopupGetText('Shift step:', title='Shift', icon='math.ico')
                    arr = arr + int(step)
                    window['-out-'].update(one_dim_analysis(arr))
                else:
                    sg.popup('Only for 1-dimension')
        except:
            sg.popup_quick('Some error occured.\nNO DATA!', icon='math.ico')
            pass

    if event == 'Standartize':
        try:
            if arr[0] is not None:
                last_event = event
                arr = (arr - statistics.mean(arr)) / statistics.stdev(arr)
                window['-out-'].update(one_dim_analysis(arr))
                try:
                    arr2 = (arr2 - statistics.mean(arr2)) / statistics.stdev(arr2)
                    arr1 = (arr1 - statistics.mean(arr1)) / statistics.stdev(arr1)
                    window['-out-'].update(three_dim_analysis(arr, arr1, arr2))
                except:
                    try:
                        arr1 = (arr1 - statistics.mean(arr1)) / statistics.stdev(arr1)
                        window['-out-'].update(two_dim_analysis(arr, arr1))
                    except:
                        pass
            # else:
            #     sg.popup('Only for 1-dimension', icon='math.ico')

        except:
            sg.popup('Some error occured.', icon='math.ico')
            pass

    if event == 'Clear':
        try:
            if arr[0] is not None:
                window['-out-'].update('')
                arr = []
                window['-Image-'].update(visible=True)
                window['-FILE-'].update(visible=False)
                try:
                    arr2 = []
                    arr1 = []
                except:
                    arr1 = []
        except:
            sg.popup_quick('Some error occured.\nNO DATA!')
            pass

    if event == 'Open':
        try:
            path = sg.popup_get_file('Open...', icon='math.ico')
            dimension = check_for_columns(path)
            if dimension > 2:
                arr, arr1, arr2 = np.loadtxt(path, unpack=True)
                df = datafr(arr, arr1, arr2)
                sg.popup_ok('3-dimensional', icon='math.ico')
                window['-out-'].update(three_dim_analysis(arr, arr1, arr2))
                window['-FILE-'].update(df.head(len(arr)))

                # T = np.array([arr, arr1, arr2])
                # reshaper = []
                # for i in range(len(arr)):
                #     reshaper.append(str(T[0:3, i]))
                # reshaper.clear()
            elif dimension == 2:
                arr2 = []
                arr, arr1 = np.loadtxt(path, unpack=True)
                d = {'X': arr, 'Y': arr1}
                df = pd.DataFrame(data=d)

                sg.popup_ok('2-dimensional', icon='math.ico')
                window['-out-'].update(two_dim_analysis(arr, arr1))
                window['-FILE-'].update(df.head(len(arr)))

                ###Старий метод для виводу інформації
                # T = np.array([arr, arr1])

                # reshaper = []
                # for i in range(len(arr)):
                #     reshaper.append(str(T[0:2, i]))
                # reshaper.clear()
            else:
                arr = np.sort(np.loadtxt(path))
                sg.popup_ok('1-dimensional', icon='math.ico')
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
                window['-out-'].update(one_dim_analysis(arr))
                window['-FILE-'].update(arr.reshape((len(arr), 1)))

            window['-Image-'].update(visible=False)
            window['-FILE-'].update(visible=True)

        except:
            if event == 'Cancel':
                pass

    if event == 'Save':
        try:
            if arr[0] is not None:
                if dimension == 1:
                    with open('sortedarray.txt', 'w') as f:
                        for item in arr:
                            f.write(str(item) + '\n')
                    sg.popup('File saved as:\n'
                             'sortedarray.txt', icon='math.ico')
                else:
                    sg.popup_quick('Only for 1-dim', icon='math.ico')
        except:
            sg.popup_quick('Nothing to save', icon='math.ico')

    if event == 'Scatter matrix':
        try:
            scatter_matrix(arr, arr1, arr2)
        except:
            sg.popup('Something went wrong', icon='math.ico')
            pass

    if event == 'Linear Regression':
        try:
            linear_reg3d(arr, arr1, arr2)
        except:
            sg.popup('Something went wrong', icon='math.ico')
            pass

    if event == 'Corr.Matr':
        try:
            corr_matr(arr, arr1, arr2)
        except:
            sg.popup('Something went wrong', icon='math.ico')
            pass

    # if event == 'Radar':
    #     try:
    #         radar(arr, arr1, arr2)
    #     except:
    #         pass

    if event == 'Parallel Coords':
        # try:
        #     parallel_coord(arr, arr1, arr2)
        # except:
        #     sg.popup('Something went wrong', icon='math.ico')
        #     pass
        parallel_coord(arr,arr1,arr2)

    if event == 'Diagnostic diagram':
        try:
            diagnos_diag(arr, arr1, arr2)
        except:
            sg.popup('Something went wrong', icon='math.ico')
            pass

    if event == 'Heatmap':
        try:
            heatmap(arr, arr1, arr2)
        except:
            sg.popup('Something went wrong', icon='math.ico')
            pass

    if event == 'Bubble plot':
        try:
            bubbleplot(arr, arr1, arr2)
        except:
            sg.popup('Something went wrong', icon='math.ico')
            pass

    if event == 'Barplot':
        try:
            barplot(arr, arr1, arr2)
        except:
            sg.popup('Something went wrong', icon='math.ico')
            pass

    if event == 'Graph':
        try:
            if arr[0] is not None:
                if dimension > 2:
                    sg.popup('You have multidimensional array', title='Oops', icon='math.ico')
                if dimension == 2:
                    two_dimens_graph(arr, arr1)
                if dimension == 1:
                    one_dimens_graph(arr)
        except:
            sg.popup_quick('Some error occured.\nNO DATA!', icon='math.ico')

    if event == sg.WIN_CLOSED or event == 'Exit':
        break
