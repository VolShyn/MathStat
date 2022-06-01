import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn import linear_model
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
                      ['Scatter matrix', 'Linear Regression', 'Parallel Coords', 'Heatmap',
                       'Diagnostic diagram', 'Bubble plot']]],
            ['Edit', ['Standartize', 'Shift', 'Clear', 'Undo'], ],
            ['Tests',
             ['Corr.Matr', 'General Average', 'Pearson Chi-squared', 'ЁKolmogorov-Smirnov', 'F-test for 2 variances',
              'Bartletts',
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
     sg.Multiline(size=(80, 30), visible=False, key='-FILE-')]
]

window = sg.Window('Mathematical Statistics', layout, resizable=True, finalize=True, font="Courier 12",
                   icon='math.ico',
                   default_element_size=(6, 1),
                   default_button_element_size=(10, 1), size=(800, 450))

# x = [random.random() for i in range(1, 25)]
# y = [random.random() for i in range(1, 25)]

while True:
    event, values = window.read()
    help_info(event)

    if event == 'General Average':
        try:
            if dimension == 2:
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
            if dimension == 2:
                pvar = statistics.pvariance(arr)
                pvar1 = statistics.pvariance(arr1)
                if pvar > pvar1:
                    sg.popup_ok(f'F-test: {pvar / pvar1:.4f}', title='Answer', icon='math.ico')
                else:
                    sg.popup_ok(f'F-test: {pvar1 / pvar:.4f}', title='Answer', icon='math.ico')
            else:
                sg.popup('You need 2-dimensional array', title='Oops!', icon='math.ico')
        except:
            pass

    if event == 'Bartletts':
        try:
            if dimension == 2:
                sg.popup(f'{bartletts(arr, arr1)}', title='Answer', icon='math.ico')
            else:
                sg.popup('You need 2-dimensional array', title='Oops!', icon='math.ico')
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
            if dimension == 2:
                fvalue, pvalue = stats.f_oneway(arr, arr1)
                sg.popup(f'fvalue: {fvalue:.6f}\npvalue: {pvalue:.6f}', title='Answer', icon='math.ico')
            else:
                sg.popup('You need 2-dimensional array', title='Oops!', icon='math.ico')
        except:
            pass

    if event == 'Kruskal-Wallis (H)':
        try:
            if dimension == 2:
                st, pvalue = scipy.stats.kruskal(arr, arr1)
                sg.popup(f'H: {st:.6f}\npvalue: {pvalue:.6f}', title='Answer', icon='math.ico')
            else:
                sg.popup('You need 2-dimensional array', title='Oops!', icon='math.ico')
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
                if dimension > 2:
                    sg.popup_ok('Oops! You have Multi-dimensional array.', title='Oops!', icon='math.ico')
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
                df = pd.read_fwf(path, header=None)
                sg.popup_ok('Multi-dimensional', icon='math.ico')
                # window['-out-'].update(three_dim_analysis(arr, arr1, arr2))
                window['-FILE-'].update(df.head(len(df[0])))

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
            sns.set_palette('colorblind')
            sns.pairplot(df, height=3)
            plt.show()
        except:
            sg.popup('Something went wrong', title = 'Oops', icon='math.ico')
            pass

    if event == 'Linear Regression':
        try:
            X,Y = ''.join(sg.popup_get_text('X,Y?').split())
            col = ''.join(sg.popup_get_text('Z?').split())
            X = df[[int(X), int(Y)]].values.reshape(-1, 2)
            Y = df[int(col[0])]

            x = X[:, 0]
            y = X[:, 1]
            z = Y

            xx_pred = np.linspace(abs(np.min(x)), abs(np.max(x)), 30)  # range of price values
            yy_pred = np.linspace(abs(np.min(y)), abs(np.max(y)), 30)  # range of advertising values
            xx_pred, yy_pred = np.meshgrid(xx_pred, yy_pred)

            model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
            ols = linear_model.LinearRegression()
            model = ols.fit(X, Y)
            predicted = model.predict(model_viz)
            r2 = model.score(X, Y)
            plt.style.use('seaborn-colorblind')
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x, y, z, color='r', zorder=15, linestyle='none', marker='x', alpha=0.7)
            ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0, 0, 0, 0), s=20,
                       edgecolor='#70b3f0')
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_zlabel('Z', fontsize=12)
            ax.locator_params(nbins=4, axis='x')
            ax.locator_params(nbins=5, axis='x')
            fig.suptitle('Regression model  ($R^2 = %.2f$)' % r2, fontsize=15, color='k')

            fig.tight_layout()
            plt.show()
        except:
            sg.popup('Something went wrong', title='Oops', icon='math.ico')
            pass

    if event == 'Corr.Matr':
        try:
            corr = df.corr()
            sg.Print(f'{corr}', size=(65, 10), font='Courier 12', no_titlebar=True, resizable=True, grab_anywhere=True)

            # making mask
            mask = np.zeros_like(corr, dtype=bool)
            np.fill_diagonal(mask, val=True)

            fig, ax = plt.subplots(figsize=(6, 4))

            cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)
            cmap.set_bad('grey')

            sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)
            fig.suptitle('Pearson correlation coefficient matrix', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=10)

            plt.show()
        except:
            sg.popup('Something went wrong', title='Oops', icon='math.ico')
            pass

    if event == 'Parallel Coords':
        try:
            # Копія через функцію, щоб не були пов'язані, бо ресетаємо індекси, не треба нам такого в ориг. датафреймі
            df1 = df.copy()
            df1.reset_index(inplace=True)
            pd.plotting.parallel_coordinates(df1, 'index')
            plt.gca().legend_.remove()
            plt.show()
        except:
            sg.popup('Something went wrong', icon='math.ico')
            pass

    if event == 'Diagnostic diagram':
        try:
            col = ''.join(sg.popup_get_text('Which columns do yo want?').split())
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.scatter(df[int(col[0])], df[int(col[1])], marker='x', c='r')
            ax.set_xlabel('{col}'.format(col=col[0]))
            ax.set_ylabel('{col1}'.format(col1=col[1]))
            plt.show()
        except:
            sg.popup('Something went wrong', title='Oops', icon='math.ico')
            pass

    if event == 'Heatmap':
        try:
            sns.heatmap(df[0:10], annot=True, fmt=".1f")
            plt.show()
        except:
            sg.popup('Something went wrong', title='Oops', icon='math.ico')
            pass

    if event == 'Bubble plot':
        try:
            # Х,Y,Z ввод. користувачем, після чого Z перетвор. у матрицю цілих чисел
            # Робимо новий датафрейм з X,Y,Z користувача, колір рандом, розмір бульбашок - Z

            X,Y = ''.join(sg.popup_get_text('Arguments(X,Y)?').split())
            col = ''.join(sg.popup_get_text('Bubble Size(Z)').split())
            colors = np.random.rand(len(df[0]))
            col = np.array(df[int(col[0])])
            col = col.astype(int)

            d = pd.DataFrame({'X': df[int(X)], 'Y': df[int(Y)], 'Colors': colors, 'Bubble size': col})
            plt.scatter('X', 'Y',
                        s='Bubble size',
                        alpha=0.5,
                        c='Colors',
                        data=d)
            plt.xlabel("X", size=16)
            plt.ylabel("y", size=16)
            plt.title("Bubble Plot ", size=18)
            plt.show()
        except:
            sg.popup('Something went wrong', title='Oops', icon='math.ico')
            pass

    if event == 'Graph':
        try:
            if dimension > 2:
                sg.popup('You have multi-dimensional array', title='Oops', icon='math.ico')
            elif dimension == 2:
                two_dimens_graph(arr, arr1)
            elif dimension == 1:
                one_dimens_graph(arr)
            else:
                sg.popup_quick('NO DATA.', title='Oops', icon='math.ico')
        except:
            sg.popup_quick('Some error occured.', title='Ooops', icon='math.ico')

    if event == sg.WIN_CLOSED or event == 'Exit':
        break
