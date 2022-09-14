from help_info import help_info
from one_dim import one_dim_analysis, one_dimens_graph
from two_dim import two_dim_analysis, two_dimens_graph
from stats import describing, find_anomalies, cov_mat, cumulative_coef
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import seaborn as sns
from sklearn import linear_model
import PySimpleGUI as sg
import pandas as pd

# seaborn gives 'FutureWarning', importing warnings to get rid of it
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

sg.theme("Default1")
sg.SetOptions(element_padding=(3, 3))

menu_def = [['Файл', ['Відкрити', 'Зберегти', 'Вихід']],
            ['Показати', ['Графік', '---', 'МГК', '---', 'Багатовимірні',
                          ['Діагн.діаграма розкиду', 'Лінійна регресія', 'Паралельні кординати', 'Теплова карта',
                           'Діагностична діаграма', 'Бульбашковий']]],
            ['Змінити', ['Стандартизація', 'Зсув', 'Повернути', 'Аномалії', 'Очистити'], ],
            ['Тести',
             ['Кор.Матр', 'Частк.Кор.', 'Множин.Кор.', '---', 'Генеральне середнє', 'Краскела — Уоліса (H)',
              'F-тест для 2-ух дисп.',
              'Однофакторний дисперсійний аналіз(ANOVA)',
              'Бартлетта', 'Крит. узгодженості Колмогорова',
              'Критерій Уілкоксона', '---', 'Критерій знаків', 'Критерій узгодженості Пірсона',
              'T-тест', ['1-ознака', '2-ознаки']]],
            ['Інфо', ['Log()', 'Tests',
                      ['General average', 'Pearson', 'Kolmogorov', 'F-test', 'Bartletts test', 'Wilcoxon', 'Sign test',
                       'Single factor',
                       'H-test',
                       'T-test'], 'Anomalies', 'Inverse transform sampling', 'Laplace table', 'About...']]]

layout = [
    [sg.Menu(menu_def)],
    [sg.Multiline(tooltip='Статистичний аналіз', size=(18, 40), key='-out-', disabled=True, no_scrollbar=True,
                  font='Courier 12'),
     sg.Multiline(size=(80, 30), visible=False, key='-FILE-')]
]

window = sg.Window('Математична статистика', layout, resizable=True, finalize=True, font="Courier 12",
                   icon='icon/math.ico',
                   default_element_size=(6, 1),
                   default_button_element_size=(10, 1), size=(800, 450))

while True:
    event, values = window.read()
    help_info(event)

    """
    Main events as Open, Save etc
    """

    if event == 'Відкрити':
        try:
            path = sg.popup_get_file('Відкрити...', icon='icon/math.ico')
            df = pd.read_fwf(path, header=None)
            if df.shape[1] > 2:
                sg.popup_no_buttons('⠀⠀⠀⠀⠀Багатовимірний⠀⠀⠀⠀⠀⠀', title='Успіх', icon='icon/math.ico', auto_close=True,
                                    auto_close_duration=2)
                window['-FILE-'].update(df.head(len(df)))
                window['-out-'].update(describing(df))

            elif df.shape[1] == 2:
                sg.popup_no_buttons('⠀⠀⠀⠀⠀⠀⠀Двовимірний⠀⠀⠀⠀⠀⠀⠀', title='Успіх', icon='icon/math.ico', auto_close=True,
                                    auto_close_duration=2)
                window['-out-'].update(two_dim_analysis(df))
                window['-FILE-'].update(df.head(len(df)))
            else:
                sg.popup_no_buttons('⠀⠀⠀⠀⠀⠀⠀Одновимірний⠀⠀⠀⠀⠀⠀⠀', title='Успіх', icon='icon/math.ico', auto_close=True,
                                    auto_close_duration=2)
                window['-out-'].update(one_dim_analysis(df))
                window['-FILE-'].update(df.head(len(df)))
            window['-FILE-'].update(visible=True)
        except:
            pass

    if event == 'Зберегти':
        try:
            if df.shape[1] > 2:
                np.savetxt('saveMn.txt', df.values, fmt='%4f')
                sg.popup_no_buttons('⠀⠀Збережено як:\n⠀⠀'
                                    'saveMn.txt', icon='icon/math.ico', auto_close=True,
                                    auto_close_duration=3)
            elif df.shape[1] == 2:
                np.savetxt('save2n.txt', df.values, fmt='%4f')
                sg.popup_no_buttons('⠀⠀Збережено як:\n⠀⠀'
                                    'save2n.txt', icon='icon/math.ico', auto_close=True,
                                    auto_close_duration=3)
            else:
                np.savetxt('save1n.txt', df.values, fmt='%4f')
                sg.popup_no_buttons('⠀⠀Збережено як:\n⠀⠀'
                                    'save1n.txt', icon='icon/math.ico', auto_close=True,
                                    auto_close_duration=3)
        except:
            sg.popup_quick('⠀⠀⠀⠀⠀Нічого зберігати!⠀⠀⠀⠀⠀', title='???', icon='icon/math.ico', button_type=5,
                           auto_close=True, auto_close_duration=3)

    if event == 'Повернути':
        try:
            df = dfcop
            window['-FILE-'].update(df)
            last_event = event
            del dfcop
        except:
            sg.popup_quick('⠀⠀⠀⠀⠀⠀Нічого повертати!⠀⠀⠀⠀⠀⠀', title='???', icon='icon/math.ico', button_type=5,
                           auto_close=True, auto_close_duration=3)

    if event == 'Зсув':
        try:
            dfcop = df.copy()
            step = sg.PopupGetText('Крок: ', title='Крок', icon='icon/math.ico')
            df = df + int(step)
            window['-FILE-'].update(df.head(len(df[0])))
        except:
            sg.popup_quick('⠀⠀⠀⠀⠀⠀⠀Сталася помилка!⠀⠀⠀⠀⠀⠀⠀', title='Помилка!', icon='icon/math.ico', button_type=5,
                           auto_close=True, auto_close_duration=3)

    if event == 'Стандартизація':
        try:
            dfcop = df.copy()
            df = (df - df.mean()) / df.std()  # Z-score
            window['-FILE-'].update(df)
        except:
            sg.popup_quick('⠀⠀⠀⠀⠀⠀⠀Сталася помилка!⠀⠀⠀⠀⠀⠀⠀', title='Помилка!', icon='icon/math.ico', button_type=5,
                           auto_close=True, auto_close_duration=3)

    if event == 'Очистити':
        try:
            window['-out-'].update('')
            window['-FILE-'].update(visible=False)
            del df
        except:
            sg.popup_quick('⠀⠀⠀⠀⠀⠀⠀Сталася помилка!⠀⠀⠀⠀⠀⠀⠀', title='Помилка!', icon='icon/math.ico', button_type=5,
                           auto_close=True, auto_close_duration=3)

    if event == 'Аномалії':
        try:
            anom = find_anomalies(df)  # 3 sigma rule to find anomalies
            if anom:  # Check if our list is empty or not
                sg.popup('Знайдено аномалії: \n'
                         f'{anom}', icon='icon/math.ico', title='Ахтунг!')
            else:
                sg.popup('Аномалій не знайдено \n', icon='icon/math.ico', title='Ок', button_type=5, auto_close=True,
                         auto_close_duration=3)
        except:
            sg.popup_quick('⠀⠀⠀⠀⠀⠀⠀Сталася помилка!⠀⠀⠀⠀⠀⠀⠀', title='Вупс...', icon='icon/math.ico', button_type=5,
                           auto_close=True, auto_close_duration=3)

    """
    1 AND 2 DIM. events
    Hypothesis checking 
    """

    if event == 'Графік':
        try:
            if df.shape[1] > 2:
                sg.popup('Багатовимірна вибірка!', title='Упс..', icon='icon/math.ico')
            elif df.shape[1] == 2:
                two_dimens_graph(df)
            else:
                one_dimens_graph(df)
        except:
            sg.popup_quick('⠀⠀⠀⠀⠀⠀⠀Сталася помилка!⠀⠀⠀⠀⠀⠀⠀', title='Вупс...', icon='icon/math.ico', button_type=5,
                           auto_close=True, auto_close_duration=3)

    if event == 'Генеральне середнє':
        try:
            if dimension == 2:
                Z_t, pvalue = stats.ttest_ind(df[0], df[1])
                alpha = sg.popup_get_text('Статистична значущість(alpha): ', title='Стат.знач.',
                                          icon='icon/math.ico')
                Z_phi = (1 - (2 * float(alpha))) / 2
                sg.popup_ok(f'Z(t): {abs(Z_t):.4f} \nZ(Phi): {Z_phi:.4f}\n'
                            '*Якщо Z(t) > Z(Phi) дивіться таблицю лапласа', title='В-дь', icon='icon/math.ico')
            else:
                sg.popup('Тільки двовимірні', title='Упс..', icon='icon/math.ico')
        except:
            pass

    if event == 'Критерій узгодженості Пірсона':
        try:
            if dimension == 1:
                sg.popup(f'Хі-квадрат: {chi_squared(arr)}', title='В-дь', icon='icon/math.ico')
            else:
                sg.popup('Тільки для одновимірних вибірок')

        except:
            sg.popup('Щось пішло не так...', title='Упс..', icon='icon/math.ico')

    if event == 'Крит. узгодженості Колмогорова':
        try:
            if dimension == 1:
                sg.popup(f'Критерій Колмогорова(1-dim): {kolmogorov_1dim(arr)}', title='В-дь', icon='icon/math.ico')
            elif dimension == 2:
                sg.popup(f'Критерій Колмогорова(2-dim): {kolmogorov_2dim(arr, arr1)}', title='В-дь',
                         icon='icon/math.ico')
            else:
                sg.popup('Тільки одно/дво вимірні', title='Упс..', icon='icon/math.ico')
        except:
            pass

    if event == 'F-тест для 2-ух дисп.':
        try:
            if dimension == 2:
                pvar = statistics.pvariance(arr)
                pvar1 = statistics.pvariance(arr1)
                if pvar > pvar1:
                    sg.popup_ok(f'Ф-тест: {pvar / pvar1:.4f}', title='В-дь', icon='icon/math.ico')
                else:
                    sg.popup_ok(f'Ф-тест: {pvar1 / pvar:.4f}', title='В-дь', icon='icon/math.ico')
            else:
                sg.popup('Тільки для двовимірних', title='Упс..', icon='icon/math.ico')
        except:
            pass

    if event == 'Бартлетта':
        try:
            if dimension == 2:
                sg.popup(f'{bartletts(arr, arr1)}', title='В-дь', icon='icon/math.ico')
            else:
                sg.popup('Тільки для двовимірних', title='Упс..', icon='icon/math.ico')
        except:
            pass

    if event == 'Критерій Уілкоксона':
        try:
            if dimension == 2:
                w, p = stats.wilcoxon(arr, arr1)
                sg.popup(f'W: {w:.4f}\npvalue: {p:.6f}', title='В-дь', icon='icon/math.ico')
            elif dimension == 1:
                w, p = stats.wilcoxon(arr)
                sg.popup(f'W: {w:.4f}\npvalue: {p:.6f}', title='В-дь', icon='icon/math.ico')
            else:
                sg.popup('Тільки одно/дво вимірні ', title='Упс!.', icon='icon/math.ico')
        except:
            pass

    if event == 'Критерій знаків':
        try:
            if dimension == 1:
                sg.popup_ok(f'{sign(arr)}', title='В-дь', icon='icon/math.ico')
            else:
                sg.popup_ok('Тільки для одновимірних', title='Упс..', icon='icon/math.ico')
        except:
            pass

    if event == 'Однофакторний дисперсійний аналіз(ANOVA)':
        try:
            if df.shape[1] == 2:
                fvalue, pvalue = stats.f_oneway(arr, arr1)
                sg.popup(f'F-значення: {fvalue:.6f}\np-значення: {pvalue:.6f}', title='В-дь', icon='icon/math.ico')
            else:
                sg.popup('Тільки для двовимірних', title='Упс..', icon='icon/math.ico')
        except:
            pass

    if event == 'Краскела — Уоліса (H)':
        try:
            if df.shape[1] == 2:
                st, pvalue = scipy.stats.kruskal(arr, arr1)
                sg.popup(f'H: {st:.6f}\np-значення: {pvalue:.6f}', title='В-дь', icon='icon/math.ico')
            else:
                sg.popup('Тільки для двовимірних', title='Упс!.', icon='icon/math.ico')
        except:
            pass

    if event == '2-ознаки':
        try:
            if arr[0] is not None:
                if df.shape[1] == 2:
                    try:
                        t, pvalue = stats.ttest_ind(arr, arr1)
                        sg.popup_ok(f't: {t:.4f}\np-значення: {pvalue:.7f}', title='2-виміри', icon='icon/math.ico')
                    finally:
                        pass
        except:
            sg.popup_quick('Сталася помилка!', icon='icon/math.ico')
            pass

    if event == '1-ознака':
        try:
            if arr[0] is not None:
                if df.shape[1] == 1:
                    value = sg.PopupGetText('Значення: ', default_text=f'{np.mean(arr):.4f}',
                                            title='Очікуване значення',
                                            icon='icon/math.ico')
                    diff = np.mean(arr) - float(value)
                    SE = statistics.stdev(arr) / pow(len(arr), 1 / 2)
                    t = diff / SE
                    try:
                        tscore, pvalue = stats.ttest_1samp(arr, float(value))
                        sg.popup_ok(f't: {t:.4f}\nP-значення: {pvalue:.4f}', title='В-дь', icon='icon/math.ico')
                    finally:
                        pass
                else:
                    sg.popup('Тільк для одновимірних', icon='icon/math.ico')
        except:
            sg.popup_quick('Сталася помилка!', icon='icon/math.ico')
            pass

    """
    Multi dim. events
    """

    if event == 'Діагн.діаграма розкиду':
        try:
            sns.set_palette('colorblind')
            sns.pairplot(df, height=3)
            plt.show()
        except:
            sg.popup('Щось пішло не так...', title='Упс..', icon='icon/math.ico')
            pass

    if event == 'Лінійна регресія':
        """
        try щоб прога не вилітала при кожній помилці, іф для роботи лише з N-вимір. вибірками
        Як фічерси беремо X. Будуємо площину через xx_pred,  yy_pred... Далі підставляємо модель, і будуємо графік 
        """
        try:
            if df.shape[1] > 2:
                X, Y = ''.join(
                    sg.popup_get_text('Незалежні?', title='Attributes (Independent)', icon='icon/math.ico').split())
                col = ''.join(sg.popup_get_text('Залежна?', title='Label (Dependent)', icon='icon/math.ico').split())

            X = df[[int(X), int(Y)]].values.reshape(-1, 2)
            Y = df[int(col[0])]

            x = X[:, 0]
            y = X[:, 1]
            z = Y


            def estimate_coef(x, y):
                # number of observations/points
                n = np.size(x)

                # mean of x and y vector
                m_x = np.mean(x)
                m_y = np.mean(y)

                # calculating cross-deviation and deviation about x
                SS_xy = np.sum(y * x) - n * m_y * m_x
                SS_xx = np.sum(x * x) - n * m_x * m_x

                # calculating regression coefficients
                b_1 = SS_xy / SS_xx
                b_0 = m_y - b_1 * m_x

                return (b_0, b_1)


            xx_pred = np.linspace(np.min(x), np.max(x), 30)
            yy_pred = np.linspace(np.min(y), np.max(y), 30)
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
            fig.suptitle(
                f'R = {r2:.2f}\ny = {ols.intercept_:.3f} + {ols.coef_[0]:.3f}x + {ols.coef_[1]:.3f}x1',
                fontsize=15, color='k')

            fig.tight_layout()
            plt.show()
        except:
            if df.shape[1] <= 2:
                sg.popup('Тільки для багатовимірних!', title='Упс..', icon='icon/math.ico')
            pass

    if event == 'Кор.Матр':
        try:
            corr = df.corr()
            sg.Print(f'{corr}', size=(65, 10), font='Courier 12', resizable=True, grab_anywhere=True)

            # making mask
            mask = np.zeros_like(corr, dtype=bool)
            np.fill_diagonal(mask, val=True)

            fig, ax = plt.subplots(figsize=(6, 4))

            cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)
            cmap.set_bad('grey')

            sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)
            fig.suptitle('Матриця кореляції пірсона', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=10)

            plt.show()
        except:
            sg.popup('Щось пішло не так...', title='Упс..', icon='icon/math.ico')
            pass

    if event == 'Частк.Кор.':
        try:
            save = ''
            if df.shape[1] > 2:
                X, Y = ''.join(sg.popup_get_text('X,Y?', title='X,Y', icon='icon/math.ico').split())
                Z = ''.join(sg.popup_get_text('Z?', title='Z', icon='icon/math.ico').split())
            import pingouin as pg

            p_corr = pg.partial_corr(data=df, x=int(X), y=int(Y), covar=int(Z)).round(3)
            save += 'Corr: ' + str(p_corr.to_numpy()[0][1]) + '\n'
            signif = p_corr.to_numpy()[0][3]

            if signif > 0.05:
                save += 'Не Значущий'
            else:
                save += 'Значущий'

            sg.popup(save, title='Відповідь:', icon='icon/math.ico')
        except:
            sg.popup('Щось пішло не так...', title='Упс..', icon='icon/math.ico')
            pass

    if event == 'Множин.Кор.':
        try:
            if df.shape[1] > 2:
                delete_n = int(
                    ''.join(sg.popup_get_text('Обрати:', title='Which array to delete?', icon='icon/math.ico').split()))
            corr = df.corr()
            det0 = np.linalg.det(corr)
            corr = corr.drop(corr.index[delete_n])
            del corr[delete_n]
            det1 = np.linalg.det(corr)
            f = ((len(df) - df.shape[1] - 1) / df.shape[1]) * (det0 / (1 - det0))
            if f > 0.05:
                sg.popup_ok(f'Multiple.Correlation: {(1 - (det0 / det1)) ** 1 / 2:.3f}\nЗначущий', title='Corr',
                            icon='icon/math.ico')
            else:
                sg.popup_ok(f'Multiple.Correlation: {(1 - (det0 / det1)) ** 1 / 2:.3f}\nНе значущий', title='Corr',
                            icon='icon/math.ico')
        except:
            sg.popup('Щось пішло не так...', title='Упс..', icon='icon/math.ico')
            pass

    if event == 'Паралельні кординати':
        if df.shape[1] > 2:
            bins = np.linspace(df[0].min(), df[0].max(), 4)
            labels = ['low', 'medium', 'max']
            df['cutted'] = pd.cut(df[0], bins, labels=labels, include_lowest=True)
            pd.plotting.parallel_coordinates(df, 'cutted', color=('#556270', '#4ECDC4', '#C7F464'))
            plt.show()
            df.drop(columns=['cutted'], inplace=True)
            # fig = px.parallel_coordinates(df, color=0)
            # fig.write_image("yourfile.png")
            # img = cv2.imread('yourfile.png')
            # plt.imshow(img)
            # plt.show()
        else:
            sg.popup('Only for multi-dim', icon='icon/math.ico')

    if event == 'Діагностична діаграма':
        try:
            if df.shape[1] > 2:
                col = ''.join(sg.popup_get_text('Which columns do you want?').split())
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.scatter(df[int(col[0])], df[int(col[1])], marker='x', c='r')
            ax.set_xlabel('{col}'.format(col=col[0]))
            ax.set_ylabel('{col1}'.format(col1=col[1]))
            plt.show()
        except:
            sg.popup('Щось пішло не так...', title='Упс..', icon='icon/math.ico')
            pass

    if event == 'Теплова карта':
        try:
            sns.heatmap(df[0:10], annot=True, fmt=".1f")
            plt.show()
        except:
            sg.popup('Щось пішло не так...', title='Упс..', icon='icon/math.ico')
            pass

    if event == 'Бульбашковий':
        try:

            if df.shape[1] > 2:
                X, Y = ''.join(sg.popup_get_text('Arguments(X,Y)?').split())
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
            sg.popup('Щось пішло не так...', title='Упс..', icon='icon/math.ico')
            pass

    if event == 'МГК':
        try:
            # Отримуємо матрицю коваріації завдяки функції
            matrix = cov_mat(df.to_numpy())
            # Рахуємо власні числа та власні вектори
            eigen_values, eigen_vectors = np.linalg.eigh(matrix)

            # Сортуємо індекси, вектори та власні числа, щоб було зручно виводити
            _ = np.argsort(eigen_values)[::-1]
            sorted_eigenvalue = np.around(eigen_values[_], decimals=3)
            sorted_eigenvectors = np.around(eigen_vectors[:, _], decimals=3)

            window['-FILE-'].update(f'Власні вектори: \n{sorted_eigenvectors}\n\n Власні числа: \n{sorted_eigenvalue}\n\n Частка %: \n{cumulative_coef(eigen_values)}')
            main_comp = int(''.join(sg.popup_get_text('К-сть головних компонент?', icon='icon/math.ico').split()))

            ev_subset = sorted_eigenvectors[:, 0:main_comp]

            # Трансформуємо дані. Скалярний добуток:(отримана вибірка^T * початкова вибірка^T)^T
            transform_df = np.dot(ev_subset.transpose(), df.transpose()).transpose()

            # Візуалізація
            fig = plt.figure(figsize=(9, 5))
            if main_comp == 2:
                ax = fig.add_subplot(121)
                ax.set_title('Normal')
                sns.histplot(x=df[0], y=df[1], bins=12, alpha=0.65, cmap='coolwarm', cbar=True)
                sns.scatterplot(x=df[0], y=df[1], s=25, color='black', alpha=0.6)

                ax = fig.add_subplot(122)
                ax.set_title('PCA')
                sns.histplot(x=transform_df[:, 0], y=transform_df[:, 1], bins=10, alpha=0.65, cmap='coolwarm')
                sns.scatterplot(x=transform_df[:, 0], y= transform_df[:, 1], s=25, color='black', alpha=0.65)

            if main_comp == 3:
                ax = plt.axes(projection='3d')

                ax.set_title('PCA')
                ax.set_xlabel('x', rotation=150)
                ax.set_ylabel('y')
                ax.set_zlabel('z', rotation=60)

                ax.scatter3D(transform_df[:, 0], transform_df[:, 1], transform_df[:, 2], cmap='coolwarm')
                ax.scatter3D(df[0], df[1], df[2])
            plt.show()

        except:
            pass

        try:
            window['-FILE-'].update(df.head(len(df)))
        except:
            sg.popup('Немає даних', title='Ооу', icon='icon/math.ico')

    if event == sg.WIN_CLOSED or event == 'Вихід':
        break
