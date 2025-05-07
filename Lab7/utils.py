import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools
import warnings
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import numpy as np

def plot_per_dates(X, Y, title, ylabel, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(X, Y)
    plt.xlabel('Дата')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(11))  # Show only 10 ticks on x-axis
    plt.tight_layout()
    plt.grid()
    plt.show()

def print_fuller_test(fuller_test):
    adf, p_value = fuller_test[0], fuller_test[1]
    print(f'ADF=: {adf}')
    print(f'p=: {p_value}')

    if p_value < 0.05:
        print('Часовий ряд стаціонарний')
    else:
        print('Часовий ряд не стаціонарний')

def best_arima_coeffs(d, test, train, range_limit = 4):
    p_range = range(1, range_limit)
    q_range = range(1, range_limit)
    pdq_combinations = list(itertools.product(p_range, [d], q_range))

    results = []
    best_aic = float('inf')
    best_order = None

    warnings.filterwarnings('ignore')
    for pdq in pdq_combinations:
        try:
            model = ARIMA(train['DailyConfirmed'], order=pdq)
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            error = rmse(test['DailyConfirmed'], predictions)
            aic = model_fit.aic
            results.append([pdq, aic, error])

            if aic < best_aic:
                best_aic = aic
                best_order = pdq

            print(f'ARIMA{pdq} : AIC={aic:.2f}, RMSE={error:.2f}')
        except:
            continue

    print(f'Найраща ARIMA{best_order}; AIC={best_aic:.2f}')
    return best_order

def plot_prediction(predict, test, train, title):
    plt.figure(figsize=(16, 10))
    plt.plot(train['Date'], train['DailyConfirmed'], label='Тренувальні дані', color='orange')
    plt.plot(test['Date'], test['DailyConfirmed'], label='Тренувальні дані', color='orange', linestyle='--')
    plt.plot(test['Date'], predict, label='Прогноз', color='g')
    plt.xlabel('Дата')
    plt.ylabel('К-сть підтверджених випадків щодня')
    plt.title(title)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(11))
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def best_alpha(train, test):
    best_alpha = None
    best_rmse = float('inf')

    for alpha in np.arange(0.1, 1.0, 0.1):
        model = SimpleExpSmoothing(train['DailyConfirmed'].values)
        model_fit = model.fit(smoothing_level=alpha, optimized=False)
        pred = model_fit.forecast(len(test))
        error = rmse(test['DailyConfirmed'].values, pred)
        print(f'α={alpha:.1f}, RMSE={error:.2f}')

        if error < best_rmse:
            best_rmse = error
            best_alpha = alpha

    print(f'Найкращий α={best_alpha}, RMSE={best_rmse:.2f}')
    return best_alpha