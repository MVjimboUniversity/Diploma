import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from scipy.optimize import fsolve

# Задаем параметр альфа - цену доставки строительных материалов
alpha = 0.1


# Задаем beta и beta_np - непрерывные функции, определяющие цену укладки дорожного полотна
def beta(x, y):
    return 1 + math.sin(5*x) * math.sin(y)


def beta_np(x, y):
    return 1 + np.sin(5*x) * np.sin(y)


# Частная производная beta по x
def betax(x, y):
    return 5 * math.cos(5*x) * math.sin(y)


# Частная производная beta по y
def betay(x, y):
    return math.sin(5*x) * math.cos(y)


def f1(x):
    return ([x[0]+x[0]*x[1]-2, x[0]-x[1]-2])


def diffy1(S, N, y):
    up = np.dot(np.dot(S, S), y)
    down = np.dot(S, np.ones(N))
    res = (1 - up[N-1])/down[N-1]
    return res


def diff1(S, N, y):
    y1 = diffy1(S, N, y)
    res = y1 * np.ones(N) + np.dot(S, y)
    return res


def func_y(S, N, y):
    y1 = diffy1(S, N, y)
    res = y1 * np.dot(S, np.ones(N)) + np.dot(np.dot(S, S), y)
    return res


# Формула Симпсона
def simpson(N, y_dash):
    integ = 0
    h = 1 / (N - 1)
    F = lambda x: (1 + x ** 2) ** 0.5

    for i in range(1, int(np.floor((N - 1) / 2)) + 1):
        integ = integ + F(y_dash[2 * i - 2]) + 4 * F(y_dash[2 * i - 1]) + F(y_dash[2 * i])
    return h * integ / 3


# Итоговая система нелинейных уравнений
def root(yd2, *args):
    (S, N, alpha) = args
    h = 1/(N-1)
    yd1 = diff1(S, N, yd2)
    y = func_y(S, N, yd2)
    simp = simpson(N, yd1)
    res = np.zeros(N)
    for i in range(N):
        j = i + 1
        res[i] = (yd2[i] / (1 + (yd1[i]) ** 2)) * (alpha*simp + beta(j*h, y[i])) + yd1[i] * betax(j*h, y[i])\
                 - betay(j*h, y[i])
    return res


def optimize_find_path(N):
    """
    Возвращает координаты оптимальной траектории, и проекцию на поверхность,
    определяющую цену укладки дорожного полотна

    Аргументы:
        N -- размер сетки, вводимой в рассматриваемом пространстве
    """
    x = np.linspace(0, 1, N)

    Z = np.vander(x, increasing=True)

    B = np.zeros((N, N), dtype=np.float64)
    for j in range(N):
            B[:,j] = (np.power(x, (j+1)) - np.power(x[0], (j+1)))/(j+1)

    S = np.dot(B, np.linalg.inv(Z))

    yd2 = np.zeros(N)
    # Решение системы нелинейных уравнений
    yd2 = fsolve(root, yd2, args=(S, N, alpha))

    y = func_y(S, N, yd2)
    beta = beta_np(x, y)
    return x, y, beta


def count_cost(x, y, beta):
    """
    Возвращает стоимсоть построенной дороги

    Аргументы:
        x -- координаты x траектории дороги
        y -- координаты y траектории дороги
        beta -- значения beta(x, y), соответствующие каждой точки траектрии дороги
    """
    cost_for_build = (beta[1:] + beta[:-1])/2
    length_by_x = x[1:] - x[:-1]
    length_by_y = y[1:] - y[:-1]
    length = np.sqrt(length_by_x**2 + length_by_y**2)
    cost_beta = cost_for_build * length
    N = x.shape[0]
    length_accum = np.zeros(N-1)
    for i in range(N-1):
        if i == 0:
            length_accum[i] = length[i]
        else:
            length_accum[i] = length_accum[i-1] + length[i]
    cost_alpha = np.zeros(N-1)
    for i in range(N-1):
        if i != 0:
            cost_alpha[i] = alpha * length_accum[i-1] * length[i]
    cost = np.sum(cost_alpha) + np.sum(cost_beta)
    return cost


def optimize_find_path_time_cost(N, repeat_times=1):
    """
    Возвращает траекторию дороги, ее проекцию на beta, среднее время работы метода, стоимсоть построенной дороги

    Аргументы:
        N -- размер сетки, вводимой в рассматриваемом пространстве
        repeat_times -- количество повторений метода(default 1)
    """
    sum_time = 0
    for _ in range(repeat_times):
        start_time = time.time()
        x_path, y_path, beta_path = optimize_find_path(N)
        cost = count_cost(x_path, y_path, beta_path)
        end_time = time.time()
        sum_time += (end_time - start_time)
    return x_path, y_path, beta_path, sum_time/repeat_times, cost


def optimize_find_best(params, repeat_times=1):
    """
    Ищет лучший размер сетки N

    Аргументы:
        params -- набор значений для праметра N
        repeat_times -- количество повторений метода(default 1)
    """
    data = np.zeros((3, len(params)))
    best_cost = 10000
    best_N = 0
    for i, N in enumerate(params):
        x_path, y_path, beta_path, work_time, cost = optimize_find_path_time_cost(N, repeat_times)
        data[0, i] = N
        data[1, i] = work_time
        data[2, i] = cost
        if best_cost > cost:
            best_cost = cost
            best_N = N
    return data, best_N


def main():
    # Ищем стоимость траектории и время работы метода для различных размеров сетки
    data, best_N = optimize_find_best(range(5, 26), repeat_times=10)
    # Сохраняем результаты
    np.savetxt("optimize.csv", data, delimiter=";", fmt="%.3f")
    print(best_N)

    # Ищем оптимальную по стоймости траекторию и проецируем ее на поверхность
    x_path, y_path, beta_path = optimize_find_path(best_N)

    xx, yy = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
    _beta = beta_np(xx, yy)

    fig = plt.figure(figsize=(12, 5))
    fig.canvas.manager.set_window_title("Optimization")
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title("Проекция на поверхность")
    ax.plot_wireframe(xx, yy, _beta, alpha=0.25)
    ax.plot(x_path, y_path, beta_path, color='red')

    ax = fig.add_subplot(122)
    ax.set_title("Вид сверху")
    im = ax.contourf(xx, yy, _beta, np.linspace(0, 2, 100), cmap="Greys")
    ax.plot(x_path, y_path, color='red')
    ax.grid()
    plt.colorbar(im, label=r'$z=\beta(x,y)$')

    plt.show()


if __name__ == "__main__":
    main()
