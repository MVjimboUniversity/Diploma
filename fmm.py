import math
import time
import numpy as np
from queue import PriorityQueue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata

# Задаем параметр альфа - цену доставки строительных материалов
alpha = 0.1
# Задаем координаты начальной и конечной точек
start_point = (0, 0)
end_point = (1, 1)


# Задаем beta и beta_np - непрерывные функции, определяющие цену укладки дорожного полотна
def beta(x, y):
    return 1 + math.sin(5*x) * math.sin(y)


def beta_np(x, y):
    return 1 + np.sin(5*x) * np.sin(y)


def add(g, N):
    """
    Задает связи в графе g для 4 соседей в равномерной квадратной сетке

    Аргументы:
        g -- граф
        N -- размер вводимой сетки
    """
    x = g.X
    y = g.Y
    size = g.d_length
    for i in range(N):
        for j in range(N):
            if (i - 1) >= 0:
                g.add_edge(i * N + j, (i - 1) * N + j, size * (beta(x[j], y[i]) + beta(x[j], y[i - 1])) / 2)
            if (j - 1) >= 0:
                g.add_edge(i * N + j, i * N + j - 1, size * (beta(x[j], y[i]) + beta(x[j - 1], y[i])) / 2)
            if (j + 1) < N:
                g.add_edge(i * N + j, i * N + j + 1, size * (beta(x[j], y[i]) + beta(x[j + 1], y[i])) / 2)
            if (i + 1) < N:
                g.add_edge(i * N + j, (i + 1) * N + j, size * (beta(x[j], y[i]) + beta(x[j], y[i + 1])) / 2)

# Класс для хранения графа
class GridFMM(object):
    def __init__(self, N, func, X, Y):
        self.N = N
        self.func = func
        self.edges = []
        for i in range(N * N):
            self.edges.append({})
        self.d_length = X[1] - X[0]
        self.X = X
        self.Y = Y

    def get_val(self, node):
        x = node % self.N
        y = math.floor(node / self.N)
        return self.func(self.X[x], self.Y[y])

    def add_edge(self, num_1, num_2, weight):
        self.edges[num_1].update({num_2: weight})

    def neighbours(self, node):
        return list(self.edges[node].keys())

    def weight(self, node_1, node_2):
        return self.edges[node_1].get(node_2)

    def convert_xy(self, node):
        return node % self.N, math.floor(node / self.N)

    def convert_index(self, x, y):
        return y * self.N + x


def get_info(neighbour, current, graph, cost_so_far, length_so_far):
    d = cost_so_far[neighbour]
    d_weight = graph.weight(neighbour, current)
    d_length_so_far = length_so_far[neighbour]
    return d, d_weight, d_length_so_far

# Метод  FMM
def fmm_find_cost_matrix(grid, start, goal):
    """
    Возвращает дискретную поверхность накопленной стоимости

    Аргументы:
        grid -- граф
        start -- начальная вершина графа
        goal -- конечная вершина графа
    """
    N = grid.N
    is_visited = [False for _ in range(goal + 1)]
    is_visited[start] = True
    cost_so_far = [float('inf') for _ in range(goal + 1)]
    cost_so_far[start] = 0
    length_so_far = [float('inf') for _ in range(goal + 1)]
    length_so_far[start] = 0
    # Используем очередь с приоритетами
    frontier = PriorityQueue()
    frontier.put(start)

    while not frontier.empty():
        cur = frontier.get()

        for current in grid.neighbours(cur):
            x, y = grid.convert_xy(current)
            if is_visited[current]:
                continue
            # Ищем соседа с минимальным знаением стоимости по горизонтали (вдоль оси Ox)
            if x == 0:
                dx, dx_weight, dx_length_so_far = get_info(grid.convert_index(x + 1, y), current, grid, cost_so_far,
                                                         length_so_far)
            elif x == N - 1:
                dx, dx_weight, dx_length_so_far = get_info(grid.convert_index(x - 1, y), current, grid, cost_so_far,
                                                         length_so_far)
            else:
                back = cost_so_far[grid.convert_index(x - 1, y)]
                frw = cost_so_far[grid.convert_index(x + 1, y)]
                if frw < back:
                    dx, dx_weight, dx_length_so_far = get_info(grid.convert_index(x + 1, y), current, grid, cost_so_far,
                                                             length_so_far)
                else:
                    dx, dx_weight, dx_length_so_far = get_info(grid.convert_index(x - 1, y), current, grid, cost_so_far,
                                                             length_so_far)

            # Ищем соседа с минимальным знаением стоимости по вертикали (вдоль оси Oy)
            if y == 0:
                dy, dy_weight, dy_length_so_far = get_info(grid.convert_index(x, y + 1), current, grid, cost_so_far,
                                                         length_so_far)
            elif y == N - 1:
                dy, dy_weight, dy_length_so_far = get_info(grid.convert_index(x, y - 1), current, grid, cost_so_far,
                                                         length_so_far)
            else:
                back = cost_so_far[grid.convert_index(x, y - 1)]
                frw = cost_so_far[grid.convert_index(x, y + 1)]
                if frw < back:
                    dy, dy_weight, dy_length_so_far = get_info(grid.convert_index(x, y + 1), current, grid, cost_so_far,
                                                             length_so_far)
                else:
                    dy, dy_weight, dy_length_so_far = get_info(grid.convert_index(x, y - 1), current, grid, cost_so_far,
                                                             length_so_far)

            # Ищем дискриминант
            d_length = grid.d_length
            cur_val = grid.get_val(current) * d_length
            D = 2 * cur_val ** 2 - (dx - dy) ** 2
            if D >= 0:
                D_2 = D ** (1 / 2)
                L = alpha * (d_length * dx_length_so_far + d_length * dy_length_so_far)
                new_cost = L / 2 + (dx + dy + D_2) / 2
                if new_cost < cost_so_far[current]:
                    cost_so_far[current] = new_cost
                    length_so_far[current] = (dx_length_so_far + dy_length_so_far) / 2 + d_length
                    frontier.put(current, new_cost)
                    is_visited[current] = True
            else:
                new_cost_x = alpha * dx_length_so_far * d_length + dx + dx_weight
                new_cost_y = alpha * dy_length_so_far * d_length + dy + dy_weight
                if new_cost_x < new_cost_y and new_cost_x < cost_so_far[current]:
                    cost_so_far[current] = new_cost_x
                    length_so_far[current] = dx_length_so_far + d_length
                    frontier.put(current, new_cost_x)
                    is_visited[current] = True
                elif new_cost_y <= new_cost_x and new_cost_y < cost_so_far[current]:
                    cost_so_far[current] = new_cost_y
                    length_so_far[current] = dy_length_so_far + d_length
                    frontier.put(current, new_cost_y)
                    is_visited[current] = True

            if current == goal:
                break
    cost_matrix = np.array(cost_so_far).reshape((N, N))
    return cost_matrix


def fmm_reconstruct_path(cost_matrix, start_point, end_point, x, y):
    """
    Строит траекторию дороги по дискретной поверхности накопленной стоимости

    Аргументы:
        cost_matrix -- дискретная поверхность накопленной стоимости
        start_point -- начальная точка
        end_point -- конечная точка
        x -- значения вдоль оси Ox для сетки
        y -- значения вдоль оси Oy для сетки
    """
    grad = np.gradient(cost_matrix)
    grad = np.dstack((grad[1], grad[0]))
    grad_norm = np.linalg.norm(grad, axis=2)
    grad_normalized = grad / np.dstack((grad_norm, grad_norm))
    x0 = np.array(start_point)
    xx, yy = np.meshgrid(x, y)
    points = np.dstack((xx, yy)).reshape((-1, 2))
    grad_normalized_vector = np.reshape(grad_normalized, (-1, 2))
    path = []
    t = .08
    path.append(np.array(end_point))
    for i in range(int(2 / t)):
        direction = griddata(points, grad_normalized_vector, path[-1], method='linear')[0]
        new_point = path[-1] - t * direction
        if np.linalg.norm(path[-1] - x0) < 0.1:
            break
        if new_point[0] < 0.:
            new_point[0] = 0.
            path.append(new_point)
            break
        if new_point[1] < 0.:
            new_point[1] = 0.
            path.append(new_point)
            break
        path.append(new_point)
    path.append(x0)
    path = np.array(path)
    return path[:, 0], path[:, 1], beta_np(path[:, 0], path[:, 1])


def fmm_find_path(N=20, repeat_times=1):
    """
    Возвращает траекторию дороги, время работы метода и стоимость дороги

    Аргументы:
        N -- размер сетки (default 20)
        repeat_times -- количество повторений (default 1)
    """
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    sum_time = 0
    for _ in range(repeat_times):
        grid = GridFMM(N, beta, x, y)
        start_time = time.time()
        add(grid, N)
        cost = fmm_find_cost_matrix(grid, 0, N * N - 1)
        x_path, y_path, beta_path = fmm_reconstruct_path(cost, start_point, end_point, x, y)
        end_time = time.time()
        sum_time += (end_time - start_time)
    return x_path, y_path, beta_path, sum_time / repeat_times, cost[N - 1, N - 1]


def fmm_find_data(params, repeat_times=1):
    data = np.zeros((3, len(params)))
    for i, N in enumerate(params):
        x_path, y_path, beta_path, work_time, cost = fmm_find_path(N, repeat_times)
        data[0, i] = N
        data[1, i] = work_time
        data[2, i] = cost
    return data


def data_draw_paths(parameters, draw_parameters):
    """
    Проецируем на поверхность траектории построенные для различных размеров сетки N, получаемых из parameters

    Аргументы:
        parameters -- список значений для параметра N (размера сетки) для сохранения в файл
        draw_parameters -- список значений для параметра N (размера сетки) для проекции
    """
    data = fmm_find_data(parameters)
    np.savetxt("fmm.csv", data, delimiter=";", fmt="%.3f")

    xx, yy = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
    _beta = beta_np(xx, yy)

    fig = plt.figure(figsize=(12, 5))
    fig.canvas.manager.set_window_title(f"FMM")
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Проекция на поверхность")
    ax1.plot_wireframe(xx, yy, _beta, alpha=0.25)

    ax2 = fig.add_subplot(122)
    ax2.set_title("Вид сверху")
    ax2.grid()
    im = ax2.contourf(xx, yy, _beta, np.linspace(0, 2, 100), cmap="Greys")

    for N in draw_parameters:
        x_path, y_path, beta_path, _, cost = fmm_find_path(N)
        ax1.plot(x_path, y_path, beta_path)
        ax2.plot(x_path, y_path, label=f"N = {N}")
    ax2.legend()
    plt.colorbar(im, label=r'$z=\beta(x,y)$')
    plt.show()


def main():
    data_draw_paths([10, 20, 50, 100, 500], [20, 50, 100])


if __name__ == "__main__":
    main()
