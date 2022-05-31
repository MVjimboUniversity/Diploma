import math
import time
from queue import PriorityQueue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

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


# Класс для хранения графа
class Graph(object):
    def __init__(self, N, X, Y):
        self.N = N
        self.edges = []
        for i in range(N * N):
            self.edges.append({})
        self.edges_length = []
        for i in range(N * N):
            self.edges_length.append({})
        self.X = X
        self.Y = Y
        self.d_length = X[1] - X[0]

    def add_edge(self, num_1, num_2, weight, length):
        self.edges[num_1].update({num_2: weight})
        self.edges_length[num_1].update({num_2: length})

    def neighbours(self, node):
        return list(self.edges[node].keys())

    def weight(self, node_1, node_2):
        return self.edges[node_1].get(node_2)

    def length(self, node_1, node_2):
        return self.edges_length[node_1].get(node_2)

    def convert_xy(self, node):
        return node % self.N, math.floor(node / self.N)

    def convert_index(self, x, y):
        return y * self.N + x

    def get_values(self, node):
        x_ind, y_ind = self.convert_xy(node)
        return self.X[x_ind], self.Y[y_ind]


def add4(g, N):
    """
    Задает связи в графе g для 4 соседей в равномерной квадратной сетке

    Аргументы:
        g -- граф
        N -- размер вводимой сетки
    """
    x = g.X
    y = g.Y
    x_size = g.d_length
    y_size = g.d_length
    for i in range(N):
        for j in range(N):
            if (i-1) >= 0:
                g.add_edge(i*N+j, (i-1)*N+j, y_size*(beta(x[j], y[i])+beta(x[j], y[i-1]))/2, y_size)
            if (j-1) >= 0:
                g.add_edge(i*N+j, i*N+j-1, x_size*(beta(x[j], y[i])+beta(x[j-1], y[i]))/2, x_size)
            if (j+1) < N:
                g.add_edge(i*N+j, i*N+j+1, x_size*(beta(x[j], y[i])+beta(x[j+1], y[i]))/2, x_size)
            if (i+1) < N:
                g.add_edge(i*N+j, (i+1)*N+j, y_size*(beta(x[j], y[i])+beta(x[j], y[i+1]))/2, y_size)


def add8(g, N):
    """
    Задает связи в графе g для 8 соседей в равномерной квадратной сетке

    Аргументы:
        g -- граф
        N -- размер вводимой сетки
    """
    add4(g, N)
    x = g.X
    y = g.Y
    x_size = g.d_length
    y_size = g.d_length
    diag_size = (x_size**2 + y_size**2)**(1/2)
    for i in range(N):
        for j in range(N):
            if (i-1) >= 0 and (j-1) >= 0:
                g.add_edge(i*N+j, (i-1)*N+j-1, diag_size*(beta(x[j], y[i])+beta(x[j-1], y[i-1]))/2, diag_size)
            if (i-1) >= 0 and (j+1) < N:
                g.add_edge(i*N+j, (i-1)*N+j+1, diag_size*(beta(x[j], y[i])+beta(x[j+1], y[i-1]))/2, diag_size)
            if (i+1) < N and (j+1) < N:
                g.add_edge(i*N+j, (i+1)*N+j+1, diag_size*(beta(x[j], y[i])+beta(x[j+1], y[i+1]))/2, diag_size)
            if (i+1) < N and (j-1) >= 0:
                g.add_edge(i*N+j, (i+1)*N+j-1, diag_size*(beta(x[j], y[i])+beta(x[j-1], y[i+1]))/2, diag_size)


def add16(g, N):
    """
    Задает связи в графе g для 16 соседей в равномерной квадратной сетке

    Аргументы:
        g -- граф
        N -- размер вводимой сетки
    """
    add8(g, N)
    x = g.X
    y = g.Y
    x_size = g.d_length
    y_size = g.d_length
    diag_x_size = ((2*x_size)**2 + y_size**2)**(1/2)
    diag_y_size = (x_size**2 + (2*y_size)**2)**(1/2)
    for i in range(N):
        for j in range(N):
            if (i-1) >= 0 and (j-2) >= 0:
                g.add_edge(i*N+j, (i-1)*N+j-2, diag_x_size*(beta(x[j], y[i])+beta(x[j-2], y[i-1]))/2, diag_x_size)
            if (i-1) >= 0 and (j+2) < N:
                g.add_edge(i*N+j, (i-1)*N+j+2, diag_x_size*(beta(x[j], y[i])+beta(x[j+2], y[i-1]))/2, diag_x_size)
            if (i+1) < N and (j+2) < N:
                g.add_edge(i*N+j, (i+1)*N+j+2, diag_x_size*(beta(x[j], y[i])+beta(x[j+2], y[i+1]))/2, diag_x_size)
            if (i+1) < N and (j-2) >= 0:
                g.add_edge(i*N+j, (i+1)*N+j-2, diag_x_size*(beta(x[j], y[i])+beta(x[j-2], y[i+1]))/2, diag_x_size)
            if (i-2) >= 0 and (j-1) >= 0:
                g.add_edge(i*N+j, (i-2)*N+j-1, diag_y_size*(beta(x[j], y[i])+beta(x[j-1], y[i-2]))/2, diag_y_size)
            if (i-2) >= 0 and (j+1) < N:
                g.add_edge(i*N+j, (i-2)*N+j+1, diag_y_size*(beta(x[j], y[i])+beta(x[j+1], y[i-2]))/2, diag_y_size)
            if (i+2) < N and (j+1) < N:
                g.add_edge(i*N+j, (i+2)*N+j+1, diag_y_size*(beta(x[j], y[i])+beta(x[j+1], y[i+2]))/2, diag_y_size)
            if (i+2) < N and (j-1) >= 0:
                g.add_edge(i*N+j, (i+2)*N+j-1, diag_y_size*(beta(x[j], y[i])+beta(x[j-1], y[i+2]))/2, diag_y_size)


# Эвристическая фанкция - расстояние городских кварталов
def heuristic(g, a, b):
    a_x, a_y = g.get_values(a)
    b_x, b_y = g.get_values(b)
    return abs(a_x - b_x) + abs(a_y - b_y)


# Алгоритм A*
def a_star_search(graph, start, goal):
    """
    Возвращает матрицу переходов и дискретную поверхность накопленной стоимости

    Аргументы:
        graph -- граф
        start -- начальная вершина графа
        goal -- конечная вершина графа
    """
    # Используем очередь с приоритетами
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = [float('inf') for i in range(goal+1)]
    came_from[start] = start
    cost_so_far = [float('inf') for i in range(goal+1)]
    cost_so_far[start] = 0
    length_so_far = [float('inf') for i in range(goal+1)]
    length_so_far[start] = 0
    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break
        for next in graph.neighbours(current):
            new_cost = alpha*length_so_far[current]*graph.length(current, next) + cost_so_far[current] + graph.weight(current, next)
            if new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                length_so_far[next] = length_so_far[current] + graph.length(current, next)
                priority = new_cost + heuristic(graph, goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    return came_from, cost_so_far


def decode_path(g, path):
    x_points = np.zeros(len(path))
    y_points = np.zeros(len(path))
    for i, node in enumerate(path):
        x_points[i], y_points[i] = g.get_values(node)
    beta_points = beta_np(x_points, y_points)
    return x_points, y_points, beta_points


def reconstruct_path(g, came_from, start, goal):
    """
    Возвращает список вершин, являющимся кратчайшим путем в графе

    Аргументы:
        g -- граф
        came_from -- матрица переходов
        start -- начальная вершина
        goal -- конечная вершина
    """
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path = path[::-1]
    path = decode_path(g, path)
    return path


def a_star_find_path(add_connections, N=20, repeat_times=1):
    """
    Возвращает траекторию дороги, время работы метода и стоимость дороги

    Аргументы:
        add_conection -- тип соединения (add4, add8, add16)
        N -- размер сетки (default 20)
        repeat_times -- количество повторений (default 1)
    """
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    sum_time = 0
    for _ in range(repeat_times):
        graph = Graph(N, x, y)
        start_time = time.time()
        add_connections(graph, N)
        came_from, cost = a_star_search(graph, 0, N * N - 1)
        x_path, y_path, beta_path = reconstruct_path(graph, came_from, 0, N * N - 1)
        end_time = time.time()
        sum_time += (end_time - start_time)

    return x_path, y_path, beta_path, sum_time/repeat_times, cost[-1]


def a_star_find_data(params, add_connections, repeat_times=1):
    data = np.zeros((3, len(params)))
    for i, N in enumerate(params):
        x_path, y_path, beta_path, work_time, cost = a_star_find_path(add_connections, N, repeat_times)
        data[0, i] = N
        data[1, i] = work_time
        data[2, i] = cost
    return data


def data_draw_paths(num_of_connections, add_connection, parameters, draw_parameters):
    """
    Проецируем на поверхность траектории построенные для различных размеров сетки N, получаемых из parameters

    Аргументы:
        num_of_connections -- кол-во соседей (4, 8, 16)
        add_conection -- тип соединения (add4, add8, add16)
        parameters -- список значений для параметра N (размера сетки) для сохранения в файл
        draw_parameters -- список значений для параметра N (размера сетки) для проекции
    """
    data = a_star_find_data(parameters, add_connection)
    np.savetxt(f"a_star_{num_of_connections}.csv", data, delimiter=";", fmt="%.3f")

    xx, yy = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
    _beta = beta_np(xx, yy)

    fig = plt.figure(figsize=(12, 5))
    fig.canvas.manager.set_window_title(f"A{num_of_connections}")
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Проекция на поверхность")
    ax1.plot_wireframe(xx, yy, _beta, alpha=0.25)

    ax2 = fig.add_subplot(122)
    ax2.set_title("Вид сверху")
    ax2.grid()
    im = ax2.contourf(xx, yy, _beta, np.linspace(0, 2, 100), cmap="Greys")

    for N in draw_parameters:
        x_path, y_path, beta_path, _, cost = a_star_find_path(add_connection, N)
        ax1.plot(x_path, y_path, beta_path)
        ax2.plot(x_path, y_path, label=f"N = {N}")
    ax2.legend()
    plt.colorbar(im, label=r'$z=\beta(x,y)$')
    plt.show()


def main():
    data_draw_paths(4, add4, [10, 20, 50, 100, 500], [20, 100, 500])
    data_draw_paths(8, add8, [10, 20, 50, 100, 500], [20, 100, 500])
    data_draw_paths(16, add16, [10, 20, 50, 100, 500], [20, 100, 500])


if __name__ == "__main__":
    main()
