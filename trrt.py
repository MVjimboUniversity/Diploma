import math
import random
import time
import numpy as np
import kdtree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Задаем параметр альфа - цену доставки строительных материалов
alpha = 0.1


# Задаем beta и beta_np - непрерывные функции, определяющие цену укладки дорожного полотна
def beta(x, y):
    return 1 + math.sin(5*x) * math.sin(y)


def beta_np(x, y):
    return 1 + np.sin(5*x) * np.sin(y)


# Задаем координаты начальной и конечной точек
start_point = np.array([0., 0.])
goal_point = np.array([1., 1.])

# Задаем границы рассматриваемого пространства
x_edges = np.array([0., 1.])
y_edges = np.array([0., 1.])


# Класс для хранения вершин дерева
class Node:
    def __init__(self, point, id):
        self.id = id
        self.point = point

    def __getitem__(self, item):
        return self.point[item]

    def __len__(self):
        return len(self.point)


def distance(point_1, point_2):
    """
    Вычисляет евклидово расстояние

    Аргументы:
        point_1 -- координаты первой точки
        point_2 -- координаты второй точки
    """
    square = np.square(point_1 - point_2)
    sum_square = np.sum(square)
    return np.sqrt(sum_square)


def stop_condition(new_node, goal, stop_size):
    """Возвращает True, если расстояние между new_node и goal меньше stop_size"""
    if distance(new_node, goal) <= stop_size:
        return True
    return False


def generate_rand_point(x_edges, y_edges):
    """Генерация рандомной точки в рассматриваемой области"""
    x = random.uniform(x_edges[0], x_edges[1]+0.1)
    y = random.uniform(y_edges[0], y_edges[1]+0.1)
    return np.array([x, y])


def find_best_neighbour(tree, point, cost):
    """
    Возвращает соседа к point с наименьшей стоймостью дороги среди 3 ближайших вершин в рассматриваемом пространстве

    Аргументы:
        tree -- kd-дерево
        point -- координаты точки
        cost -- массив стоимостей для дерева
    """
    neighbours = tree.search_knn(point, 3)
    min_cost = 1000000
    best_neighbour = None
    for neighbour in neighbours:
        id = neighbour[0].data.id
        if min_cost > cost[id]:
            min_cost = cost[id]
            best_neighbour = neighbour[0].data
    return best_neighbour


def count_new_point(nearest_point, rand_point, delta):
    """
    Определяем координаты новой вершины

    Аргументы:
        nearest_point -- координаты близжайшей вершины дерева
        rand_point --  координаты случайной точки
        delta -- шаг приращения
    """
    if distance(nearest_point, rand_point) <= delta:
        return rand_point
    angle = math.atan2(rand_point[1] - nearest_point[1], rand_point[0] - nearest_point[0])
    x_delta = delta * math.cos(angle)
    y_delta = delta * math.sin(angle)
    return np.array([nearest_point[0] + x_delta, nearest_point[1] + y_delta])


def out_of_area(point):
    """Проверка на выход за пределы рассматриваемой области"""
    if x_edges[0] <= point[0] <= x_edges[1] and y_edges[0] <= point[1] <= y_edges[1]:
        return False
    return True


def transition_test(nearest_point, new_point, cost_func, K, T, nFail, alpha_trrt, nFail_max):
    """Возвращаем True, если принято решение добавить вершину в дерево"""
    near_value = cost_func(nearest_point[0], nearest_point[1])
    new_value = cost_func(new_point[0], new_point[1])
    if new_value < near_value:
        return True, T, nFail
    delta_value = (new_value - near_value) / distance(nearest_point, new_point)
    p = math.exp(- delta_value / (K * T))
    if random.random() < p:
        T = T / alpha_trrt
        nFail = 0
        return True, T, nFail
    else:
        if nFail > nFail_max:
            T = T * alpha_trrt
            nFail = 0
        else:
            nFail = nFail + 1
        return False, T, nFail


def min_expand_control(nearest_point, new_point, exploration, refinement, delta, ratio_e_r):
    """Возвращаем True, если принято решение добавить вершину в дерево"""
    d = distance(nearest_point, new_point)
    if d >= delta:
        exploration += 1
        return True, exploration, refinement
    else:
        refinement += 1
        if exploration / refinement < ratio_e_r:
            refinement -= 1
            return False, exploration, refinement
        return True, exploration, refinement


# Метод T-RRT
def trrt(start, goal, cost_func, T=(10 ** (-2)), alpha_trrt=2, delta=0.05, stop_size=0.06, nFail_max=1, ratio_e_r=1):
    """
    Вовращает накопленную стоимсть для вершин дерева и построенное дерево

    Аргументы:
        start -- координаты начальной точки
        goal -- координаты конечной точки
        cost_func -- функция стоимости
        T -- температура
        alpha_trrt, delta, stop_size, nFail_max, ratio_e_r -- гиперпараметры
    """
    exploration = 0
    refinement = 0
    K = (beta(start[0], start[1]) + beta(goal[0], goal[1])) / 2
    nFail = 0

    start_node = Node(start, 0)
    new_node_verified = start_node
    cost_so_far = [0]
    length_so_far = [0]

    # Строим дерево
    nodes = [start_node]
    parent_tree = [None]
    # Используем kd-дерево для определения близжайших соседей
    kd_tree = kdtree.create([start_node], dimensions=2)

    counter_id = 1
    while not stop_condition(new_node_verified.point, goal, stop_size):
        # Генерируем rand_point, ищем nearest_point и создаем new_point
        rand_point = generate_rand_point(x_edges, y_edges)
        nearest_node = find_best_neighbour(kd_tree, rand_point, cost_so_far)
        new_point = count_new_point(nearest_node.point, rand_point, delta)
        # Если new_point удовлетворяет всем условиям, то добавляем ее в дерево
        if out_of_area(new_point):
            continue
        transitionTest, T, nFail = transition_test(nearest_node.point, new_point, cost_func, K, T, nFail,
                                                   alpha_trrt, nFail_max)
        minExpandControl, exploration, refinement = min_expand_control(nearest_node.point, new_point, exploration,
                                                                       refinement, delta, ratio_e_r)
        if transitionTest and minExpandControl:
            new_node = Node(new_point, counter_id)
            new_node_verified = new_node

            nodes.append(new_node)
            kd_tree.add(new_node)
            parent_tree.append(nearest_node.id)

            d = distance(nearest_node.point, new_point)
            c = (cost_func(nearest_node.point[0], nearest_node.point[1]) + cost_func(new_point[0], new_point[1])) / 2
            new_node_cost = alpha * length_so_far[nearest_node.id] * d + cost_so_far[nearest_node.id] + c * d

            length_so_far.append(length_so_far[nearest_node.id] + d)
            cost_so_far.append(new_node_cost)

            counter_id += 1

    goal_node = Node(goal, counter_id)
    nodes.append(goal_node)
    parent_tree.append(new_node_verified.id)

    d = distance(new_node_verified.point, goal)
    c = (cost_func(new_node_verified.point[0], new_node_verified.point[1]) + cost_func(goal[0], goal[1])) / 2
    new_node_cost = alpha * length_so_far[new_node_verified.id] * d + cost_so_far[new_node_verified.id] + c * d
    cost_so_far.append(new_node_cost)

    return cost_so_far, parent_tree, nodes


def reconstruct_path(path_tree, nodes):
    """Восстанавливаем путь, соостоящий из вершин дерева"""
    path = [nodes[-1].point]
    node = path_tree[-1]
    while node is not None:
        path.append(nodes[node].point)
        node = path_tree[node]
    return np.array(path[::-1])


def trrt_find_path(delta=0.05, stop_size=0.06):
    """Ищем траекторию дороги, время работы, стоимость дороги и дерево"""
    start_time = time.time()
    cost_so_far, parent_tree, nodes = trrt(start_point, goal_point, beta, delta=delta, stop_size=stop_size)
    path = reconstruct_path(parent_tree, nodes)
    end_time = time.time()
    work_time = end_time - start_time
    return path[:, 0],  path[:, 1], beta_np(path[:, 0],  path[:, 1]), work_time, cost_so_far[-1], parent_tree, nodes


def trrt_find_best_path(num_repeats=10, draw_tree=False, delta=0.05, stop_size=0.06):
    """Ищем лучшую по стоимости дороги за num_repeats повторений"""
    best_cost = 100000
    sum_cost = 0
    sum_time = 0
    for _ in range(num_repeats):
        x_path, y_path, beta_path, work_time, cost, parent_tree, nodes = trrt_find_path(delta, stop_size)
        sum_time += work_time
        sum_cost += cost
        if best_cost > cost:
            best_x_path = x_path
            best_y_path = y_path
            best_beta_path = beta_path
            best_cost = cost
            best_parent_tree = parent_tree
            best_nodes = nodes
    if draw_tree:
        show_tree(best_x_path, best_y_path, best_parent_tree, best_nodes)
    return best_x_path, best_y_path, best_beta_path, sum_time/num_repeats, best_cost, sum_cost/num_repeats


def get_tree(parent_tree, nodes):
    tree = []
    for i, node in enumerate(nodes):
        parent_id = parent_tree[i]
        if parent_id is not None:
            parent = nodes[parent_id]
            tree.append((node.point[0], parent.point[0]))
            tree.append((node.point[1], parent.point[1]))
            tree.append('b')
    return tree


def show_tree(x_path, y_path, parent_tree, nodes):
    """Отрисовываем дерево, построеноое методом T-RRT"""
    tree = get_tree(parent_tree, nodes)

    xx, yy = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
    _beta = beta_np(xx, yy)

    plt.figure(figsize=(6, 5))
    plt.title("Итоговое дерево")
    plt.grid()
    plt.contourf(xx, yy, _beta, np.linspace(0, 2, 100), cmap="Greys")
    plt.plot(*tree)
    plt.plot(x_path, y_path, color='red')
    plt.colorbar(label=r'$z=\beta(x,y)$')
    plt.show()


def trrt_find_data(params, repeat_times=1):
    data = np.zeros((4, len(params)))
    for i, (delta, s_s) in enumerate(params):
        x_path, y_path, beta_path, work_time, best_cost, avg_cost = trrt_find_best_path(repeat_times, delta=delta,
                                                                                        stop_size=s_s)
        data[0, i] = delta
        data[1, i] = work_time
        data[2, i] = best_cost
        data[3, i] = avg_cost
    return data


def data_draw_paths(num_repeats, parameters, draw_parameters):
    """
    Проецируем на поверхность траектории построенные для различных значений шага delta, получаемых из parameters

    Аргументы:
        num_repeats -- количество повторений
        parameters -- список значений для параметра delta (шаг приращения) для сохранения в файл
        draw_parameters -- список значений для параметра delta (шаг приращения) для проекции
    """
    data = trrt_find_data(parameters, num_repeats)
    np.savetxt(f"trrt_delta.csv", data, delimiter=";", fmt="%.3f")

    xx, yy = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
    _beta = beta_np(xx, yy)

    fig = plt.figure(figsize=(12, 5))
    fig.canvas.manager.set_window_title(f"T-RRT")
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Проекция на поверхность")
    ax1.plot_wireframe(xx, yy, _beta, alpha=0.25)

    ax2 = fig.add_subplot(122)
    ax2.set_title("Вид сверху")
    ax2.grid()
    im = ax2.contourf(xx, yy, _beta, np.linspace(0, 2, 100), cmap="Greys")

    for N in draw_parameters:
        x_path, y_path, beta_path, _, _, _ = trrt_find_best_path(num_repeats, delta=N[0], stop_size=N[1])
        ax1.plot(x_path, y_path, beta_path)
        ax2.plot(x_path, y_path, label=f"$\\delta={N[0]}$")
    ax2.legend()
    plt.colorbar(im, label=r'$z=\beta(x,y)$')
    plt.show()


def main():
    x_path, y_path, beta_path, avg_time, best_cost, avg_cost = trrt_find_best_path(10, True)
    data = np.array([[avg_time], [best_cost], [avg_cost]])
    np.savetxt(f"trrt.csv", data, delimiter=";", fmt="%.3f")

    data_draw_paths(10, ((0.02, 0.04), (0.05, 0.06), (0.1, 0.11)), ((0.02, 0.04), (0.05, 0.06), (0.1, 0.11)))


if __name__ == "__main__":
    main()
