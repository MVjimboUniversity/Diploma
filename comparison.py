import numpy as np
import matplotlib.pyplot as plt
from a_star import a_star_find_path, add4, add8, add16
from fmm import fmm_find_path
from optimization import optimize_find_path_time_cost
from trrt import trrt_find_best_path


def beta_np(x, y):
    return 1 + np.sin(5*x) * np.sin(y)


def draw_paths(paths):
    """Проецируем на поверхность траектории из paths"""
    xx, yy = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
    _beta = beta_np(xx, yy)

    fig = plt.figure(figsize=(12, 5))
    fig.canvas.manager.set_window_title(f"Comparision")
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Проекция на поверхность")
    ax1.plot_wireframe(xx, yy, _beta, alpha=0.25)

    ax2 = fig.add_subplot(122)
    ax2.set_title("Вид сверху")
    ax2.grid()
    im = ax2.contourf(xx, yy, _beta, np.linspace(0, 2, 100), cmap="Greys")
    for path in paths:
        ax1.plot(path[0], path[1], path[2])
        ax2.plot(path[0], path[1], label=path[3])
    ax2.legend()
    plt.colorbar(im, label=r'$z=\beta(x,y)$')
    plt.show()


# Сравниваем методы A* и FMM
def compare_a_star_fmm(N):
    x_path_4, y_path_4, beta_path_4, time_4, cost_4 = a_star_find_path(add4, N)
    x_path_8, y_path_8, beta_path_8, time_8, cost_8 = a_star_find_path(add8, N)
    x_path_fmm, y_path_fmm, beta_path_fmm, time_fmm, cost_fmm = fmm_find_path(N)
    data = np.array([[cost_4, time_4],
                     [cost_8, time_8],
                     [cost_fmm, time_fmm]])
    np.savetxt(f"comparision_A_star_FMM.csv", data, delimiter=";", fmt="%.3f")

    paths = [(x_path_4, y_path_4, beta_path_4, "A* 4 соседа"),
             (x_path_8, y_path_8, beta_path_8, "A* 8 соседей"),
             (x_path_fmm, y_path_fmm, beta_path_fmm, "FMM")]

    draw_paths(paths)


# Сравниваем все методы
def compare_all(N, N_opt, TRRT_repeats):
    x_path_8, y_path_8, beta_path_8, time_8, cost_8 = a_star_find_path(add8, N)
    x_path_16, y_path_16, beta_path_16, time_16, cost_16 = a_star_find_path(add16, N)
    x_path_fmm, y_path_fmm, beta_path_fmm, time_fmm, cost_fmm = fmm_find_path(N)
    x_path_opt, y_path_opt, beta_path_opt, time_opt, cost_opt = optimize_find_path_time_cost(N_opt)
    x_path_trrt, y_path_trrt, beta_path_trrt, avg_time_trrt, best_cost_trrt, avg_time_trrt = trrt_find_best_path(TRRT_repeats)
    data = np.array([[cost_8, time_8],
                     [cost_fmm, time_fmm],
                     [cost_16, time_16],
                     [cost_opt, time_opt],
                     [best_cost_trrt, avg_time_trrt]])
    np.savetxt(f"comparision_All.csv", data, delimiter=";", fmt="%.3f")

    paths = [(x_path_8, y_path_8, beta_path_8, "A* 8 соседей"),
             (x_path_16, y_path_16, beta_path_16, "A* 16 соседей"),
             (x_path_fmm, y_path_fmm, beta_path_fmm, "FMM"),
             (x_path_opt, y_path_opt, beta_path_opt, "Оптимиз. метод"),
             (x_path_trrt, y_path_trrt, beta_path_trrt, "T-RRT")]

    draw_paths(paths)


if __name__ == "__main__":
    compare_a_star_fmm(50)
    compare_all(50, 23, 10)
