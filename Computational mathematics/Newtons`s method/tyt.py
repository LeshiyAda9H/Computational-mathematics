import numpy as np
from numpy import linalg
import math
import time
from lu import *

# каждый вызов func = 138 действий
# каждый вызов derivative = 201 действие
# каждое матричное прозведение dev на func = 100 действий
# каждое LU = 330 действий



def func(x):
    rows, _ = x.shape
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = [x[i, 0] for i in range(rows)]
    return np.mat([
        math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440,
        math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994,
        x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904,
        2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757,
        math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862,
        math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115,
        x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014,
        x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(x2 * x7 + x10) + 3.5668321989693809040,
        7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499,
        x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]).transpose()
def derivative(x):
    rows, _ = x.shape
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = [x[i, 0] for i in range(rows)]
    return np.mat([[-x2 * math.sin(x2 * x1), -x1 * math.sin(x2 * x1), 3 * math.exp(-3 * x3), x5 ** 2, 2 * x4 * x5,
                       -1, 0, -2 * math.cosh(2 * x8) * x9, -math.sinh(2 * x8), 2],
                      [x2 * math.cos(x2 * x1), x1 * math.cos(x2 * x1), x9 * x7, 0, 6 * x5,
                       -math.exp(-x10 + x6) - x8 - 1, x3 * x9, -x6, x3 * x7, math.exp(-x10 + x6)],
                      [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                      [-x5 / (x3 + x1) ** 2, -2 * x2 * math.cos(x2 ** 2), -x5 / (x3 + x1) ** 2, -2 * math.sin(-x9 + x4),
                       1.0 / (x3 + x1), 0, -2 * math.cos(x7 * x10) * x10 * math.sin(x7 * x10), -1,
                       2 * math.sin(-x9 + x4), -2 * math.cos(x7 * x10) * x7 * math.sin(x7 * x10)],
                      [2 * x8, -2 * math.sin(x2), 2 * x8, 1.0 / (-x9 + x4) ** 2, math.cos(x5),
                       x7 * math.exp(-x7 * (-x10 + x6)), -(x10 - x6) * math.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1,
                       -1.0 / (-x9 + x4) ** 2, -x7 * math.exp(-x7 * (-x10 + x6))],
                      [math.exp(x1 - x4 - x9), -1.5 * x10 * math.sin(3 * x10 * x2), -x6, -math.exp(x1 - x4 - x9),
                       2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -math.exp(x1 - x4 - x9), -1.5 * x2 * math.sin(3 * x10 * x2)],
                      [math.cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * math.sin(x4), x10 / x5 ** 2 * math.cos(x10 / x5 + x8),
                       -math.cos(x4), x2 ** 3, -math.cos(x10 / x5 + x8), 0, -1.0 / x5 * math.cos(x10 / x5 + x8)],
                      [2 * x5 * (x1 - 2 * x6), -x7 * math.exp(x2 * x7 + x10), -2 * math.cos(-x9 + x3), 1.5,
                       (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * math.exp(x2 * x7 + x10), 0, 2 * math.cos(-x9 + x3),
                       -math.exp(x2 * x7 + x10)],
                      [-3, -2 * x8 * x10 * x7, 0, math.exp(x5 + x4), math.exp(x5 + x4),
                       -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
                      [x10, x9, -x8, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7,
                       math.cos(x4 + x5 + x6) * x7, math.sin(x4 + x5 + x6), -x3, x2, x1]])

def newton():
    start_time = time.time()
    x_0 = np.mat([[0.5], [0.5], [1.5], [-1], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
    сounter_iteration  = 0
    сounter_operation = 0
    epsilon = 1e-9

    while True:
        сounter_iteration  += 1

        # вызов derivative + LUpq
        L, U, m_row, m_col = LUpq(derivative(x_0))

        # вызов func + решение слау
        x, operations = pluqb(m_row, L, U, m_col, -func(x_0))

        сounter_operation += operations + 201 + 138 + 330

        x += x_0
        error = x - x_0

        if np.linalg.norm(error) < epsilon:
            ans = time.time() - start_time
            print()
            print("─" * 73)
            print("Значение корня:")
            print(x.T)
            print("─" * 73)
            print("Значение функции:")
            print(func(x).T)
            print("─" * 73)
            print("☞Количество итераций:", сounter_iteration)
            print("☞Количество операций:", сounter_operation)
            print("☞Затраченное время:", ans)
            print("─" * 43)
            print()
            break

        x_0 = x

def modified_newtone():
    x_0 = np.mat([[0.5], [0.5], [1.5], [-1], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
    eps = 1e-9
    start_time = time.time()
    сounter_iteration = 0
    сounter_operation = 0

    # Посчитаем один раз матрицу А как функцию производных
    L, U, m_row, m_col = LUpq(derivative(x_0))
    сounter_operation += 330 + 201

    while True:
        сounter_iteration += 1
        x_1, operations = pluqb(m_row, L, U, m_col, -func(x_0))
        x_1 += x_0

        error = x_1 - x_0

        сounter_operation += operations + 138

        if np.linalg.norm(error) < eps:
            ans = time.time() - start_time
            print()
            print("─" * 73)
            print("Значение корня:")
            print(x_1.T)
            print("─" * 73)
            print("Значение функции:")
            print(func(x_1).T)
            print("─" * 73)
            print("☞Количество итераций:", сounter_iteration)
            print("☞Количество операций:", сounter_operation)
            print("☞Затраченное время:", ans)
            print("─" * 43)
            print()
            break
        x_0 = x_1

def k_newton():
    x0 = np.mat([[0.5], [0.5], [1.5], [-1], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
    k = 5
    l, u, p, q = LUpq(derivative(x0))

    count_operation = 330 + 201 #dev и LU
    count_iterations = 0

    eps = 1e-9

    start_time = time.time()

    while True:
        count_iterations += 1

        x, operations = pluqb(p, l, u, q, -func(x0))
        x += x0

        count_operation += operations + 138

        if np.linalg.norm(x - x0) < eps:
            ans = time.time() - start_time
            print()
            print("─" * 73)
            print("Значение корня:")
            print(x.T)
            print("─" * 73)
            print("Значение функции:")
            print(func(x).T)
            print("─" * 73)
            print("☞Количество итераций:", count_iterations)
            print("☞Количество операций:", count_operation)
            print("☞Затраченное время:", ans)
            print("─" * 43)
            print()
            break

        x0 = x

        if count_iterations < k:
            l, u, p, q = LUpq(derivative(x0))
            count_operation += 330 + 201 #вызов dev и LU


def m_newton():
    x0 = np.mat([[0.5], [0.5], [1.5], [-1], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
    m = 5
    l, u, p, q = LUpq(derivative(x0))


    count_operation = 330 + 201  #LU и dev
    count_iterations = 0

    eps = 1e-9

    start_time = time.time()

    while True:
        count_iterations += 1
        x, operations = pluqb(p, l, u, q, -func(x0))
        x += x0
        count_operation += operations + 138 #решение слау и вызов func

        if np.linalg.norm(x - x0) < eps:
            ans = time.time() - start_time
            print()
            print("─" * 73)
            print("Значение корня:")
            print(x.T)
            print("─" * 73)
            print("Значение функции:")
            print(func(x).T)
            print("─" * 73)
            print("☞Количество итераций:", count_iterations)
            print("☞Количество операций:", count_operation)
            print("☞Затраченное время:", ans)
            print("─" * 43)
            print()
            break
        x0 = x

        if count_iterations % m == 0:
            l, u, p, q = LUpq(derivative(x0))
            count_operation += 330 + 201 #вызов dev и LU


def k_m_newton():
    x0 = np.mat([[0.5], [0.5], [1.5], [-1], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
    m = 5
    k = 5

    l, u, p, q = LUpq(derivative(x0))

    count_operation = 330 + 201 #вызов dev и LU
    count_iterations = 0
    eps = 1e-9

    start_time = time.time()

    while True:
        count_iterations += 1
        x, operations = pluqb(p, l, u, q, -func(x0))
        x += x0
        count_operation += operations + 138

        if np.linalg.norm(x - x0) < eps:
            ans = time.time() - start_time
            print()
            print("─" * 73)
            print("Значение корня:")
            print(x.T)
            print("─" * 73)
            print("Значение функции:")
            print(func(x).T)
            print("─" * 73)
            print("☞Количество итераций:", count_iterations)
            print("☞Количество операций:", count_operation)
            print("☞Затраченное время:", ans)
            print("─" * 43)
            print()
            break
        x0 = x

        if count_iterations % m == 0 or count_iterations < k:
            l, u, p, q = LUpq(derivative(x0))
            count_operation += 330 + 201 #вызов dev и LU
