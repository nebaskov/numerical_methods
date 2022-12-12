import numpy as np
import sympy as sp


def calculate_function(expression, value):
    """Вычисление значения заданной функции в конкретных точках.
    --------------------------------------------------------------
       Parameters:

       expression: функция, значение которой необходимо вычислить 
       
       value: значение переменной в функции 

       Returns:

       result: численное значение функции в заданной точке или интервале.
       """

    # lambdify - встроенная функция для вычисления значений символьных уравнений
    x = sp.Symbol("x")
    func = sp.lambdify(args=x, expr=expression)
    result = func(value)
    return result


def NewtonCotes(express, left_border:float, right_border:float, count: int, order: int):

    newton_weight = np.array([[1, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 4, 1, 0, 0, 0],
                   [1, 3, 3, 1, 0, 0],
                   [7, 32, 12 , 32, 7, 0],
                   [19,75, 50, 50, 75, 19]])

    sub_step = (right_border - left_border) / count
    h = sub_step / order
    xi_arrr = np.arange(left_border, right_border, h)
    xi_arr = np.reshape(xi_arrr, (count, order))
    iterable = [newton_weight[order][i] * calculate_function(express, xi_arr[j][i]) for j in range(count) for i in range(order)]
    res_arr = np.fromiter(iterable, dtype=float)
    cn = newton_weight[order].sum()
    result = sub_step / cn * res_arr.sum()
    return result


def Gauss(express, left_border, right_border, count, order):

    gauss_weight = np.array([
        [2, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0.5555556, 0.8888889, 0.5555556, 0, 0, 0],
        [0.3478548, 0.6521451, 0.6521451, 0.3478548, 0, 0],
        [0.4786287, 0.2369269, 0.5688888, 0.2369269, 0.4786287, 0],
        [0.1713245, 0.3607616, 0.4679140, 0.4679140, 0.3607616, 0.1713245]])

    sub_step = (right_border - left_border) / count
    h = sub_step / order
    xi_arrrr = np.arange(left_border, right_border, h)
    xi_arrr = np.reshape(xi_arrrr, (count, order))
    xi_arr = [np.divide(xi_arrr[j] - xi_arrr[j].mean(), xi_arrr[j].std()) for j in range(count)]
    res_arr = np.fromiter([gauss_weight[order][i] * calculate_function(express, xi_arr[j][i]) for j in range(count) for i in range(order)], dtype=float)
    result = (right_border - left_border) / (2 * count) * res_arr.sum()
    return result