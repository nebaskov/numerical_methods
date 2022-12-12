import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# блок функций
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
    func = sp.lambdify(args=x, expr=expression)
    result = func(value)
    return result


def NewtonCotes(express, a, b, count_arr, order_arr):
    """Вычисление интеграла по методу Ньютона-Котеса
    -------------------------------------
    Parameters:
    
    expression: заданная функция 

    a: левая граница заданного интервала 

    b: правая граница заданного интервала 

    count: число подинтервалов 
    
    order: количество точек в подинтервале (порядок метода) 
    
    weights: coefficient matrix
    
    Returns: 

    Численное значение интеграла
    """
    result_arr = np.zeros((len(order_arr), len(count_arr)))
    for p in range(len(order_arr)):
        for k in range(len(count_arr)):
            count = count_arr[k]
            order = order_arr[p]
            int_width = (b-a) / count
            h = int_width / order
            x_j = a
            res = 0
            C_n = 0

            for j in range(count):
                x_j += j * int_width
                sum = 0
                for i in range(order):
                    x_i = x_j + i * h
                    sum += newton_weight[p][i] * calculate_function(express, x_i)

            for i in range(order+1):
                C_n += newton_weight[p][i]
            
            result = res * order * h / C_n
            result_arr[p][k] = result
    return result_arr



def Gauss(express, a, b, count_arr, order_arr):
    """Вычисление интеграла по методу Гаусса
        -------------------------------------
        :param \n
        expression: заданная функция \n
        a: левая граница заданного интервала \n
        b: правая граница заданного интервала \n
        count: число подинтервалов \n
        order: количество точек в подинтервале (порядок метода) \n
        weights: coefficient matrix \n
        starting_x_i: initial x_i matrix for Gauss method \n
        :return \n
        Численное значение интеграла
        """

    result_arr = np.zeros((len(order_arr), len(count_arr)))
    for p in range(len(order_arr)):
        for k in range(len(count_arr)):
            count = count_arr[k]
            order = order_arr[p]
            int_width = (b-a) / count
            h = int_width / order

            x_j = a
            x_i = np.zeros(shape=(count, order))
            res = 0

            for j in range(count):
                std = x_i[j].std()
                mean = x_i[j].mean()
                x_i[j] = x_i[j] - mean
                x_i[j] = np.divide(x_i[j], std)

            for j in range(count):
                x_j += j * int_width
                for i in range(order):
                    x_i[j][i] = x_j + i * h

            for j in range(count):
                r = 0
                for i in range(order):
                    r += gauss_weight[order][i] * calculate_function(express, x_i[j][i])
                    res += r
            
            result = (b-a) / (2 * count) * res

            result_arr[p][k] = result

    return result_arr

    
newton_weight = np.array([[1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 4, 1, 0, 0, 0],
                [1, 3, 3, 1, 0, 0],
                [7, 32, 12 , 32, 7, 0],
                [19,75, 50, 50, 75, 19]])

Cn = np.array([1, 2, 6, 8, 90, 288])

gauss_weight = np.array([
    [2, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0.5555556, 0.8888889, 0.5555556, 0, 0, 0],
    [0.3478548, 0.6521451, 0.6521451, 0.3478548, 0, 0],
    [0.4786287, 0.2369269, 0.5688888, 0.2369269, 0.4786287, 0],
    [0.1713245, 0.3607616, 0.4679140, 0.4679140, 0.3607616, 0.1713245]
])

init_xi = np.array([
    [0, 0, 0, 0, 0, 0],
    [-0.5773503, 0.5773503, 0, 0, 0, 0],
    [-0.7745967, 0, 0.7745967, 0, 0, 0],
    [-0.8611363, -0.3399810, 0.3399810, 0.8611363, 0, 0],
    [-0.9061798, -0.5384693, 0, 0.5384693, 0.9061798, 0],
    [-0.9324700, -0.6612094, -0.2386142, 0.2386142, 0.6612094, 0.9324700]
])

x = sp.Symbol("x")
function = sp.sin(x) ** 2
right_border = np.pi / 3
left_border = - np.pi / 2

n_order_array = np.array([1, 2, 3, 4, 5], dtype=np.int8)
g_order_array = np.array([1, 2, 3, 4, 5, 6], dtype=np.int8)
count_array = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int32)

analytical_result = calculate_function(sp.integrate(function, x), right_border) \
                    - calculate_function(sp.integrate(function, x), left_border)

newton_result = NewtonCotes(function, left_border, right_border, count_array, n_order_array)
gauss_result = Gauss(function, left_border, right_border, count_array, g_order_array)

