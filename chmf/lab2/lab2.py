import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
from integral import NewtonCotes, Gauss, calculate_function


# x = sp.Symbol("x")
# function = sp.sin(x) ** 2
# right_border = math.pi / 3
# left_border = - math.pi / 2

# newton_order = np.arange(1, 6, 1, dtype=np.int8)
# gauss_order = np.arange(1, 7, 1, dtype=np.int8)

# N = np.arange(start=1e3, stop=5e3, step=1e3, dtype=np.int32)

# analytical_result = calculate_function(sp.integrate(function, x), right_border) \
#                     - calculate_function(sp.integrate(function, x), left_border)

# # newton_result = NewtonCotes(function, left_border, right_border, 2000, 3)
# # gauss_result = Gauss(function, left_border, right_border, 2000, 3)

# count_arr = np.arange(100, 10000, 100, dtype=int)
# newton_order_arr = np.arange(1, 6, dtype=int)
# gauss_order_arr = np.arange(1, 6, dtype=int)

# newton_res_arr = np.array([[NewtonCotes(function, left_border, right_border, cnt, ordr) for cnt in count_arr] for ordr in newton_order_arr])
# gauss_res_arr = np.array([[Gauss(function, left_border, right_border, cnt, ordr) for cnt in count_arr] for ordr in gauss_order_arr])
# newton_error = np.abs(newton_res_arr - analytical_result)
# gauss_error = np.abs(gauss_res_arr - analytical_result)
# print(newton_res_arr.shape)

# newton_plot = plt.figure()
# for order in newton_order_arr:
#     plt.plot(count_arr, newton_error[order-1], label=f'Для {order} порядка')
# plt.xlabel('Количество подинтервалов (count)')
# plt.ylabel('Ошибка')
# plt.legend(loc="upper right")
# plt.title('Зависимости точности численного интегрирования от шага для метода Ньютона-Котеса')
# plt.show()

# gauss_plot = plt.figure()
# for order in gauss_order_arr:
#     plt.plot(count_arr, gauss_error[order-1], marker='o', linestyle='', label=f'Для {order} порядка')
# plt.xlabel('Количество подинтервалов (counts)')
# plt.ylabel('Ошибка')
# plt.legend(loc="upper right")
# plt.title('Зависимости точности численного интегрирования от шага для метода Гаусса')
# plt.show()

with open(os.path.join(os.getcwd(), 'psf_c40_03.txt'), 'r') as file:
    lines = []
    for line in file:
        values = [value.strip() for value in  line.split()]
        lines.append(values)
    array = np.array(lines, dtype='float64')
    x = np.arange(start=-6.375, stop=(6.375 + 0.05), step=0.05, dtype='float64')
    y = np.arange(start=-6.375, stop=(6.375 + 0.05), step=0.05, dtype='float64')
    r = np.arange(start=0, stop=(6.375 + 0.05), step=0.05, dtype='float64')
    
    N = len(x)

    frt = plt.figure()
    plt.plot(x, array[:][N // 2])
    plt.title('Сечение ФРТ')
    
    full_eng = array.sum()
    ecf_array = np.zeros(shape=r.shape)
    radius_array = np.zeros(shape=r.shape)

    for n in range(len(r)):
        local_en = 0
        for i in range(len(x)):
            for j in range(len(y)):
                if np.square(x[i]) + np.square(y[j]) <= np.square(r[n]):
                    local_en += array[j][i]
                ecf = local_en / full_eng
            ecf_array[n] =  ecf
    
    ecf_fig = plt.figure()
    plt.plot(r, ecf_array)
    plt.title('ФКЭ')
    plt.show()
