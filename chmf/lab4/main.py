import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def function(x):
    return np.divide(np.sin(x), x)


def lin_inter(x, y, n):
    points = y
    approx_func = np.array([])

    for i in range(n):
        if i < n-1:
            c1 = (points[i+1] - points[i]) / (x[i+1] - x[i-1])
            c0 = points[i] - c1 * x[i]
            res = c0 + c1 * x[i]
            approx_func = np.append(approx_func, res)

        else:
            approx_func = np.append(approx_func, points[-1])

    return approx_func


def squar_inter(x, y, n):
    points = y
    approx_func = np.array([])

    for j in range(n):
        left = j 
        i = j + 1
        right = j + 2
        if right < n:
            c2 = (points[right] - points[left]) / ((x[right] - x[left]) * (x[right] - x[i])) \
                - (points[i] - points[left]) / ((x[i] - x[left]) * (x[right] - x[i]))

            c1 = (points[i] - points[left]) / (x[i] - x[left]) - c2 * (x[i] + x[left])

            c0 = points[left] - c1 * x[left] - c2 * np.square(x[left])

            res = c0 + c1 * x[left] + c2 * np.square(x[left])
            approx_func = np.append(approx_func, res)
        else:
            approx_func = np.append(approx_func, points[-2])
            approx_func = np.append(approx_func, points[-1])
            
    return np.unique(approx_func, axis=0)


a = - np.pi
b = np.pi
n1 = 10
n2 = 50

x1 = np.linspace(a, b, n1)
x2 = np.linspace(a, b, n2)

func_1 = function(x1)

lin_inter_1 = lin_inter(x1, function(x1), n1)
lin_inter_2 = lin_inter(x2, function(x2), n2)

sq_inter_1 = squar_inter(x1, function(x1), n1)
sq_inter_2 = squar_inter(x2, function(x2), n2)

lin_built_in_interp_1 = sp.interpolate.interp1d(x1, function(x1), kind = 'linear')(x1)
cub_built_in_interp_1 = sp.interpolate.interp1d(x1, function(x1), kind = 'cubic')(x1)
spl_built_in_interp_1 = sp.interpolate.UnivariateSpline(x1, function(x1))(x1)

func_2 = function(x2)

lin_built_in_interp_2 = sp.interpolate.interp1d(x1, function(x1), kind = 'linear')(x2)
cub_built_in_interp_2 = sp.interpolate.interp1d(x1, function(x1), kind = 'cubic')(x2)
spl_built_in_interp_2 = sp.interpolate.UnivariateSpline(x2, function(x2))(x2)

fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(x1, func_1,'o', label='исходные точки')
ax[0, 0].plot(x1, func_1, label='функция')
ax[0, 0].plot(x1, lin_inter_1, label='кусочно-линейная интерполяция')
ax[0, 0].plot(x1, sq_inter_1, label='кусочно-квадратичная интерполяция')
ax[0, 0].title.set_text('Интерполяция')
ax[0, 0].legend(loc='upper right')

ax[0, 1].plot(x1, func_1,'o', label='исходные точки')
ax[0, 1].plot(x1, func_1, label='функция')
ax[0, 1].plot(x1, lin_built_in_interp_1, label='linear')
ax[0, 1].plot(x1, cub_built_in_interp_1, label='cubic')
ax[0, 1].plot(x1, spl_built_in_interp_1, label='spline')
ax[0, 1].title.set_text('Интерполяция Scipy')
ax[0, 1].legend(loc='upper right')


ax[1, 0].plot(x2, func_2,'o', label='исходные точки')
ax[1, 0].plot(x2, func_2, label='функция')
ax[1, 0].plot(x2, lin_inter_2, label='кусочно-линейная интерполяция')
ax[1, 0].plot(x2[np.where(sq_inter_2 == sq_inter_2)], sq_inter_2, label='кусочно-квадратичная интерполяция')
ax[1, 0].title.set_text('Интерполяция')
ax[1, 0].legend(loc='upper right')

ax[1, 1].plot(x2, func_2,'o', label='исходные точки')
ax[1, 1].plot(x2, func_2, label='функция')
ax[1, 1].plot(x2, lin_built_in_interp_2, label='linear')
ax[1, 1].plot(x2, cub_built_in_interp_2, label='cubic')
ax[1, 1].plot(x2, spl_built_in_interp_2, label='spline')
ax[1, 1].title.set_text('Интерполяция Scipy')
ax[1, 1].legend(loc='upper right')

fig.suptitle('Графики интерполяции для 10 (левый ряд) и для 50 (правый ряд) точек.')
plt.show()