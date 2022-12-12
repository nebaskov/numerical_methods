import re
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def function(x):
    return np.divide(np.sin(x), x) 


def linear(x, y, n):
    new_y = y[0]
    new_x = x[0]
    for i in range(len(x)-1):
        a1 = (y[i + 1] - y[i])/(x[i+1] - x[i])
        a0 = y[i] - a1 * x[i]
        x_len50 = np.linspace(x[i], x[i+1], num = n)[1:]
        new_x = np.append(new_x, x_len50)
        f_xi = a0 + a1*x_len50
        new_y = np.append(new_y, f_xi)
        
    return new_x, new_y


def square(x, y, n):
    new_y = y[0]
    new_x = x[0]
    for i in range(len(x)-2):
        a2 = (y[i+2] - y[i]) / ((x[i+2] - x[i])*(x[i+2] - x[i+1])) -\
             (y[i+1] - y[i]) / ((x[i+1] - x[i])*(x[i+2] - x[i+1]))
        a1 = (y[i+1] - y[i]) / (x[i+1] - x[i]) - a2 * (x[i+1] + x[i])
        a0 = y[i] - a1 * x[i] - a2 * np.square(x[i])
        local_x = np.linspace(x[i], x[i+1], num = n)[1:]
        new_x = np.append(new_x, local_x)
        f_xi = a0 + a1 * local_x + a2 * np.square(local_x)
        new_y = np.append(new_y, f_xi)
    local_x = np.linspace(x[x.shape[0]-2], x[x.shape[0]-1], num = n)[1:]
    new_x = np.append(new_x, local_x)
    f_xi = a0 + a1 * local_x + a2 * np.square(local_x)
    new_y = np.append(new_y, f_xi)

    return new_x, new_y


def msa(y, y_true):
    return np.sqrt(np.sum(np.square(y_true - y)) / y_true.shape[0])


left = - np.pi
right = np.pi

x0 = np.linspace(left, right, 10)
x1 = np.linspace(left, right, 50)
inter_x = np.linspace(left, right, 100)

y0 = function(x0)
y1 = function(x1)

ln_x0, lin_inter_y0 = linear(x0, y0, 100)
ln_x1, lin_inter_y1 = linear(x1, y1, 100)
sq_x0 ,sq_inter_y0 = square(x0, y0, 100)
sq_x1, sq_inter_y1 = square(x1, y1, 100)

lin_err0 = msa(lin_inter_y0, function(ln_x0))
lin_err1 = msa(lin_inter_y1, function(ln_x1))
sq_err0 = msa(sq_inter_y0, function(sq_x0))
sq_err1 = msa(sq_inter_y1, function(sq_x1))

print('Ошибка кусочно-линейной интерполяции 10 точек: ', lin_err0)
print('Ошибка кусочно-линейной интерполяции 50 точек: ', lin_err1)
print('Ошибка кусочно-квадратичной интерполяции 10 точек: ', sq_err0)
print('Ошибка кусочно-квадратичной интерполяции 50 точек: ', sq_err1)

lin_built_in_interp_1 = sp.interpolate.interp1d(x0, function(x0), kind = 'linear')(inter_x)
cub_built_in_interp_1 = sp.interpolate.interp1d(x0, function(x0), kind = 'cubic')(inter_x)
spl_built_in_interp_1 = sp.interpolate.UnivariateSpline(x0, function(x0))(inter_x)

lin_built_in_interp_2 = sp.interpolate.interp1d(x1, function(x1), kind = 'linear')(inter_x)
cub_built_in_interp_2 = sp.interpolate.interp1d(x1, function(x1), kind = 'cubic')(inter_x)
spl_built_in_interp_2 = sp.interpolate.UnivariateSpline(x1, function(x1))(inter_x)

lin_err0 = msa(lin_built_in_interp_1, function(inter_x))
cub_err0 = msa(cub_built_in_interp_1, function(inter_x))
spl_err0 = msa(spl_built_in_interp_1, function(inter_x))
lin_err1 = msa(lin_built_in_interp_2, function(inter_x))
cub_err1 = msa(cub_built_in_interp_2, function(inter_x))
spl_err1 = msa(spl_built_in_interp_2, function(inter_x))

print('Ошибка линейной интерполяции scipy 10 точек: ', lin_err0)
print('Ошибка линейной интерполяции scipy 50 точек: ', lin_err1)
print('Ошибка cubic интерполяции scipy 10 точек: ', cub_err0)
print('Ошибка cubic интерполяции scipy 50 точек: ', cub_err1)
print('Ошибка spline интерполяции scipy 10 точек: ', spl_err0)
print('Ошибка spline интерполяции scipy 50 точек: ', spl_err1)


fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(x0, y0,'o', label='исходные точки')
ax[0, 0].plot(x0, y0, label='функция')
ax[0, 0].plot(ln_x0, lin_inter_y0, label='кусочно-линейная интерполяция')
ax[0, 0].plot(sq_x0, sq_inter_y0, label='кусочно-квадратичная интерполяция')
ax[0, 0].title.set_text('Интерполяция')
ax[0, 0].legend(loc='upper right')

ax[0, 1].plot(x0, y0,'o', label='исходные точки')
ax[0, 1].plot(x0, y0, label='функция')
ax[0, 1].plot(inter_x, lin_built_in_interp_1, label='linear')
ax[0, 1].plot(inter_x, cub_built_in_interp_1, label='cubic')
ax[0, 1].plot(inter_x, spl_built_in_interp_1, label='spline')
ax[0, 1].title.set_text('Интерполяция Scipy')
ax[0, 1].legend(loc='upper right')


ax[1, 0].plot(x1, y1,'o', label='исходные точки')
ax[1, 0].plot(x1, y1, label='функция')
ax[1, 0].plot(ln_x1, lin_inter_y1, label='кусочно-линейная интерполяция')
ax[1, 0].plot(sq_x1, sq_inter_y1, label='кусочно-квадратичная интерполяция')
ax[1, 0].title.set_text('Интерполяция')
ax[1, 0].legend(loc='upper right')

ax[1, 1].plot(x1, y1,'o', label='исходные точки')
ax[1, 1].plot(x1, y1, label='функция')
ax[1, 1].plot(inter_x, lin_built_in_interp_2, label='linear')
ax[1, 1].plot(inter_x, cub_built_in_interp_2, label='cubic')
ax[1, 1].plot(inter_x, spl_built_in_interp_2, label='spline')
ax[1, 1].title.set_text('Интерполяция Scipy')
ax[1, 1].legend(loc='upper right')

fig.suptitle('Графики интерполяции для 10 (верхний ряд) и для 50 (нижний ряд) точек.')
plt.show()

# читаем файл с выборкой, переход на размерность 1024
# num = re.compile(r'(?:([-]?\d+(?:\.\d+)+(?:e[+-]?\d+))|([-]?\d+(?:\.\d+)))')
path = 'C:\\Users\\nshir\\code\\shit\\chmf\\lab4\\data.txt'
dx, dy = np.array([]), np.array([])
with open(path, 'r') as file:
    for line in file:
        # groups = num.findall(line)
        content = [float(element.strip()) for element in line.split()]
        dx = np.append(dx, content[0])
        dy = np.append(dy, content[1])

d_linx,d_lin_int = linear(dx, dy, 1024)
d_sqx, d_sq_int = square(dx, dy, 1024)

dfig = plt.figure()
plt.plot(dx, dy, 'bo', label='исходные точки')
plt.plot(d_linx, d_lin_int, label='кусочно-линейная интерполяция')
plt.plot(d_sqx, d_sq_int, label='кусочно-квадратичная интерполяция')
plt.title('График интерполяции для выборки')
plt.legend(loc='upper right')
plt.grid()
plt.show()