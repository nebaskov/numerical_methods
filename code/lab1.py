from sympy import *   # модуль, позволяющий преобразовывать уравнения в буквенной форме
import numpy as np # модуль для работы с векторами
from matplotlib import pyplot as plt # модуль, позволяющий строить графики


# запишем исходное уравнение W(p_x, p_y)
px, py = symbols("px py")
w = (10 * py * px ** 4) + (20 * (px ** 2) * (py ** 3)) + 10 * (py ** 5) \
        - 12 * (px ** 2) * py - 12 * (py ** 3) + 3 * py


# вычисление значений функции аберрации
def calc_w(px, py):
    result = (10 * py * px ** 4) + (20 * (px ** 2) * (py ** 3)) + 10 * (py ** 5) \
        - 12 * (px ** 2) * py - 12 * (py ** 3) + 3 * py
    return result



def calc_wx(x, wx):
    result = wx.subs({px: x, py: 0})
    return result


def calc_wy(y, wy):
    result = wy.subs({px: 0, py: 1})
    return result


calc_dwx = lambdify(px, w)
calc_dwy = lambdify(py, w)

dw_dx =  w.diff(px)
dw_dy = w.diff(py)
# print(dw_dx)
# print(dw_dy)

px_space = list(range(0, 11))
py_space = list(range(-1, 2, 10))

calc_dwx_list = []
calc_dwy_list = []


# for px in px_space:
#     calc = calc_wx(px, dw_dx)
#     calc_dwx_list.append(calc)

# print(calc_dwx_list)

# for py in py_space:
#     calc = calc_wy(y=py, wy=dw_dy)
#     calc_dwy_list.append(calc)

dwdx_value = dw_dy.subs({px: px_space, py: 0})
dwdy_value = dw_dy.subs({px: 0, py: py_space})
for i in py_space:
    dw_dy_value = dwdy_value.subs(py, 0)
    calc_dwy_list.append(dw_dy_value)

print(calc_dwy_list)

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1 = plt.plot(px_space, calc_dwx_list, "b", label="производная для p_x")
# ax1.legend(loc="upper right")
ax2 = plt.plot(py_space, calc_dwy_list, "r", label="производная для p_y")
# ax2.legend(loc="upper right")
plt.suptitle("Графики для производных")
plt.show()