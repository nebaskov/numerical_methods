import numpy as np
import matplotlib.pyplot as plt

# одномерное преобразование Фурье

steps = 512
step = np.sqrt(1 / steps)

x_max = step * steps / 2
x = np.arange(-x_max, x_max - step, step)

# заданная функция согласно варианта
f_cos = np.cos(np.pi * x)

f_rect = np.fft.fftshift(f_cos)
f_f_rect = np.fft.fft(f_rect)

f_rect = np.fft.fftshift(f_f_rect)
f_rect = f_rect / np.sqrt(steps)

plt.figure()
plt.plot(x, f_cos, label=r'$\cos(\pi \cdot х)$')
plt.plot(x, f_rect, label= 'о/п Фурье')
plt.legend(loc="upper right")
plt.title(r'График преобразований Фурье для фукнции $\cos(\pi \cdot x)$')
plt.grid()
plt.show()

rectangle = np.where(abs(x)<=0.5, 1, 0)
f_rectangle = np.fft.fftshift(rectangle)
f_f_rectangle = np.fft.fft(f_rectangle)
f_rectangle = np.fft.fftshift(f_f_rectangle)
f_rectangle = f_rectangle / np.sqrt(steps)

plt.figure()
plt.plot(x, rectangle, label= 'rect(х)')
plt.plot(x, f_rectangle, label= 'о/п Фурье')
plt.legend(loc="upper right")
plt.title('График преобразований Фурье для фукнции rect(x)')
plt.grid()
plt.show()

a_right_shift = np.where(x>=0.5, 1, 0)
b_right_shift = np.where(x<=1.5, 1, 0)

a_left_shift = np.where(x<=-0.5, 1, 0)
b_left_shift = np.where(x>=-1.5, 1, 0)

for i in range(len(a_right_shift)): a_right_shift[i] = 1 if (a_right_shift[i] == b_right_shift[i]) else 0

for i in range(len(a_right_shift)): a_left_shift[i] = 1 if (a_left_shift[i] == b_left_shift[i]) else 0

rectangle = a_right_shift
f_rectangle = np.fft.fftshift(rectangle)
f_f_rectangle = np.fft.fft(f_rectangle)
im_f_rectangle = np.fft.ifft(f_rectangle)
im_f_rectangle = np.fft.fftshift(im_f_rectangle)
im_f_rectangle = im_f_rectangle * np.sqrt(steps)
f_rectangle = np.fft.fftshift(f_f_rectangle)
f_rectangle = f_rectangle / np.sqrt(steps)
fig_n, ax = plt.subplots(1, 2)
ax[0].plot(x, rectangle)
ax[0].plot(x, f_rectangle, label= 'о/п Фурье')
ax[0].plot(x, im_f_rectangle, label= 'обратное о/п Фурье')
ax[0].title.set_text('График преобразований Фурье для фукнции rect(x-1)')
ax[0].legend(loc="upper right")
ax[0].grid()

rectangle = a_left_shift
f_rectangle = np.fft.fftshift(rectangle)
f_f_rectangle = np.fft.fft(f_rectangle)
im_f_rectangle = np.fft.ifft(f_rectangle)
im_f_rectangle = np.fft.fftshift(im_f_rectangle)
im_f_rectangle = im_f_rectangle * np.sqrt(steps)
f_rectangle = np.fft.fftshift(f_f_rectangle)
f_rectangle = f_rectangle / np.sqrt(steps)

ax[1].plot(x, rectangle)
ax[1].plot(x, f_rectangle, label= 'о/п Фурье')
ax[1].plot(x, im_f_rectangle, label= 'обратное о/п Фурье')
ax[1].title.set_text('График преобразований Фурье для фукнции rect(x+1)')
ax[1].legend(loc="upper right")
ax[1].grid()
plt.show()

rectangle = np.where(abs(x)<=0.25, 1, 0)
_rect_ = np.fft.fftshift(rectangle)

im_f_rectrangle = np.fft.ifft(_rect_)
im_f_rectrangle = np.fft.fftshift(im_f_rectrangle)
im_f_rectrangle = im_f_rectrangle*np.sqrt(steps)

f_rectangle = np.fft.fft(_rect_)
f_rectangle = np.fft.fftshift(f_rectangle)
f_rectangle = f_rectangle / np.sqrt(steps)

plt.figure()
plt.plot(x, rectangle, label= 'rect(2x)')
plt.plot(x, f_rectangle, label= 'о/п Фурье')
plt.legend(loc="upper right")
plt.title('График преобразований Фурье для фукнции rect(2x)')
plt.grid()
plt.show()

rectangle = np.where(abs(x)<=1, 1, 0)
_rect_ = np.fft.fftshift(rectangle)

im_f_rectrangle = np.fft.ifft(_rect_)
im_f_rectrangle = np.fft.fftshift(im_f_rectrangle)
im_f_rectrangle = im_f_rectrangle*np.sqrt(steps)

f_rectangle = np.fft.fft(_rect_)
f_rectangle = np.fft.fftshift(f_rectangle)
f_rectangle = f_rectangle / np.sqrt(steps)

plt.figure()
plt.plot(x, rectangle, label= 'rect(x/2)')
plt.plot(x, f_rectangle, label= 'о/п Фурье')
plt.legend(loc="upper right")
plt.title('График преобразований Фурье для фукнции rect(x/2)')
plt.grid()
plt.show()

rectangle = 1-np.where(abs(x)<=0.5, 1, 0)
f_rectangle = np.fft.fftshift(rectangle)
f_f_rectangle = np.fft.fft(f_rectangle)

f_rectangle = np.fft.fftshift(f_f_rectangle)
f_rectangle = f_rectangle / np.sqrt(steps)

plt.figure()
plt.plot(x, rectangle, label= '1-rect(х)')
plt.plot(x, f_rectangle, label= 'о/п Фурье')
plt.title('График преобразований Фурье для фукнции 1-rect(x)')
plt.legend(loc="upper right")
plt.grid()
plt.show()

# двумерное преобразование Фурье
function = np.zeros(shape=(steps, steps))

arr = np.arange(-x_max, x_max, step)
x, y = np.meshgrid(arr, arr)

for i in range(steps):
    x_one = x[1][i]
    for j in range(steps):
        y_one = y[j][1]

        function[i][j] = 1 if (np.square(x_one) + np.square(y_one) < 1) else 0

real_func = function.real
transformy = np.fft.fftshift(real_func)
stransformy =  np.fft.fft2(transformy)
transformy = np.fft.fftshift(stransformy)
transformy = transformy / steps
intensity = abs(transformy) * abs(transformy)

fig_nn, ax = plt.subplots(2, 2)
ax[0, 0].imshow(function, cmap='gray', interpolation='nearest')
ax[0, 0].title.set_text('Полутоновое отображение зрачка')

ax[0, 1].plot(function[:, 257])
ax[0, 1].title.set_text('Сечение')

ax[1, 0].imshow(intensity, cmap='gray', interpolation='nearest')
ax[1, 0].title.set_text('Полутоновое отображение ФРТ')

ax[1, 1].plot(intensity[:, 257])
ax[1, 1].title.set_text('Сечение ФРТ')
fig_nn.suptitle('Графики для двумерного преобразования Фурье функции circ(p)')
plt.show()

function = np.zeros(shape=(steps, steps))

arr = np.arange(-x_max, x_max, step)
x, y = np.meshgrid(arr, arr)

for i in range(steps):
    x_one = x[1][i]
    for j in range(steps):
        y_one = y[j][1]

        function[i][j] = 1 if (np.square(x_one) + np.square(y_one) < 1 and np.square(x_one) + np.square(y_one) > np.square(0.9)) else 0

real_func = function.real
transformy = np.fft.fftshift(real_func)
stransformy =  np.fft.fft2(transformy)
transformy = np.fft.fftshift(stransformy)

# нормировка, steps=512
transformy = transformy / steps
intensity = abs(transformy) * abs(transformy)



fig_nn, ax = plt.subplots(2, 2)
ax[0, 0].imshow(function, cmap='gray', interpolation='nearest')
ax[0, 0].title.set_text('Полутоновое отображение зрачка')

ax[0, 1].plot(function[:, 257])
ax[0, 1].title.set_text('Сечение')

ax[1, 0].imshow(intensity, cmap='gray', interpolation='nearest')
ax[1, 0].title.set_text('Полутоновое отображение ФРТ')

ax[1, 1].plot(intensity[:, 257])
ax[1, 1].title.set_text('Сечение ФРТ')
fig_nn.suptitle('Графики для двумерного преобразования Фурье функции circ(p) - circ(p/e)')
plt.show()  