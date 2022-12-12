import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# исходные данные
data = pd.DataFrame(index=['t', 'r', 'C', 'D', 'e', 'F', 'g', 'i'], columns=['n', 'lambda', 'q'])
data['n'] = [1.599483, 1.608862, 1.611600, 1.616403, 1.620541, 1.628440, 1.638265, 1.662241]
data['lambda'] = [1.01398, 0.7065188, 0.6562725, 0.5892938, 0.546074, 0.4861327, 0.4358343, 0.3650146]
data.loc[['i', 't', 'r'], 'q'] = 1
data.loc[['g', 'F', 'e', 'D', 'C'], 'q'] = 10
data['q'] = data['q'].astype('int')

# диагональная матрица Q
Q = np.diag(data['q'].to_numpy())
lambda02 = 0.028

# переменная L
L = np.power(np.power(data['lambda'].to_numpy(), 2) - lambda02, -1)

# вычисление матрицы Lambda
Lamb = np.transpose(np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                    np.power(data['lambda'].to_numpy(), 2),
                     np.power(data['lambda'].to_numpy(), 4),
                     L,
                     np.power(L, 2),
                     np.power(L, 3)], dtype=float))

# print(Lamb) 

# вычисление коэффициентов
M = np.matmul(
                    np.matmul(
                        np.matmul(
                            np.linalg.inv(
                                np.matmul(
                                    np.matmul(
                                        np.transpose(Lamb), np.power(Q, 2)),
                                         Lamb)),
                                          np.transpose(Lamb)),
                                           np.power(Q, 2)),
                                            data['n'].to_numpy())


# print(M)
# точки по заданию
test_arr = np.array([0.5875618, 0.4046561, 0.64380])

# массив для построения графика
full_test_arr = np.arange(0.37, 1.2, step=0.01)

test_L = np.power(np.power(test_arr, 2) - lambda02, -1)

full_test_L = np.power(np.power(full_test_arr, 2) - lambda02, -1)

test_Lamb = np.transpose(np.array([[1, 1, 1],
                    np.power(test_arr, 2),
                     np.power(test_arr, 4),
                     test_L,
                     np.power(test_L, 2),
                     np.power(test_L, 3)], dtype=float))

full_test_Lamb = np.transpose(np.array([[1] * len(full_test_arr),
                    np.power(full_test_arr, 2),
                     np.power(full_test_arr, 4),
                     full_test_L,
                     np.power(full_test_L, 2),
                     np.power(full_test_L, 3)], dtype=float))

test_result = np.matmul(test_Lamb, M)

full_test_result = np.matmul(full_test_Lamb, M)

# стекло F2 glassBank
compar_arr = np.array([1.6165511, 1.6468077, 1.6123800])

# погрешность
confusion_arr = abs(compar_arr - test_result)
print('Ошибка для lambda_d, lambda_h, lambda=0.64380 составляет: \n', confusion_arr)
print('n вычисленные: \n', test_result)

# вспомогательный блок чтобы нанести точки на график
test_series = pd.DataFrame(index=['d', 'h', 'test'], columns=data.columns)
test_series.loc['d', 'lambda'] = 0.5875618
test_series.loc['h', 'lambda'] = 0.4046561
test_series.loc['test', 'lambda'] = 0.64380

test_series.loc['d', 'n'] = 1.6165511
test_series.loc['h', 'n'] = 1.6468077
test_series.loc['test', 'n'] = 1.6123800

test_dataset = pd.concat([data, test_series])
test_dataset.sort_values(by='lambda')


# график
plt.plot(full_test_arr, full_test_result, label='аппроксимация')
plt.plot(test_dataset['lambda'], test_dataset['n'], 'go', label='тестовые точки')
plt.plot(data['lambda'], data['n'], 'ro', label='исходные точки')
plt.title(r'График $n(\lambda)$')
plt.legend(loc='upper right')
plt.xlabel('Длина волны, мкм')
plt.ylabel('n')
plt.grid()
plt.show()
