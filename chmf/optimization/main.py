import os
import numpy as np
import scipy as sp


def en_in_r(x, y, psf: np.array, r):
    full_eng = psf.sum()
    local_en = 0
    for i in range(len(x)):
        for j in range(len(y)):
            if np.square(x[i]) + np.square(y[j]) <= np.square(r):
                local_en += psf[j][i]
            ecf = local_en / full_eng
    
    return ecf


def r_for_en(x, y, psf: np.array, concentr):
    r = 0
    ecf = 0
    local_en = 0
    full_eng = psf.sum()
    for i in range(len(x)):
        for j in range(len(y)):
            local_en += psf[j][i]
            ecf = local_en / full_eng
            if ecf <= concentr + 1e-3 or ecf >= concentr - 1e-3:
                break
    return r


def half_split(x, y, psf: np.array, conc, xr, xm, xl):
    f = 0

    while abs(f-conc) > 1e-3:
        fl = en_in_r(x, y, psf, xl)
        fm = en_in_r(x, y, psf, xm)
        fr = en_in_r(x, y, psf, xr)
        dl = abs(fl-conc)
        dm = abs(fm-conc)
        dr = abs(fr-conc)

        if dl < dm and dm < dr:
            f = fl
            r = xl
            xr = xm
            xm = (xr - xl) / 2

        elif dl > dm and dl > dr:
            f = fr
            r = xr
            xl = xm
            xm = (xr - xl) / 2
        
        elif dm < dl and dm < dr:
            f = fm
            r = xm
    
    return r


with open(os.path.join(os.getcwd(), 'chmf\\optimization\\psf_c00_00.txt'), 'r') as fl:
    lines = []
    for line in fl:
        values = [value.strip() for value in  line.split()]
        lines.append(values)
    psf_arr = np.array(lines, dtype='float64')


# вариант 5
# стекло 1, стекло 2 = БФ7, ТФ7
# параметры оптимизации: r1, r2
# ограничения: |r| > 30

ecf = 0.83
r1 = 100
r2 = -100
r3 = -116.16
d1 = 7
d2 = 5
f = 70

x = np.arange(start=-6.375, stop=(6.375 + 0.05), step=0.05, dtype='float64')
y = np.arange(start=-6.375, stop=(6.375 + 0.05), step=0.05, dtype='float64')
    
print(half_split(x, y, psf_arr, ecf, r3, r2, r1))