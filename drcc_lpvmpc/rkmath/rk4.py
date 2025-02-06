""" Runge Kutta sixth order integration.
    Uses same syntax as scipy.integrate.odeint
    `fun` should be of the form fun(t, y, *args)
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np


def odeintRK4(fun, y0, t, args=()):
    gamma = np.asarray([1/6, 1/3, 1/3, 1/6])
    # gamma = np.asarray([1, 0, 0, 0])

    y_next = np.zeros([len(t)-1, len(y0)])
    # print("time length",len(t))
    for i in range(len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*fun(t[i], y0, *args)
        k2 = h*fun(t[i]+h/2, y0+k1/2, *args)
        k3 = h*fun(t[i]+h/2, y0+k2/2, *args)
        k4 = h*fun(t[i]+h, y0+k3, *args)
        K = np.asarray([k1, k2, k3, k4])
        y_next[i,:] = y0 + gamma@K
        y0 = y0 + gamma@K
    return y_next