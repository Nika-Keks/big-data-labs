import numpy as np


def huber(x: np.array, k: float):
    phi = lambda elem: elem if abs(elem) < k else k * np.sign(elem)

    return np.mean([phi(elem) for elem in x])

def sigm3_rull(x: np.array):

    xmean = np.mean(x)
    xvar = np.var(x)
    is_in_3sigm = lambda xi: np.abs(xi - xmean) < 3*xvar

    clear_res = [val for val in x if is_in_3sigm(val)]

    return clear_res

def double_stage_mean(x: np.array):
    x = sigm3_rull(x)
    return np.mean(x)

