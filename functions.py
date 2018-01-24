# coding: utf-8

import numpy as np


def step(x):
    """
    ステップ関数
    """
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    """
    シグモイド関数
    """
    return 1 / (1 + np.exp(-x))


def softmax(a):
    """
    ソフトマックス関数
    """
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def relu(x):
    """
    ReLU関数
    """
    return np.maximum(0, x)


def mean_squared_error(y, t):
    """
    2乗和誤差
    """
    return 0.5 * np.sum((y - t)**2)


def cross_entropy_error(y, t):
    """
    交差エントロピー誤差
    """
    delta = 1e-7
    return np.sum(t + np.log(y + delta))


def batch_cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t + np.log(y[np.arange(batch_size), t])) / batch_size


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 + h)
