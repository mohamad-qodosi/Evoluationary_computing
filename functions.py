import numpy as np


def ackley(X):
    p1 = -0.2 * np.sqrt(np.mean(X ** 2, axis=-1))
    p2 = np.mean(np.cos(2 * np.pi * X), axis=-1)

    t1 = -20 * np.exp(p1)
    t2 = np.exp(p2)

    return t1 - t2 + 20 + np.e


def rastrigin(X):
    X_pow2 = X ** 2

    T1 = 10 * (X_pow2)
    C1 = (X < -5.12) + (X > 5.12).astype(np.int8)

    T2 = (X_pow2) - 10 * np.cos(2 * np.pi * X)
    C2 = 1 - C1

    T = (T1 * C1) + (T2 * C2)

    return T.sum(axis=-1) + 10 * T.shape[-1]


def schwefel(X):
    T1 = 0.02 * (X ** 2)
    C1 = (X < -500) + (X > 500).astype(np.int8)

    T2 = -X * np.sin(np.sqrt(np.abs(X)))
    C2 = 1 - C1

    T = (T1 * C1) + (T2 * C2)

    return T.sum(axis=-1) + 418.9829 * T.shape[-1]


def Griewangk(X):
    return 1 + np.sum((X ** 2), axis=-1) / 4000 - np.prod(np.cos(X / ((np.arange(X.shape[-1]) + 1) ** 0.5)))
