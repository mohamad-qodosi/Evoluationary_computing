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


def quadratic(X):
    return np.sum(X ** 2, axis=-1) / X.shape[-1]


benchmark_functions = {'ackley': {'function': ackley,
                                  'initialization_range': [-32.768, +32.768],
                                  'global_min': 0},

                       'rastrigin': {'function': rastrigin,
                                     'initialization_range': [-5.12, +5.12],
                                     'global_min': 0},

                       'schwefel': {'function': schwefel,
                                    'initialization_range': [-500.0, +500.0],
                                    'global_min': 420.9687},

                       'quadratic': {'function': quadratic,
                                    'initialization_range': [-5.0, +5.0],
                                    'global_min': 0}}

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for benchmark_function in benchmark_functions.values():
        print(benchmark_function)
        x = np.arange(benchmark_function['initialization_range'][0] * 1.5,
                      benchmark_function['initialization_range'][1] * 1.5,
                      0.01).reshape(-1, 1)

        y = benchmark_function['function'](x)
        # print(x)
        # print(y)

        plt.figure()
        plt.vlines(benchmark_function['initialization_range'], y.min(), y.max(), linestyles='dashed')
        plt.plot(x, y)
    plt.show()

