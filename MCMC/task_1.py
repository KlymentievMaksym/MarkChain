import numpy as np
import matplotlib.pyplot as plt

def distribution(n: int, start_x: int = 1) -> np.ndarray:
    x = np.zeros(n + 1, dtype=int)
    x[0] = start_x

    for iter in range(1, n + 1):
        if x[iter-1] == 0:
            prob = (1/2)**(5/2)
            if np.random.rand() < prob: x[iter] = 1
        else:
            u = np.random.rand()
            if u < 1/2: x[iter] = x[iter-1] - 1
            else:
                p = (x[iter-1]/(x[iter-1] + 1)) ** (3/2)

                v = np.random.rand()
                if v < p: x[iter] = x[iter-1] + 1
                else: x[iter] = x[iter-1]
    return x

def task_1(i_lims: list[int, int] = [1, 1000], n: int = int(1e6), start_pos: list[int] = [1, 10, 100, 500]):
    fig, [logs, bars] = plt.subplots(1, 2)

    i = np.arange(i_lims[0], i_lims[1]+1)

    pi_ = i**(-3/2)
    pi = pi_ / pi_.sum()

    logs.plot(i, pi, '.', label='"Справжній"', linewidth=2, markersize=1)
    
    for pos in start_pos:
        samples = distribution(n, pos)
        hist, counts = np.histogram(samples, i_lims[1])
        mcmc = hist / hist.sum()
        # counts = np.bincount(samples, minlength=i_lims[1]+1)[1:]
        # emp = counts / counts.sum()

        logs.plot(i, mcmc, '.', label=f'Через алгоритм, стартова позиція: {pos}', linewidth=1, markersize=0.8)

    logs.set_xlim(*i_lims)
    logs.set_yscale('log')
    logs.legend()
    logs.set_title('Порівняння теоретичного і через алгоритм (логарифмічна шкала)')
    logs.xlabel('Значення');
    logs.ylabel('Ймовірність')
    plt.show()