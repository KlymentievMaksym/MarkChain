import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba

@numba.njit
def distribution_step(x, iter):
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

def distribution(n: int, start_x: int = 1) -> np.ndarray:
    x = np.zeros(n + 1, dtype=np.int64)
    x[0] = start_x

    for iter in tqdm(range(1, n + 1), leave=False):
        distribution_step(x, iter)
    return x

def task_1(i_lims: list[int, int] = [1, 1000], n: int = int(1e6), start_pos: list[int] = [1, 10, 100, 500], tries: int = 1, figsize: tuple = (8, 4)):
    fig, [logs, bars] = plt.subplots(1, 2, figsize=figsize)

    i = np.arange(i_lims[0], i_lims[1]+1)

    pi_ = i**(-3/2)
    pi = pi_ / pi_.sum()

    logs.plot(i, pi, '.', label='"Справжній"', linewidth=2, markersize=1)
    bars.bar(i, pi, width=1, label='"Справжній"', alpha=0.8, edgecolor='black', linewidth=0.8)
    
    samples = np.zeros((tries, i_lims[1]), dtype=float)
    for pos in start_pos:
        for trie in tqdm(range(tries), ):
            samples_single = distribution(n, pos)
            hist, counts = np.histogram(samples_single, i_lims[1])
            mcmc = hist / hist.sum()
            # print(mcmc)
            samples[trie] = mcmc.copy()
            # print(samples[trie])
        mcmc = np.mean(samples, axis=0)
        # print(mcmc.shape)
        # print(mcmc)

        logs.plot(i, mcmc, '.', label=f'Через алгоритм, стартова позиція: {pos}', linewidth=1, markersize=0.8)
        bars.bar(i, mcmc, width=1, label=f'Через алгоритм, стартова позиція: {pos}', alpha=0.3)


    logs.set_xlim(i_lims[0] - 20, i_lims[1] + 20)
    logs.set_yscale('log')
    
    bars.set_xlim(i_lims[0], 20)

    handles, labels = bars.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper right', ncol=1)

    plt.suptitle('Порівняння теоретичного і через алгоритм')

    logs.set_xlabel('Значення');
    bars.set_xlabel('Значення');
    logs.set_ylabel('Ймовірність')
    bars.set_ylabel('Ймовірність')

    plt.show()


if __name__ == '__main__':
    task_1(i_lims = [1, 1000], n = int(1e6), start_pos = [1, 10, 100, 500], tries = 1)

    