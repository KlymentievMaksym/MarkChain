import numpy as np
import matplotlib.pyplot as plt
import tqdm
import numba as nb

@nb.njit
def _ising_2d_energy(grid: np.ndarray, grid_size: int, beta: float):
    i, j = np.random.randint(0, grid_size, 2)
    spin = grid[i, j]

    ipp = (i + 1) % grid_size
    jpp = (j + 1) % grid_size
    inn = (i - 1)
    jnn = (j - 1)
    neighbors = grid[ipp, j] + grid[i, jpp] + grid[inn, j] + grid[i, jnn]

    energy = 2 * spin * neighbors
    if energy < 0 or np.random.rand() < np.exp(-beta * energy):
        spin = -spin
    grid[i, j] = spin
    return grid

def _ising_2d_simulation(beta: float, grid_size: int = 50, iterations: int = 100000):
    grid = np.random.choice([-1, 1], size=(grid_size, grid_size)).astype(float)
    start_grid = grid.copy()

    for _ in tqdm.trange(iterations):
        grid = _ising_2d_energy(grid, grid_size, beta)

    return start_grid, grid

def task_3(betas: list, grid_size: int = 50, iterations: int = 100000):
    for beta in betas:
        start, end = _ising_2d_simulation(beta, grid_size, iterations)

        plt.figure(figsize=(10, 5))
        plt.suptitle(f'Beta = {beta}', fontsize=16)

        cmap='viridis'
        plt.subplot(1, 2, 1)
        plt.title("start")
        plt.imshow(start, cmap=cmap, interpolation='nearest')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("end")
        plt.imshow(end, cmap=cmap, interpolation='nearest')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    task_3([0.1, 0.4, 1.5], grid_size=200, iterations=int(1e6))