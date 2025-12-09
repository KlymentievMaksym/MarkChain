import numpy as np
import matplotlib.pyplot as plt


def _ising_2d_simulation(beta=0.4, L=50, steps=100000):
    grid = np.random.choice([-1, 1], size=(L, L))
    start_grid = grid.copy()
    
    for _ in range(steps):
        i, j = np.random.randint(0, L, 2)
        S = grid[i, j]
        neighbors = (grid[(i+1)%L, j] + grid[(i-1)%L, j] + grid[i, (j+1)%L] + grid[i, (j-1)%L])
        dE = 2 * S * neighbors
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            grid[i, j] = -S
            
    return start_grid, grid

def task_3(betas: list, L: int = 50, steps: int = 100000):
    for b in betas:
        start, end = _ising_2d_simulation(beta=b, L=L, steps=steps)
        
        plt.figure(figsize=(10, 5))
        plt.suptitle(f'Beta = {b}', fontsize=16)

        cmap='binary'
        plt.subplot(1, 2, 1)
        plt.title("start")
        plt.imshow(start, cmap=cmap, interpolation='nearest')
        plt.axis('off')

        # Графік 2: End
        plt.subplot(1, 2, 2)
        plt.title("end")
        plt.imshow(end, cmap=cmap, interpolation='nearest')
        plt.axis('off')

        plt.tight_layout()
        plt.show()