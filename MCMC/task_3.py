import numpy as np
import matplotlib.pyplot as plt


def ising_1d(L=30, beta=0.4, steps=100000):
    grid = np.random.choice([-1, 1], size=(L, L))
    
    for _ in range(steps):
        i, j = np.random.randint(0, L, 2)
        S = grid[i, j]
        neighbors = grid[(i+1)%L, j] + grid[(i-1)%L, j] + grid[i, (j+1)%L] + grid[i, (j-1)%L]
        dE = 2 * S * neighbors
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            grid[i, j] = -S
            
    return grid

def task_3(betas = [-1.5, -0.5, -0.1, 0, 0.1, 0.5, 1.5], **kwargs):
    plt.figure(**kwargs)
    for idx, b in enumerate(betas):
        final_grid = ising_1d(beta=b)
        plt.subplot(len(betas)//2, len(betas)//2, idx+1)
        plt.imshow(final_grid, cmap='binary', interpolation='nearest')
        plt.title(f'Beta = {b}')
        plt.axis('off')
    plt.show()