import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import tqdm
import numba as nb


def task_4(a=2, b=5, n_samples=50000):
    samples = []
    current = 0.5
    delta = 0.1
    
    def target(x):
        if x < 0 or x > 1: return 0
        return x**(a-1) * (1-x)**(b-1)
    
    for _ in range(n_samples):
        proposal = current + np.random.uniform(-delta, delta)
        
        ratio = target(proposal) / target(current)
        if np.random.rand() < min(1, ratio):
            current = proposal
        samples.append(current)
        
    plt.figure(figsize=(10, 5))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='green', label='MCMC')
    x = np.linspace(0, 1, 100)
    plt.plot(x, beta.pdf(x, a, b), 'r-', lw=2, label=f'True Beta({a},{b})')
    plt.title(f'Beta({a}, {b})')
    plt.legend()
    plt.show()