import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import tqdm
import numba as nb

def task_4(a=2, b=5, iterations=50000, bins=50):
    samples = []
    current = np.random.uniform()
    
    def target(x):
        if x < 0 or x > 1: return 0
        return x**(a-1) * (1-x)**(b-1)
    
    for _ in tqdm.trange(iterations):
        # proposal = current + np.random.uniform(-delta, delta)
        proposal = np.random.uniform()

        p_current = target(current)
        p_proposal = target(proposal)
        
        if p_current == 0:
            ratio = 1
        else:
            ratio = p_proposal / p_current

        if np.random.rand() < min(1, ratio):
            current = proposal
        samples.append(current)

    hist_values, bin_edges = np.histogram(samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    true_pdf_values = beta.pdf(bin_centers, a, b)
    error_line = np.abs(hist_values - true_pdf_values)

    plt.figure(figsize=(10, 5))
    plt.hist(samples, bins=bins, density=True, alpha=0.6, color='green', label='MCMC')
    x = np.linspace(0, 1, 100)
    plt.plot(bin_centers, true_pdf_values, 'r-', lw=2, label=f'True Beta({a},{b})')
    plt.plot(bin_centers, error_line, 'b.-', lw=1.5, label='Error')
    plt.title(f'Beta({a}, {b})')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    task_4(a=.5, b=.5, iterations=50000, bins=50)