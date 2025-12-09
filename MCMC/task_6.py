import numpy as np
import matplotlib.pyplot as plt


def task_6(rho=0.7):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal(mean, cov, 2000)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10, color='purple')
    plt.title(f'2D Normal (rho={rho})')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.grid(True)
    plt.show()