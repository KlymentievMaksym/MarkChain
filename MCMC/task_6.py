import numpy as np
import matplotlib.pyplot as plt


def task_6(rhos: list = [0.2, 0.6, 0.95], size: int = 2000, bins: int = 50):
    results = {}
    for rho in rhos:

        x = np.zeros(size)
        y = np.zeros(size)

        for i in range(1, size):
            x[i] = np.random.normal(rho * y[i-1], 1-rho**2)
            y[i] = np.random.normal(rho * x[i], 1-rho**2)

        data = np.column_stack((x, y))

        results[rho] = data

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title(f"Scatter plot (rho={rho})")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].scatter(data[:,0], data[:,1], s=8, alpha=0.6)
        axes[0].grid(True)
        
        axes[1].set_title(f"Histogram of x (rho={rho})")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("Frequency")
        axes[1].hist(data[:,0], bins=bins)
        axes[1].grid(True)
        
        axes[2].set_title(f"Histogram of y (rho={rho})")
        axes[2].set_xlabel("y")
        axes[2].set_ylabel("Frequency")
        axes[2].hist(data[:,1], bins=bins)
        axes[2].grid(True)

        plt.show()
    return results


if __name__ == "__main__":
    task6_data = task_6(size=2000, rhos=(0.2, 0.6, 0.95))