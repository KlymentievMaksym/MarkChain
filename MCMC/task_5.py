import numpy as np
import matplotlib.pyplot as plt
import tqdm


def task_5(ds=[0.01, 1, 100], y = 3, mu0 = 0, sigma2 = 1, tau2 = 4, iterations = 10000):
    def log_posterior(theta):
        prior = -0.5 * (theta - mu0)**2 / tau2
        likelihood = -0.5 * (y - theta)**2 / sigma2
        return prior + likelihood

    def posterior(x, mu, sigma):
        return (1.0 / np.sqrt(2*np.pi*sigma)) * np.exp(-0.5*((x-mu)**2 / sigma))

    true_mean = 12/5
    true_dispersion = 4/5
    
    plt.figure(figsize=(12, 8))
    for idx, d in enumerate(ds):
        chain = [np.random.normal(mu0, np.sqrt(tau2))]
        accepted = 1
        
        for _ in tqdm.trange(iterations):
            current = chain[-1]
            proposal = current + np.random.normal(0, d)
            
            probability = min(1, np.exp(log_posterior(proposal) - log_posterior(current)))
            if np.random.uniform() <= probability:
                chain.append(proposal)
                accepted += 1
            else:
                chain.append(current)

        plt.subplot(2, 3, idx+1)
        if idx == 0: plt.ylabel('Theta')
        plt.plot(chain)
        plt.title(f'd = {d}, Accept: {accepted/iterations*100:.2f}%')
        plt.xlabel('Iterations')
        plt.subplot(2, 3, 3 + (idx + 1))
        if idx == 0: plt.ylabel('Theta')
        plt.hist(chain, bins=50, density=True, label='MCMC')
        hist_values, bin_centers = np.histogram(chain, bins=50)
        plt.plot(bin_centers, posterior(bin_centers, true_mean, true_dispersion), label='Theoretical')
        plt.legend()
    plt.show()


if __name__ == '__main__':
    task_5()