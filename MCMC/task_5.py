import numpy as np
import matplotlib.pyplot as plt


def task_5(d_squared_list=[0.01, 1, 100]):
    # Дано в умові
    y = 3
    mu0 = 0
    sigma2 = 1
    tau2 = 4
    
    # Апостеріорний розподіл (з точністю до константи)
    # f(theta | y) ~ N(theta | y, sigma^2) * N(theta | mu0, tau2)
    def log_posterior(theta):
        prior = -0.5 * (theta - mu0)**2 / tau2
        likelihood = -0.5 * (y - theta)**2 / sigma2
        return prior + likelihood

    # Теоретичні значення
    # Mean = (y/sigma2 + mu0/tau2) / (1/sigma2 + 1/tau2)
    # Mean = (3/1 + 0/4) / (1/1 + 1/4) = 3 / 1.25 = 2.4 (або 12/5)
    true_mean = 12/5
    
    plt.figure(figsize=(15, 4))
    
    for idx, d2 in enumerate(d_squared_list):
        d = np.sqrt(d2)
        n_iter = 1000
        chain = [0] # Старт з 0
        accepted = 0
        
        for _ in range(n_iter):
            current = chain[-1]
            proposal = current + np.random.normal(0, d)
            
            # log acceptance ratio
            log_ratio = log_posterior(proposal) - log_posterior(current)
            if np.log(np.random.rand()) < log_ratio:
                chain.append(proposal)
                accepted += 1
            else:
                chain.append(current)
        
        # Графік шляху (Trace plot)
        plt.subplot(1, 3, idx+1)
        plt.plot(chain)
        plt.axhline(true_mean, color='r', linestyle='--', label='Theory Mean (2.4)')
        plt.title(f'd^2 = {d2}, Acc: {accepted/n_iter:.2f}')
        plt.xlabel('Ітерації')
        if idx == 0: plt.ylabel('Значення Theta')
        plt.legend()
    plt.show()
