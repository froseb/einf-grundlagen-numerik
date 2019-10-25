import numpy as np
import matplotlib.pyplot as plt

def clenshaw(alphas, xs):
    if len(alphas) == 1:
        return np.repeat(alphas[0], len(xs))

    # Degree of the polynomial
    n = len(alphas)-1
    # $\beta_{i+2}$
    beta_ip2 = np.zeros(len(xs))

    # $\beta_{i+1}$
    beta_ip1 = np.repeat(alphas[n], len(xs))

    for i in range(n-1, 0, -1):
        tmp = beta_ip1
        beta_ip1 = alphas[i] + 2*xs*beta_ip1 - beta_ip2
        beta_ip2 = tmp

    return alphas[0] + xs * beta_ip1 - beta_ip2

def plot_chebyshev(n):
    for i in range(n):
        alphas = np.zeros(i+1)
        alphas[i] = 1
        xs = np.linspace(-1, 1, num=100)
        ys = clenshaw(alphas, xs)
        plt.plot(xs, ys, label="$T_" + str(i) + "$")

    plt.legend()
    plt.show()

plot_chebyshev(6)