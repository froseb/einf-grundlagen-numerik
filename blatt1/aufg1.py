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

def recursion_3term_iterative(F0, F1, beta, gamma, n, xs):
    vals_im1 = np.array([F0(x) for x in xs])
    vals_i = np.array([F1(x) for x in xs])

    for i in range(2, n+1):
        tmp = vals_i
        vals_i = beta*vals_i + gamma*vals_im1
        vals_im1 = tmp

    return vals_i

def pn_tschebyscheff_iterative(alphas, xs):
    pass