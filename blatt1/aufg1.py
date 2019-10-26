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

def recursion_3term_iterative(F0, F1, beta, gamma, n, xs):
    vals_im1 = np.array([F0(x) for x in xs])
    vals_i = np.array([F1(x) for x in xs])

    if n == 0:
        return vals_im1

    for i in range(2, n+1):
        tmp = vals_i
        vals_i = beta*vals_i + gamma*vals_im1
        vals_im1 = tmp

    return vals_i

def pn_tschebyscheff_iterative(alphas, xs):
    beta = 2*xs
    gamma = -1

    if type(alphas) != np.ndarray:
        alphas = np.array(alphas)

    Ts = np.column_stack([recursion_3term_iterative(lambda x: 1, lambda x: x, beta, gamma, i, xs) for i in range(len(alphas))])
    return Ts @ alphas

def plot_chebyshev(n, iterative=False):
    for i in range(n):
        alphas = np.zeros(i+1)
        alphas[i] = 1
        xs = np.linspace(-1, 1, num=100)
        if iterative:
            ys = pn_tschebyscheff_iterative(alphas, xs)
        else:
            ys = clenshaw(alphas, xs)
        plt.plot(xs, ys, label="$T_" + str(i) + "$")
    
    plt.title("First $" + str(n) + "$ Chebyshev Polynomials of the first kind")
    plt.legend()
    plt.savefig("chebyshev_polynomials.pdf")
    plt.show()

plot_chebyshev(6)

alphas = np.array([0, 0, 2/3, 0, 4/14, 0, 23/96])
xs = np.linspace(0, 10, 100)
clenshaw_ys = clenshaw(alphas, xs)
iterative_ys = pn_tschebyscheff_iterative(alphas, xs)
errors = np.abs(clenshaw_ys - iterative_ys)
plt.figure(figsize=(8, 7))
plt.plot(xs, errors)
plt.title("Error Between Clenshaw and Iterative Polynomial Evaluation for $\\bf \\alpha \\sf = \\left(0, 0, \\frac{2}{3}, 0, \\frac{4}{14}, 0, \\frac{23}{96}\\right)$")
plt.savefig("error_clenshaw_iterative.pdf")
plt.show()