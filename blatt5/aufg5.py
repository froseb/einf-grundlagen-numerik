import numpy as np
import matplotlib.pyplot as plt
from warnings import warn


def solve_system_cg(A, b, x_0, kmax=105, eps=1e-8, return_errors=False):
    errs = list()
    
    x = x_0
    r = b - A @ x
    d = r
    Ad = A @ d
    beta = 0

    for _ in range(kmax):
        Ad_old = Ad
        Ad = A @ d
        Ar = Ad - beta * Ad_old

        err_squared = abs(r @ Ar)

        if return_errors:
            errs.append(err_squared ** 0.5)
        if err_squared <= eps*eps:
            break

        # Variable Updates
        rr_old = r @ r
        alpha = (rr_old) / (d @ Ad)
        x = x + alpha * d
        r = r - alpha * Ad
        beta = (r @ r) / (rr_old)
        d = r + beta * d

    if err_squared > eps*eps:
        warn("Maximum number of iterations exceeded without sufficient convergence. Residual Error: " + str(err_squared ** .5))
    return x if not return_errors else (x, errs)

A = np.loadtxt("laplacian_200.txt", delimiter=',')
b = np.arange(200)
x, errs = solve_system_cg(-1*A, -1*b, np.zeros(len(b)), return_errors=True)
print(x)
plt.semilogy(errs)
plt.show()
