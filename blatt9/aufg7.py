import numpy as np
import matplotlib.pyplot as plt

def compute_gerschgorin_circles(A):
    return A.diagonal(), np.sum(abs(A), axis=1) - abs(A.diagonal())

def bound_of_eigenvalues(A, upper_bound=True):
    ms, rs = compute_gerschgorin_circles(A)
    sign = 1 if upper_bound else -1
    candidates = ms + sign * rs
    return np.max(candidates) if upper_bound else np.min(candidates)

if __name__ == "__main__":
    A = np.loadtxt("gerschgorin_50.txt", delimiter=',')
    ms, rs = compute_gerschgorin_circles(A)

    colors = plt.cm.rainbow(np.linspace(0, 1, 50))
    fig, ax = plt.subplots()
    for m, r, c in zip(ms, rs, colors):
        ax.add_artist(plt.Circle((m.real, m.imag), r, color=c))

    lower, upper = bound_of_eigenvalues(A, upper_bound=False), bound_of_eigenvalues(A)

    max_radius = np.max(rs)
    plt.ylim((-max_radius, max_radius))
    plt.xlim((lower, upper))
    plt.show()