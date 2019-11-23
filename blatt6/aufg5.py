import numpy as np
import matplotlib.pyplot as plt

def power_iteration(A, v_0 = None, kmax=100000, eps=1e-8):
    if v_0 is None:
        try:
            v_0 = np.ones(np.shape(A)[1])
        except:
            raise Exception("Wrong input matrix")

    v = v_0

    for _ in range(kmax):
        v_old = v

        Av = A @ v
        v = Av / np.linalg.norm(Av)

        if np.linalg.norm(v-v_old) < eps:
            break

    return v, v @ Av / np.linalg.norm(v)

def rank_1_update(A, lam, ev):
    return A - lam * ev[np.newaxis].T @ ev[np.newaxis]

def compute_first_eigenpairs(A, v_0, m, max_it):
    m = min(m, len(A))

    eigenvectors = []
    eigenvalues = []

    for _ in range(m):
        ev, lmb = power_iteration(A, v_0, max_it)
        eigenvectors.append(ev)
        eigenvalues.append(lmb)
        A = rank_1_update(A, lmb, ev)

    return eigenvectors, eigenvalues

A = np.loadtxt("matrix_power.txt", delimiter=',')
v, lmb = power_iteration(A)
print(lmb)

v, lmb = compute_first_eigenpairs(A, np.ones(np.shape(A)[1]), 100, 10000)
print(lmb)