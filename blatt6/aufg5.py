import numpy as np
import matplotlib.pyplot as plt

def power_iteration(A, v_0 = None, kmax=1e6, eps=1e-8):
    if v_0 == None:
        try:
            v_0 = np.zeros(np.shape(A)[1])
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

A = np.loadtxt("matrix_power.txt")
v, lmb = power_iteration(A)
print(v, lmb)

def rank_1_update(A, lam, ev):
    return A - lam * v @ v

def compute_first_eigenpairs(A, v_0, m, max_it):
    m = max(m, len(A))

    eigenvectors = []
    eigenvalues = []

    for i in range(m):
        ev, lmb = power_iteration(A, v_0, max_it)
        eigenvectors.append(v)
        eigenvalues.append(lmb)
        A = rank_1_update(A, lmb, ev)

