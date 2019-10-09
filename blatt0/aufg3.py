import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def gauss_seidel(A, b, num_it = 1000):
    D = np.diag(np.diag(A))
    L = np.tril(A) - D
    U = np.triu(A) - D

    x = np.zeros(A.shape[1])
    
    errors = list()
    for i in range(1000):
        x = la.solve(L + D, -U @ x + b, lower=True)
        errors.append(la.norm(A@x-b))
        if errors[-1] < 1e-8:
            break

    plt.figure()
    plt.semilogy(errors)
    plt.title('Gauss Seidel Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('$| A x_i - b |$')
    return x

def jacobi(A, b, num_it = 1000):
    D = np.diag(np.diag(A))
    L = np.tril(A) - D
    U = np.triu(A) - D

    x = np.zeros(A.shape[1])
    
    errors = list()
    for i in range(num_it):
        x_old = x
        x = la.solve(D, -(L+U) @ x + b, lower=True)
        errors.append(la.norm(A@x-b))
        if errors[-1] < 1e-8:
            break

    plt.figure()
    plt.semilogy(errors)
    plt.title('Jacobi Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('$| A x_i - b |$')
    return x

T = np.array([[8, -1, 0, 0], [-1, 8, -1, 0], [0, -1, 8, -1], [0, 0, -1, 8]])
A = np.block([[T,                -np.eye(4),       np.zeros((4, 4)), np.zeros((4, 4))],
              [-np.eye(4),       T,                -np.eye(4),       np.zeros((4, 4))],
              [np.zeros((4, 4)), -np.eye(4),       T,                -np.eye(4)      ],
              [np.zeros((4, 4)), np.zeros((4, 4)), -np.eye(4),       T               ]])

b1 = np.array([6, 5, 5, 6])
b2 = np.array([5, 4, 4, 5])
b = np.concatenate((b1, b2, b1, b2))

x_gs = gauss_seidel(A, b)
x_j = jacobi(A, b)

plt.show()

print(A@x_gs)
print(A@x_j)
print(b)