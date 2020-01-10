import numpy as np
import scipy.sparse as spa
import matplotlib.pyplot as plt
import time

def compute_householder_matrix_reduced(v):
    """Computes the Householder matrix that maps v to the vector w = (w[0], 0, ..., 0) in R^(len(v))
    """
    # Define the norm at which a vector is assumed to be equal to zero
    eps = np.linalg.norm(A) * 1e-20

    # Unit vector in the first direction
    e = np.zeros(len(v))
    e[0] = 1
    # Normalize v
    v = v / np.linalg.norm(v)
    norm = np.linalg.norm(e - v)
    if abs(norm) < eps:
        # Return identity if e-v is close to 0
        return np.eye(len(v))
    # Compute direction between e and v
    v = (e - v) / norm
    return np.eye(len(v)) - 2 * v[np.newaxis].T @ v[np.newaxis]


def compute_householder_matrix_full(v, m):
    """Computes the Householder matrix that maps v to the vector w = (w[0], ..., w[m], 0, ..., 0) in R^(len(v))
    """
    n = len(v)
    # Create block for the full householder matrix
    return np.block([[np.eye(m), np.zeros((m, n-m))],
                     [np.zeros((n-m, m)), compute_householder_matrix_reduced(v[m:])]])


def compute_qrdecomp_householder(A):
    """Computes the QR decomposition of a matrix A^(n x n) in R (i. e. matrices Q and R,
    where Q.T @ Q == Q @ Q.T == np.eye(n) and R is a right upper triangular matrix with A == Q @ R)
    using Householder matrices
    """
    n = np.shape(A)[1]
    Q = np.eye(n)
    R = np.copy(A)

    # iterate through all columns
    for i in range(n):
        v = R[:, i]
        H = compute_householder_matrix_full(v, i)
        Q = Q @ H.T
        R = H @ R

    return Q, R

def compute_qrdecomp_householder_optimized(A):
    """Optimized version of compute_qrdecomp_householder that does not explicitly compute the householder matrix

    Computes the QR decomposition of a matrix A^(n x n) in R (i. e. matrices Q and R,
    where Q.T @ Q == Q @ Q.T == np.eye(n) and R is a right upper triangular matrix with A == Q @ R)
    using Householder matrices
    """
    n = np.shape(A)[1]
    Q = np.eye(n)
    R = np.copy(A)

    eps = np.linalg.norm(A) * 1e-20

    # iterate through all columns
    for i in range(n):
        # Get the ith column of A
        v = R[:, i]
        vk = v[i:]
        vk = vk / np.linalg.norm(vk)
        # Unit vector in first direction
        e = np.zeros(len(vk)); e[0] = 1
        norm = np.linalg.norm(e - vk)
        if abs(norm) < eps:
            # Do nothing if e-vk is close to 0
            continue
        # Compute the reflection direction of the householder transformation
        reflection_direction = (e - vk) / norm

        # Update Q and R without explicitely using the householder matrix -> less computation
        Q[:, i:] = ((Q.T)[i:, :] - 2 * reflection_direction[np.newaxis].T @ (reflection_direction[np.newaxis] @ (Q.T)[i:, :])).T
        R[i:, i:] = R[i:, i:] - 2 * reflection_direction[np.newaxis].T @ (reflection_direction[np.newaxis] @ R[i:, i:])

    return Q, R

def compute_givens_rotation(A, i, j):
    """Computes the Givens rotation G in the subspace generated by the ith and jth standard basis vectors
    such that for B = G @ A, we have:
    B[i, j] = 0
    G is returned as a sparse matrix for more efficient multiplication
    """
    n = np.shape(A)[0]
    G = np.eye(n)
    norm = np.linalg.norm(np.array([A[j, j], A[i, j]]))
    a = A[j, j] / norm
    b = -A[i, j] / norm
    G[i, i] = a; G[j, j] = a; G[i, j] = b; G[j, i] = -b
    G = spa.csr_matrix(G)
    return G

def compute_qrdecomp_givens(A):
    """Computes the QR decomposition of a square matrix A using Givens rotations
    """
    n = np.shape(A)[1]
    Q = np.eye(n)
    R = np.copy(A)
    for i in range(n):
        for j in range(i+1, n):
            G = compute_givens_rotation(A, i, j)
            Q = G.dot(Q.T).T
            R = G.dot(R)
    return Q, R


if __name__ == "__main__":
    A = np.loadtxt("matrix_qr2.txt", delimiter=',')
    start = time.time()
    Q, R = compute_qrdecomp_householder_optimized(A)
    end = time.time()
    print("Execution time for Householder QR decomposition: {:5.3f} seconds".format(end - start))

    start = time.time()
    Q, R = compute_qrdecomp_givens(A)
    end = time.time()
    print("Execution time for Givens QR decomposition: {:5.3f} seconds".format(end - start))