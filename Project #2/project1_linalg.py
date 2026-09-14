
import numpy as np


def infNorm(M):
    return np.max(np.sum(np.abs(M), axis=1))


def conditionNumber(M):
    M_inv = np.linalg.inv(M)

    maxValOg = infNorm(M)
    maxValInv = infNorm(M_inv)

    return maxValOg * maxValInv


def lu_factorize(M):
    assert M.shape[0] == M.shape[1], "Matrix ain't there, be square"

    n = M.shape[0]
    U = M.astype(float).copy()
    L = np.eye(n)
    for i in range(n):
        if U[i, i] == 0:
            raise ZeroDivisionError(f"zero pivot at index {i}: LU without pivoting fails")
        L[i+1:, i]   = U[i+1:, i] / U[i, i]               # every multiplier of column i at once
        U[i+1:, i:] -= np.outer(L[i+1:, i], U[i, i:])      # rank-1 update of the remaining submatrix
        U[i+1:, i]   = 0.0                                 # exact zeros below the diagonal
    return L, U


def forward_substitute(L, b):
    n = L.shape[0]
    y = np.zeros(n, dtype=float)
    for k in range(n):
        if L[k, k] == 0:
            raise ZeroDivisionError(f"singular L at index {k}")
        y[k] = (b[k] - L[k, :k] @ y[:k]) / L[k, k]
    return y


def back_substitute(U, y):
    n = U.shape[0]
    x = np.zeros(n, dtype=float)
    for k in range(n-1, -1, -1):
        if U[k, k] == 0:
            raise ZeroDivisionError(f"singular U at index {k}")
        x[k] = (y[k] - U[k, k+1:] @ x[k+1:]) / U[k, k]
    return x


def lu_solve(A, b):
    L, U = lu_factorize(A)
    return back_substitute(U, forward_substitute(L, b))


def sign(x):
    return 1.0 if x >= 0 else -1.0


def householder_QR_slow(A):
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m)

    assert m >= n, "Needs more rows than columns"

    for k in range(min(n, m - 1)):
        a_k = -sign(R[k, k]) * np.sqrt(np.sum(R[k:, k] ** 2))
        v_k = R[k:, k].copy()
        v_k[0] -= a_k
        b_k = v_k.T @ v_k

        if b_k == 0:
            continue

        for j in range(k, n):
            phi_j = v_k @ R[k:, j]
            R[k:, j] -= (2 * phi_j / b_k) * v_k

        for j in range(m):
            phi_j = v_k @ Q[k:, j]
            Q[k:, j] -= (2 * phi_j / b_k) * v_k

    Q = Q.T

    # Checks
    assert np.allclose(Q.T @ Q, np.eye(m)), "Q not orthogonal"
    assert np.allclose(Q.T @ Q, Q @ Q.T), "Q not orthogonal"
    assert np.allclose(Q @ R, A), "M != QR"

    return Q, R


def householder_QR_fast(A):
    m, n = A.shape
    R = A.astype(float).copy()
    V = np.zeros((m, n))

    assert m >= n, "Needs more rows than columns"

    for k in range(min(n, m - 1)):
        a_k = -sign(R[k, k]) * np.sqrt(np.sum(R[k:, k] ** 2))
        v_k = R[k:, k].copy()
        v_k[0] -= a_k
        b_k = v_k @ v_k

        if b_k == 0:
            continue

        for j in range(k, n):
            phi_j = v_k @ R[k:, j]
            R[k:, j] -= (2 * phi_j / b_k) * v_k

        V[k:, k] = v_k

    return V, R


def least_squares(A, b):
    assert A.shape[0] >= A.shape[1], "More columns than rows"
    assert A.shape[0] == b.shape[0], "A rows != b rows"
    assert b.shape[1] == 1, "b not column vector"

    m, n = A.shape
    Q, R = householder_QR_slow(A)

    QTb = (Q.T @ b).flatten()

    x = back_substitute(R[:n, :n], QTb[:n])
    x = x.reshape(n, 1)

    r = b - A @ x

    return x, r
