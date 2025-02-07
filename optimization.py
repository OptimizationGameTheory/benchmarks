import numpy as np


def gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_h = np.copy(x)
        x_h[i] += h
        grad[i] = (f(x_h) - f(x)) / h
    return grad


def hessian(f, x, h=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_ij = np.copy(x)
            if i == j:
                x_ij[i] += h
                f_ih = f(x_ij)
                x_ij[i] -= 2 * h
                f_i_h = f(x_ij)
                f_orig = f(x)
                hess[i, j] = (f_ih - 2 * f_orig + f_i_h) / h ** 2
            else:
                x_ij[i] += h
                x_ij[j] += h
                f_ij = f(x_ij)
                x_ij[j] -= 2 * h
                f_i_j = f(x_ij)
                x_ij[i] -= 2 * h
                x_ij[j] += 2 * h
                f_i_j_neg = f(x_ij)
                x_ij[j] -= 2 * h
                f_ij_neg = f(x_ij)
                hess[i, j] = (f_ij - f_i_j - f_i_j_neg + f_ij_neg) / (4 * h ** 2)
    return hess


def steepest_descent(f, x0, alpha=0.1, grad_function=gradient, convergence_tol=1e-6, max_iter=1000):
    x = np.copy(x0)
    for i in range(max_iter):
        grad = grad_function(f, x)
        x_new = x - alpha * grad
        if np.linalg.norm(x_new - x) < convergence_tol:
            print(f"Converged in {i + 1} iterations")
            return x_new
        x = x_new
    print("Reached maximum iterations")
    return x


def newton(f, x0, grad_function=gradient, hessian_function=hessian, convergence_tol=1e-6, max_iter=100):
    x = np.copy(x0)
    for i in range(max_iter):
        grad = grad_function(f, x)
        hess = hessian_function(f, x)
        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            raise ValueError("Hessian is not invertible. No solution found.")

        x_new = x - np.dot(hess_inv, grad)
        if np.linalg.norm(x_new - x) < convergence_tol:
            print(f"Converged in {i + 1} iterations")
            return x_new
        x = x_new
    raise ValueError("Maximum iterations reached. No solution found.")
