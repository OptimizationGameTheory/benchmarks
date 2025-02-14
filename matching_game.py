import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from optimization import steepest_descent, newton

def matching_potential(x, n, C, mu, lam):
    """
    Computes the potential function for the stable matching game.

    This function enforces row and column constraints, ensuring a valid assignment.
    
    Parameters:
        x (numpy array): Flattened assignment matrix of shape (n*n,).
        n (int): Number of men and women.
        C (numpy array): Cost matrix of shape (n, n).
        mu (float): Penalty parameter for constraint violations.
        lam (float): Regularization parameter to ensure convexity.
    
    Returns:
        float: Value of the potential function.
    """
    X = x.reshape((n, n))
    cost_term = np.sum(C * X)
    reg_term = (lam / 2.0) * np.sum(x ** 2)
    penalty_rows = np.sum((np.sum(X, axis=1) - 1) ** 2)
    penalty_cols = np.sum((np.sum(X, axis=0) - 1) ** 2)
    penalty_bounds = np.sum(np.maximum(0, -X) ** 2 + np.maximum(0, X - 1) ** 2)
    penalty = penalty_rows + penalty_cols + penalty_bounds
    return cost_term + reg_term + (mu / 2.0) * penalty

def gradient_matching_potential(x, n, C, mu, lam):
    """
    Computes the gradient of the matching potential function.
    
    Parameters:
        x (numpy array): Flattened assignment matrix of shape (n*n,).
        n (int): Number of men and women.
        C (numpy array): Cost matrix of shape (n, n).
        mu (float): Penalty parameter.
        lam (float): Regularization parameter.
    
    Returns:
        numpy array: Gradient vector of shape (n*n,).
    """
    X = x.reshape((n, n))
    grad_cost_term = C.flatten()
    grad_reg_term = lam * x
    row_sums = np.sum(X, axis=1) - 1
    grad_penalty_rows = np.zeros_like(X)
    for i in range(n):
        grad_penalty_rows[i, :] = 2 * row_sums[i]
    col_sums = np.sum(X, axis=0) - 1
    grad_penalty_cols = np.zeros_like(X)
    for j in range(n):
        grad_penalty_cols[:, j] = 2 * col_sums[j]
    grad_penalty_bounds = 2 * np.maximum(0, -X) - 2 * np.maximum(0, X - 1)
    grad_penalty = grad_penalty_rows + grad_penalty_cols + grad_penalty_bounds
    return grad_cost_term + grad_reg_term + (mu / 2.0) * grad_penalty.flatten()

def hessian_matching_potential(x, n, C, mu, lam):
    """
    Computes the Hessian matrix of the matching potential function.
    
    Parameters:
        x (numpy array): Flattened assignment matrix of shape (n*n,).
        n (int): Number of men and women.
        C (numpy array): Cost matrix of shape (n, n).
        mu (float): Penalty parameter.
        lam (float): Regularization parameter.
    
    Returns:
        numpy array: Hessian matrix of shape (n*n, n*n).
    """
    X = x.reshape((n, n))
    H = lam * np.eye(n * n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                H[i * n + j, i * n + k] += mu * 2
    for j in range(n):
        for i in range(n):
            for k in range(n):
                H[i * n + j, k * n + j] += mu * 2
    for i in range(n):
        for j in range(n):
            if X[i, j] < 0 or X[i, j] > 1:
                H[i * n + j, i * n + j] += mu * 2
    return H
import argparse
import numpy as np

from scipy.optimize import linear_sum_assignment
from optimization import steepest_descent, newton

def load_cost_matrix(file_path, n):
    """
    Loads a cost matrix from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
        n (int): Expected size of the matrix (n x n).
    
    Returns:
        numpy.ndarray: Cost matrix of shape (n, n).
    """
    M = np.loadtxt(file_path, delimiter=',')
    if M.shape != (n, n):
        raise ValueError(f"Cost matrix shape {M.shape} does not match expected size ({n}, {n}).")
    return M

def project_onto_simplex(v, z=1):
    """
    Projects a vector onto the simplex defined by sum(x) = z and 0 <= x <= 1.
    
    Parameters:
        v (numpy.ndarray): Input vector.
        z (float): Constraint sum (default is 1).
    
    Returns:
        numpy.ndarray: Projected vector.
    """
    n = len(v)
    if np.sum(v) == z and np.alltrue((v >= 0) & (v <= 1)):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    w = np.minimum(w, 1)
    while np.sum(w) != z:
        w = np.maximum(w - (np.sum(w) - z) / n, 0)
    return w

def matching_projection(x, n):
    """
    Projects an assignment matrix onto the simplex.
    
    Parameters:
        x (numpy.ndarray): Flattened assignment matrix.
        n (int): Number of entities (men/women).
    
    Returns:
        numpy.ndarray: Flattened projected matrix.
    """
    X = x.reshape((n, n))
    for i in range(n):
        X[i] = project_onto_simplex(X[i])
    return X.flatten()

def solve_matching_game(n, cost_matrix, mu, lam, alpha, tol, max_iter, max_iter_newton):
    """
    Solves the stable matching problem using numerical optimization techniques.
    
    Parameters:
        n (int): Number of men/women.
        cost_matrix (numpy.ndarray): Cost matrix of shape (n, n).
        mu (float): Penalty parameter.
        lam (float): Regularization parameter.
        alpha (float): Step size for steepest descent.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum iterations for steepest descent.
        max_iter_newton (int): Maximum iterations for Newton's method.
    """
    print("Stable Matching Game")
    print("======================")
    print(f"Number of men/women: {n}")
    print("Cost matrix (C):")
    print(cost_matrix)
    print()
    potential_func = lambda x: matching_potential(x, n, cost_matrix, mu, lam)
    x0 = np.full((n, n), 1.0 / n).flatten()
    print("Using Steepest Descent:")
    x_sd = steepest_descent(potential_func, x0, alpha=alpha, convergence_tol=tol, max_iter=max_iter, visualize=True, N=n, game_type='matching')
    X_sd = x_sd.reshape((n, n))
    print("Solution from Steepest Descent (assignment matrix):")
    print(X_sd)
    print(f"Potential value: {potential_func(x_sd):.6f}")
    print()
    print("Using Newton's Method:")
    try:
        x_newton = newton(potential_func, x0, convergence_tol=tol, max_iter=max_iter_newton, visualize=True, N=n, game_type='matching', regularization=1e-5, projection=lambda x: matching_projection(x, n))
        X_newton = x_newton.reshape((n, n))
        print("Solution from Newton's Method (assignment matrix):")
        print(X_newton)
        print(f"Potential value: {potential_func(x_newton):.6f}")
    except ValueError as e:
        print("Newton's Method encountered an error:", e)
        X_newton = None
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    X_opt = np.zeros((n, n))
    for i, j in zip(row_ind, col_ind):
        X_opt[i, j] = 1
    optimal_cost = np.sum(cost_matrix * X_opt)
    print()
    print("Analytical Optimal Solution (via Hungarian Algorithm):")
    print("Optimal assignment matrix:")
    print(X_opt)
    print(f"Optimal total cost: {optimal_cost:.6f}")
    def linear_cost(x):
        X = x.reshape((n, n))
        return np.sum(np.abs(cost_matrix * X))
    cost_sd = linear_cost(x_sd)
    print()
    print("Comparison of Linear Cost Terms:")
    print(f"Steepest Descent Linear Cost: {cost_sd:.6f}")
    if X_newton is not None:
        cost_newton = linear_cost(X_newton)
        print(f"Newton's Method Linear Cost: {cost_newton:.6f}")
    print(f"Analytical Optimal Linear Cost: {optimal_cost:.6f}")

def main():
    """
    Parses command-line arguments and runs the matching game solver.
    """
    parser = argparse.ArgumentParser(description="Stable Matching Game via Continuous Optimization.")
    args = parser.parse_args()
    n = args.n
    if args.cost_file:
        cost_matrix = load_cost_matrix(args.cost_file, n)
    else:
        np.random.seed(args.seed)
        cost_matrix = np.random.uniform(0, 10, size=(n, n))
    solve_matching_game(n, cost_matrix, args.mu, args.lam, args.alpha, args.tol, args.max_iter, args.max_iter_newton)

if __name__ == '__main__':
    main()
