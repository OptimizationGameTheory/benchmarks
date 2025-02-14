import numpy as np
import argparse
from optimization import steepest_descent, newton
from scipy.optimize import linear_sum_assignment


def matching_potential(x, n, C, mu, lam):
    """
    Computes the potential function for the stable matching game.

    In this formulation, we consider a matching between n men and n women.
    Let x be a vector of length n*n that represents the flattened assignment matrix X,
    where X[i,j] indicates the (continuous) degree to which man i is matched with woman j.
    
    The potential function is defined as:
    
        Phi(x) = <C, X> + (lambda/2) * ||x||^2 
                 + (mu/2) * { penalty_rows + penalty_cols + penalty_bounds }
    
    where:
      - <C, X> = sum_{i,j} C[i,j]*X[i,j] is the total matching cost,
      - penalty_rows = sum_{i=1}^n (sum_j X[i,j] - 1)^2 ensures each man is assigned exactly one partner,
      - penalty_cols = sum_{j=1}^n (sum_i X[i,j] - 1)^2 ensures each woman is assigned exactly one partner,
      - penalty_bounds = sum_{i,j} (max(0, -X[i,j])^2 + max(0, X[i,j]-1)^2) penalizes assignments outside [0,1].
    
    Parameters:
      x  : 1D numpy array of length n*n (flattened assignment matrix).
      n  : Number of men/women.
      C  : Cost matrix of shape (n, n), where C[i, j] is the cost (or dissatisfaction)
           of matching man i with woman j.
      mu : Penalty parameter for constraint violations.
      lam: Regularization parameter to ensure strict convexity.
      
    Returns:
      The scalar value of the potential function.
    """
    X = x.reshape((n, n))
    # Linear cost term.
    cost_term = np.sum(C * X)
    # Regularization term.
    reg_term = (lam / 2.0) * np.sum(x ** 2)
    # Penalty for row sum constraints: each man must be matched exactly once.
    penalty_rows = np.sum((np.sum(X, axis=1) - 1) ** 2)
    # Penalty for column sum constraints: each woman must be matched exactly once.
    penalty_cols = np.sum((np.sum(X, axis=0) - 1) ** 2)
    # Penalty for bounds: each entry must lie in [0, 1].
    penalty_bounds = np.sum(np.maximum(0, -X) ** 2 + np.maximum(0, X - 1) ** 2)
    penalty = penalty_rows + penalty_cols + penalty_bounds
    return cost_term + reg_term + (mu / 2.0) * penalty


def gradient_matching_potential(x, n, C, mu, lam):
    X = x.reshape((n, n))

    # Gradient of the linear cost term.
    grad_cost_term = C.flatten()

    # Gradient of the regularization term.
    grad_reg_term = lam * x

    # Gradient of the penalty for row sum constraints.
    row_sums = np.sum(X, axis=1) - 1
    grad_penalty_rows = np.zeros_like(X)
    for i in range(n):
        grad_penalty_rows[i, :] = 2 * row_sums[i]

    # Gradient of the penalty for column sum constraints.
    col_sums = np.sum(X, axis=0) - 1
    grad_penalty_cols = np.zeros_like(X)
    for j in range(n):
        grad_penalty_cols[:, j] = 2 * col_sums[j]

    # Gradient of the penalty for bounds.
    grad_penalty_bounds = 2 * np.maximum(0, -X) - 2 * np.maximum(0, X - 1)

    # Total gradient.
    grad_penalty = grad_penalty_rows + grad_penalty_cols + grad_penalty_bounds

    grad_total = grad_cost_term + grad_reg_term + (mu / 2.0) * grad_penalty.flatten()

    return grad_total


def hessian_matching_potential(x, n, C, mu, lam):
    X = x.reshape((n, n))

    # Initialize the Hessian matrix with regularization term (lam * I)
    H = lam * np.eye(n * n)

    # Hessian of the penalty for row sum constraints
    for i in range(n):
        for j in range(n):
            for k in range(n):
                H[i * n + j, i * n + k] += mu * 2

    # Hessian of the penalty for column sum constraints
    for j in range(n):
        for i in range(n):
            for k in range(n):
                H[i * n + j, k * n + j] += mu * 2

    # Hessian of the penalty for bounds
    for i in range(n):
        for j in range(n):
            if X[i, j] < 0:
                H[i * n + j, i * n + j] += mu * 2
            elif X[i, j] > 1:
                H[i * n + j, i * n + j] += mu * 2

    return H


def load_cost_matrix(file_path, n):
    """
    Loads a cost matrix from a CSV file.

    Parameters:
      file_path: Path to the CSV file.
      n        : Expected size of the matrix (n x n).
      
    Returns:
      A numpy array of shape (n, n).
    """
    M = np.loadtxt(file_path, delimiter=',')
    if M.shape != (n, n):
        raise ValueError(f"Cost matrix shape {M.shape} does not match expected size ({n}, {n}).")
    return M


def solve_matching_game(n, cost_matrix, mu, lam, alpha, tol, max_iter, max_iter_newton):
    print("Stable Matching Game")
    print("======================")
    print(f"Number of men/women: {n}")
    print("Cost matrix (C):")
    print(cost_matrix)
    print()

    # Define the potential function as a function of x only.
    potential_func = lambda x: matching_potential(x, n, cost_matrix, mu, lam)
    gradient_func = lambda f, x: gradient_matching_potential(x, n, cost_matrix, mu, lam)
    hessian_func = lambda f, x: hessian_matching_potential(x, n, cost_matrix, mu, lam)


    # Initial guess: uniform assignment (each man assigns 1/n to every woman).
    x0 = np.full((n, n), 1.0 / n).flatten()

    # Solve using Steepest Descent.
    print("Using Steepest Descent:")
    x_sd = steepest_descent(potential_func, x0, alpha=alpha, convergence_tol=tol, max_iter=max_iter, visualize=True,
                            N=n, game_type='matching')
    X_sd = x_sd.reshape((n, n))
    print("Solution from Steepest Descent (assignment matrix):")
    print(X_sd)
    print(f"Potential value: {potential_func(x_sd):.6f}")
    print()

    # Solve using Newton's Method.
    print("Using Newton's Method:")
    try:
        x_newton = newton(potential_func, x0, convergence_tol=tol, max_iter=max_iter_newton, visualize=True, N=n, game_type='matching')
        X_newton = x_newton.reshape((n, n))
        print("Solution from Newton's Method (assignment matrix):")
        print(X_newton)
        print(f"Potential value: {potential_func(x_newton):.6f}")
    except ValueError as e:
        print("Newton's Method encountered an error:", e)
        X_newton = None
    
    # Compute the analytical optimal solution using the Hungarian Algorithm.
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

    # For comparison, compute the linear cost term from the numerical solutions.
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