import argparse
import numpy as np
from optimization import steepest_descent, newton

def potential_allocation(x, valuations, mu=100.0):
    """
    Computes the potential function for the optimal auction design (allocations only).
    
    Parameters:
        x (numpy array): Allocation vector of length n.
        valuations (numpy array): Valuations of each bidder.
        mu (float): Penalty parameter for constraints.
    
    Returns:
        float: Value of the potential function.
    """
    n = len(valuations)
    
    # Revenue term: Minimize -sum(v_i * x_i) to maximize revenue
    revenue_term = -np.sum(x * valuations)
    
    # Penalty for allocation bounds: Ensure x_i is within [0,1]
    penalty_alloc_bounds = np.sum(np.maximum(0, -x) ** 2) + np.sum(np.maximum(0, x - 1) ** 2)
    
    # Penalty for total allocation constraint: sum(x) must be <= 1
    penalty_total_alloc = np.maximum(0, np.sum(x) - 1) ** 2
    
    penalty = penalty_alloc_bounds + penalty_total_alloc
    return revenue_term + (mu / 2.0) * penalty

def analytical_optimal_solution(v):
    """
    Computes the analytical optimal allocation for the auction problem.
    
    Parameters:
        v (numpy array): Valuations of bidders.
    
    Returns:
        numpy array: Optimal allocation vector.
    """
    n = len(v)
    x_opt = np.zeros(n)
    i_star = np.argmax(v)
    x_opt[i_star] = 1.0
    return x_opt

def auction_projection(x):
    """
    Projects the allocation vector onto the feasible set.
    
    Parameters:
        x (numpy array): Allocation vector.
    
    Returns:
        numpy array: Feasible allocation vector.
    """
    x = np.clip(x, 0, 1)  # Enforce bounds [0,1]
    if np.sum(x) > 1:
        x /= np.sum(x)  # Normalize to ensure sum(x) <= 1
    return x
import argparse
import numpy as np

from optimization import steepest_descent, newton

def solve_auction_game(valuations, mu, alpha, tol, max_iter, max_iter_newton):
    """
    Solves the auction game using Steepest Descent and Newton's Method.
    
    Parameters:
        valuations (numpy array): Array of bidder valuations.
        mu (float): Penalty parameter for constraints.
        alpha (float): Step size for steepest descent.
        tol (float): Convergence tolerance for optimization methods.
        max_iter (int): Maximum iterations for steepest descent.
        max_iter_newton (int): Maximum iterations for Newton's method.
    """
    n = len(valuations)

    # Define the potential function as a lambda that depends only on x.
    potential_func = lambda x: potential_allocation(x, valuations, mu=mu)

    # Initial guess: equal allocation to all bidders
    x0 = np.ones(n) / n

    print("=== Optimal Auction Design Optimization (Allocations Only) ===")
    print("Bidder valuations:", valuations)
    print("Initial guess (allocations):", x0)

    # Solve using Steepest Descent
    print("\n--- Running Steepest Descent ---")
    x_sd = steepest_descent(
        potential_func, x0, alpha=alpha, convergence_tol=tol, max_iter=max_iter,
        visualize=True, N=n, valuations=valuations, game_type='auction', projection=auction_projection
    )
    print("Steepest Descent Solution (x):", x_sd)
    print("Potential function value (Steepest Descent):", potential_func(x_sd))
    payments_sd = np.maximum(0, x_sd * valuations)
    revenue_sd = np.sum(payments_sd)
    print("Achieved Revenue (Steepest Descent):", revenue_sd)
    print("Payments (Steepest Descent):", payments_sd)

    # Solve using Newton's Method
    print("\n--- Running Newton's Method ---")
    try:
        x_newton = newton(
            potential_func, x0, convergence_tol=tol, max_iter=max_iter_newton,
            visualize=True, N=n, valuations=valuations, game_type='auction',
            regularization=1e-5, projection=auction_projection
        )
        print("Newton's Method Solution (x):", x_newton)
        print("Potential function value (Newton):", potential_func(x_newton))
        payments_newton = np.maximum(0, x_newton * valuations)
        revenue_newton = np.sum(payments_newton)
        print("Achieved Revenue (Newton):", revenue_newton)
        print("Payments (Newton):", payments_newton)
    except Exception as e:
        print("Newton's method failed with error:", e)
        x_newton = None

    # Compute the analytical (optimal) solution
    x_opt = analytical_optimal_solution(valuations)
    payments_opt = x_opt * valuations
    optimal_revenue = np.sum(payments_opt)
    print("\n--- Analytical Optimal Solution ---")
    print("Analytical Optimal (x):", x_opt)
    print("Payments (Analytical):", payments_opt)
    print("Optimal Revenue:", optimal_revenue)

    # Compare numerical results with the analytical optimal solution
    print("\n--- Comparison ---")
    print("Steepest Descent Revenue Error: {:.6f}".format(abs(optimal_revenue - revenue_sd)))
    if x_newton is not None:
        print("Newton's Method Revenue Error: {:.6f}".format(abs(optimal_revenue - revenue_newton)))
    else:
        print("Newton's Method did not produce a solution.")

def main():
    """
    Parses command-line arguments and runs the auction game solver.
    """
    parser = argparse.ArgumentParser(description="Optimize auction design using only allocation optimization.")
    parser.add_argument("--valuations", type=str, required=True, help="Comma-separated list of bidder valuations")
    parser.add_argument("--mu", type=float, default=100.0, help="Penalty parameter")
    parser.add_argument("--alpha", type=float, default=0.005, help="Step size for steepest descent")
    parser.add_argument("--tol", type=float, default=1e-8, help="Convergence tolerance")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for steepest descent")
    parser.add_argument("--max_iter_newton", type=int, default=100, help="Max iterations for Newton's method")

    args = parser.parse_args()
    valuations = np.array([float(val.strip()) for val in args.valuations.split(',')])
    solve_auction_game(valuations, args.mu, args.alpha, args.tol, args.max_iter, args.max_iter_newton)

if __name__ == '__main__':
    main()
