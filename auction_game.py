import numpy as np
from optimization import steepest_descent, newton


def potential(z, valuations, mu=100.0):
    """
    Potential function for the optimal auction design.
    
    Parameters:
      z  : 1D numpy array of length 2*n, where the first n entries are allocations x and
           the next n entries are payments p.
      valuations  : 1D numpy array of bidder valuations.
      mu : Penalty parameter.
    
    Returns:
      Scalar value of the potential function.
    """
    n = len(valuations)
    x = z[:n]
    p = z[n:]

    # Term to maximize revenue: we minimize (-sum(p))
    revenue_term = -np.sum(p)

    # Penalty for allocation bounds: x_i must be between 0 and 1.
    penalty_alloc_bounds = np.sum(np.maximum(0, -x) ** 2) + np.sum(np.maximum(0, x - 1) ** 2)

    # Penalty for total allocation: sum(x) must be <= 1.
    penalty_total_alloc = np.maximum(0, np.sum(x) - 1) ** 2

    # Penalty for payment non-negativity: p_i must be >= 0.
    penalty_payment_nonneg = np.sum(np.maximum(0, -p) ** 2)

    # Penalty for individual rationality: p_i must be <= v_i * x_i.
    penalty_ir = np.sum(np.maximum(0, p - valuations * x) ** 2)

    penalty = penalty_alloc_bounds + penalty_total_alloc + penalty_payment_nonneg + penalty_ir
    return revenue_term + (mu / 2.0) * penalty


def analytical_optimal_solution(v):
    """
    Computes the analytical optimal solution for the auction problem.
    
    Given valuations v, the optimum is to award the entire good to the bidder with the highest v.
    
    Returns:
      A numpy array z_opt = [x_opt, p_opt] of length 2*n.
    """
    n = len(v)
    x_opt = np.zeros(n)
    p_opt = np.zeros(n)
    i_star = np.argmax(v)
    x_opt[i_star] = 1.0
    p_opt[i_star] = v[i_star]
    return np.concatenate([x_opt, p_opt])


def main(args):
    # Parse bidder valuations from the input string.
    try:
        v_list = [float(val.strip()) for val in args.valuations.split(',')]
    except Exception as e:
        raise ValueError("Error parsing valuations. Ensure they are comma-separated numbers.") from e
    valuations = np.array(v_list)
    n = len(valuations)

    # Define the potential function as a lambda that depends only on z.
    potential_func = lambda z: potential(z, valuations, mu=args.mu)

    # Initial guess: a feasible point (e.g., equal allocation and zero payments)
    x0 = np.ones(n) / n
    p0 = np.zeros(n)
    z0 = np.concatenate([x0, p0])

    print("=== Optimal Auction Design Optimization ===")
    print("Bidder valuations:", valuations)
    print("Initial guess (allocations, payments):", z0)

    # Solve using Steepest Descent
    print("\n--- Running Steepest Descent ---")
    z_sd = steepest_descent(potential_func, z0,
                            alpha=args.alpha,
                            convergence_tol=args.tol,
                            max_iter=args.max_iter,
                            visualize=True,
                            N=n,
                            valuations=valuations,
                            game_type='auction')
    print("Steepest Descent Solution (x and p):", z_sd)
    print("Potential function value (Steepest Descent):", potential_func(z_sd))
    revenue_sd = np.sum(z_sd[n:])
    print("Achieved Revenue (Steepest Descent):", revenue_sd)

    # Solve using Newton's Method
    print("\n--- Running Newton's Method ---")
    try:
        z_newton = newton(potential_func, z0, convergence_tol=args.tol, max_iter=args.max_iter_newton, visualize=True,
                          N=n, valuations=valuations, game_type='auction')
        print("Newton's Method Solution (x and p):", z_newton)
        print("Potential function value (Newton):", potential_func(z_newton))
        revenue_newton = np.sum(z_newton[n:])
        print("Achieved Revenue (Newton):", revenue_newton)
    except Exception as e:
        print("Newton's method failed with error:", e)
        z_newton = None

    # Compute the analytical (optimal) solution.
    z_opt = analytical_optimal_solution(valuations)
    optimal_revenue = np.sum(z_opt[n:])
    print("\n--- Analytical Optimal Solution ---")
    print("Analytical Optimal (x and p):", z_opt)
    print("Optimal Revenue:", optimal_revenue)

    # Compare the revenues obtained by the numerical methods to the analytical optimum.
    print("\n--- Comparison ---")
    print("Steepest Descent Revenue Error: {:.6f}".format(abs(optimal_revenue - revenue_sd)))
    if z_newton is not None:
        print("Newton's Method Revenue Error: {:.6f}".format(abs(optimal_revenue - revenue_newton)))
    else:
        print("Newton's Method did not produce a solution.")

# if __name__ == '__main__':
#     main()
