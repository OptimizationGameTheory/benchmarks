import numpy as np
import argparse
from optimization import steepest_descent, newton

# -------------------------------
# Define the Potential Function
# -------------------------------
def congestion_potential(x, N, a1, b1, a2, b2):
    """
    Computes the potential function for the congestion game.
    
    Parameters:
        x (np.ndarray): A one-dimensional numpy array where x[0] represents the
                        number of drivers choosing route 1. The number of drivers on
                        route 2 is then (N - x[0]).
        N (int): Total number of drivers.
        a1, b1 (float): Parameters for the latency function of route 1.
        a2, b2 (float): Parameters for the latency function of route 2.
        
    Returns:
        float: The value of the potential function.
    """
    x_val = x[0]
    x2 = N - x_val  # drivers on route 2
    potential_route1 = 0.5 * a1 * x_val**2 + b1 * x_val
    potential_route2 = 0.5 * a2 * x2**2 + b2 * x2
    return potential_route1 + potential_route2

# -------------------------------
# Main Routine
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Solve the Congestion Game using Optimization Techniques.")
    parser.add_argument('--N', type=int, default=100, help='Total number of drivers')
    parser.add_argument('--a1', type=float, default=1.0, help='Coefficient for latency function of route 1')
    parser.add_argument('--b1', type=float, default=0.0, help='Intercept for latency function of route 1')
    parser.add_argument('--a2', type=float, default=2.0, help='Coefficient for latency function of route 2')
    parser.add_argument('--b2', type=float, default=10.0, help='Intercept for latency function of route 2')
    parser.add_argument('--alpha', type=float, default=0.01, help='Step size for steepest descent')
    parser.add_argument('--tol', type=float, default=1e-8, help='Convergence tolerance')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations for steepest descent')
    parser.add_argument('--max_iter_newton', type=int, default=100, help='Maximum iterations for Newton’s method')

    args = parser.parse_args()

    # Initial guess: assume initially half the drivers choose route 1.
    x0 = np.array([args.N / 2])

    print("\nCongestion Game on a Network")
    print("------------------------------")
    print(f"Total drivers: {args.N}")
    print(f"Route 1: L(x) = {args.a1} * x + {args.b1}")
    print(f"Route 2: L(x) = {args.a2} * x + {args.b2}")
    print()

    # Using Steepest Descent
    print("Using Steepest Descent:")
    x_sd = steepest_descent(lambda x: congestion_potential(x, args.N, args.a1, args.b1, args.a2, args.b2),
                            x0, alpha=args.alpha, convergence_tol=args.tol, max_iter=args.max_iter)
    print(f"Equilibrium (load on route 1): {x_sd[0]:.6f}")
    print(f"Drivers on route 2: {args.N - x_sd[0]:.6f}")
    print(f"Potential value: {congestion_potential(x_sd, args.N, args.a1, args.b1, args.a2, args.b2):.6f}")
    print()

    # Using Newton's Method
    print("Using Newton's Method:")
    try:
        x_newton = newton(lambda x: congestion_potential(x, args.N, args.a1, args.b1, args.a2, args.b2),
                           x0, convergence_tol=args.tol, max_iter=args.max_iter_newton)
        print(f"Equilibrium (load on route 1): {x_newton[0]:.6f}")
        print(f"Drivers on route 2: {args.N - x_newton[0]:.6f}")
        print(f"Potential value: {congestion_potential(x_newton, args.N, args.a1, args.b1, args.a2, args.b2):.6f}")
    except ValueError as e:
        print("Newton's Method encountered an error:", e)

    # Expected Equilibrium Calculation
    expected_x = (args.a2 * args.N + args.b2 - args.b1) / (args.a1 + args.a2)
    print("\nExpected Equilibrium (analytical):")
    print(f"Load on route 1: {expected_x:.6f}")
    print(f"Load on route 2: {args.N - expected_x:.6f}")


if __name__ == '__main__':
    main()