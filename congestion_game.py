import numpy as np
import argparse
from optimization import steepest_descent, newton


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
    potential_route1 = 0.5 * a1 * x_val ** 2 + b1 * x_val
    potential_route2 = 0.5 * a2 * x2 ** 2 + b2 * x2
    return potential_route1 + potential_route2


def solve_congestion_game(N, a1, b1, a2, b2, alpha, tol, max_iter, max_iter_newton):
    x0 = np.array([N / 2])

    print("\nCongestion Game on a Network")
    print("------------------------------")
    print(f"Total drivers: {N}")
    print(f"Route 1: L(x) = {a1} * x + {b1}")
    print(f"Route 2: L(x) = {a2} * x + {b2}")
    print()

    # Using Steepest Descent
    print("Using Steepest Descent:")
    x_sd = steepest_descent(lambda x: congestion_potential(x, N, a1, b1, a2, b2), x0, alpha=alpha, convergence_tol=tol,
                            max_iter=max_iter, visualize=True, N=N, game_type='congestion')
    print(f"Equilibrium (load on route 1): {x_sd[0]:.6f}")
    print(f"Drivers on route 2: {N - x_sd[0]:.6f}")
    print(f"Potential value: {congestion_potential(x_sd, N, a1, b1, a2, b2):.6f}")
    print()

    # Using Newton's Method
    print("Using Newton's Method:")
    try:
        x_newton = newton(lambda x: congestion_potential(x, N, a1, b1, a2, b2), x0, convergence_tol=tol,
                          max_iter=max_iter_newton, visualize=True, N=N, game_type='congestion')
        print(f"Equilibrium (load on route 1): {x_newton[0]:.6f}")
        print(f"Drivers on route 2: {N - x_newton[0]:.6f}")
        print(f"Potential value: {congestion_potential(x_newton, N, a1, b1, a2, b2):.6f}")
    except ValueError as e:
        print("Newton's Method encountered an error:", e)

    # Expected Equilibrium Calculation
    expected_x = (a2 * N + b2 - b1) / (a1 + a2)
    print("\nExpected Equilibrium (analytical):")
    print(f"Load on route 1: {expected_x:.6f}")
    print(f"Load on route 2: {N - expected_x:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Solve the Congestion Game using Optimization Techniques.")
    args = parser.parse_args()
    solve_congestion_game(args.N, args.a1, args.b1, args.a2, args.b2, args.alpha, args.tol, args.max_iter, args.max_iter_newton)


if __name__ == '__main__':
    main()