import argparse

import numpy as np

from congestion_game import solve_congestion_game
from auction_game import main as auction_main
from matching_game import solve_matching_game, load_cost_matrix


def add_parser_args(parser):
    subparsers = parser.add_subparsers(dest="game", help="The game to solve")
    # Congestion game arguments
    congestion_parser = subparsers.add_parser('congestion', help='Solve the congestion game')
    congestion_parser.add_argument('--N', type=int, default=100, help='Total number of drivers')
    congestion_parser.add_argument('--a1', type=float, default=1.0, help='Coefficient for latency function of route 1')
    congestion_parser.add_argument('--b1', type=float, default=0.0, help='Intercept for latency function of route 1')
    congestion_parser.add_argument('--a2', type=float, default=2.0, help='Coefficient for latency function of route 2')
    congestion_parser.add_argument('--b2', type=float, default=10.0, help='Intercept for latency function of route 2')
    congestion_parser.add_argument('--alpha', type=float, default=0.01, help='Step size for steepest descent')
    congestion_parser.add_argument('--tol', type=float, default=1e-8, help='Convergence tolerance')
    congestion_parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations for steepest descent')
    congestion_parser.add_argument('--max_iter_newton', type=int, default=100,
                                   help='Maximum iterations for Newton’s method')
    # Auction game arguments
    auction_parser = subparsers.add_parser('auction', help='Solve the auction game')
    auction_parser.add_argument('--valuations', type=str, required=True,
                                help="Comma separated bidder valuations, e.g. '10,20,15'")
    auction_parser.add_argument('--mu', type=float, default=10.0, help="Penalty parameter mu (default: 10.0)")
    auction_parser.add_argument('--alpha', type=float, default=0.001,
                                help="Step size for steepest descent (default: 0.001)")
    auction_parser.add_argument('--tol', type=float, default=1e-6, help="Convergence tolerance (default: 1e-6)")
    auction_parser.add_argument('--max_iter', type=int, default=1000,
                                help="Maximum iterations for steepest descent (default: 1000)")
    auction_parser.add_argument('--max_iter_newton', type=int, default=100,
                                help="Maximum iterations for Newton's method (default: 100)")
    # Matching game arguments
    matching_parser = subparsers.add_parser('matching', help='Solve the matching game')
    matching_parser.add_argument('--n', type=int, default=5, help='Number of men/women (default: 5)')
    matching_parser.add_argument('--cost_file', type=str, default=None,
                                 help='Path to CSV file containing the cost matrix (optional)')
    matching_parser.add_argument('--mu', type=float, default=1000.0, help='Penalty parameter mu (default: 1000.0)')
    matching_parser.add_argument('--lam', type=float, default=0.1,
                                 help='Regularization parameter lambda (default: 0.1)')
    matching_parser.add_argument('--alpha', type=float, default=0.001,
                                 help='Step size for steepest descent (default: 0.001)')
    matching_parser.add_argument('--tol', type=float, default=1e-6, help='Convergence tolerance (default: 1e-6)')
    matching_parser.add_argument('--max_iter', type=int, default=2000,
                                 help='Maximum iterations for steepest descent (default: 2000)')
    matching_parser.add_argument('--max_iter_newton', type=int, default=100,
                                 help='Maximum iterations for Newton’s method (default: 100)')
    matching_parser.add_argument('--seed', type=int, default=42,
                                 help='Random seed for cost matrix generation (default: 42)')


def main():
    parser = argparse.ArgumentParser(description="Solve various games using optimization techniques.")
    add_parser_args(parser)
    args = parser.parse_args()

    if args.game == 'congestion':
        solve_congestion_game(args.N, args.a1, args.b1, args.a2, args.b2, args.alpha, args.tol, args.max_iter,
                              args.max_iter_newton)
    elif args.game == 'auction':
        auction_main(args)
    elif args.game == 'matching':
        if args.cost_file:
            cost_matrix = load_cost_matrix(args.cost_file, args.n)
        else:
            np.random.seed(args.seed)
            cost_matrix = np.random.uniform(0, 10, size=(args.n, args.n))
        solve_matching_game(args.n, cost_matrix, args.mu, args.lam, args.alpha, args.tol, args.max_iter,
                            args.max_iter_newton)


if __name__ == '__main__':
    main()
