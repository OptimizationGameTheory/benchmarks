import numpy as np
from matplotlib import pyplot as plt
from visualization import plot_congestion_distribution, plot_auction_allocation, plot_matching_assignment

def gradient(f, x, h=1e-5):
    """
    Computes the numerical gradient of a function f at point x using finite differences.
    
    Parameters:
        f (function): Function whose gradient is to be computed.
        x (numpy array): Point at which the gradient is computed.
        h (float): Step size for finite differences.
    
    Returns:
        numpy array: Computed gradient.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_h = np.copy(x)
        x_h[i] += h
        grad[i] = (f(x_h) - f(x)) / h
    return grad

def hessian(f, x, h=1e-5):
    """
    Computes the numerical Hessian matrix of a function f at point x using finite differences.
    
    Parameters:
        f (function): Function whose Hessian is to be computed.
        x (numpy array): Point at which the Hessian is computed.
        h (float): Step size for finite differences.
    
    Returns:
        numpy array: Computed Hessian matrix.
    """
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

def visualize_game(x, iteration, method, N, game_type, final=False, valuations=None):
    """
    Visualizes the optimization progress at each iteration.
    
    Parameters:
        x (numpy array): Current state of the system.
        iteration (int): Current iteration number.
        method (str): Optimization method (Steepest Descent or Newton's Method).
        N (int): Problem size.
        game_type (str): Type of game being solved.
        final (bool): Flag to indicate final iteration visualization.
        valuations (numpy array, optional): Bidder valuations (for auction game).
    """
    if game_type == 'congestion':
        plot_congestion_distribution(x, iteration, method, N, final)
    elif game_type == 'auction':
        plot_auction_allocation(x, valuations, iteration, method, final)
    elif game_type == 'matching':
        plot_matching_assignment(x, N, iteration, method, final)
    
    if iteration % 10 == 0 or final:
        plt.savefig(f"visualizations/{game_type}_{method}_iteration_{iteration}.png")

def backtracking_line_search(f, x, grad, alpha=0.01, rho=0.5, c=1e-4, max_iter=10):
    """
    Performs backtracking line search to determine an appropriate step size.
    
    Parameters:
        f (function): Objective function.
        x (numpy array): Current state.
        grad (numpy array): Gradient at x.
        alpha (float): Initial step size.
        rho (float): Reduction factor for step size.
        c (float): Tolerance parameter.
        max_iter (int): Maximum iterations.
    
    Returns:
        float: Optimized step size.
    """
    for _ in range(max_iter):
        if f(x - alpha * grad) > f(x) - c * alpha * np.dot(grad, grad):
            alpha *= rho
        else:
            break
    return alpha

def steepest_descent(f, x0, alpha=0.1, grad_function=gradient, convergence_tol=1e-6, max_iter=1000, visualize=False,
                     N=None, valuations=None, game_type=None, projection=None):
    """
    Implements the Steepest Descent optimization method.
    
    Parameters:
        f (function): Objective function to minimize.
        x0 (numpy array): Initial guess for the optimization.
        alpha (float): Step size for gradient descent (if negative, backtracking is enabled).
        grad_function (function): Function to compute gradient.
        convergence_tol (float): Convergence tolerance for stopping criterion.
        max_iter (int): Maximum number of iterations.
        visualize (bool): Whether to visualize iterations.
        N (int, optional): Problem size (used for visualization).
        valuations (numpy array, optional): Additional parameter for auction game visualization.
        game_type (str, optional): Type of game being solved.
        projection (function, optional): Projection function to maintain feasibility.
    
    Returns:
        numpy array: Optimized solution.
    """
    should_backtrack = (alpha < 0)
    x = np.copy(x0)
    history = []
    converged = False
    
    for i in range(max_iter):
        history.append(x)
        if visualize and N is not None:
            visualize_game(x, i, "Steepest Descent", N=N, valuations=valuations, game_type=game_type, final=converged)
        
        if converged:
            print(f"Converged in {i} iterations")
            return x
        
        grad = grad_function(f, x)
        
        if should_backtrack:
            alpha = backtracking_line_search(f, x, grad)
        
        x_new = x - alpha * grad
        
        if projection is not None:
            x_new = projection(x_new)
        
        if np.linalg.norm(x_new - x) < convergence_tol:
            converged = True
        
        x = x_new
    
    print("Reached maximum iterations")
    if visualize and N is not None:
        visualize_game(x, max_iter, "Steepest Descent", N=N, valuations=valuations, game_type=game_type, final=True)
    
    return x

def newton(f, x0, grad_function=gradient, hessian_function=hessian, convergence_tol=1e-6, max_iter=100, visualize=False,
           valuations=None, N=None, game_type=None, regularization=0.0, projection=None):
    """
    Implements Newton's method for optimization.
    
    Parameters:
        f (function): Objective function to minimize.
        x0 (numpy array): Initial guess for the optimization.
        grad_function (function): Function to compute gradient.
        hessian_function (function): Function to compute Hessian.
        convergence_tol (float): Convergence tolerance for stopping criterion.
        max_iter (int): Maximum number of iterations.
        visualize (bool): Whether to visualize iterations.
        valuations (numpy array, optional): Additional parameter for auction game visualization.
        game_type (str, optional): Type of game being solved.
        regularization (float, optional): Regularization term for Hessian matrix to avoid singularity.
        projection (function, optional): Projection function to maintain feasibility.
    
    Returns:
        numpy array: Optimized solution.
    """
    x = np.copy(x0)
    history = []
    converged = False
    
    for i in range(max_iter):
        history.append(x)
        if visualize and N is not None:
            visualize_game(x, i, "Newton Method", N=N, valuations=valuations, game_type=game_type, final=converged)
        
        if converged:
            print(f"Converged in {i} iterations")
            return x
        
        grad = grad_function(f, x)
        hess = hessian_function(f, x) + regularization * np.eye(len(x))
        
        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            raise ValueError("Hessian is not invertible. No solution found.")
        
        x_new = x - np.dot(hess_inv, grad)
        
        if projection is not None:
            x_new = projection(x_new)
        
        if np.linalg.norm(x_new - x) < convergence_tol:
            converged = True
        
        x = x_new
    
    print("Reached maximum iterations.")
    if visualize and N is not None:
        visualize_game(x, max_iter, "Newton Method", N=N, valuations=valuations, game_type=game_type, final=True)
    
    return x
