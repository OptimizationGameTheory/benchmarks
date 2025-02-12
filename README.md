# Introduction to Optimization: Optimization Techniques in Game Theoretic Problems

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Code Architecture](#code-architecture)
4. [Theory and Implementations](#theory-and-implementations)
5. [Results and Discussion](#results-and-discussion)
6. [Future Directions](#future-directions)

## Overview

This repository presents a collection of optimization-driven implementations for classical game theoretic problems. The project focuses on three key problems:

- **Auction Design:** Optimizing revenue in auctions by allocating a divisible good among multiple bidders while enforcing individual rationality and allocation constraints.

- **Congestion on Networks:** Modeling route-choice behavior among drivers with congestion-dependent latency functions to determine an equilibrium distribution.

- **Stable Matching:** Formulating a continuous relaxation of the assignment problem to compute near-optimal matchings between two sets (e.g., men and women) while enforcing one-to-one pairing constraints.

Each problem is formulated as a continuous optimization task where objective functions are minimized using numerical techniques—primarily the Steepest Descent and Newton's Methods. Analytical solutions or benchmarks (such as the Hungarian algorithm for stable matching) are used to validate the numerical results. This project is intended for researchers and practitioners interested in the interplay between optimization theory and game theory.

## Dependencies

### Core Requirements
| Package | Version |
|---------|---------|
| Python  | ≥3.8    |
| NumPy   | ≥1.21.0 |
| SciPy   | ≥1.7.0  |
| pandas  | ≥3.10.0 |
| Matplotlib | ≥2.2.3 |

To fulfill all the requirements, after installing Python 3.8, install the packages with:

```bash
pip3 install numpy scipy pandas matplotlib
```

## Code Architecture

```sh
optimization-game-theory/
├── auction_game.py        # Auction design implementation
├── congestion_game.py     # Congestion on networks implementation
├── main.py                # The driver code for the results
├── matching_game.py       # Stable matching implementation
├── optimization.py        # Shared optimization routines
├── README.md              # Project documentation
└── visualization.py       # Visualization tools of the project
```

The `main.py` script serves as the central driver for the project, coordinating the execution of different game-theoretic experiments based on command-line arguments. It directs the flow of execution by calling the appropriate optimization routines from `optimization.py` and subsequently invokes `visualization.py` to generate convergence plots and other relevant visual outputs. This modular design ensures a reproducible workflow and facilitates a clear demonstration of the algorithmic performance across various problem settings.

## Theory and Implementations

### Problem I: Auction Design

#### Problem Formulation

We consider an auction setting where an auctioneer allocates a **divisible** good to *n* bidders, each possessing a private valuation $v_i$. The auctioneer determines the allocation $x_i$ (the fraction of the good assigned to bidder $i$) and the corresponding payment $p_i$. The objective is to maximize total revenue  
$$
R = \sum_{i=1}^{n} p_i
$$  
Since conventional optimization routines are designed for minimization, the problem is reformulated by minimizing the negative revenue with added penalty terms to enforce feasibility constraints:  
$$
\Phi(z) = -\sum_{i=1}^{n} p_i + \frac{\mu}{2} \text{ (Penalty Terms)}
$$
where $\mu$ is a penalty parameter enforcing the constraints.

The design also imposes key constraints:

- **Individual Rationality:** $p_i \leq v_i \, x_i$ for each bidder.  

- **Allocation Constraints:** Each allocation $x_i$ must be in $[0,1]$ and the total allocation must satisfy $\sum_{i=1}^{n} x_i \leq 1$.

- **Nonnegative Payments:** $p_i \geq 0$.  
The analytical optimal solution is obtained by allocating the entire good to the bidder with the highest valuation, i.e., if $i^\ast = \arg\max(v_i)$ then set $x_{i^\ast} = 1$ and $p_{i^\ast} = v_{i^\ast}$ (with all other $x_i$ and $p_i$ set to zero), yielding $R^\ast = \max(v_i)$.

Numerical optimization is performed using both the Steepest Descent and Newton’s Methods. These routines iteratively minimize the potential function, with parameters such as the step size (`alpha`), convergence tolerance (`tol`), and maximum iterations adjusted to ensure reliable convergence. The numerical solution is then compared against the analytical optimal allocation to evaluate performance.

#### Implementation Usage

Execute the script from the command line. For example, to run the auction design for three bidders with valuations 10, 20, and 15, use:  

```sh
python auction_game.py --valuations "10,20,15"
```  

Command-line arguments include: 

- `--valuations`: Comma-separated list of bidder valuations (e.g., `"10,20,15"`).  
- `--mu`: Penalty coefficient (default: `100.0`).  
- `--alpha`: Step size for steepest descent (default: `0.01`).  
- `--tol`: Convergence criterion (default: `1e-6`).  
- `--max_iter`: Maximum iterations for steepest descent (default: `1000`).  
- `--max_iter_newton`: Maximum iterations for Newton’s method (default: `100`).  

For example, to use custom parameters, run:  

```sh
python auction_game.py --valuations "15,25,30" --mu 200 --alpha 0.005 --tol 1e-8
```  

---

### Problem II: Congestion on Networks

#### Problem Formulation

We consider a congestion game on a network with two routes. Route 1 has a latency function given by  
$$
L_1(x) = a_1 x + b_1,
$$  
and Route 2 has  
$$
L_2(x) = a_2 x + b_2.
$$  
There are $N$ drivers who must be allocated between these routes. The equilibrium is determined by minimizing the potential function:  
$$
\Phi(x) = \frac{1}{2} a_1 x^2 + b_1 x + \frac{1}{2} a_2 (N - x)^2 + b_2 (N - x),
$$  
where $x$ represents the number of drivers on Route 1 (and $N - x$ on Route 2). The analytical equilibrium is obtained by setting the derivative to zero, leading to  
$$
(a_1 + a_2)x = a_2 N + b_2 - b_1,
$$  
and therefore,  
$$
x^\ast = \frac{a_2 N + b_2 - b_1}{a_1 + a_2}.
$$

The potential function is minimized using numerical optimization techniques—specifically, the Steepest Descent and Newton’s Methods—to determine the optimal distribution of drivers. As with the auction design, penalty functions and convergence parameters (`alpha`, `tol`, `max_iter`, etc.) are tuned to ensure that the algorithm converges to the analytical equilibrium. The computed equilibrium distribution is then compared with the expected analytical result.

#### Implementation Usage

Run the congestion game script from the terminal. For example, to simulate 100 drivers with specified latency parameters, execute:

```sh
python congestion_game.py --N 100 --a1 1.0 --b1 0.0 --a2 2.0 --b2 10.0
``` 

Command-line arguments include:

- `--N`: Number of drivers (default: `100`).  
- `--a1`, `--b1`: Parameters for Route 1's latency function.  
- `--a2`, `--b2`: Parameters for Route 2's latency function.  
- `--alpha`: Step size for steepest descent (default: `0.01`).  
- `--tol`: Convergence tolerance (default: `1e-8`).  
- `--max_iter`: Maximum iterations for steepest descent (default: `1000`).  
- `--max_iter_newton`: Maximum iterations for Newton’s method (default: `100`).

---

### Problem III: Stable Matching

#### Problem Formulation

We consider a matching market with **n men** and **n women**. A cost matrix $C$ of size $n \times n$ is provided, where $C[i,j]$ represents the cost (or dissatisfaction) of matching man $i$ with woman $j$. The goal is to minimize the total matching cost subject to assignment constraints (each man is matched to exactly one woman, and vice versa) and feasibility bounds (each element $X[i,j]$ of the assignment matrix must lie within $[0, 1]$). The continuous formulation defines the potential function as:  
$$
\Phi(x) = \langle C, X \rangle + \frac{\lambda}{2} \|x\|^2 + \frac{\mu}{2}\Biggl[ \sum_{i=1}^{n} \Bigl(\sum_{j=1}^{n} X_{ij} - 1\Bigr)^2 + \sum_{j=1}^{n} \Bigl(\sum_{i=1}^{n} X_{ij} - 1\Bigr)^2 + \sum_{i,j}\Bigl(\max(0, -X_{ij})^2 + \max(0, X_{ij}-1)^2\Bigr) \Biggr],
$$  
where $\langle C, X \rangle$ denotes the total matching cost, $\lambda$ is a regularization parameter ensuring strict convexity, and $\mu$ is a penalty parameter enforcing the matching constraints. The analytical optimal solution is obtained via the Hungarian algorithm, which computes a binary assignment matrix $X^\ast$ that minimizes $\langle C, X^\ast \rangle$ under the one-to-one matching constraints.

The potential function for the stable matching problem is minimized using numerical optimization techniques—Steepest Descent and Newton’s Methods. The implementation integrates penalty terms to enforce the assignment and bound constraints, and uses a regularization parameter $\lambda$ for convexity. The continuous solution is then compared against the optimal binary matching computed using the Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`).

#### Implementation Usage

Run the stable matching script from the command line. For instance, to run with a randomly generated $5\times5$ cost matrix, simply execute:

```sh
python matching_game.py
```

To run with a custom cost matrix from a CSV file (where each row represents a man and each column a woman), use:

```sh
python matching_game.py --n 4 --cost_file cost_matrix.csv
```

An example `cost_matrix.csv` might contain:

```csv
3.2, 1.5, 4.8, 2.0
2.1, 3.0, 1.2, 4.5
4.5, 2.3, 3.7, 1.8
1.0, 4.2, 2.5, 3.3
```

Additional command-line arguments include:

- `--n`: Number of men/women (matrix size, default: `5`).  
- `--cost_file`: Path to a CSV file containing the cost matrix.  
- `--mu`: Penalty parameter enforcing assignment constraints (default: `1000.0`).  
- `--lam`: Regularization parameter ensuring strict convexity (default: `0.01`).  
- `--alpha`: Step size for steepest descent (default: `0.001`).  
- `--tol`: Convergence tolerance (default: `1e-6`).  
- `--max_iter`: Maximum iterations for steepest descent (default: `2000`).  
- `--max_iter_newton`: Maximum iterations for Newton’s method (default: `100`).  
- `--seed`: Random seed for reproducibility (default: `42`).

For example, to adjust parameters, run:

```sh
python matching_game.py --n 6 --mu 2000 --lam 0.2 --alpha 0.0005 --max_iter 5000
```

## Results and Discussion

Our experimental analysis indicates that the effectiveness of the optimization methods varies with the characteristics of the underlying problem. In the Auction Design problem, the objective function is highly sensitive to the enforcement of economic constraints via penalty terms. Both the steepest descent and Newton’s methods can converge to a solution that approximates the analytical optimum; however, steepest descent tends to be more robust in scenarios where the penalty terms introduce mild nonlinearity, while Newton’s method may deliver faster convergence when the Hessian is well-conditioned. Consequently, careful parameter tuning is essential in auction design to balance convergence speed against numerical stability.

For the Congestion on Networks problem, the potential function exhibits a clear quadratic structure. In such settings, Newton’s method benefits from the strong curvature information provided by the Hessian and can converge rapidly. Nevertheless, steepest descent offers a more straightforward implementation with robust performance, albeit at the cost of increased iterations. The inherent structure of the congestion problem renders both methods effective, with the choice largely depending on the computational resources available and the desired convergence rate.

In contrast, the Stable Matching problem poses unique challenges due to the near-linear nature of its potential function. Under these conditions, the reliance of Newton’s method on second-order derivative information becomes less beneficial, leading to instability or non-convergence. Steepest descent, although generally slower and more sensitive to parameter choices, demonstrates a higher degree of robustness for this problem class. However, the continuous relaxation inherent in the formulation can lead to significant approximation errors when compared to the discrete optimal solution. Overall, the matching problem appears to favor methods that prioritize stability over rapid convergence.

## Future Directions

Advancing the current framework can be approached along several complementary axes:

- **Parameter Tuning:**
Future work will explore adaptive strategies for dynamically adjusting penalty parameters ($\mu$) and regularization terms ($\lambda$). Such adaptive schemes could automatically balance the trade-off between constraint enforcement and objective minimization, thereby enhancing convergence rates and solution accuracy.

- **Algorithmic Enhancements:**
The incorporation of advanced optimization techniques is a promising avenue. Implementing quasi-Newton methods (e.g., BFGS and L-BFGS) and integrating trust-region frameworks could provide improved convergence behavior, particularly in problems where the Hessian information is unreliable. Additionally, exploring variants of stochastic gradient descent may prove beneficial in handling large-scale or noisy optimization scenarios.

- **Model Extensions:**
Extending the current models to more complex real-world applications is an important direction. Potential extensions include multi-item auctions with combinatorial bids, network games that involve multiple origin-destination pairs, and many-to-many matching markets. Moreover, developing robust post-processing methods to round continuous solutions into feasible binary assignments will enhance the practical applicability of the matching framework.

- **Computational Enhancements:**
To address scalability issues, future efforts could focus on parallel implementations and GPU acceleration (e.g., using CuPy) to manage large-scale problems more efficiently. Such computational improvements would allow the framework to be applied to higher-dimensional and more computationally intensive problems without compromising performance.