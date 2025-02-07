# Introduction to Optimization: Optimization Techniques in Game Theoretic Problems

---
---

## Optimal Auction Design

### Overview
This part of the project investigates the application of numerical optimization techniques to the problem of optimal auction design for revenue maximization. The auctioneer seeks to allocate a divisible good among multiple bidders while satisfying key economic constraints such as individual rationality and feasibility. 

The optimization problem is addressed using:
- **Steepest Descent Method**
- **Newton's Method**

The performance of these numerical techniques is benchmarked against the **analytical optimal solution**, which strategically assigns the entire good to the highest bidder.

---

### Problem Formulation
We consider an auction setting where an auctioneer allocates a **divisible** good to $n$ bidders, each possessing a private valuation $v_i$. The auctioneer determines the allocation $x_i$ (representing the fraction of the good assigned to bidder $i$) and the corresponding payment $p_i$ imposed on each participant.

#### **Objective Function**
The objective is to maximize total revenue:
$$
R = \sum_{i=1}^{n} p_i
$$

Given that conventional optimization routines are designed for **minimization**, we recast the problem as minimizing the negative revenue:
$$
\Phi(z) = -\sum_{i=1}^n p_i + \frac{\mu}{2} \text{(Penalty Terms)}
$$
where $μ$ is a penalty parameter that enforces feasibility constraints within the optimization framework.

#### **Constraints**
The auction must adhere to fundamental economic principles, leading to the following constraints:

1. **Individual Rationality (IR):** Each bidder must derive non-negative utility:
$$
p_i \leq v_i x_i
$$
2. **Allocation Constraints:**
   - Each allocation fraction must be within the interval $[0,1]$.
   - The total allocation cannot exceed the available supply:
     $$
     \sum_{i=1}^{n} x_i \leq 1
     $$
3. **Nonnegative Payments:**
$$
p_i \geq 0
$$

---

### Analytical Optimal Solution
The optimal allocation strategy follows from classical auction theory: the entire good should be allocated to the highest bidder.

- **Let** $i^* = \argmax (v_i)$.
- Assign $x_{i^*} = 1$, $p_{i^*} = v_{i^*}$
- Set $x_i = 0$, $p_i = 0$ for all $i \neq i^*$

This results in an optimal revenue of: $R^* = \max(v_i)$

---

### Computational Implementation
#### **Dependencies**
Ensure that Python 3 and NumPy are installed:
```sh
pip install numpy
```

#### **Usage Instructions**
Execute the script from the command line:
```sh
python auction_game.py --valuations "10,20,15"
```
This command runs the optimization for three bidders with valuations `{10, 20, 15}`.

#### **Command-line Arguments**
- `--valuations`: Comma-separated list of bidder valuations (e.g., `10,20,15`)
- `--mu`: Penalty coefficient (default: `100.0`)
- `--alpha`: Step size for steepest descent (default: `0.01`)
- `--tol`: Convergence criterion (default: `1e-6`)
- `--max_iter`: Maximum iterations for steepest descent (default: `1000`)
- `--max_iter_newton`: Maximum iterations for Newton’s method (default: `100`)

Example with custom parameters:
```sh
python auction_game.py --valuations "15,25,30" --mu 200 --alpha 0.005 --tol 1e-8
```

#### **Expected Output**
- **Steepest Descent Solution**: Computed allocation `x` and payments `p`
- **Newton’s Method Solution**: Alternative computed solution, if convergence is achieved
- **Analytical Optimal Solution**: Benchmark revenue and allocations
- **Performance Analysis**: Deviation of numerical solutions from the theoretical optimum

---

### Code Architecture
```sh
├── auction_game.py           # Main execution script
├── optimization.py           # Implementation of numerical optimization methods
└── README.md                 # Documentation
```

---

### Results & Discussion
- **Penalty Function Approach:** The quadratic penalty formulation ensures soft constraint enforcement while maintaining numerical stability.
- **Steepest Descent vs. Newton’s Method:**
  - The steepest descent method demonstrates robustness but requires careful step size tuning for convergence.
  - Newton’s method converges more rapidly when the Hessian is well-conditioned, but may suffer from numerical instability in ill-conditioned cases.
- **Comparison with Analytical Solution:** The numerical methods approximate the optimal allocation but may exhibit small deviations due to penalty parameter selection and numerical precision.

---

### Future Directions
- **Adaptive Penalty Methods:** Implementing dynamic adjustment schemes for $μ$ to enhance constraint adherence.
- **Alternative Optimization Techniques:** Employing constrained solvers such as `scipy.optimize.minimize` to directly handle feasibility constraints.
- **Graphical Convergence Analysis:** Visualizing algorithmic convergence behavior across different initial conditions and parameter settings.

---
---

## Congestion Game on Networks

### Overview
This part of the project implements an optimization-based approach to solve the **Congestion Game on a Network**, where drivers choose between two routes, and congestion impacts their travel times. The objective is to find the equilibrium distribution of drivers across the two routes using numerical optimization techniques.

We use:
- **Steepest Descent Method**
- **Newton's Method**

We compare these numerical solutions to the **analytical equilibrium solution** obtained from setting the potential function's derivative to zero.

---

### Problem Formulation
#### **Network Setup**
- We consider two routes with congestion-dependent latencies:
  - **Route 1:** Latency function: $L_1(x) = a_1 x + b_1$
  - **Route 2:** Latency function: $L_2(x) = a_2 x + b_2$
- There are $N$ drivers who must be allocated between these two routes.

#### **Potential Function**
The equilibrium can be found by minimizing the **potential function**:
$$
\Phi(x) = \frac{1}{2} a_1 x^2 + b_1 x + \frac{1}{2} a_2 (N - x)^2 + b_2 (N - x)
$$
where $x$ is the number of drivers on route 1, and $N-x$ is the number on route 2.

#### **Analytical Optimal Solution**
Setting the derivative of the potential function to zero:
$$
(a_1 + a_2)x = a_2 N + b_2 - b_1
$$
Solving for $x$, we obtain the expected equilibrium:
$$
x^* = \frac{a_2 N + b_2 - b_1}{a_1 + a_2}
$$

---

### Computational Implementation
#### **Dependencies**
Ensure Python 3 and NumPy are installed:
```sh
pip install numpy
```

#### **Usage**
Run the script from the terminal:
```sh
python congestion_game.py --N 100 --a1 1.0 --b1 0.0 --a2 2.0 --b2 10.0
```

#### **Command-line Arguments**
- `--N`: Number of drivers (default: `100`)
- `--a1`, `--b1`: Parameters for route 1's latency function
- `--a2`, `--b2`: Parameters for route 2's latency function
- `--alpha`: Step size for steepest descent (default: `0.01`)
- `--tol`: Convergence tolerance (default: `1e-8`)
- `--max_iter`: Maximum iterations for steepest descent (default: `1000`)
- `--max_iter_newton`: Maximum iterations for Newton’s method (default: `100`)

---

### Code Architecture
```sh
├── congestion_game.py      # Main script (runs optimization)
├── optimization.py         # Contains steepest descent & Newton’s method
└── README.md               # Documentation
```

---

### Results & Discussion
- **Steepest Descent vs Newton’s Method:**
  - Steepest descent converges steadily but may require more iterations.
  - Newton’s method, if the Hessian is well-conditioned, converges much faster.
- **Comparison to Analytical Solution:**
  - The best solution aligns with the analytical equilibrium.
  - The error between computed and theoretical equilibrium is minimal.

---

### Future Directions
- **Adaptive Step Sizes:** Improve steepest descent convergence.
- **Graphical Analysis:** Visualize convergence and solution distributions.
- **Multi-route Generalization:** Extend the model to more than two routes.

---
---

## Stable Matching Game

### Overview
This part of the project applies continuous optimization techniques to solve the **Stable Matching Game**. We formulate a continuous relaxation of the classic assignment (or stable matching) problem by defining a potential function that integrates the matching cost with penalty terms enforcing one-to-one matching constraints and feasibility bounds. Two optimization methods—**Steepest Descent** and **Newton's Method**—are employed to minimize the potential function. The obtained solutions are then compared with the analytical optimal matching computed via the Hungarian algorithm.

---

### Problem Formulation
We consider a matching market consisting of **n men** and **n women**. A cost matrix $C$ of size $n \times n$ is provided, where $C[i,j]$ represents the cost (or dissatisfaction) of matching man $i$ with woman $j$. Our goal is to determine a matching that minimizes the total cost subject to the following conditions:

- **Assignment Constraints:** Each man must be matched to exactly one woman, and each woman to exactly one man.
- **Feasibility Bounds:** Each entry of the assignment matrix $X$ (with $X[i,j]$ indicating the degree of matching) must lie in the interval $[0, 1]$.

To solve this, we define a potential function $\Phi(x)$ over a flattened vector $x$ (which corresponds to the matrix $X$) as follows:

$$
\Phi(x) = \langle C, X \rangle + \frac{\lambda}{2} \|x\|^2 + \frac{\mu}{2}\Bigg[ \sum_{i=1}^{n} \Big(\sum_{j=1}^{n} X_{ij} - 1\Big)^2 + \sum_{j=1}^{n} \Big(\sum_{i=1}^{n} X_{ij} - 1\Big)^2 + \sum_{i,j}\Big(\max(0, -X_{ij})^2 + \max(0, X_{ij}-1)^2\Big) \Bigg]
$$

Here, $\langle C, X \rangle$ denotes the total matching cost, $\lambda$ is a regularization parameter ensuring strict convexity, and $\mu$ is a penalty parameter enforcing the assignment and bounds constraints.

---

### Analytical Optimal Solution
The classical assignment problem can be solved optimally using the Hungarian algorithm. This method computes the binary assignment matrix $X^*$ that minimizes the total cost $\langle C, X^* \rangle$ while satisfying the one-to-one matching constraints. This optimal solution serves as a benchmark for our continuous optimization approach.

---

### Computational Implementation
The implementation comprises:
- **Continuous Optimization:** The potential function is minimized using two numerical methods:
  - **Steepest Descent:** Iteratively updates the solution in the direction of the negative gradient.
  - **Newton's Method:** Utilizes second-order (Hessian) information to achieve faster convergence when feasible.
- **Penalty Method:** Constraint violations (row sums, column sums, and variable bounds) are penalized quadratically.
- **Optimal Matching:** The Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`) is used to compute the analytical optimal solution.

The code accepts input parameters such as the number of agents $n$, an optional CSV file for the cost matrix, penalty and regularization parameters, and optimization parameters (step size, tolerance, and maximum iterations).

#### Dependencies

To run the stable matching game optimization, you need Python 3 and the following dependencies:

- **NumPy** (`pip install numpy`) – for numerical computations
- **SciPy** (`pip install scipy`) – for optimization routines (Hungarian algorithm)
- **argparse** (included in Python standard library) – for command-line argument parsing

Ensure you have these installed before running the script.

---

#### Usage

Run the script from the command line using:

```sh
python matching_game.py --n <num_agents> --cost_file <path_to_csv> [options]
```

##### Example 1: Run with a randomly generated cost matrix (default size 5x5)

```sh
python matching_game.py
```

##### Example 2: Run with a custom cost matrix from a CSV file

If you want to use a custom cost matrix, you can run the following command:

```sh
python matching_game.py --n 4 --cost_file cost_matrix.csv
```

An example `cost_matrix.csv` ($4\times4$ matrix) is as follows:

```csv
3.2, 1.5, 4.8, 2.0
2.1, 3.0, 1.2, 4.5
4.5, 2.3, 3.7, 1.8
1.0, 4.2, 2.5, 3.3
```
Note that Each row corresponds to a man, each column corresponds to a woman, and each cell $(i, j)$ contains the cost of matching man $i$ with woman $j$. The file has exactly $n$ rows and $n$ columns (where $n$ matches the `--n` argument). Values are separated by commas (,). No extra spaces or blank lines are present in the file.

##### Example 3: Adjust penalty and optimization parameters

```sh
python matching_game.py --n 6 --mu 2000 --lam 0.2 --alpha 0.0005 --max_iter 5000
```

#### Command-Line Arguments

- `--n`: Number of men/women in the matching problem (i.e., matrix size $n \times n$). (default: `5`)
- `--cost_file`: Path to a CSV file containing the cost matrix. If not provided, a random matrix is generated.
- `--mu`: Penalty parameter enforcing assignment constraints. (default: `1000.0`)
- `--lam`: Regularization parameter ensuring strict convexity. (default: `0.01`)
- `--alpha`: Step size for the steepest descent method. (default: `0.001`)
- `--tol`: Convergence tolerance for stopping the optimization. (default: `1e-6`)
- `--max_iter`: Maximum iterations for steepest descent (default: `2000`)
- `--max_iter_newton`: Maximum iterations for Newton’s method (default: `100`)
- `--seed`: Random seed for reproducibility when generating cost matrices. (default: `42`)

---

### Code Architecture
```sh
├── matching_game.py      # Main script (runs the optimization for the stable matching game)
├── optimization.py       # Contains implementations of steepest descent & Newton's method
└── README.md             # This documentation
```

---

### Results & Discussion

- **Numerical Methods:** Both steepest descent and Newton's method yield continuous assignment matrices that approximate the optimal matching. In practice, these continuous solutions are close to binary values due to the heavy penalty on constraint violations.
- **Comparison:** The linear cost term computed from the numerical solutions is compared with the optimal cost from the Hungarian algorithm. Typically, the potential function minimizers achieve a linear cost near the optimal cost, validating the continuous formulation.
- **Trade-offs:** Steepest descent is robust but may require a carefully tuned step size, while Newton's method can converge faster provided that the Hessian is well-conditioned.

---

### Future Directions

- **Parameter Tuning:** Investigate adaptive strategies for adjusting the penalty $\mu$ and regularization $\lambda$ parameters.
- **Algorithm Enhancements:** Explore advanced optimization algorithms (e.g., quasi-Newton methods or constrained optimization solvers) to improve convergence.
- **Extension to Larger Markets:** Generalize the approach to handle larger matching markets and many-to-many matchings.
Post-processing: Develop effective rounding techniques to convert continuous solutions into feasible binary matchings.

---
