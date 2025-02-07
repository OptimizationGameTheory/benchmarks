# Introduction to Optimization: Optimization Techniques in Game Theoretic Problems

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
python auction_optimization.py --valuations "10,20,15"
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
python auction_optimization.py --valuations "15,25,30" --mu 200 --alpha 0.005 --tol 1e-8
```

#### **Expected Output**
- **Steepest Descent Solution**: Computed allocation `x` and payments `p`
- **Newton’s Method Solution**: Alternative computed solution, if convergence is achieved
- **Analytical Optimal Solution**: Benchmark revenue and allocations
- **Performance Analysis**: Deviation of numerical solutions from the theoretical optimum

---

### Code Architecture
```sh
├── auction_optimization.py   # Main execution script
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

## Congestion Game on Networks

### Overview
This project implements an optimization-based approach to solve the **Congestion Game on a Network**, where drivers choose between two routes, and congestion impacts their travel times. The objective is to find the equilibrium distribution of drivers across the two routes using numerical optimization techniques.

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
