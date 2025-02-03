import numpy as np

n = 3
valuations = np.array([0.15, 0.7, 0.9])
x = np.ones(n) / n
alpha = 0.01
num_iterations = 1000


def utility(v, x, p):
    return v * x - p


def revenue(x):
    return np.sum(np.minimum(x * valuations, valuations))


def revenue_gradient(x):
    grad = np.zeros(n)
    for i in range(n):
        if x[i] * valuations[i] <= valuations[i]:
            grad[i] = valuations[i]
        else:
            grad[i] = 0
    return grad


for iteration in range(num_iterations):
    grad = revenue_gradient(x)
    x = x + alpha * grad

    # Projection onto the feasible set (non-negative allocations and sum <= 1)
    x = np.maximum(0, x)
    if np.sum(x) > 1:
        x = x / np.sum(x)

    p = np.minimum(x * valuations, valuations)
    for i in range(n):
        p = np.minimum(x * valuations, valuations)
        if utility(valuations[i], x[i], p[i]) < 0:
            x[i] = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                if utility(valuations[i], x[i], p[i]) < utility(valuations[i], x[j], p[j]):
                    x[i] = 0

optimal_payment = np.minimum(x * valuations, valuations)

print("Allocation:", x)
print("Payment:", optimal_payment)
print("Revenue:", revenue(x))
