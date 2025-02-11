import matplotlib.pyplot as plt


def plot_congestion_distribution(x, iteration, method, N, final=False):
    if iteration == 0:
        plt.ion()
        fig, ax = plt.subplots()
        plot_congestion_distribution.fig = fig
        plot_congestion_distribution.ax = ax
    ax = plot_congestion_distribution.ax
    ax.clear()
    route1_drivers = x[0]
    route2_drivers = N - x[0]
    ax.bar(['Route 1', 'Route 2'], [route1_drivers, route2_drivers], color=['blue', 'orange'])
    ax.set_xlabel('Routes')
    ax.set_ylabel('Number of Drivers')
    ax.set_title(f'{method} - Iteration {iteration}' + (' (Final)' if final else ''))
    ax.set_ylim(0, N)
    plt.pause(0.005)


def plot_auction_allocation(z, valuations, iteration, method, final=False):
    if iteration == 0:
        plt.ion()
        fig, ax1 = plt.subplots()
        plot_auction_allocation.fig = fig
        plot_auction_allocation.ax1 = ax1
        plot_auction_allocation.ax2 = ax1.twinx()
    ax1 = plot_auction_allocation.ax1
    ax2 = plot_auction_allocation.ax2
    ax1.clear()
    ax2.clear()
    n = len(valuations)
    x = z[:n]
    payments = z[n:]

    color = 'tab:blue'
    ax1.bar(range(n), x, color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:orange'
    ax2.plot(range(n), payments, color=color, marker='o', linestyle='-', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_title(f'{method} - Iteration {iteration}' + (' (Final)' if final else ''))
    ax1.set_xticks(range(n))
    ax1.set_xticklabels([f'Bidder {i+1}' for i in range(n)])
    ax1.set_xlim(-0.5, n - 0.5)

    ax1.legend(['Allocations'], loc='upper left')
    ax2.legend(['Payments'], loc='upper right')
    plt.tight_layout()
    plt.pause(0.5)


def plot_matching_assignment(x, n, iteration, method, final=False):
    if iteration == 0:
        plt.ion()
        fig, ax = plt.subplots()
        plot_matching_assignment.fig = fig
        plot_matching_assignment.ax = ax
    ax = plot_matching_assignment.ax
    # fig = plot_matching_assignment.fig
    ax.clear()
    X = x.reshape((n, n))
    ax.matshow(X, cmap='viridis')
    ax.set_title(f'{method} - Iteration {iteration}' + (' (Final)' if final else ''))
    ax.set_xlabel("Women")
    ax.set_ylabel("Men")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    plt.pause(0.2)