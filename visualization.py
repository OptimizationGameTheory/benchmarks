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


def plot_auction_allocation(z, v, iteration, method, final=False):
    if iteration == 0:
        plt.ion()
        fig, ax1 = plt.subplots()
        plot_auction_allocation.fig = fig
        plot_auction_allocation.ax1 = ax1
    ax1 = plot_auction_allocation.ax1
    ax1.clear()
    n = len(v)
    x = z[:n]
    p = z[n:]

    color = 'tab:blue'
    ax1.set_xlabel('Bidders')
    ax1.set_ylabel('Allocations', color=color)
    ax1.bar(range(n), x, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Payments', color=color)
    ax2.plot(range(n), p, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'{method} - Iteration {iteration}' + (' (Final)' if final else ''))
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
    cax = ax.matshow(X, cmap='viridis')
    # fig.colorbar(cax)
    ax.set_title(f'{method} - Iteration {iteration}' + (' (Final)' if final else ''))
    ax.set_xlabel("Women")
    ax.set_ylabel("Men")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    plt.pause(0.2)