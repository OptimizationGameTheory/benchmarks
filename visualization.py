import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize


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

def plot_auction_allocation(x, valuations, iteration, method, final=False):
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
    payments = np.maximum(0, x * valuations)

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
    X = x.reshape((n, n))
    if iteration == 0:
        plt.ion()
        fig, ax = plt.subplots()
        plot_matching_assignment.fig = fig
        plot_matching_assignment.ax = ax
        # Create the table once
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        labels = [f'Quality {i}' for i in range(n)]
        table_data = [[label] for label in labels]
        plot_matching_assignment.table = plt.table(cellText=table_data, colLabels=['Match Quality'],
                                                   cellColours=[[color] for color in colors], cellLoc='center',
                                                   loc='right', bbox=[1.1, 0.1, 0.2, 0.8])
        plot_matching_assignment.table.auto_set_font_size(False)
        plot_matching_assignment.table.set_fontsize(10)
        # Set normalization for color bar to cover the full range [0, 1]
        plot_matching_assignment.norm = Normalize(vmin=0, vmax=1)
    ax = plot_matching_assignment.ax
    ax.clear()
    cax = ax.matshow(X, cmap='viridis', norm=plot_matching_assignment.norm)
    ax.set_title(f'{method} - Iteration {iteration}' + (' (Final)' if final else ''))
    ax.set_xlabel("Women")
    ax.set_ylabel("Men")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    # Add color legend with fixed range and labels
    if iteration == 0:
        cbar = plot_matching_assignment.fig.colorbar(cax, ax=ax, orientation='vertical')
        cbar.set_label('Match Quality')
        plot_matching_assignment.cbar = cbar
    cbar = plot_matching_assignment.cbar
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'])

    if iteration % 30 == 0:
        plt.pause(0.001)
