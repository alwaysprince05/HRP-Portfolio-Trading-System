"""
visualize.py
Module for plotting results and heatmaps.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_all_together(cumrets_dict, drawdowns_dict, corr):
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    # Cumulative Returns
    for label, cumrets in cumrets_dict.items():
        axes[0].plot(cumrets, label=label)
    axes[0].set_title('Cumulative Returns')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend()

    # Drawdowns
    for label, drawdown in drawdowns_dict.items():
        axes[1].plot(drawdown, label=label)
    axes[1].set_title('Drawdowns')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Drawdown')
    axes[1].legend()

    # Correlation Heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[2])
    axes[2].set_title('Correlation Matrix Heatmap')

    plt.tight_layout()
    plt.show()
