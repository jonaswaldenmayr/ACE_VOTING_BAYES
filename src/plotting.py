from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable


def plot_time_series(Y, x=None, title=None, xlabel='Year', ylabel='Value', 
                     style='-', figsize=(10, 6), show: bool = True):
    time_steps = x if x is not None else np.arange(2020, 2020 + len(Y)*10, 10)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_steps, Y, style, label=ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    if show:
        plt.show()
    return fig, ax