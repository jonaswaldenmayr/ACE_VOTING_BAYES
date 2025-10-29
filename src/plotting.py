from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

def log_election(store: List[Dict[str, Any]], t: int, E_star: float,
                 xi_H: float, xi_L: float, vote_share: float | None = None) -> None:
    store.append({"t": t, "E_star": E_star, "xi_H": xi_H, "xi_L": xi_L, "vote_share": vote_share})

def plot_election_series(elections: List[Dict[str, Any]], years: np.ndarray) -> None:
    if not elections:
        return
    E_star = np.array([e["E_star"] for e in elections])
    xi_H   = np.array([e["xi_H"]   for e in elections])
    xi_L   = np.array([e["xi_L"]   for e in elections])

    # beliefs (both ξ on the same axes)
    fig, ax = plt.subplots()
    ax.plot(years, xi_H, label=r"$\hat{\xi}_H$")
    ax.plot(years, xi_L, label=r"$\hat{\xi}_L$")
    ax.set_title("Beliefs")
    ax.set_xlabel("Year")
    ax.set_ylabel(r"$\hat{\xi}$")
    ax.legend()
    plt.tight_layout()

    # E* (separate figure)
    fig, ax = plt.subplots()
    ax.plot(years, E_star, label=r"$E^*$")
    ax.set_title("Elected Policy $E^*$")
    ax.set_xlabel("Year")
    ax.set_ylabel("E*")
    ax.legend()
    plt.tight_layout()


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