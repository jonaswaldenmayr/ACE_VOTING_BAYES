from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from typing import List, Dict, Any
import re
import numpy as np
from matplotlib.ticker import ScalarFormatter

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
    plt.savefig(f"pdf/{re.sub(r'[^A-Za-z0-9_-]+', '_', title.strip() or 'plot')}.pdf", dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    return fig, ax

def plot_time_series_multi(series: dict[str, list[float] | np.ndarray],
                           x: list[float] | np.ndarray | None = None,
                           title: str = "",
                           xlabel: str = "",
                           ylabel: str = "",
                           x_min: float | None = None,
                           y_min: float | None = None,
                           colors: dict[str, str] | None = None,
                           y_line: float | None = None,
                           y_line_color: str = "gray",
                           y_line_style: str = "--",
                           y_line_width: float = 1.2,
                           ) -> None:
    plt.figure()
    for label, y in series.items():
        y = np.asarray(y)
        x_vals = np.arange(len(y)) if x is None else np.asarray(x)[:len(y)]
        color = colors.get(label) if colors and label in colors else None
        plt.plot(x_vals, y, label=label,color=color, linewidth= 1.8)

    if y_line is not None:
        plt.axhline(
            y=y_line,
            color=y_line_color,
            linestyle=y_line_style,
            linewidth=y_line_width,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if x_min is not None:
        plt.xlim(left=x_min)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    plt.savefig("test.pdf", dpi=300, bbox_inches="tight")
    plt.show()



def plot_time_series_multi(series: dict[str, list[float] | np.ndarray],
                           x: list[float] | np.ndarray | None = None,
                           title: str = "",
                           xlabel: str = "",
                           ylabel: str = "",
                           x_min: float | None = None,
                           y_min: float | None = None,
                           colors: dict[str, str] | None = None,
                           y_line: float | None = None,
                           y_line_color: str = "gray",
                           y_line_style: str = "--",
                           y_line_width: float = 1.2,
                           y_horizontal_label: str | None = None,
                            legend_loc: str = "best",
                           ) -> None:
    plt.figure()
    for label, y in series.items():
        y = np.asarray(y)
        x_vals = np.arange(len(y)) if x is None else np.asarray(x)[:len(y)]
        color = colors.get(label) if colors and label in colors else None
        plt.plot(x_vals, y, label=label,color=color, linewidth= 1.8)

    if y_line is not None:
        plt.axhline(
            y=y_line,
            color=y_line_color,
            linestyle=y_line_style,
            linewidth=y_line_width,
            label=y_horizontal_label if y_horizontal_label else None,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend(loc=legend_loc, frameon=True, framealpha=0.9, edgecolor="gray")


    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if x_min is not None:
        plt.xlim(left=x_min)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    plt.savefig(f"pdf/{re.sub(r'[^A-Za-z0-9_-]+', '_', title.strip() or 'plot')}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def plot_dual_axis(
    x: list[float] | np.ndarray,
    y_left: list[float] | np.ndarray,
    y_right: list[float] | np.ndarray,
    label_left: str,
    label_right: str,
    title: str = "",
    color_left: str = "#127FBF",
    color_right: str = "#F9703E",
    xlabel: str = "Year",
    figsize: tuple[int, int] = (9, 5),
) -> None:
    """Plot two time series with separate y-axes (left & right)."""
    fig, ax1 = plt.subplots(figsize=figsize)

    # Left axis
    ax1.plot(x, y_left, color=color_left, label=label_left, linewidth=1.8)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(label_left, color=color_left)
    ax1.tick_params(axis="y", labelcolor=color_left)

    # Right axis (twin)
    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color=color_right, label=label_right, linewidth=1.8, linestyle="--")
    ax2.set_ylabel(label_right, color=color_right)
    ax2.tick_params(axis="y", labelcolor=color_right)

    # Title and grid
    fig.suptitle(title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f"pdf/{re.sub(r'[^A-Za-z0-9_-]+', '_', title.strip() or 'plot')}.pdf", dpi=300, bbox_inches='tight')
    plt.show()



def plot_dual_axis_beliefs_M(
    years: np.ndarray,
    M_t: np.ndarray,
    elections: list[dict],
    title: str = "Atmospheric Carbon and Average Beliefs",
    color_M: str = "#127FBF",     # blue
    color_xi: str = "#F9703E",    # orange
    xlabel: str = "Year",
    figsize: tuple[int, int] = (9, 5),
) -> None:
    """Plot M_t (left axis) and average beliefs (right axis)."""
    xi_H = np.array([e["xi_H"] for e in elections])
    xi_L = np.array([e["xi_L"] for e in elections])
    xi_avg = 0.5 * (xi_H + xi_L)

    # Align time steps with elections
    x = years[:len(elections)]

    fig, ax1 = plt.subplots(figsize=figsize)

    # Left y-axis: M_t
    ax1.plot(x, M_t[1:len(x)+1], color=color_M, linewidth=1.8, label="Atmospheric Carbon (GtC)")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Atmospheric Carbon (GtC)", color=color_M)
    ax1.tick_params(axis="y", labelcolor=color_M)

    # Right y-axis: average beliefs
    ax2 = ax1.twinx()
    ax2.plot(x, xi_avg, color=color_xi, linewidth=1.8, linestyle="--", label=r"Average $\hat{\xi}$")
    ax2.set_ylabel(r"Average $\hat{\xi}$", color=color_xi)
    ax2.tick_params(axis="y", labelcolor=color_xi)

    # Title, grid, and layout
    fig.suptitle(title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

def plot_beliefs_and_damages(
    years: np.ndarray,
    elections: list[dict],
    damages: np.ndarray,
    title: str = "Beliefs and Damages",
    color_H: str = "#127FBF",   # blue
    color_L: str = "#F9703E",   # orange
    color_D: str = "#E63946",   # red (damages)
    xlabel: str = "Year",
    figsize: tuple[int, int] = (9, 5),
    save_pdf: bool = False,
) -> None:
    """Plot ξ_H, ξ_L (left y-axis) and damages D_t (right y-axis)."""
    xi_H = np.array([e["xi_H"] for e in elections])
    xi_L = np.array([e["xi_L"] for e in elections])
    x = years[:len(elections)]

    fig, ax1 = plt.subplots(figsize=figsize)

    # --- Left y-axis: beliefs ---
    ax1.plot(x, xi_H, color=color_H, linewidth=1.8, linestyle="--",label=r"$\hat{\xi}_H$")
    ax1.plot(x, xi_L, color=color_L, linewidth=1.8, linestyle="--",label=r"$\hat{\xi}_L$")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r"Beliefs $\hat{\xi}_j$", color=color_H)
    ax1.tick_params(axis="y", labelcolor=color_H)

    # --- Right y-axis: damages ---
    ax2 = ax1.twinx()
    ax2.plot(years[:len(damages)], damages[:len(x)], color=color_D, linewidth=1.8, label="Damages")
    ax2.set_ylabel("Damages (fraction of GDP)", color=color_D)
    ax2.tick_params(axis="y", labelcolor=color_D)

    # --- Titles, grid, and legend ---
    fig.suptitle(title)
    ax1.grid(True, alpha=0.3)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    plt.savefig(f"pdf/{re.sub(r'[^A-Za-z0-9_-]+', '_', title.strip() or 'plot')}.pdf", dpi=300, bbox_inches='tight')


    plt.show()


def plot_all_beliefs_and_damages(
    years: np.ndarray,
    elections: list[dict],
    damages: np.ndarray,
    title: str = "Beliefs (High, Low, Avg) and Damages over Time",
    color_H: str = "#127FBF",    # blue
    color_L: str = "#F9703E",    # or
    color_avg: str = "#6c757d",  # neutral gray
    color_D: str = "#E63946",    # red
    xlabel: str = "Year",
    figsize: tuple[int, int] = (9, 5),
    save_pdf: bool = False,
) -> None:
    """Plot ξ_H, ξ_L, average ξ (left y-axis) and damages D_t (right y-axis)."""
    xi_H = np.array([e["xi_H"] for e in elections])
    xi_L = np.array([e["xi_L"] for e in elections])
    xi_avg = 0.5 * (xi_H + xi_L)
    x = years[:len(elections)]

    fig, ax1 = plt.subplots(figsize=figsize)

    # --- Left y-axis: beliefs ---
    ax1.plot(x, xi_H, color=color_H, linewidth=1.8, linestyle=":",label=r"$\hat{\xi}_H$")
    ax1.plot(x, xi_L, color=color_L, linewidth=1.8, linestyle=":", label=r"$\hat{\xi}_L$")
    ax1.plot(x, xi_avg, color=color_avg, linewidth=1.6, linestyle="--", label=r"$\hat{\xi}$ (avg)")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r"Beliefs $\hat{\xi}$", color=color_H)
    ax1.tick_params(axis="y", labelcolor=color_H)

    # --- Right y-axis: damages ---
    ax2 = ax1.twinx()
    ax2.plot(years[:len(damages)], damages[:len(x)], color=color_D, linewidth=1.8, label="Damages")
    ax2.set_ylabel("Damages (fraction of GDP)", color=color_D)
    ax2.tick_params(axis="y", labelcolor=color_D)

    # --- Title, grid, legend ---
    fig.suptitle(title)
    ax1.grid(True, alpha=0.3)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    if save_pdf:
        fig.savefig("beliefs_all_and_damages.pdf", dpi=300, bbox_inches="tight")

    plt.savefig(f"pdf/{re.sub(r'[^A-Za-z0-9_-]+', '_', title.strip() or 'plot')}.pdf", dpi=300, bbox_inches='tight')
    plt.show()



import numpy as np
import matplotlib.pyplot as plt

def plot_scenarios_flexible(
    series: dict[str, np.ndarray | list[float]],
    start_year: int = 2020,
    step: int = 10,
    title: str = "",
    xlabel: str = "Year",
    ylabel: str = "",
    colors: dict[str, str] | None = None,
    y_line: float | None = None,
    y_line_label: str | None = None,
    y_line_style: str = "--",
    y_line_width: float = 1.2,
    y_line_color: str = "gray",
    y_min: float | None = None,
    x_min: float | None = None,
    legend_loc: str = "best",
    figsize: tuple[int, int] = (9, 5),
    save_path: str | None = "test.pdf",
    log_y: bool = False,
):
    """Plot multiple scenario series of potentially unequal lengths."""
    if not series:
        raise ValueError("`series` is empty. Provide at least one series.")

    fig, ax = plt.subplots(figsize=figsize)

    plotted_any = False
    for label, y in series.items():
        if y is None:
            continue
        y_arr = np.asarray(y, dtype=float).ravel()
        if y_arr.size == 0 or np.all(~np.isfinite(y_arr)):
            continue

        # Build an x for this series individually
        x_vals = np.arange(start_year, start_year + step * y_arr.size, step)

        color = colors.get(label) if (colors and label in colors) else None
        ax.plot(x_vals, y_arr, label=label, linewidth=1.8, color=color)
        plotted_any = True

    if not plotted_any:
        raise ValueError("No plottable data found in `series` (all empty or invalid).")

    # Optional horizontal reference line
    if y_line is not None:
        ax.axhline(
            y=y_line,
            linestyle=y_line_style,
            linewidth=y_line_width,
            color=y_line_color,
            label=y_line_label if y_line_label else None,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_min is not None:
        ax.set_xlim(left=x_min)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    ax.grid(True, alpha=0.3)

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='y')
    ax.legend(loc=legend_loc, frameon=True, framealpha=0.9, edgecolor="gray")
    fig.tight_layout()

    plt.savefig(f"pdf/{re.sub(r'[^A-Za-z0-9_-]+', '_', title.strip() or 'plot')}.pdf", dpi=300, bbox_inches='tight')


    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")  # explicitly ensure linear scaling

    plt.show()
