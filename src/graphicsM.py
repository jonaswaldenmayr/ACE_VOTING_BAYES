



from gettext import npgettext
from plotting import plot_time_series_multi
import numpy as np
from matplotlib.ticker import ScalarFormatter


from plotting import plot_scenarios_flexible


bau = [986.37, 5777.37, 9310.37, 11915.69, 13836.92, 15253.68, 16298.43, 17068.86, 17636.99, 18055.94, 18364.89, 18592.71, 18760.71, 18884.60, 18975.96, 19043.33, 19093.01, 19129.65, 19156.66, 19176.59, 19191.28, 19202.11, 19210.10, 19215.99, 19220.34, 19223.54, 19225.90, 19227.64, 19228.93, 19229.88, 19230.57]
votingO =        [986.37, 1702.25, 2182.25, 2436.65, 2560.94, 2652.60, 2720.19, 2770.03, 2806.79, 2833.89, 2853.88, 2868.62, 2879.49, 2887.50, 2893.41, 2897.77, 2900.99, 2903.36, 2905.10, 2906.39, 2907.34, 2908.04, 2908.56, 2908.94, 2909.22, 2909.43, 2909.58, 2909.70, 2909.78, 2909.84, 2909.89]
votingOP =        [828.37,1685.46, 2193.58, 2550.54, 2792.60, 2952.60, 3056.41, 2684.71, 2396.34, 2175.82, 2008.17, 1880.99, 1784.49, 1711.17, 1655.29, 1612.53, 1579.61, 1554.10, 1534.14, 1518.37, 1505.75, 1495.51, 1487.09, 1480.03, 1474.03, 1468.84, 1464.28, 1460.20, 1456.52, 1453.14, 1450.02]
extremeSCCavg =   [986.37,1177.37, 1318.22, 1422.09, 1498.68, 1555.17, 1596.82, 1627.53, 1650.18, 1666.88, 1679.20, 1688.28, 1694.98, 1699.92, 1703.56, 1706.25, 1708.23, 1709.69, 1710.77, 1711.56, 1712.15, 1712.58, 1712.90, 1713.13, 1713.30, 1713.43, 1713.53, 1713.60, 1713.65, 1713.68, 1713.71]
# extreme = [        986.37,1427.37, 1752.58, 1992.39, 2169.24, 2299.65, 2395.82, 2466.73, 2519.03, 2557.59, 2586.03, 2607.00, 2622.47, 2633.87, 2642.28, 2648.48, 2653.05, 2656.43, 2658.91, 2660.75, 2662.10, 2663.10, 2663.83, 2664.37, 2664.77, 2665.07, 2665.29, 2665.45, 2665.56, 2665.65, 2665.72]

candidates = [
    ("BAU", "bau"),
    ("Office Motivated Parties", "votingO"),       # note: zero, not letter O
    ("Office & Policy Motivated Parties", "votingOP"),
    ("SCC","extremeSCCavg"),
    # ("SCC", "extreme"),
]
series = {}
for name, var in candidates:
    if var in locals():
        series[name] = np.asarray(locals()[var])
    else:
        print(f"⚠️ Missing variable: {var}")

if not series:
    raise ValueError("No valid series were found. Define at least one before building `series`.")

# if "SCC" in series:
#     series["SCC"] = np.asarray(series["SCC"]) * 0.85
if "Office & Policy Motivated Parties" in series:
    series["Office & Policy Motivated Parties"] = np.asarray(series["Office & Policy Motivated Parties"]) * 1.19


n = len(next(iter(series.values())))
print(f"✅ Found {len(series)} series, length = {n}")
years = np.arange(2020, 2020 + 10 * n, 10)

# Colors for the scenarios (only applied to ones present)
palette = {
    "BAU": "#E63946",      # neutral gray
    "Office Motivated Parties": "#6c757d", # blue127FBF
    "Office & Policy Motivated Parties": "#2196F3",# green
    "SCC": "#59db35",
    # "SCC": "#59db35",  # red
}
colors = {k: palette[k] for k in series.keys() if k in palette}

plot_scenarios_flexible(
    series,
    start_year=2020,
    step=10,
    title="Carbon Stock across emission scenarios",
    xlabel="Year",
    ylabel="GDP",
    colors=colors,
    legend_loc="upper right",
    save_path="pdf/M_scenarios.pdf",
    log_y=True,
)
