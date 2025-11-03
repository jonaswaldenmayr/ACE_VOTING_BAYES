


from gettext import npgettext
from plotting import plot_time_series_multi
import numpy as np
from matplotlib.ticker import ScalarFormatter


from plotting import plot_scenarios_flexible


bau = [4484.63, 9280.42, 9885.20, 9319.90, 8967.44, 9053.39, 9564.60, 10479.81, 11805.97, 13578.79, 15859.90, 18735.97, 22320.06, 26754.83, 32217.45, 38926.21, 47148.84, 57213.03, 69519.25, 84556.60, 102922.11, 125344.42, 152712.52, 186111.03, 226863.10, 276582.78, 337238.90, 411233.04, 501494.54]

votingO =       [2206.54, 3575.16, 4492.57, 5255.41, 6072.53, 7170.28, 8694.52,   10627.11, 13007.15, 15914.90, 19459.98, 23780.42, 29046.11, 35464.82, 43290.15, 52831.48, 64466.12, 78654.30,  95957.32, 117059.78, 142796.69, 174186.46, 212471.27, 259166.18, 316119.13, 385584.02, 470309.99, 573650.13, 699694.26]
#extremeSCCavg = [2171.21, 3614.40, 4783.27, 5927.29, 7217.83, 8758.09, 10628.79,  12911.47, 15699.43, 19104.35, 23261.68, 28336.42, 34529.75, 42087.11, 51307.83, 62557.08, 76280.28, 93020.74, 113441.15, 138349.86, 168732.81, 205792.58, 250996.00, 306132.19, 373383.34, 455411.15, 555462.08, 677495.92, 826342.45]
votingOP      = [2723.71, 4841.96, 6298.94, 7540.26, 8892.32, 10528.04, 10175.25, 11384.87, 13764.73, 16980.51, 21009.21, 25948.56, 31965.58, 39283.28, 48181.68, 59005.66, 72177.35, 88212.08, 107738.53, 131523.30, 160501.08, 195811.49, 238843.91, 291292.32, 355222.03, 433151.17, 528149.97, 643961.87, 785151.21]
extreme       = [2478.95, 4294.10, 5663.40, 6931.84, 8340.02, 10021.39, 12071.45, 14582.27, 17657.69, 21421.60, 26024.30, 31648.90, 38518.82, 46906.67, 57145.05, 69639.76, 84885.80, 100487.06, 126180.14,153863.56, 187633.27, 228826.11, 279072.64, 340361.73, 415119.23, 506304.03, 617525.12, 753184.47, 918651.21]



# ---- After your bau / voting0 / votingOP / extremeSCCavg / extreme arrays ----


# Collect available series safely (only those you actually defined)
candidates = [
    ("BAU", "bau"),
    ("Office Motivated Parties", "votingO"),       # note: zero, not letter O
    ("Office & Policy Motivated Parties", "votingOP"),
    ("SCC", "extreme"),
]
series = {}
for name, var in candidates:
    if var in locals():
        series[name] = np.asarray(locals()[var])
    else:
        print(f"⚠️ Missing variable: {var}")

if not series:
    raise ValueError("No valid series were found. Define at least one before building `series`.")

n = len(next(iter(series.values())))
print(f"✅ Found {len(series)} series, length = {n}")
years = np.arange(2020, 2020 + 10 * n, 10)

# Colors for the scenarios (only applied to ones present)
palette = {
    "BAU": "#E63946",      # neutral gray
    "Office Motivated Parties": "#6c757d", # blue127FBF
    "Office & Policy Motivated Parties": "#2196F3",# green
    "SCC": "#59db35",  # red
}
colors = {k: palette[k] for k in series.keys() if k in palette}

plot_scenarios_flexible(
    series,
    start_year=2020,
    step=10,
    title="GDP across emission scenarios",
    xlabel="Year",
    ylabel="GDP",
    y_min=0,
    colors=colors,
    legend_loc="best",
    save_path="pdf/gdp_scenarios.pdf",
    log_y=False,
)
