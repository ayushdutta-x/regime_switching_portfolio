"""
sanity_check_plot.py
---------------------
Phase 2 checkpoint (per project spec): visually confirm our volatility
and momentum features actually spike/dip during known crisis periods.
This is NOT optional - an unvalidated feature feeding into the HMM
later is how you end up with a regime classifier that looks fine
statistically but is actually meaningless.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.csv"
FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(df.index, df["volatility_21d"], color="darkred", linewidth=0.8)
axes[0].set_title("21-day Realized Volatility (equity)")
axes[0].axvspan("2020-02-15", "2020-04-30", alpha=0.2, color="red", label="COVID crash")
axes[0].axvspan("2022-01-01", "2022-06-30", alpha=0.2, color="orange", label="2022 rate hikes")
axes[0].legend(loc="upper right")

axes[1].plot(df.index, df["india_vix"], color="purple", linewidth=0.8)
axes[1].set_title("India VIX")
axes[1].axvspan("2020-02-15", "2020-04-30", alpha=0.2, color="red")
axes[1].axvspan("2022-01-01", "2022-06-30", alpha=0.2, color="orange")

axes[2].plot(df.index, df["momentum_1m"], color="steelblue", linewidth=0.8)
axes[2].axhline(0, color="black", linewidth=0.5)
axes[2].set_title("1-Month Momentum (rolling mean of equity returns)")
axes[2].axvspan("2020-02-15", "2020-04-30", alpha=0.2, color="red")
axes[2].axvspan("2022-01-01", "2022-06-30", alpha=0.2, color="orange")

plt.tight_layout()
out_path = FIG_DIR / "feature_sanity_check.png"
plt.savefig(out_path, dpi=150)
print(f"Saved plot to: {out_path}")

# Numeric sanity check too, not just visual
covid_window = df.loc["2020-02-15":"2020-04-30"]
normal_window = df.loc["2019-01-01":"2019-12-31"]

print(f"\nMean 21d volatility during COVID crash: {covid_window['volatility_21d'].mean():.5f}")
print(f"Mean 21d volatility during 2019 (calm year): {normal_window['volatility_21d'].mean():.5f}")
print(f"Mean India VIX during COVID crash: {covid_window['india_vix'].mean():.2f}")
print(f"Mean India VIX during 2019 (calm year): {normal_window['india_vix'].mean():.2f}")
print(f"Mean 1m momentum during COVID crash: {covid_window['momentum_1m'].mean():.5f}")
print(f"Mean 1m momentum during 2019 (calm year): {normal_window['momentum_1m'].mean():.5f}")