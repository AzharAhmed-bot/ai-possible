import matplotlib.pyplot as plt
import numpy as np

# ── Shared data ───────────────────────────────────────────────────────────────
months = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

total_profit = [15200, 18400, 22100, 19800, 25600, 28300,
                31200, 29700, 26800, 23400, 20100, 35600]

face_cream = [1500, 1800, 2100, 2400, 1900, 2700,
              3100, 2800, 2300, 2000, 1700, 3500]
face_wash  = [1200, 1400, 1600, 1900, 1700, 2100,
              2400, 2200, 1900, 1600, 1400, 2800]

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Data Visualizations — 8 Tasks", fontsize=16, fontweight="bold")


# ── Task 1: Political parties pie chart (2017) ────────────────────────────────
ax = axes[0, 0]
parties = ["CDU/CSU", "SPD", "AfD", "FDP", "Linke", "Grüne", "Others"]
votes   = [26.8, 20.5, 12.6, 10.7, 9.2, 8.9, 11.3]
ax.pie(votes, labels=parties, autopct="%1.1f%%", startangle=140)
ax.set_title("Task 1: 2017 Election Vote Share")


# ── Task 2: First 10 Fibonacci numbers — line + dots ─────────────────────────
ax = axes[0, 1]
fib = [0, 1]
for _ in range(8):
    fib.append(fib[-1] + fib[-2])
x = range(1, 11)
ax.plot(x, fib, color="steelblue", linewidth=2, label="Line")
ax.scatter(x, fib, color="orange", s=80, zorder=3, label="Dots")
ax.set_title("Task 2: First 10 Fibonacci Numbers")
ax.set_xlabel("Index")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)


# ── Task 3: Total profit line plot ───────────────────────────────────────────
ax = axes[0, 2]
ax.plot(months, total_profit, color="green", marker="o", linewidth=2)
ax.set_title("Task 3: Total Profit by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Profit (USD)")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, linestyle="--", alpha=0.5)


# ── Task 4: Units sold per year pie chart ────────────────────────────────────
ax = axes[1, 0]
products = ["Face Cream", "Face Wash", "Moisturiser", "Sun Screen", "Lip Balm"]
units    = [35, 25, 20, 12, 8]
ax.pie(units, labels=products, autopct="%1.1f%%", startangle=90)
ax.set_title("Task 4: Units Sold per Year by Product")


# ── Task 5: Profit line plot with style properties ───────────────────────────
ax = axes[1, 1]
ax.plot(months, total_profit,
        color="purple",
        linewidth=2.5,
        linestyle="--",
        marker="D",
        markersize=8,
        markerfacecolor="red",
        markeredgecolor="black",
        markeredgewidth=1,
        label="Profit")
ax.set_title("Task 5: Styled Profit Line Plot")
ax.set_xlabel("Month")
ax.set_ylabel("Profit (USD)")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()


# ── Task 6: Face cream & face wash bar chart ─────────────────────────────────
ax = axes[1, 2]
x6 = np.arange(len(months))
w  = 0.35
ax.bar(x6 - w/2, face_cream, w, label="Face Cream", color="steelblue")
ax.bar(x6 + w/2, face_wash,  w, label="Face Wash",  color="coral")
ax.set_title("Task 6: Face Cream & Face Wash Sales")
ax.set_xlabel("Month")
ax.set_ylabel("Units Sold")
ax.set_xticks(x6)
ax.set_xticklabels(months, rotation=45, ha="right")
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.5)


# ── Task 7: Total profit histogram ───────────────────────────────────────────
ax = axes[2, 0]
ax.hist(total_profit,bins=6, fibcolor="teal", edgecolor="black",)
ax.axvline(np.mean(total_profit), color="red",    linestyle="--", label=f"Mean ${np.mean(total_profit):,.0f}")
ax.axvline(np.median(total_profit), color="orange", linestyle=":",  label=f"Median ${np.median(total_profit):,.0f}")
ax.set_title("Task 7: Profit Distribution Histogram")
ax.set_xlabel("Profit (USD)")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)


# ── Task 8: Student scores box plot ──────────────────────────────────────────
ax = axes[2, 1]
np.random.seed(42)
scores_a = np.concatenate([np.random.normal(75, 8, 50),  [45, 98, 99]])
scores_b = np.concatenate([np.random.beta(2, 6, 50)*100, [95, 97]])
scores_c = np.concatenate([np.random.normal(65, 15, 50), [15, 18, 99]])

ax.boxplot([scores_a, scores_b, scores_c],
           tick_labels=["Class A\n(Normal)", "Class B\n(Skewed)", "Class C\n(Outliers)"],
           patch_artist=True,
           boxprops=dict(facecolor="lightblue"),
           medianprops=dict(color="red", linewidth=2),
           flierprops=dict(marker="o", markerfacecolor="red", markersize=6))
ax.set_title("Task 8: Student Scores Box Plot")
ax.set_ylabel("Score")
ax.grid(True, axis="y", linestyle="--", alpha=0.5)

# Explanation text
ax.text(0.98, 0.97,
        "Skewness: median position in box\n"
        "Outliers: dots beyond whiskers (1.5×IQR)",
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))


# ── Hide unused subplot ───────────────────────────────────────────────────────
axes[2, 2].set_visible(False)

plt.tight_layout()
plt.savefig("plot.png", dpi=150, bbox_inches="tight")