import matplotlib as _mpl
_mpl.use("Agg")  # headless backend: save figures to files instead of opening a window
import matplotlib.pyplot as plt
import os as _os
_os.makedirs("plots", exist_ok=True)
_FIG_N = [0]
def _save():
    """Save the current figure to plots/figure_NN.png (replaces plt.show)."""
    _FIG_N[0] += 1
    plt.savefig(_os.path.join("plots", f"figure_{_FIG_N[0]:02d}.png"), dpi=150, bbox_inches="tight")
    plt.close()
import numpy as np

# 2017 UK General Election approximate vote share
parties = ['Conservative', 'Labour', 'SNP', 'Lib Dems', 'DUP', 'Others']
votes   = [42.3, 40.0, 3.0, 7.4, 0.9, 6.4]  # percentages
colors  = ['#0087DC', '#E4003B', '#FDF38E', '#FAA61A', '#D46A4C', '#AAAAAA']
explode = (0.05, 0.05, 0, 0, 0, 0)  # slightly separate top two parties

plt.figure(figsize=(8, 8))
plt.pie(
    votes,
    labels=parties,
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    startangle=140,
    shadow=True
)
plt.title('2017 General Election — Vote Share by Party', fontsize=14, fontweight='bold')
plt.axis('equal')   # keeps the pie circular
plt.tight_layout()
_save()

# Generate the first 10 Fibonacci numbers
def fibonacci(n):
    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib[:n]

n = 10
fibs = fibonacci(n)
x = list(range(1, n + 1))   # positions 1..10

plt.figure(figsize=(10, 5))

# Continuous line
plt.plot(x, fibs, color='steelblue', linewidth=2, label='Continuous line')

# Independent dots
plt.scatter(x, fibs, color='red', s=80, zorder=5, label='Independent dots')

plt.title('First 10 Fibonacci Numbers', fontsize=14)
plt.xlabel('Index')
plt.ylabel('Value')
plt.xticks(x)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
_save()

# Sample monthly profit data (replace with actual CSV read if available)
months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']
total_profit = [15000, 18000, 12000, 22000, 25000, 21000,
                28000, 30000, 27000, 32000, 35000, 40000]

plt.figure(figsize=(12, 5))
plt.plot(months, total_profit, marker='o', color='green', linewidth=2)
plt.fill_between(months, total_profit, alpha=0.15, color='green')  # shade under line
plt.title('Total Profit per Month', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Profit ($)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
_save()

# Sample product sales data per year
products = ['Face Cream', 'Face Wash', 'Toothpaste', 'Shampoo', 'Body Lotion']
units_sold = [3200, 4100, 2800, 5000, 3700]  # total units across all years

plt.figure(figsize=(8, 8))
plt.pie(
    units_sold,
    labels=products,
    autopct='%1.1f%%',
    startangle=90,
    shadow=True,
    colors=plt.cm.Pastel1.colors
)
plt.title('Units Sold per Product (% of Total)', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
_save()

# Same profit data, now with explicit style properties applied
months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']
total_profit = [15000, 18000, 12000, 22000, 25000, 21000,
                28000, 30000, 27000, 32000, 35000, 40000]

plt.figure(figsize=(12, 5))
plt.plot(
    months, total_profit,
    color='blue',         # line colour
    linewidth=2.5,        # line thickness
    linestyle='--',       # dashed line style
    marker='D',           # diamond markers
    markersize=8,         # marker size
    markerfacecolor='red' # marker fill colour
)
plt.title('Monthly Profit (Styled)', fontsize=16, fontweight='bold', color='darkblue')
plt.xlabel('Month', fontsize=12, labelpad=8)
plt.ylabel('Profit ($)', fontsize=12, labelpad=8)
plt.xticks(rotation=45)
plt.grid(True, color='grey', linestyle=':', linewidth=0.7)
plt.tight_layout()
_save()

# Monthly sales data for Face Cream and Face Wash
months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']
face_cream  = [2500, 3000, 2200, 3400, 3800, 3100,
               4000, 4200, 3900, 4500, 5000, 5500]
face_wash   = [1800, 2100, 1900, 2600, 2800, 2400,
               3000, 3200, 2900, 3400, 3700, 4000]

x = np.arange(len(months))
width = 0.35  # width of each bar group

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width/2, face_cream, width, label='Face Cream', color='#FF6B9D')
bars2 = ax.bar(x + width/2, face_wash,  width, label='Face Wash',  color='#4ECDC4')

ax.set_title('Face Cream vs Face Wash Monthly Sales', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Units Sold')
ax.set_xticks(x)
ax.set_xticklabels(months)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
_save()

# Monthly profit values — histogram shows which profit ranges occur most
total_profit = [15000, 18000, 12000, 22000, 25000, 21000,
                28000, 30000, 27000, 32000, 35000, 40000]

plt.figure(figsize=(10, 5))
plt.hist(
    total_profit,
    bins=6,           # 6 profit range buckets
    color='steelblue',
    edgecolor='black',
    alpha=0.85
)
plt.title('Profit Distribution — Most Common Profit Ranges', fontsize=14)
plt.xlabel('Profit Range ($)')
plt.ylabel('Frequency (Number of Months)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
_save()

np.random.seed(42)

# Generate a dataset of student scores with some outliers
scores = np.concatenate([
    np.random.normal(loc=65, scale=12, size=80),  # main cluster around 65
    [10, 12, 98, 99, 100]                          # deliberate outliers
])
scores = np.clip(scores, 0, 100)  # keep scores within 0-100

plt.figure(figsize=(8, 6))
bp = plt.boxplot(
    scores,
    vert=True,
    patch_artist=True,                          # filled box
    boxprops=dict(facecolor='lightblue', color='navy'),
    medianprops=dict(color='red', linewidth=2), # median line in red
    flierprops=dict(marker='o', color='orange', markersize=8)  # outlier dots
)

plt.title('Box Plot of Student Scores', fontsize=14)
plt.ylabel('Score (out of 100)')
plt.xticks([1], ['Students'])

# Annotate key parts
plt.annotate('Outliers (beyond 1.5×IQR)', xy=(1.05, 98), fontsize=9, color='darkorange')
plt.annotate('Median (red line)', xy=(1.05, 65), fontsize=9, color='red')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
_save()
