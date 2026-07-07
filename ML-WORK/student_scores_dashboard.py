import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

np.random.seed(42)

# Generate 200 students with mostly normal scores, then inject outliers
n_students = 200
n_outliers = 12

# Base scores: normally distributed around 65, std=12
base_scores = np.random.normal(loc=65, scale=12, size=n_students - n_outliers)
base_scores = np.clip(base_scores, 30, 95)

# Outliers: very high (near perfect) and very low (fail badly)
high_outliers = np.random.uniform(97, 100, size=6)
low_outliers = np.random.uniform(0, 15, size=6)

all_scores = np.concatenate([base_scores, high_outliers, low_outliers])
np.random.shuffle(all_scores)
all_scores = np.round(all_scores, 1)

# Assign grade letters
def grade(score):
    if score >= 70: return "A"
    elif score >= 60: return "B"
    elif score >= 50: return "C"
    elif score >= 40: return "D"
    else: return "F"

student_ids = [f"STU{str(i+1).zfill(3)}" for i in range(n_students)]
grades = [grade(s) for s in all_scores]

df = pd.DataFrame({"StudentID": student_ids, "Score": all_scores, "Grade": grades})

# Detect outliers via IQR
Q1, Q3 = df["Score"].quantile(0.25), df["Score"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df["IsOutlier"] = (df["Score"] < lower) | (df["Score"] > upper)

# ── Dashboard layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Student Score Dashboard  |  200 Students", fontsize=18,
             fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.97, top=0.92, bottom=0.08)

DARK = "#1a1a2e"
ACCENT = "#e94560"
BLUE = "#0f3460"
GREEN = "#16213e"
TEXT = "#e0e0e0"
GRID = "#2a2a3e"

def style_ax(ax, title):
    ax.set_facecolor(DARK)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

# 1. Histogram with KDE
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Score Distribution")
n_bins = 20
counts, bins, patches = ax1.hist(df["Score"], bins=n_bins, color=BLUE,
                                  edgecolor="#4a4a6a", alpha=0.85)
# Colour outlier bins
for patch, left in zip(patches, bins[:-1]):
    if left < lower or left >= upper:
        patch.set_facecolor(ACCENT)
        patch.set_alpha(0.9)

# KDE overlay
kde_x = np.linspace(0, 100, 300)
kde = stats.gaussian_kde(df["Score"])
ax1.plot(kde_x, kde(kde_x) * n_students * (bins[1] - bins[0]),
         color="#f5a623", linewidth=2, label="KDE")
ax1.axvline(lower, color=ACCENT, linestyle="--", linewidth=1.2, label=f"IQR fence ({lower:.1f})")
ax1.axvline(upper, color=ACCENT, linestyle="--", linewidth=1.2, label=f"IQR fence ({upper:.1f})")
ax1.set_xlabel("Score"); ax1.set_ylabel("Count")
ax1.legend(fontsize=7.5, facecolor=GREEN, labelcolor=TEXT, framealpha=0.8)

# 2. Box plot
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Box Plot (with Outliers)")
bp = ax2.boxplot(df["Score"], vert=True, patch_artist=True,
                 medianprops=dict(color="#f5a623", linewidth=2),
                 boxprops=dict(facecolor=BLUE, color="#4a4a6a"),
                 whiskerprops=dict(color=TEXT),
                 capprops=dict(color=TEXT),
                 flierprops=dict(marker="o", color=ACCENT,
                                 markerfacecolor=ACCENT, markersize=7))
ax2.set_ylabel("Score"); ax2.set_xticks([])

# Annotate outlier count
n_out = df["IsOutlier"].sum()
ax2.text(1.25, df.loc[df["IsOutlier"], "Score"].max(),
         f"  {n_out} outliers", color=ACCENT, fontsize=9, va="center")

# 3. Grade pie chart
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "Grade Distribution")
grade_counts = df["Grade"].value_counts().reindex(["A", "B", "C", "D", "F"]).fillna(0)
colors_pie = ["#4ade80", "#60a5fa", "#facc15", "#fb923c", ACCENT]
wedges, texts, autotexts = ax3.pie(
    grade_counts, labels=grade_counts.index, autopct="%1.1f%%",
    colors=colors_pie, startangle=90,
    textprops={"color": TEXT, "fontsize": 10},
    wedgeprops={"edgecolor": "#0f0f1a", "linewidth": 1.5})
for at in autotexts:
    at.set_fontsize(8)

# 4. Scatter: rank vs score (outliers highlighted)
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, "Score by Student Rank")
df_sorted = df.sort_values("Score").reset_index(drop=True)
normal = df_sorted[~df_sorted["IsOutlier"]]
outliers = df_sorted[df_sorted["IsOutlier"]]
ax4.scatter(normal.index, normal["Score"], color=BLUE, s=18,
            alpha=0.8, edgecolors="#4a4a6a", linewidths=0.4, label="Normal")
ax4.scatter(outliers.index, outliers["Score"], color=ACCENT, s=45,
            zorder=5, edgecolors="white", linewidths=0.6, label="Outlier")
ax4.axhline(df["Score"].mean(), color="#f5a623", linewidth=1.2,
            linestyle="--", label=f"Mean ({df['Score'].mean():.1f})")
ax4.set_xlabel("Rank"); ax4.set_ylabel("Score")
ax4.legend(fontsize=8, facecolor=GREEN, labelcolor=TEXT, framealpha=0.8)

# 5. Bar chart: grade counts
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, "Students per Grade")
bars = ax5.bar(grade_counts.index, grade_counts.values,
               color=colors_pie, edgecolor="#0f0f1a", linewidth=1.2)
for bar, val in zip(bars, grade_counts.values):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             str(int(val)), ha="center", va="bottom", color=TEXT, fontsize=10,
             fontweight="bold")
ax5.set_xlabel("Grade"); ax5.set_ylabel("Count")
ax5.set_ylim(0, grade_counts.max() + 15)

# 6. Stats summary panel
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(DARK)
for spine in ax6.spines.values():
    spine.set_edgecolor(GRID)
ax6.set_xticks([]); ax6.set_yticks([])
ax6.set_title("Summary Statistics", color=TEXT, fontsize=11,
              fontweight="bold", pad=8)

summary = {
    "Total Students": n_students,
    "Mean Score":     f"{df['Score'].mean():.2f}",
    "Median Score":   f"{df['Score'].median():.2f}",
    "Std Dev":        f"{df['Score'].std():.2f}",
    "Min Score":      f"{df['Score'].min():.1f}",
    "Max Score":      f"{df['Score'].max():.1f}",
    "IQR":            f"{IQR:.2f}",
    "Lower Fence":    f"{lower:.2f}",
    "Upper Fence":    f"{upper:.2f}",
    "Outlier Count":  int(df["IsOutlier"].sum()),
    "Pass Rate (≥50)": f"{(df['Score'] >= 50).mean()*100:.1f}%",
}

y_pos = 0.95
for key, val in summary.items():
    color = ACCENT if "Outlier" in key else ("#4ade80" if "Pass" in key else TEXT)
    ax6.text(0.08, y_pos, key, transform=ax6.transAxes,
             fontsize=9.5, color="#a0a0c0", va="top")
    ax6.text(0.92, y_pos, str(val), transform=ax6.transAxes,
             fontsize=9.5, color=color, va="top", ha="right", fontweight="bold")
    ax6.axhline(y=y_pos - 0.01, xmin=0.05, xmax=0.95,
                color=GRID, linewidth=0.5, transform=ax6.transAxes)
    y_pos -= 0.085

plt.savefig("student_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Dashboard saved to student_dashboard.png")
plt.show()
