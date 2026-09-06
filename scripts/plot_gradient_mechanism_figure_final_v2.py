import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/root/autodl-tmp/clip-lt"

VANILLA_DIR = os.path.join(
    ROOT,
    "results/coop_imagenet_lt/gradient_probe_epoch5"
)

BALANCED_DIR = os.path.join(
    ROOT,
    "results/balanced_coop_imagenet_lt_gamma05/"
    "gradient_probe_epoch9_balanced"
)

OUT_DIR = os.path.join(
    ROOT,
    "results/gradient_mechanism_figure"
)
os.makedirs(OUT_DIR, exist_ok=True)

van_summary = pd.read_csv(
    os.path.join(VANILLA_DIR, "gradient_probe_summary.csv")
)
van_batch = pd.read_csv(
    os.path.join(VANILLA_DIR, "gradient_probe_per_batch.csv")
)
van_cos = pd.read_csv(
    os.path.join(VANILLA_DIR, "gradient_probe_cosines.csv")
)
bal_cos = pd.read_csv(
    os.path.join(BALANCED_DIR, "gradient_probe_cosines.csv")
)

GROUPS = ["Many", "Medium", "Few"]


def summary_value(df, group, col):
    row = df[df["group"] == group]
    return float(row[col].iloc[0])


def cosine_value(df, a, b, col):
    row = df[
        ((df["group_a"] == a) & (df["group_b"] == b))
        | ((df["group_a"] == b) & (df["group_b"] == a))
    ]
    return float(row[col].iloc[0])


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.8,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 7.8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "Many": "#4C72B0",
    "Medium": "#55A868",
    "Few": "#C44E52",
}

batch_presence = [
    100.0 * summary_value(
        van_summary, g, "batches_present"
    ) / 500.0
    for g in GROUPS
]

sample_share = [
    100.0 * summary_value(
        van_summary, g, "sample_share"
    )
    for g in GROUPS
]

grad_means = [
    summary_value(
        van_summary, g, "avg_group_mean_grad_norm"
    )
    for g in GROUPS
]

contrib_means = [
    summary_value(
        van_summary, g, "avg_batch_contribution_norm"
    )
    for g in GROUPS
]

van_neg = 100.0 * cosine_value(
    van_cos, "Many", "Few", "negative_fraction"
)
bal_neg = 100.0 * cosine_value(
    bal_cos, "Many", "Few", "negative_fraction"
)

van_mean_cos = cosine_value(
    van_cos, "Many", "Few", "mean_cosine"
)
bal_mean_cos = cosine_value(
    bal_cos, "Many", "Few", "mean_cosine"
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.15, 5.05)
)

ax_a, ax_b, ax_c, ax_d = axes.flatten()

# ============================================================
# (a) Temporal sparsity
# ============================================================
x = np.arange(3)

bars = ax_a.bar(
    x,
    batch_presence,
    width=0.58,
    color=[COLORS[g] for g in GROUPS],
    edgecolor="black",
    linewidth=0.6,
)

ax_a.set_title("(a) Temporal sparsity", pad=8)
ax_a.set_ylabel("Batch presence (%)")
ax_a.set_xticks(x)
ax_a.set_xticklabels(GROUPS)
ax_a.set_ylim(0, 108)

ax_a.axhline(
    100,
    linestyle="--",
    linewidth=0.8,
    color="0.7",
    zorder=0
)

for bar, value in zip(bars, batch_presence):
    ax_a.text(
        bar.get_x() + bar.get_width()/2,
        value + 1.5,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold"
    )

few_bar = bars[2]

ax_a.text(
    few_bar.get_x() + few_bar.get_width()/2,
    8.7,
    f"{sample_share[2]:.2f}%\nof samples",
    ha="center",
    va="center",
    fontsize=7.3,
    color="white",
    fontweight="bold"
)

# moved upward/left for cleaner spacing
ax_a.annotate(
    "Absent in 81.8%\nof mini-batches",
    xy=(2, batch_presence[2]),
    xytext=(1.10, 60),
    fontsize=7.8,
    ha="center",
    va="center",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=0.9,
        color="0.25"
    )
)

# ============================================================
# (b) Per-occurrence gradient strength
# ============================================================
grad_data = [
    van_batch.loc[
        van_batch["group"] == g,
        "raw_grad_norm"
    ].values
    for g in GROUPS
]

bp = ax_b.boxplot(
    grad_data,
    positions=[1, 2, 3],
    widths=0.52,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(
        color="black",
        linewidth=1.0
    ),
    whiskerprops=dict(
        linewidth=0.8
    ),
    capprops=dict(
        linewidth=0.8
    )
)

for patch, g in zip(bp["boxes"], GROUPS):
    patch.set_facecolor(COLORS[g])
    patch.set_alpha(0.78)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.6)

rng = np.random.default_rng(7)

for i, (g, vals) in enumerate(
    zip(GROUPS, grad_data),
    start=1
):
    jitter = rng.normal(
        0,
        0.045,
        len(vals)
    )

    ax_b.scatter(
        np.full(len(vals), i) + jitter,
        vals,
        s=5,
        alpha=0.14,
        color=COLORS[g],
        edgecolors="none",
        rasterized=True
    )

ax_b.scatter(
    [1, 2, 3],
    grad_means,
    marker="D",
    s=23,
    color="black",
    zorder=5,
    label="Mean"
)

# use point-offset annotations to avoid overlap
for i, value in enumerate(grad_means, start=1):
    ax_b.annotate(
        f"{value:.2f}",
        xy=(i, value),
        xytext=(7, 5),
        textcoords="offset points",
        fontsize=7.2,
        ha="left",
        va="bottom"
    )

ax_b.set_title(
    "(b) Per-occurrence gradient strength",
    pad=8
)
ax_b.set_ylabel(r"Raw gradient norm $\|g\|_2$")
ax_b.set_xticks([1, 2, 3])
ax_b.set_xticklabels(GROUPS)
ax_b.set_ylim(0, 90)

ax_b.annotate(
    "Strong when observed",
    xy=(3, 18),
    xytext=(2.12, 45),
    fontsize=7.8,
    ha="center",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=0.9,
        color="0.25"
    )
)

ax_b.legend(
    loc="upper left",
    frameon=False,
    handletextpad=0.5
)

# ============================================================
# (c) Effective batch contribution
# ============================================================
bars = ax_c.bar(
    x,
    contrib_means,
    width=0.58,
    color=[COLORS[g] for g in GROUPS],
    edgecolor="black",
    linewidth=0.6,
)

ax_c.set_title(
    "(c) Effective batch contribution",
    pad=8
)
ax_c.set_ylabel("Contribution norm")
ax_c.set_xticks(x)
ax_c.set_xticklabels(GROUPS)
ax_c.set_ylim(0, 2.7)

for bar, value in zip(bars, contrib_means):
    ax_c.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.06,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=8.1,
        fontweight="bold"
    )

ax_c.annotate(
    "Strong, but diluted\nby sparse occurrence",
    xy=(2, contrib_means[2]),
    xytext=(1.18, 2.25),
    fontsize=7.6,
    ha="center",
    va="center",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=0.9,
        color="0.25"
    )
)

# ============================================================
# (d) Persistent head-tail conflict
# ============================================================
methods = [
    "Vanilla\nCoOp",
    "Reweighted\nCoOp"
]

neg_rates = [
    van_neg,
    bal_neg
]

mean_cosines = [
    van_mean_cos,
    bal_mean_cos
]

bars = ax_d.bar(
    [0, 1],
    neg_rates,
    width=0.55,
    color=["#7F7F7F", "#B6B6B6"],
    edgecolor="black",
    linewidth=0.6,
)

ax_d.set_title(
    "(d) Persistent head–tail conflict",
    pad=8
)

ax_d.set_ylabel(
    "Conflicting Many–Few batches (%)"
)

ax_d.set_xticks([0, 1])
ax_d.set_xticklabels(methods)
ax_d.set_ylim(0, 70)

# no 50% reference line

for i, (bar, rate, cos) in enumerate(
    zip(bars, neg_rates, mean_cosines)
):
    ax_d.text(
        bar.get_x() + bar.get_width()/2,
        rate + 1.4,
        f"{rate:.2f}%",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold"
    )

    ax_d.text(
        bar.get_x() + bar.get_width()/2,
        33,
        f"mean cosine\n{cos:.3f}",
        ha="center",
        va="center",
        fontsize=7.2,
        color="white" if i == 0 else "black",
        fontweight="bold"
    )

# ============================================================
# Common formatting
# ============================================================
for ax in axes.flatten():
    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.5,
        alpha=0.35
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        width=0.8,
        length=3.5
    )

fig.subplots_adjust(
    left=0.09,
    right=0.985,
    bottom=0.09,
    top=0.965,
    wspace=0.24,
    hspace=0.42
)

pdf_path = os.path.join(
    OUT_DIR,
    "gradient_mechanism_figure_final_v2.pdf"
)

png_path = os.path.join(
    OUT_DIR,
    "gradient_mechanism_figure_final_v2.png"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

print("=" * 76)
print("FINAL V2 FIGURE SAVED")
print("=" * 76)
print("PDF:", pdf_path)
print("PNG:", png_path)

print()
print(
    f"Few batch presence: "
    f"{batch_presence[2]:.2f}%"
)
print(
    f"Few sample share: "
    f"{sample_share[2]:.2f}%"
)
print(
    f"Few raw gradient mean: "
    f"{grad_means[2]:.4f}"
)
print(
    f"Many raw gradient mean: "
    f"{grad_means[0]:.4f}"
)
print(
    f"Vanilla Many-Few conflict: "
    f"{van_neg:.2f}%"
)
print(
    f"Reweighted Many-Few conflict: "
    f"{bal_neg:.2f}%"
)
