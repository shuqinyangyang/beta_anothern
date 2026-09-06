import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
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

OUT_DIR = os.path.join(ROOT, "results/gradient_mechanism_figure")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================
van_summary = pd.read_csv(
    os.path.join(VANILLA_DIR, "gradient_probe_summary.csv")
)
van_batch = pd.read_csv(
    os.path.join(VANILLA_DIR, "gradient_probe_per_batch.csv")
)
van_cos = pd.read_csv(
    os.path.join(VANILLA_DIR, "gradient_probe_cosines.csv")
)

bal_summary = pd.read_csv(
    os.path.join(BALANCED_DIR, "gradient_probe_summary.csv")
)
bal_cos = pd.read_csv(
    os.path.join(BALANCED_DIR, "gradient_probe_cosines.csv")
)

GROUPS = ["Many", "Medium", "Few"]

# ============================================================
# Helper functions
# ============================================================
def summary_value(df, group, column):
    return float(
        df.loc[df["group"] == group, column].iloc[0]
    )


def cosine_value(df, a, b, column):
    row = df[
        ((df["group_a"] == a) & (df["group_b"] == b))
        | ((df["group_a"] == b) & (df["group_b"] == a))
    ]
    if len(row) != 1:
        raise RuntimeError(f"Cannot uniquely find cosine pair {a}-{b}")
    return float(row[column].iloc[0])


# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

colors = {
    "Many": "#4C72B0",
    "Medium": "#55A868",
    "Few": "#C44E52",
}

# ============================================================
# Figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8))
ax_a, ax_b, ax_c, ax_d = axes.flatten()

# ============================================================
# (a) Temporal sparsity
# ============================================================
batch_presence = [
    100.0 * summary_value(van_summary, g, "batches_present") / 500.0
    for g in GROUPS
]

sample_share = [
    100.0 * summary_value(van_summary, g, "sample_share")
    for g in GROUPS
]

x = np.arange(len(GROUPS))

bars = ax_a.bar(
    x,
    batch_presence,
    width=0.62,
    color=[colors[g] for g in GROUPS],
    edgecolor="black",
    linewidth=0.6,
)

ax_a.set_title("(a) Temporal sparsity")
ax_a.set_ylabel("Batch presence (%)")
ax_a.set_xticks(x)
ax_a.set_xticklabels(GROUPS)
ax_a.set_ylim(0, 112)
ax_a.axhline(100, linestyle="--", linewidth=0.8, color="0.7")

for i, (bar, presence, share) in enumerate(
    zip(bars, batch_presence, sample_share)
):
    ax_a.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 2.0,
        f"{presence:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold"
    )

    ax_a.text(
        bar.get_x() + bar.get_width()/2,
        max(3, bar.get_height()*0.52),
        f"{share:.2f}%\nof samples",
        ha="center",
        va="center",
        fontsize=7.5,
        color="white",
        fontweight="bold"
    )

ax_a.annotate(
    "Few absent in 81.8%\nof mini-batches",
    xy=(2, batch_presence[2]),
    xytext=(1.28, 55),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.0,
        color="0.25"
    ),
    fontsize=8,
    ha="center"
)

# ============================================================
# (b) Raw gradient norm distribution
# ============================================================
grad_data = [
    van_batch.loc[
        van_batch["group"] == g, "raw_grad_norm"
    ].values
    for g in GROUPS
]

bp = ax_b.boxplot(
    grad_data,
    positions=np.arange(1, 4),
    widths=0.55,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.2),
    whiskerprops=dict(linewidth=0.8),
    capprops=dict(linewidth=0.8),
)

for patch, g in zip(bp["boxes"], GROUPS):
    patch.set_facecolor(colors[g])
    patch.set_alpha(0.8)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.6)

# deterministic jittered individual points
rng = np.random.default_rng(7)

for i, (g, vals) in enumerate(zip(GROUPS, grad_data), start=1):
    jitter = rng.normal(0, 0.055, len(vals))

    ax_b.scatter(
        np.full(len(vals), i) + jitter,
        vals,
        s=7,
        alpha=0.22,
        color=colors[g],
        edgecolors="none",
        rasterized=True,
    )

means = [
    summary_value(van_summary, g, "avg_group_mean_grad_norm")
    for g in GROUPS
]

ax_b.scatter(
    np.arange(1, 4),
    means,
    marker="D",
    s=27,
    color="black",
    zorder=5,
    label="Mean"
)

ax_b.set_title("(b) Per-occurrence gradient strength")
ax_b.set_ylabel(r"Raw gradient norm $\|g\|_2$")
ax_b.set_xticks([1, 2, 3])
ax_b.set_xticklabels(GROUPS)
ax_b.set_ylim(bottom=0)

for i, m in enumerate(means, start=1):
    ax_b.text(
        i + 0.10,
        m,
        f"{m:.2f}",
        fontsize=7.5,
        va="center"
    )

ax_b.annotate(
    "Strong when observed",
    xy=(3, np.median(grad_data[2])),
    xytext=(2.12, 47),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.0,
        color="0.25"
    ),
    fontsize=8
)

ax_b.legend(
    frameon=False,
    loc="upper left"
)

# ============================================================
# (c) Effective batch contribution
# ============================================================
contrib_means = [
    summary_value(
        van_summary,
        g,
        "avg_batch_contribution_norm"
    )
    for g in GROUPS
]

bars = ax_c.bar(
    x,
    contrib_means,
    width=0.62,
    color=[colors[g] for g in GROUPS],
    edgecolor="black",
    linewidth=0.6,
)

ax_c.set_title("(c) Effective batch contribution")
ax_c.set_ylabel("Contribution norm")
ax_c.set_xticks(x)
ax_c.set_xticklabels(GROUPS)
ax_c.set_ylim(0, max(contrib_means) * 1.35)

for bar, value in zip(bars, contrib_means):
    ax_c.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.05,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold"
    )

ax_c.annotate(
    "Largest raw gradient,\nbut lowest average\nbatch contribution",
    xy=(2, contrib_means[2]),
    xytext=(1.18, 2.50),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.0,
        color="0.25"
    ),
    fontsize=7.8,
    ha="center"
)

# ============================================================
# (d) Many–Few conflict before/after reweighting
# ============================================================
van_neg = 100 * cosine_value(
    van_cos, "Many", "Few", "negative_fraction"
)
bal_neg = 100 * cosine_value(
    bal_cos, "Many", "Few", "negative_fraction"
)

van_mean_cos = cosine_value(
    van_cos, "Many", "Few", "mean_cosine"
)
bal_mean_cos = cosine_value(
    bal_cos, "Many", "Few", "mean_cosine"
)

methods = ["Vanilla\nCoOp", "Reweighted\nCoOp"]
neg_rates = [van_neg, bal_neg]

bars = ax_d.bar(
    np.arange(2),
    neg_rates,
    width=0.55,
    color=["#777777", "#BBBBBB"],
    edgecolor="black",
    linewidth=0.7,
)

ax_d.set_title("(d) Persistent head–tail conflict")
ax_d.set_ylabel("Conflicting Many–Few batches (%)")
ax_d.set_xticks(np.arange(2))
ax_d.set_xticklabels(methods)
ax_d.set_ylim(0, 75)

for i, (bar, rate, cos) in enumerate(
    zip(bars, neg_rates, [van_mean_cos, bal_mean_cos])
):
    ax_d.text(
        bar.get_x() + bar.get_width()/2,
        rate + 1.3,
        f"{rate:.2f}%",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold"
    )

    ax_d.text(
        bar.get_x() + bar.get_width()/2,
        rate * 0.53,
        f"mean cos.\n{cos:.3f}",
        ha="center",
        va="center",
        fontsize=7.5,
        color="white" if i == 0 else "black",
        fontweight="bold"
    )

ax_d.axhline(
    50,
    linestyle="--",
    linewidth=0.9,
    color="0.55"
)

ax_d.text(
    1.42,
    50.8,
    "50%",
    fontsize=7,
    color="0.4"
)

ax_d.text(
    0.5,
    8,
    "Reweighting changes gradient magnitude,\n"
    "but conflict remains frequent.",
    ha="center",
    va="center",
    fontsize=7.7
)

# ============================================================
# Common styling
# ============================================================
for ax in axes.flatten():
    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.55,
        alpha=0.4
    )
    ax.set_axisbelow(True)

fig.suptitle(
    "Gradient Dynamics under Long-Tailed Prompt Learning",
    fontsize=12,
    fontweight="bold",
    y=0.995
)

fig.tight_layout(
    rect=[0, 0.015, 1, 0.965],
    h_pad=2.0,
    w_pad=1.8
)

png_path = os.path.join(
    OUT_DIR,
    "gradient_mechanism_figure.png"
)

pdf_path = os.path.join(
    OUT_DIR,
    "gradient_mechanism_figure.pdf"
)

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.close(fig)

print("=" * 80)
print("Saved gradient mechanism figure")
print("=" * 80)
print("PNG:", png_path)
print("PDF:", pdf_path)

print("\nKey statistics:")
print(f"Few batch presence : {batch_presence[2]:.2f}%")
print(f"Few sample share   : {sample_share[2]:.2f}%")
print(f"Raw grad norms     : {dict(zip(GROUPS, means))}")
print(f"Contribution norms : {dict(zip(GROUPS, contrib_means))}")
print(f"Vanilla conflict   : {van_neg:.2f}%")
print(f"Balanced conflict  : {bal_neg:.2f}%")
print(f"Vanilla mean cos   : {van_mean_cos:.4f}")
print(f"Balanced mean cos  : {bal_mean_cos:.4f}")
