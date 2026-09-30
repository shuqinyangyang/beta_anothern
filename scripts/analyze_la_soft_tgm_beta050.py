import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = "/root/autodl-tmp/clip-lt"

COOP_DIR = os.path.join(
    ROOT,
    "results/la_soft_tgm_tau10_beta050_lambda025"
)

TRAIN_LIST = os.path.join(
    ROOT,
    "SADE-AgnosticLT/data_txt/ImageNet_LT/ImageNet_LT_train.txt"
)

OUT_FILE = os.path.join(
    COOP_DIR,
    "coop_epoch_analysis.csv"
)

# ============================================================
# Get train frequency
# ============================================================

freq = {}

with open(TRAIN_LIST) as f:
    for line in f:
        path, label = line.strip().rsplit(" ", 1)
        label = int(label)
        freq[label] = freq.get(label, 0) + 1


def shot_group(n):
    if n > 100:
        return "Many"
    elif n >= 20:
        return "Medium"
    else:
        return "Few"


rows = []

# ============================================================
# Analyze epochs
# ============================================================

for epoch in range(1, 11):

    path = os.path.join(
        COOP_DIR,
        f"epoch_{epoch}_predictions.csv"
    )

    if not os.path.exists(path):
        print("Missing:", path)
        continue

    df = pd.read_csv(path)

    df["train_frequency"] = df["label"].map(freq)
    df["shot_group"] = df["train_frequency"].apply(shot_group)

    overall = df["correct"].mean()

    # class-balanced group accuracy
    class_stats = (
        df.groupby("label")
        .agg(
            accuracy=("correct", "mean"),
            mean_logit=("true_logit", "mean"),
            mean_similarity=("true_similarity", "mean"),
            mean_margin=("margin", "mean")
        )
        .reset_index()
    )

    class_stats["train_frequency"] = (
        class_stats["label"].map(freq)
    )

    class_stats["shot_group"] = (
        class_stats["train_frequency"].apply(shot_group)
    )

    group_acc = (
        class_stats
        .groupby("shot_group")["accuracy"]
        .mean()
    )

    many = group_acc.get("Many", np.nan)
    medium = group_acc.get("Medium", np.nan)
    few = group_acc.get("Few", np.nan)

    # correlation
    x = np.log10(
        class_stats["train_frequency"].values
    )

    y_acc = class_stats["accuracy"].values
    y_margin = class_stats["mean_margin"].values

    p_r, p_p = pearsonr(x, y_acc)
    s_r, s_p = spearmanr(x, y_acc)

    m_r, m_p = spearmanr(x, y_margin)

    rows.append({
        "epoch": epoch,
        "overall_acc": overall,
        "many_acc": many,
        "medium_acc": medium,
        "few_acc": few,
        "many_few_gap": many - few,
        "freq_acc_pearson": p_r,
        "freq_acc_pearson_p": p_p,
        "freq_acc_spearman": s_r,
        "freq_acc_spearman_p": s_p,
        "freq_margin_spearman": m_r,
        "freq_margin_p": m_p,
    })


result = pd.DataFrame(rows)

print("\n" + "=" * 100)
print("COOP EPOCH ANALYSIS")
print("=" * 100)

print(
    result.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 100)
print("BEST EPOCH")
print("=" * 100)

best = result.loc[
    result["overall_acc"].idxmax()
]

print(best.to_string())

result.to_csv(
    OUT_FILE,
    index=False
)

print("\nSaved:")
print(OUT_FILE)

