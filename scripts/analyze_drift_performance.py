import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = "/root/autodl-tmp/clip-lt"

ZS_FILE = os.path.join(
    ROOT,
    "results/imagenet_lt_clip_baseline/per_class_analysis.csv"
)

COOP_DIR = os.path.join(
    ROOT,
    "results/coop_imagenet_lt"
)

TRAIN_LIST = os.path.join(
    ROOT,
    "SADE-AgnosticLT/data_txt/ImageNet_LT/ImageNet_LT_train.txt"
)

EPOCH = 5   # best CoOp epoch


# ============================================================
# 1. Train frequency
# ============================================================

freq = {}

with open(TRAIN_LIST) as f:
    for line in f:
        _, label = line.strip().rsplit(" ", 1)
        label = int(label)
        freq[label] = freq.get(label, 0) + 1


def shot_group(n):
    if n > 100:
        return "Many"
    elif n >= 20:
        return "Medium"
    else:
        return "Few"


# ============================================================
# 2. Zero-shot class statistics
# ============================================================

zs = pd.read_csv(ZS_FILE)

zs = zs[
    [
        "label",
        "accuracy",
        "mean_margin",
        "mean_logit",
        "mean_similarity"
    ]
].copy()

zs = zs.rename(columns={
    "accuracy": "zs_acc",
    "mean_margin": "zs_margin",
    "mean_logit": "zs_logit",
    "mean_similarity": "zs_similarity"
})


# ============================================================
# 3. CoOp class statistics
# ============================================================

coop_file = os.path.join(
    COOP_DIR,
    f"epoch_{EPOCH}_predictions.csv"
)

coop_raw = pd.read_csv(coop_file)

coop = (
    coop_raw
    .groupby("label")
    .agg(
        coop_acc=("correct", "mean"),
        coop_margin=("margin", "mean"),
        coop_logit=("true_logit", "mean"),
        coop_similarity=("true_similarity", "mean")
    )
    .reset_index()
)


# ============================================================
# 4. Semantic drift
# ============================================================

zs_text = np.load(
    os.path.join(
        COOP_DIR,
        "zeroshot_text_features.npy"
    )
).astype(np.float32)

coop_text = np.load(
    os.path.join(
        COOP_DIR,
        f"epoch_{EPOCH}_text_features.npy"
    )
).astype(np.float32)

zs_text /= (
    np.linalg.norm(zs_text, axis=1, keepdims=True)
    + 1e-12
)

coop_text /= (
    np.linalg.norm(coop_text, axis=1, keepdims=True)
    + 1e-12
)

semantic_drift = (
    1.0 - np.sum(zs_text * coop_text, axis=1)
)

drift_df = pd.DataFrame({
    "label": np.arange(1000),
    "semantic_drift": semantic_drift
})


# ============================================================
# 5. Merge
# ============================================================

df = (
    zs
    .merge(coop, on="label")
    .merge(drift_df, on="label")
)

df["train_frequency"] = df["label"].map(freq)

df["shot_group"] = (
    df["train_frequency"].apply(shot_group)
)

df["delta_acc"] = (
    df["coop_acc"] - df["zs_acc"]
)

df["delta_margin"] = (
    df["coop_margin"] - df["zs_margin"]
)

df["delta_logit"] = (
    df["coop_logit"] - df["zs_logit"]
)

df["delta_similarity"] = (
    df["coop_similarity"] - df["zs_similarity"]
)


# ============================================================
# 6. Overall correlations
# ============================================================

print("=" * 100)
print(f"DRIFT vs PERFORMANCE CHANGE — COOP EPOCH {EPOCH}")
print("=" * 100)


def correlation(x, y, name):

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    print(
        f"{name:28s} | "
        f"Pearson r={pr: .4f} p={pp:.3e} | "
        f"Spearman rho={sr: .4f} p={sp:.3e}"
    )


correlation(
    df["semantic_drift"],
    df["delta_acc"],
    "drift vs delta accuracy"
)

correlation(
    df["semantic_drift"],
    df["delta_margin"],
    "drift vs delta margin"
)

correlation(
    df["semantic_drift"],
    df["delta_logit"],
    "drift vs delta logit"
)

correlation(
    df["semantic_drift"],
    df["delta_similarity"],
    "drift vs delta similarity"
)


# ============================================================
# 7. Group-level changes
# ============================================================

print("\n" + "=" * 100)
print("GROUP PERFORMANCE CHANGE")
print("=" * 100)

group_stats = (
    df.groupby("shot_group")
    .agg(
        classes=("label", "count"),

        mean_frequency=("train_frequency", "mean"),

        zs_acc=("zs_acc", "mean"),
        coop_acc=("coop_acc", "mean"),
        delta_acc=("delta_acc", "mean"),

        zs_margin=("zs_margin", "mean"),
        coop_margin=("coop_margin", "mean"),
        delta_margin=("delta_margin", "mean"),

        semantic_drift=("semantic_drift", "mean"),
    )
    .reindex(["Many", "Medium", "Few"])
)

print(
    group_stats.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 8. Correlations inside each shot group
# ============================================================

print("\n" + "=" * 100)
print("DRIFT CORRELATION WITHIN EACH GROUP")
print("=" * 100)

for group in ["Many", "Medium", "Few"]:

    sub = df[df["shot_group"] == group]

    print(
        f"\n[{group}] n={len(sub)}"
    )

    correlation(
        sub["semantic_drift"],
        sub["delta_acc"],
        "drift vs delta accuracy"
    )

    correlation(
        sub["semantic_drift"],
        sub["delta_margin"],
        "drift vs delta margin"
    )


# ============================================================
# 9. Worst degraded classes
# ============================================================

print("\n" + "=" * 100)
print("TOP 20 MOST DEGRADED CLASSES")
print("=" * 100)

cols = [
    "label",
    "train_frequency",
    "shot_group",
    "semantic_drift",
    "zs_acc",
    "coop_acc",
    "delta_acc",
    "delta_margin"
]

print(
    df.sort_values(
        "delta_acc"
    )[cols]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 10. Save
# ============================================================

out = os.path.join(
    COOP_DIR,
    f"epoch_{EPOCH}_drift_performance.csv"
)

df.to_csv(out, index=False)

print("\nSaved:")
print(out)

print("\nDONE.")
