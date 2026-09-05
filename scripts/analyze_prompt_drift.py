import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = "/root/autodl-tmp/clip-lt"
COOP_DIR = os.path.join(ROOT, "results/coop_imagenet_lt")

TRAIN_LIST = os.path.join(
    ROOT,
    "SADE-AgnosticLT/data_txt/ImageNet_LT/ImageNet_LT_train.txt"
)

# ============================================================
# 1. Class frequencies
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
# 2. Load zero-shot text features
# ============================================================

zs_path = os.path.join(
    COOP_DIR,
    "zeroshot_text_features.npy"
)

zs = np.load(zs_path).astype(np.float32)

zs = zs / (
    np.linalg.norm(zs, axis=1, keepdims=True) + 1e-12
)

print("=" * 100)
print("ZERO-SHOT FEATURES")
print("=" * 100)
print("Shape:", zs.shape)


# ============================================================
# 3. Analyze semantic drift
# ============================================================

rows = []

for epoch in range(1, 11):

    path = os.path.join(
        COOP_DIR,
        f"epoch_{epoch}_text_features.npy"
    )

    if not os.path.exists(path):
        print("Missing:", path)
        continue

    coop = np.load(path).astype(np.float32)

    coop = coop / (
        np.linalg.norm(
            coop,
            axis=1,
            keepdims=True
        ) + 1e-12
    )

    # cosine similarity between zero-shot and CoOp prototype
    cosine = np.sum(
        zs * coop,
        axis=1
    )

    # semantic drift
    drift = 1.0 - cosine

    frequencies = np.array(
        [freq[i] for i in range(1000)]
    )

    groups = np.array(
        [shot_group(freq[i]) for i in range(1000)]
    )

    log_freq = np.log10(frequencies)

    pearson_r, pearson_p = pearsonr(
        log_freq,
        drift
    )

    spearman_r, spearman_p = spearmanr(
        frequencies,
        drift
    )

    many_drift = drift[
        groups == "Many"
    ].mean()

    medium_drift = drift[
        groups == "Medium"
    ].mean()

    few_drift = drift[
        groups == "Few"
    ].mean()

    rows.append({
        "epoch": epoch,
        "mean_drift": drift.mean(),
        "many_drift": many_drift,
        "medium_drift": medium_drift,
        "few_drift": few_drift,
        "many_few_drift_gap":
            many_drift - few_drift,
        "freq_drift_pearson":
            pearson_r,
        "freq_drift_pearson_p":
            pearson_p,
        "freq_drift_spearman":
            spearman_r,
        "freq_drift_spearman_p":
            spearman_p,
    })


# ============================================================
# 4. Print results
# ============================================================

df = pd.DataFrame(rows)

print("\n" + "=" * 120)
print("PROMPT SEMANTIC DRIFT")
print("=" * 120)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 5. Save
# ============================================================

out = os.path.join(
    COOP_DIR,
    "prompt_semantic_drift.csv"
)

df.to_csv(
    out,
    index=False
)

print("\nSaved:")
print(out)


# ============================================================
# 6. Epoch 10 detailed class-level drift
# ============================================================

epoch = 10

path = os.path.join(
    COOP_DIR,
    f"epoch_{epoch}_text_features.npy"
)

if os.path.exists(path):

    coop = np.load(path).astype(np.float32)

    coop = coop / (
        np.linalg.norm(
            coop,
            axis=1,
            keepdims=True
        ) + 1e-12
    )

    cosine = np.sum(
        zs * coop,
        axis=1
    )

    drift = 1.0 - cosine

    detail = pd.DataFrame({
        "label": range(1000),
        "train_frequency": [
            freq[i] for i in range(1000)
        ],
        "shot_group": [
            shot_group(freq[i])
            for i in range(1000)
        ],
        "semantic_drift": drift
    })

    detail = detail.sort_values(
        "semantic_drift",
        ascending=False
    )

    detail_path = os.path.join(
        COOP_DIR,
        "epoch10_class_semantic_drift.csv"
    )

    detail.to_csv(
        detail_path,
        index=False
    )

    print("\n" + "=" * 120)
    print("TOP 20 MOST DRIFTED CLASSES — EPOCH 10")
    print("=" * 120)

    print(
        detail.head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nDetailed class drift saved:")
    print(detail_path)
