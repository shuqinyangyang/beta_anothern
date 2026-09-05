import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

RESULT_FILE = "/root/autodl-tmp/clip-lt/results/imagenet_lt_clip_baseline/sanity_100_images.csv"

OUT_DIR = "/root/autodl-tmp/clip-lt/results/imagenet_lt_clip_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(RESULT_FILE)

print("=" * 70)
print("Loaded:", len(df), "test images")
print("=" * 70)

# ------------------------------------------------------------
# Check columns
# ------------------------------------------------------------

print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# Per-class statistics
# ------------------------------------------------------------

class_stats = (
    df.groupby(["label", "class_name", "train_frequency"])
      .agg(
          test_images=("correct", "count"),
          accuracy=("correct", "mean"),
          mean_logit=("true_logit", "mean"),
          mean_similarity=("true_similarity", "mean"),
          mean_margin=("margin", "mean"),
      )
      .reset_index()
)

# Standard deviation
std_stats = (
    df.groupby("label")
      .agg(
          logit_std=("true_logit", "std"),
          similarity_std=("true_similarity", "std"),
          margin_std=("margin", "std"),
      )
      .reset_index()
)

class_stats = class_stats.merge(std_stats, on="label")

# ------------------------------------------------------------
# Head / Medium / Few
# ------------------------------------------------------------

def shot_group(freq):
    if freq > 100:
        return "Many"
    elif freq >= 20:
        return "Medium"
    else:
        return "Few"

class_stats["shot_group"] = class_stats["train_frequency"].apply(shot_group)

# ------------------------------------------------------------
# Save per-class table
# ------------------------------------------------------------

class_file = os.path.join(OUT_DIR, "per_class_analysis.csv")
class_stats.to_csv(class_file, index=False)

print("\nPer-class statistics saved to:")
print(class_file)

# ------------------------------------------------------------
# Overall statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HEAD / MEDIUM / FEW")
print("=" * 70)

group_stats = (
    class_stats.groupby("shot_group")
    .agg(
        classes=("label", "count"),
        mean_frequency=("train_frequency", "mean"),
        accuracy=("accuracy", "mean"),
        mean_logit=("mean_logit", "mean"),
        mean_similarity=("mean_similarity", "mean"),
        mean_margin=("mean_margin", "mean"),
    )
    .reindex(["Many", "Medium", "Few"])
)

print(group_stats.to_string(float_format=lambda x: f"{x:.4f}"))

group_file = os.path.join(OUT_DIR, "head_medium_few.csv")
group_stats.to_csv(group_file)

# ------------------------------------------------------------
# Correlations
# ------------------------------------------------------------

x = np.log10(class_stats["train_frequency"].values)

metrics = {
    "accuracy": class_stats["accuracy"].values,
    "mean_logit": class_stats["mean_logit"].values,
    "mean_similarity": class_stats["mean_similarity"].values,
    "mean_margin": class_stats["mean_margin"].values,
}

print("\n" + "=" * 70)
print("FREQUENCY CORRELATION")
print("=" * 70)

rows = []

for name, y in metrics.items():

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    print(
        f"{name:20s} | "
        f"Pearson r={pearson_r: .4f} p={pearson_p:.3e} | "
        f"Spearman rho={spearman_r: .4f} p={spearman_p:.3e}"
    )

    rows.append({
        "metric": name,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,
    })

corr_df = pd.DataFrame(rows)

corr_file = os.path.join(OUT_DIR, "frequency_correlations.csv")
corr_df.to_csv(corr_file, index=False)

print("\nCorrelation results saved to:")
print(corr_file)

# ------------------------------------------------------------
# Head / Tail sorted table
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 HEAD CLASSES")
print("=" * 70)

print(
    class_stats.sort_values("train_frequency", ascending=False)
    [
        [
            "label",
            "class_name",
            "train_frequency",
            "accuracy",
            "mean_logit",
            "mean_similarity",
            "mean_margin",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("BOTTOM 10 TAIL CLASSES")
print("=" * 70)

print(
    class_stats.sort_values("train_frequency")
    [
        [
            "label",
            "class_name",
            "train_frequency",
            "accuracy",
            "mean_logit",
            "mean_similarity",
            "mean_margin",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nDONE.")
