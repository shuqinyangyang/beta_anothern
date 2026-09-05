import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

RESULT_FILE = "/root/autodl-tmp/clip-lt/results/imagenet_lt_clip_baseline/sanity_100_images.csv"

df = pd.read_csv(RESULT_FILE)

print("=" * 70)
print("CLIP ERROR BIAS ANALYSIS")
print("=" * 70)

print("Total samples:", len(df))
print("Accuracy:", df["correct"].mean())

# ------------------------------------------------------------
# Build class frequency table
# ------------------------------------------------------------

freq = (
    df[["label", "train_frequency"]]
    .drop_duplicates()
    .set_index("label")["train_frequency"]
)

# prediction frequency
df["pred_frequency"] = df["prediction"].map(freq)

# true frequency
df["true_frequency"] = df["label"].map(freq)

# ------------------------------------------------------------
# Errors only
# ------------------------------------------------------------

errors = df[df["correct"] == 0].copy()

print("\nErrors:", len(errors))
print("Error rate:", len(errors) / len(df))

# ------------------------------------------------------------
# Frequency comparison
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRUE CLASS vs PREDICTED CLASS FREQUENCY")
print("=" * 70)

print(
    "Mean true frequency:",
    errors["true_frequency"].mean()
)

print(
    "Mean predicted frequency:",
    errors["pred_frequency"].mean()
)

print(
    "Median true frequency:",
    errors["true_frequency"].median()
)

print(
    "Median predicted frequency:",
    errors["pred_frequency"].median()
)

# log frequency
true_log = np.log10(errors["true_frequency"])
pred_log = np.log10(errors["pred_frequency"])

print(
    "\nMean log10(true frequency):",
    true_log.mean()
)

print(
    "Mean log10(pred frequency):",
    pred_log.mean()
)

# ------------------------------------------------------------
# Correlation
# ------------------------------------------------------------

r1, p1 = pearsonr(
    true_log,
    pred_log
)

r2, p2 = spearmanr(
    true_log,
    pred_log
)

print("\nCorrelation between true and predicted class frequency:")
print(
    f"Pearson r = {r1:.4f}, p = {p1:.3e}"
)

print(
    f"Spearman rho = {r2:.4f}, p = {p2:.3e}"
)

# ------------------------------------------------------------
# Tail -> Head errors
# ------------------------------------------------------------

def group(freq):
    if freq > 100:
        return "Many"
    elif freq >= 20:
        return "Medium"
    else:
        return "Few"

errors["true_group"] = errors["true_frequency"].apply(group)
errors["pred_group"] = errors["pred_frequency"].apply(group)

print("\n" + "=" * 70)
print("ERROR TRANSITION MATRIX")
print("=" * 70)

transition = pd.crosstab(
    errors["true_group"],
    errors["pred_group"],
    normalize="index"
)

transition = transition.reindex(
    index=["Many", "Medium", "Few"],
    columns=["Many", "Medium", "Few"],
    fill_value=0
)

print(
    (transition * 100).round(2).to_string()
)

# ------------------------------------------------------------
# Count transition matrix
# ------------------------------------------------------------

print("\nCounts:")

counts = pd.crosstab(
    errors["true_group"],
    errors["pred_group"]
)

counts = counts.reindex(
    index=["Many", "Medium", "Few"],
    columns=["Many", "Medium", "Few"],
    fill_value=0
)

print(counts.to_string())

# ------------------------------------------------------------
# Tail -> Many
# ------------------------------------------------------------

few_errors = errors[
    errors["true_group"] == "Few"
]

print("\n" + "=" * 70)
print("FEW -> MANY")
print("=" * 70)

if len(few_errors) > 0:

    ratio = (
        few_errors["pred_group"]
        .eq("Many")
        .mean()
    )

    print(
        "Few-class errors predicted as Many:",
        f"{ratio:.4f}"
    )

    print(
        "Few-class errors:",
        len(few_errors)
    )

# ------------------------------------------------------------
# Most common error pairs
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP ERROR PAIRS")
print("=" * 70)

pairs = (
    errors
    .groupby(
        [
            "class_name",
            "prediction_name",
            "true_frequency",
            "pred_frequency"
        ]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(
    pairs.head(30).to_string(index=False)
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUT = "/root/autodl-tmp/clip-lt/results/imagenet_lt_clip_baseline/clip_error_bias.csv"

errors.to_csv(OUT, index=False)

print("\nSaved:")
print(OUT)

print("\nDONE.")
