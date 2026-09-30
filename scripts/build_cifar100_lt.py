import os
import pickle
import numpy as np
import csv

DATA_ROOT = "data/cifar-100-python"
OUT_DIR = "data/cifar100_lt_ir100"

IMBALANCE_RATIO = 100
SEED = 1
NUM_CLASSES = 100
MAX_SAMPLES = 500

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load CIFAR-100
# --------------------------------------------------
train_file = os.path.join(DATA_ROOT, "train")

with open(train_file, "rb") as f:
    train = pickle.load(f, encoding="bytes")

labels = np.asarray(train[b"fine_labels"], dtype=np.int64)

assert len(labels) == 50000
assert len(np.unique(labels)) == NUM_CLASSES

# --------------------------------------------------
# Exponential long-tail distribution
#
# n_c = n_max * (1 / IR)^(c / (C-1))
# --------------------------------------------------
img_num_per_cls = []

for cls_idx in range(NUM_CLASSES):
    n = MAX_SAMPLES * (
        (1.0 / IMBALANCE_RATIO)
        ** (cls_idx / (NUM_CLASSES - 1))
    )
    img_num_per_cls.append(int(n))

img_num_per_cls = np.asarray(
    img_num_per_cls,
    dtype=np.int64
)

# --------------------------------------------------
# Deterministically sample each class
# --------------------------------------------------
rng = np.random.RandomState(SEED)

selected_indices = []

for cls_idx in range(NUM_CLASSES):

    cls_indices = np.where(labels == cls_idx)[0].copy()

    rng.shuffle(cls_indices)

    n = int(img_num_per_cls[cls_idx])

    selected_indices.extend(
        cls_indices[:n].tolist()
    )

selected_indices = np.asarray(
    selected_indices,
    dtype=np.int64
)

# Shuffle final LT training order
rng.shuffle(selected_indices)

selected_labels = labels[selected_indices]

# --------------------------------------------------
# Save indices
# --------------------------------------------------
np.save(
    os.path.join(
        OUT_DIR,
        "train_indices.npy"
    ),
    selected_indices
)

# --------------------------------------------------
# Save class counts
# --------------------------------------------------
counts = np.bincount(
    selected_labels,
    minlength=NUM_CLASSES
)

with open(
    os.path.join(
        OUT_DIR,
        "class_counts.csv"
    ),
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class_id",
        "num_train",
        "group"
    ])

    for c, n in enumerate(counts):

        if n > 100:
            group = "Many"
        elif n >= 20:
            group = "Medium"
        else:
            group = "Few"

        writer.writerow([
            c,
            int(n),
            group
        ])

# --------------------------------------------------
# Save manifest
# --------------------------------------------------
with open(
    os.path.join(
        OUT_DIR,
        "train_manifest.csv"
    ),
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "original_index",
        "class_id"
    ])

    for idx in selected_indices:
        writer.writerow([
            int(idx),
            int(labels[idx])
        ])

# --------------------------------------------------
# Summary
# --------------------------------------------------
many = counts[counts > 100]
medium = counts[
    (counts >= 20) &
    (counts <= 100)
]
few = counts[counts < 20]

print("=" * 60)
print("CIFAR100-LT SPLIT")
print("=" * 60)

print("IR:", IMBALANCE_RATIO)
print("Seed:", SEED)

print(
    "Original train samples:",
    len(labels)
)

print(
    "LT train samples:",
    len(selected_indices)
)

print(
    "Classes:",
    len(counts)
)

print(
    "Max samples/class:",
    counts.max()
)

print(
    "Min samples/class:",
    counts.min()
)

print(
    "Actual imbalance ratio:",
    counts.max() / counts.min()
)

print()
print(
    "Many classes (>100):",
    len(many)
)

print(
    "Medium classes (20-100):",
    len(medium)
)

print(
    "Few classes (<20):",
    len(few)
)

print()
print(
    "First 10 counts:",
    counts[:10].tolist()
)

print(
    "Last 10 counts:",
    counts[-10:].tolist()
)

print()
print(
    "Saved:",
    OUT_DIR
)
