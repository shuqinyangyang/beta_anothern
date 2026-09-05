import os
import io
import csv
import tarfile
from collections import defaultdict, Counter

import numpy as np
import torch
import clip
from PIL import Image
from scipy.io import loadmat


# ============================================================
# Paths
# ============================================================

ROOT = "/root/autodl-tmp/clip-lt"

SPLIT_ROOT = os.path.join(
    ROOT,
    "SADE-AgnosticLT/data_txt/ImageNet_LT"
)

TRAIN_LIST = os.path.join(
    SPLIT_ROOT,
    "ImageNet_LT_train.txt"
)

TEST_LIST = os.path.join(
    SPLIT_ROOT,
    "ImageNet_LT_test.txt"
)

DEVKIT = (
    ROOT +
    "/data/imagenet_devkit/"
    "ILSVRC2012_devkit_t12/data/meta.mat"
)

TRAIN_TAR = (
    "/root/autodl-pub/ImageNet/ILSVRC2012/"
    "ILSVRC2012_img_train.tar"
)

VAL_TAR = (
    "/root/autodl-pub/ImageNet/ILSVRC2012/"
    "ILSVRC2012_img_val.tar"
)

RESULT_DIR = os.path.join(
    ROOT,
    "results/imagenet_lt_clip_baseline"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 64

# First run:
# only process 100 test images.
MAX_IMAGES = 100


# ============================================================
# Read ImageNet-LT train list
# ============================================================

print("=" * 70)
print("ImageNet-LT CLIP Baseline")
print("=" * 70)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


train_items = []

with open(TRAIN_LIST, "r") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        path, label = line.rsplit(" ", 1)

        train_items.append(
            (path, int(label))
        )


# ============================================================
# Train frequency
# ============================================================

train_counts = Counter(
    label
    for _, label in train_items
)

print(
    "Train images:",
    len(train_items)
)

print(
    "Classes:",
    len(train_counts)
)

print(
    "Frequency range:",
    min(train_counts.values()),
    "~",
    max(train_counts.values())
)


# ============================================================
# ImageNet-LT label -> WNID
# ============================================================

label_to_wnid = {}

for path, label in train_items:

    if label not in label_to_wnid:

        wnid = path.split("/")[-2]

        label_to_wnid[label] = wnid


assert len(label_to_wnid) == 1000


# ============================================================
# WNID -> official ImageNet name
# ============================================================

mat = loadmat(DEVKIT)

synsets = mat["synsets"]

wnid_to_name = {}

for row in synsets:

    item = row[0]

    wnid = str(item[1][0])

    words = str(item[2][0])

    wnid_to_name[wnid] = words


# ============================================================
# Clean class names
# ============================================================

def clean_class_name(name):

    # ImageNet has multiple synonyms separated by commas.
    #
    # Use the first/common name for the initial baseline.

    name = name.split(",")[0]

    return name.strip()


class_names = []

for label in range(1000):

    wnid = label_to_wnid[label]

    name = wnid_to_name[wnid]

    name = clean_class_name(name)

    class_names.append(name)


print("\nFirst 10 prompts:")

for i in range(10):

    print(
        i,
        "->",
        class_names[i]
    )


# ============================================================
# Head / Medium / Few
# ============================================================

def shot_group(n):

    if n > 100:
        return "Many"

    if n > 20:
        return "Medium"

    return "Few"


group_counts = Counter(
    shot_group(train_counts[c])
    for c in range(1000)
)

print("\nShot groups:")

for group in [
    "Many",
    "Medium",
    "Few"
]:

    print(
        group,
        group_counts[group],
        "classes"
    )


# ============================================================
# Load CLIP
# ============================================================

print("\nLoading CLIP...")

model, preprocess = clip.load(
    "ViT-B/32",
    device=DEVICE
)

model.eval()


# ============================================================
# Text features
# ============================================================

prompts = [
    f"a photo of a {name}"
    for name in class_names
]

tokens = clip.tokenize(
    prompts
).to(DEVICE)

with torch.no_grad():

    text_features = model.encode_text(
        tokens
    )

    text_features /= (
        text_features.norm(
            dim=-1,
            keepdim=True
        )
    )

print(
    "Text features:",
    tuple(text_features.shape)
)


# ============================================================
# Read first MAX_IMAGES test entries
# ============================================================

test_items = []

with open(TEST_LIST, "r") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        path, label = line.rsplit(" ", 1)

        test_items.append(
            (path, int(label))
        )

        if len(test_items) >= MAX_IMAGES:
            break


print(
    "\nTest images for this run:",
    len(test_items)
)


# ============================================================
# Group by synset
# ============================================================

groups = defaultdict(list)

for path, label in test_items:

    parts = path.split("/")

    synset = parts[-2]

    filename = parts[-1]

    groups[synset].append(
        (filename, label)
    )


# ============================================================
# Read images from validation tar
# ============================================================

print(
    "\nOpening validation tar..."
)

outer_tar = tarfile.open(
    VAL_TAR,
    "r:"
)


# ============================================================
# Results
# ============================================================

results = []

processed = 0


# ============================================================
# Process groups
# ============================================================

for synset, samples in groups.items():

    requested = {
        filename
        for filename, _ in samples
    }

    images = {}

    prefix = synset + "/"

    for member in outer_tar:

        if not member.isfile():
            continue

        if not member.name.startswith(prefix):
            continue

        filename = os.path.basename(
            member.name
        )

        if filename not in requested:
            continue

        f = outer_tar.extractfile(member)

        if f is not None:

            images[filename] = f.read()

        if len(images) == len(requested):
            break


    batch_images = []

    batch_labels = []

    batch_names = []

    for filename, label in samples:

        if filename not in images:

            print(
                "MISSING:",
                synset,
                filename
            )

            continue

        image = Image.open(
            io.BytesIO(images[filename])
        ).convert("RGB")

        batch_images.append(
            preprocess(image)
        )

        batch_labels.append(label)

        batch_names.append(filename)


    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    for start in range(
        0,
        len(batch_images),
        BATCH_SIZE
    ):

        batch = torch.stack(
            batch_images[
                start:start+BATCH_SIZE
            ]
        ).to(DEVICE)

        labels = batch_labels[
            start:start+BATCH_SIZE
        ]

        filenames = batch_names[
            start:start+BATCH_SIZE
        ]

        with torch.no_grad():

            image_features = (
                model.encode_image(batch)
            )

            image_features /= (
                image_features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

            logits = (
                100.0 *
                image_features @
                text_features.T
            )

        predictions = logits.argmax(
            dim=1
        ).cpu().numpy()

        logits_np = logits.cpu().numpy()

        image_np = (
            image_features.cpu().numpy()
        )

        text_np = (
            text_features.cpu().numpy()
        )

        for i, label in enumerate(labels):

            pred = int(predictions[i])

            true_logit = float(
                logits_np[i, label]
            )

            true_similarity = float(
                image_np[i] @ text_np[label]
            )

            wrong = logits_np[i].copy()

            wrong[label] = -np.inf

            best_wrong = float(
                wrong.max()
            )

            margin = (
                true_logit -
                best_wrong
            )

            results.append({

                "path":
                    f"{synset}/{filenames[i]}",

                "label":
                    label,

                "wnid":
                    label_to_wnid[label],

                "class_name":
                    class_names[label],

                "train_frequency":
                    train_counts[label],

                "shot_group":
                    shot_group(
                        train_counts[label]
                    ),

                "prediction":
                    pred,

                "correct":
                    int(pred == label),

                "true_logit":
                    true_logit,

                "true_similarity":
                    true_similarity,

                "margin":
                    margin
            })

            processed += 1


outer_tar.close()


# ============================================================
# Print sanity results
# ============================================================

print("\n" + "=" * 70)
print("SANITY CHECK")
print("=" * 70)

print(
    "Processed:",
    processed
)

if results:

    accuracy = np.mean([
        r["correct"]
        for r in results
    ])

    print(
        "Accuracy:",
        f"{accuracy:.4f}"
    )

    print("\nFirst 10 predictions:")

    for r in results[:10]:

        print(
            f'{r["class_name"]:25s} | '
            f'pred={class_names[r["prediction"]]:25s} | '
            f'freq={r["train_frequency"]:4d} | '
            f'logit={r["true_logit"]:7.3f} | '
            f'sim={r["true_similarity"]:.4f} | '
            f'correct={r["correct"]}'
        )


# ============================================================
# Save CSV
# ============================================================

csv_path = os.path.join(
    RESULT_DIR,
    "sanity_100_images.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()

    writer.writerows(results)


print(
    "\nSaved:",
    csv_path
)

print(
    "\nSanity check complete."
)

