import os
import io
import csv
import tarfile
from collections import Counter

import numpy as np
import torch
import clip
from PIL import Image
from scipy.io import loadmat


# ============================================================
# Paths
# ============================================================

ROOT = "/root/autodl-tmp/clip-lt"

TRAIN_LIST = (
    ROOT +
    "/SADE-AgnosticLT/data_txt/ImageNet_LT/"
    "ImageNet_LT_train.txt"
)

TEST_LIST = (
    ROOT +
    "/SADE-AgnosticLT/data_txt/ImageNet_LT/"
    "ImageNet_LT_test.txt"
)

DEVKIT = (
    ROOT +
    "/data/imagenet_devkit/"
    "ILSVRC2012_devkit_t12/data/meta.mat"
)

VAL_TAR = (
    "/root/autodl-pub/ImageNet/ILSVRC2012/"
    "ILSVRC2012_img_val.tar"
)

RESULT_DIR = (
    ROOT +
    "/results/imagenet_lt_clip_baseline"
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

# Sanity test only
MAX_IMAGES = 50000


# ============================================================
# Print environment
# ============================================================

print("=" * 70)
print("ImageNet-LT CLIP Sanity Check")
print("=" * 70)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 1. Read ImageNet-LT train list
# ============================================================

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


train_counts = Counter(
    label
    for _, label in train_items
)

print(
    "\nTrain images:",
    len(train_items)
)

print(
    "Classes:",
    len(train_counts)
)

print(
    "Frequency:",
    min(train_counts.values()),
    "~",
    max(train_counts.values())
)


# ============================================================
# 2. ImageNet-LT label -> WNID
# ============================================================

label_to_wnid = {}

for path, label in train_items:

    if label not in label_to_wnid:

        wnid = path.split("/")[-2]

        label_to_wnid[label] = wnid


assert len(label_to_wnid) == 1000


# ============================================================
# 3. WNID -> class name
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
# 4. Class names
# ============================================================

def clean_name(name):

    # Use first synonym.
    return name.split(",")[0].strip()


class_names = []

for label in range(1000):

    wnid = label_to_wnid[label]

    name = clean_name(
        wnid_to_name[wnid]
    )

    class_names.append(name)


print("\nFirst classes:")

for i in range(10):

    print(
        f"{i:3d} | "
        f"{label_to_wnid[i]:10s} | "
        f"{class_names[i]}"
    )


# ============================================================
# 5. Shot groups
# ============================================================

def shot_group(freq):

    if freq > 100:
        return "Many"

    elif freq > 20:
        return "Medium"

    else:
        return "Few"


groups = Counter(
    shot_group(train_counts[i])
    for i in range(1000)
)

print("\nShot groups:")

for group in [
    "Many",
    "Medium",
    "Few"
]:

    print(
        f"{group:7s}: {groups[group]}"
    )


# ============================================================
# 6. Read first 100 test samples
# ============================================================

test_items = []

with open(TEST_LIST, "r") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        path, label = line.rsplit(" ", 1)

        label = int(label)

        filename = os.path.basename(path)

        wnid = path.split("/")[-2]

        test_items.append({
            "filename": filename,
            "label": label,
            "wnid": wnid,
        })

        if len(test_items) >= MAX_IMAGES:
            break


print(
    "\nTest samples:",
    len(test_items)
)


# ============================================================
# 7. Create filename -> metadata
# ============================================================

wanted = {}

for item in test_items:

    wanted[item["filename"]] = item


print(
    "Unique filenames:",
    len(wanted)
)


# ============================================================
# 8. Load CLIP
# ============================================================

print("\nLoading CLIP...")

model, preprocess = clip.load(
    "ViT-B/32",
    device=DEVICE
)

model.eval()


# ============================================================
# 9. Encode text
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
# 10. Scan validation tar ONCE
# ============================================================

print("\nScanning validation tar...")

found_images = {}

with tarfile.open(
    VAL_TAR,
    "r:"
) as tar:

    for member in tar:

        if not member.isfile():
            continue

        filename = os.path.basename(
            member.name
        )

        if filename not in wanted:
            continue

        f = tar.extractfile(member)

        if f is None:
            continue

        found_images[filename] = f.read()

        if len(found_images) == len(wanted):

            break


print(
    "Found:",
    len(found_images),
    "/",
    len(wanted)
)


missing = set(wanted) - set(found_images)

if missing:

    print("\nMissing files:")

    for filename in sorted(missing):
        print(filename)

    raise RuntimeError(
        "Some test images were not found."
    )


print(
    "All requested images found."
)


# ============================================================
# 11. CLIP inference
# ============================================================

results = []

filenames = list(wanted.keys())


for start in range(
    0,
    len(filenames),
    BATCH_SIZE
):

    batch_names = filenames[
        start:start+BATCH_SIZE
    ]

    images = []

    labels = []

    for filename in batch_names:

        item = wanted[filename]

        image = Image.open(
            io.BytesIO(
                found_images[filename]
            )
        ).convert("RGB")

        images.append(
            preprocess(image)
        )

        labels.append(
            item["label"]
        )


    batch = torch.stack(
        images
    ).to(DEVICE)


    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

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


    logits_np = (
        logits.cpu().numpy()
    )

    image_np = (
        image_features.cpu().numpy()
    )

    text_np = (
        text_features.cpu().numpy()
    )


    predictions = logits_np.argmax(
        axis=1
    )


    # --------------------------------------------------------
    # Save each result
    # --------------------------------------------------------

    for i, filename in enumerate(
        batch_names
    ):

        label = labels[i]

        pred = int(
            predictions[i]
        )

        true_logit = float(
            logits_np[i, label]
        )

        similarity = float(
            image_np[i] @
            text_np[label]
        )

        wrong_logits = (
            logits_np[i].copy()
        )

        wrong_logits[label] = -np.inf

        best_wrong = float(
            wrong_logits.max()
        )

        margin = (
            true_logit -
            best_wrong
        )

        freq = train_counts[label]

        results.append({

            "filename":
                filename,

            "label":
                label,

            "wnid":
                label_to_wnid[label],

            "class_name":
                class_names[label],

            "train_frequency":
                freq,

            "shot_group":
                shot_group(freq),

            "prediction":
                pred,

            "prediction_name":
                class_names[pred],

            "correct":
                int(pred == label),

            "true_logit":
                true_logit,

            "true_similarity":
                similarity,

            "margin":
                margin
        })


# ============================================================
# 12. Summary
# ============================================================

accuracy = np.mean([
    r["correct"]
    for r in results
])


print("\n" + "=" * 70)
print("SANITY CHECK RESULT")
print("=" * 70)

print(
    "Processed:",
    len(results)
)

print(
    "Accuracy:",
    f"{accuracy:.4f}"
)


# ============================================================
# 13. Show predictions
# ============================================================

print("\nFirst 20 predictions:")

for r in results[:20]:

    print(
        f'{r["class_name"][:25]:25s} | '
        f'pred={r["prediction_name"][:25]:25s} | '
        f'freq={r["train_frequency"]:4d} | '
        f'logit={r["true_logit"]:8.3f} | '
        f'sim={r["true_similarity"]:.4f} | '
        f'margin={r["margin"]:8.3f} | '
        f'correct={r["correct"]}'
    )


# ============================================================
# 14. Save CSV
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
    "\nResults saved to:"
)

print(csv_path)

print("\nDONE.")


