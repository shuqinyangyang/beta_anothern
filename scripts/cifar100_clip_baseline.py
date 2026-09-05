import os
import pickle
import numpy as np
import torch
import clip
from PIL import Image

# =========================
# 1. Paths
# =========================
DATA_ROOT = "/root/autodl-tmp/clip-lt/data/cifar-100-python"
RESULT_ROOT = "/root/autodl-tmp/clip-lt/results/cifar100_baseline"

os.makedirs(RESULT_ROOT, exist_ok=True)

# =========================
# 2. Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# =========================
# 3. Load CIFAR-100
# =========================
with open(os.path.join(DATA_ROOT, "test"), "rb") as f:
    test_data = pickle.load(f, encoding="bytes")

with open(os.path.join(DATA_ROOT, "meta"), "rb") as f:
    meta = pickle.load(f, encoding="bytes")

images = test_data[b"data"]
labels = np.array(test_data[b"fine_labels"])

class_names = [
    name.decode("utf-8")
    for name in meta[b"fine_label_names"]
]

print("Number of test images:", len(images))
print("Number of classes:", len(class_names))

# =========================
# 4. Load CLIP
# =========================
print("\nLoading CLIP...")

model, preprocess = clip.load(
    "ViT-B/32",
    device=device
)

model.eval()

print("CLIP loaded.")

# =========================
# 5. Text features
# =========================
texts = [
    f"a photo of a {name.replace('_', ' ')}"
    for name in class_names
]

text_tokens = clip.tokenize(texts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )

print("Text features:", text_features.shape)

# =========================
# 6. Image features
# =========================
all_image_features = []
all_logits = []

batch_size = 64

print("\nEncoding images...")

for start in range(0, len(images), batch_size):

    end = min(start + batch_size, len(images))

    batch_images = []

    for img in images[start:end]:

        # CIFAR-100 stores flattened RGB images
        img = img.reshape(3, 32, 32)
        img = np.transpose(img, (1, 2, 0))

        pil_img = Image.fromarray(img)

        batch_images.append(
            preprocess(pil_img)
        )

    batch_images = torch.stack(batch_images).to(device)

    with torch.no_grad():

        image_features = model.encode_image(
            batch_images
        )

        image_features = (
            image_features /
            image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        logits = 100.0 * (
            image_features @ text_features.T
        )

    all_image_features.append(
        image_features.cpu().numpy()
    )

    all_logits.append(
        logits.cpu().numpy()
    )

    if start % (batch_size * 10) == 0:
        print(
            f"Processed {end}/{len(images)}"
        )

# =========================
# 7. Merge results
# =========================
image_features = np.concatenate(
    all_image_features,
    axis=0
)

logits = np.concatenate(
    all_logits,
    axis=0
)

# =========================
# 8. Predictions
# =========================
predictions = logits.argmax(axis=1)

accuracy = (
    predictions == labels
).mean()

print("\n" + "=" * 60)
print("Zero-shot CLIP accuracy:", accuracy)
print("=" * 60)

# =========================
# 9. True-class similarity/logit
# =========================
true_class_logits = logits[
    np.arange(len(labels)),
    labels
]

# Because image/text features are normalized,
# cosine similarity = image_features @ text_features.T
similarity = (
    image_features @ text_features.cpu().numpy().T
)

true_class_similarity = similarity[
    np.arange(len(labels)),
    labels
]

# =========================
# 10. Classification margin
# =========================
wrong_logits = logits.copy()

wrong_logits[
    np.arange(len(labels)),
    labels
] = -np.inf

best_wrong_logits = wrong_logits.max(axis=1)

margin = (
    true_class_logits -
    best_wrong_logits
)

# =========================
# 11. Per-class statistics
# =========================
results = []

for c in range(len(class_names)):

    mask = labels == c

    class_acc = (
        predictions[mask] == c
    ).mean()

    results.append([
        c,
        class_names[c],
        int(mask.sum()),
        float(class_acc),
        float(true_class_similarity[mask].mean()),
        float(true_class_logits[mask].mean()),
        float(margin[mask].mean())
    ])

results = np.array(
    results,
    dtype=object
)

# =========================
# 12. Save everything
# =========================
np.save(
    os.path.join(
        RESULT_ROOT,
        "image_features.npy"
    ),
    image_features
)

np.save(
    os.path.join(
        RESULT_ROOT,
        "text_features.npy"
    ),
    text_features.cpu().numpy()
)

np.save(
    os.path.join(
        RESULT_ROOT,
        "logits.npy"
    ),
    logits
)

np.save(
    os.path.join(
        RESULT_ROOT,
        "labels.npy"
    ),
    labels
)

np.save(
    os.path.join(
        RESULT_ROOT,
        "predictions.npy"
    ),
    predictions
)

np.save(
    os.path.join(
        RESULT_ROOT,
        "true_class_similarity.npy"
    ),
    true_class_similarity
)

np.save(
    os.path.join(
        RESULT_ROOT,
        "true_class_logits.npy"
    ),
    true_class_logits
)

np.save(
    os.path.join(
        RESULT_ROOT,
        "margin.npy"
    ),
    margin
)

# Save readable CSV
import csv

csv_path = os.path.join(
    RESULT_ROOT,
    "per_class_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class_id",
        "class_name",
        "num_test_images",
        "accuracy",
        "mean_similarity",
        "mean_logit",
        "mean_margin"
    ])

    writer.writerows(results.tolist())

print("\nResults saved to:")
print(RESULT_ROOT)
print("\nPer-class results:")
print(csv_path)
