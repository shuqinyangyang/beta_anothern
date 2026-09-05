import os
import io
import csv
import tarfile
from collections import defaultdict

TRAIN_LIST = "/root/autodl-tmp/clip-lt/SADE-AgnosticLT/data_txt/ImageNet_LT/ImageNet_LT_train.txt"
TEST_LIST = "/root/autodl-tmp/clip-lt/SADE-AgnosticLT/data_txt/ImageNet_LT/ImageNet_LT_test.txt"

TRAIN_TAR = "/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar"
VAL_TAR = "/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar"

CACHE_ROOT = "/dev/shm/imagenet_lt"
TRAIN_OUT = os.path.join(CACHE_ROOT, "train")
VAL_OUT = os.path.join(CACHE_ROOT, "val")

os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(VAL_OUT, exist_ok=True)

# ============================================================
# Train split
# ============================================================

groups = defaultdict(list)

with open(TRAIN_LIST) as f:
    for line in f:
        path, label = line.strip().rsplit(" ", 1)
        parts = path.split("/")
        synset = parts[-2]
        filename = parts[-1]
        groups[synset].append((filename, int(label)))

print("Train classes:", len(groups))
print("Extracting ImageNet-LT train subset...")

train_manifest = []

with tarfile.open(TRAIN_TAR, "r:") as outer:

    for class_index, (synset, samples) in enumerate(groups.items()):

        wanted = {x[0]: x[1] for x in samples}

        member = outer.getmember(synset + ".tar")
        f = outer.extractfile(member)

        inner_bytes = f.read()

        class_dir = os.path.join(TRAIN_OUT, synset)
        os.makedirs(class_dir, exist_ok=True)

        found = 0

        with tarfile.open(fileobj=io.BytesIO(inner_bytes), mode="r:") as inner:

            for m in inner:
                if not m.isfile():
                    continue

                filename = os.path.basename(m.name)

                if filename not in wanted:
                    continue

                image_file = inner.extractfile(m)

                if image_file is None:
                    continue

                out_path = os.path.join(class_dir, filename)

                with open(out_path, "wb") as out:
                    out.write(image_file.read())

                train_manifest.append(
                    (out_path, wanted[filename])
                )

                found += 1

        if (class_index + 1) % 25 == 0:
            print(
                f"Train classes: {class_index+1}/1000 | "
                f"images extracted: {len(train_manifest)}"
            )

print("Train extracted:", len(train_manifest))

# ============================================================
# Test split
# ============================================================

test_meta = {}

with open(TEST_LIST) as f:
    for line in f:
        path, label = line.strip().rsplit(" ", 1)
        filename = os.path.basename(path)
        test_meta[filename] = int(label)

print("\nExtracting validation/test images...")

test_manifest = []

with tarfile.open(VAL_TAR, "r:") as tar:

    for i, member in enumerate(tar):

        if not member.isfile():
            continue

        filename = os.path.basename(member.name)

        if filename not in test_meta:
            continue

        f = tar.extractfile(member)

        if f is None:
            continue

        out_path = os.path.join(VAL_OUT, filename)

        with open(out_path, "wb") as out:
            out.write(f.read())

        test_manifest.append(
            (out_path, test_meta[filename])
        )

        if len(test_manifest) % 5000 == 0:
            print(
                "Test extracted:",
                len(test_manifest),
                "/50000"
            )

print("Test extracted:", len(test_manifest))

# ============================================================
# Save manifests
# ============================================================

for filename, rows in [
    ("train_manifest.csv", train_manifest),
    ("test_manifest.csv", test_manifest),
]:

    path = os.path.join(CACHE_ROOT, filename)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    print("Saved:", path)

print("\nDONE.")
