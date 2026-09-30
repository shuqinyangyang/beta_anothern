import os
import csv
import tarfile
from collections import defaultdict

ROOT = "/root/autodl-tmp/clip-lt"

TRAIN_LIST = os.path.join(
    ROOT,
    "SADE-AgnosticLT/data_txt/ImageNet_LT/ImageNet_LT_train.txt"
)

TEST_LIST = os.path.join(
    ROOT,
    "SADE-AgnosticLT/data_txt/ImageNet_LT/ImageNet_LT_test.txt"
)

TRAIN_TAR = (
    "/root/autodl-pub/ImageNet/ILSVRC2012/"
    "ILSVRC2012_img_train.tar"
)

VAL_TAR = (
    "/root/autodl-pub/ImageNet/ILSVRC2012/"
    "ILSVRC2012_img_val.tar"
)

OUT_ROOT = "/dev/shm/imagenet_lt"

TRAIN_OUT = os.path.join(
    OUT_ROOT,
    "train"
)

TEST_OUT = os.path.join(
    OUT_ROOT,
    "test"
)

TRAIN_MANIFEST = os.path.join(
    OUT_ROOT,
    "train_manifest.csv"
)

TEST_MANIFEST = os.path.join(
    OUT_ROOT,
    "test_manifest.csv"
)

os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(TEST_OUT, exist_ok=True)


# ============================================================
# Parse ImageNet-LT split
# ============================================================

def read_split(path):
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rel_path, label = line.rsplit(
                " ",
                1
            )

            label = int(label)

            parts = rel_path.split("/")

            # train/wnid/image.JPEG
            # val/wnid/image.JPEG
            if len(parts) != 3:
                raise RuntimeError(
                    f"Unexpected split format: {line}"
                )

            split_name = parts[0]
            wnid = parts[1]
            filename = parts[2]

            rows.append(
                {
                    "split": split_name,
                    "wnid": wnid,
                    "filename": filename,
                    "label": label,
                }
            )

    return rows


train_rows = read_split(
    TRAIN_LIST
)

test_rows = read_split(
    TEST_LIST
)

print("=" * 80)
print("ImageNet-LT split")
print("=" * 80)

print(
    "Train images:",
    len(train_rows)
)

print(
    "Test images:",
    len(test_rows)
)

print(
    "Train classes:",
    len(set(x["label"] for x in train_rows))
)

print(
    "Test classes:",
    len(set(x["label"] for x in test_rows))
)


# ============================================================
# Build train lookup
#
# ImageNet train tar:
#
# ILSVRC2012_img_train.tar
#   ├── n01440764.tar
#   ├── n01443537.tar
#   └── ...
#
# Each class tar contains:
#
# n01440764_190.JPEG
# ...
# ============================================================

train_needed = defaultdict(dict)

for row in train_rows:

    train_needed[row["wnid"]][
        row["filename"]
    ] = row["label"]


print("\n" + "=" * 80)
print("Extracting ImageNet-LT train images")
print("=" * 80)

train_found = set()

with tarfile.open(
    TRAIN_TAR,
    mode="r:"
) as outer_tar:

    class_index = 0

    for outer_member in outer_tar:

        if not outer_member.isfile():
            continue

        name = os.path.basename(
            outer_member.name
        )

        if not name.endswith(".tar"):
            continue

        wnid = name[:-4]

        if wnid not in train_needed:
            continue

        class_index += 1

        needed = train_needed[wnid]

        class_dir = os.path.join(
            TRAIN_OUT,
            wnid
        )

        os.makedirs(
            class_dir,
            exist_ok=True
        )

        class_file = outer_tar.extractfile(
            outer_member
        )

        if class_file is None:
            raise RuntimeError(
                f"Cannot read class tar: {wnid}"
            )

        # Streaming inner tar avoids loading a whole
        # ImageNet class archive into RAM.
        with tarfile.open(
            fileobj=class_file,
            mode="r|*"
        ) as inner_tar:

            for img_member in inner_tar:

                if not img_member.isfile():
                    continue

                filename = os.path.basename(
                    img_member.name
                )

                if filename not in needed:
                    continue

                out_path = os.path.join(
                    class_dir,
                    filename
                )

                # Resume-safe.
                if not os.path.exists(out_path):

                    src = inner_tar.extractfile(
                        img_member
                    )

                    if src is None:
                        raise RuntimeError(
                            f"Cannot extract {filename}"
                        )

                    with open(
                        out_path,
                        "wb"
                    ) as dst:
                        dst.write(
                            src.read()
                        )

                train_found.add(
                    (wnid, filename)
                )

        if (
            class_index % 25 == 0
            or class_index == len(train_needed)
        ):
            print(
                f"Classes processed: "
                f"{class_index}/{len(train_needed)} "
                f"| images found: "
                f"{len(train_found)}/{len(train_rows)}",
                flush=True
            )


# ============================================================
# Verify train extraction
# ============================================================

train_missing = []

for row in train_rows:

    key = (
        row["wnid"],
        row["filename"]
    )

    out_path = os.path.join(
        TRAIN_OUT,
        row["wnid"],
        row["filename"]
    )

    if (
        key not in train_found
        and not os.path.exists(out_path)
    ):
        train_missing.append(row)


print(
    "\nTrain missing:",
    len(train_missing)
)

if train_missing:

    print(
        "First missing examples:"
    )

    for x in train_missing[:20]:
        print(x)

    raise RuntimeError(
        "Train extraction incomplete."
    )


# ============================================================
# Extract ImageNet-LT test images
#
# Validation tar is flat:
#
# ILSVRC2012_val_000xxxxx.JPEG
# ============================================================

test_needed = {}

for row in test_rows:

    filename = row["filename"]

    if filename in test_needed:
        raise RuntimeError(
            f"Duplicate validation filename: "
            f"{filename}"
        )

    test_needed[filename] = row


print("\n" + "=" * 80)
print("Extracting ImageNet-LT test images")
print("=" * 80)

test_found = set()

with tarfile.open(
    VAL_TAR,
    mode="r:"
) as val_tar:

    for idx, member in enumerate(
        val_tar,
        start=1
    ):

        if not member.isfile():
            continue

        filename = os.path.basename(
            member.name
        )

        if filename not in test_needed:
            continue

        row = test_needed[
            filename
        ]

        class_dir = os.path.join(
            TEST_OUT,
            row["wnid"]
        )

        os.makedirs(
            class_dir,
            exist_ok=True
        )

        out_path = os.path.join(
            class_dir,
            filename
        )

        if not os.path.exists(out_path):

            src = val_tar.extractfile(
                member
            )

            if src is None:
                raise RuntimeError(
                    f"Cannot extract "
                    f"{filename}"
                )

            with open(
                out_path,
                "wb"
            ) as dst:
                dst.write(
                    src.read()
                )

        test_found.add(
            filename
        )

        if (
            len(test_found) % 5000 == 0
        ):
            print(
                f"Test images found: "
                f"{len(test_found)}/"
                f"{len(test_rows)}",
                flush=True
            )


# ============================================================
# Verify test extraction
# ============================================================

test_missing = []

for row in test_rows:

    out_path = os.path.join(
        TEST_OUT,
        row["wnid"],
        row["filename"]
    )

    if not os.path.exists(
        out_path
    ):
        test_missing.append(
            row
        )


print(
    "\nTest missing:",
    len(test_missing)
)

if test_missing:

    print(
        "First missing examples:"
    )

    for x in test_missing[:20]:
        print(x)

    raise RuntimeError(
        "Test extraction incomplete."
    )


# ============================================================
# Generate manifests
#
# Required columns:
#
# path,label
# ============================================================

def write_manifest(
    rows,
    root_dir,
    manifest_path
):

    with open(
        manifest_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "path",
            "label"
        ])

        for row in rows:

            path = os.path.join(
                root_dir,
                row["wnid"],
                row["filename"]
            )

            writer.writerow([
                path,
                row["label"]
            ])


write_manifest(
    train_rows,
    TRAIN_OUT,
    TRAIN_MANIFEST
)

write_manifest(
    test_rows,
    TEST_OUT,
    TEST_MANIFEST
)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(
    "Train manifest:",
    TRAIN_MANIFEST
)

print(
    "Test manifest:",
    TEST_MANIFEST
)

print(
    "Train rows:",
    len(train_rows)
)

print(
    "Test rows:",
    len(test_rows)
)
