import os
import csv
import math
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint_sequential

import clip
from PIL import Image
from scipy.io import loadmat


# ============================================================
# Config
# ============================================================

ROOT = "/root/autodl-tmp/clip-lt"

TRAIN_MANIFEST = "/dev/shm/imagenet_lt/train_manifest.csv"
TEST_MANIFEST = "/dev/shm/imagenet_lt/test_manifest.csv"

TRAIN_LIST = (
    ROOT +
    "/SADE-AgnosticLT/data_txt/ImageNet_LT/"
    "ImageNet_LT_train.txt"
)

META_MAT = (
    ROOT +
    "/data/imagenet_devkit/"
    "ILSVRC2012_devkit_t12/data/meta.mat"
)

OUT_DIR = (
    ROOT +
    "/results/balanced_coop_imagenet_lt_gamma05"
)

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda"

N_CTX = 16

BATCH_SIZE = 16

LR = 0.002

EPOCHS = 10

NUM_WORKERS = 8

SEED = 1


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# ImageNet-LT class mapping
# ============================================================

label_to_wnid = {}

with open(TRAIN_LIST) as f:

    for line in f:

        path, label = line.strip().rsplit(" ", 1)

        label = int(label)

        if label not in label_to_wnid:
            label_to_wnid[label] = path.split("/")[-2]


mat = loadmat(META_MAT)

wnid_to_name = {}

for row in mat["synsets"]:

    item = row[0]

    wnid = str(item[1][0])
    words = str(item[2][0])

    wnid_to_name[wnid] = words


def clean_name(x):
    return x.split(",")[0].strip()


class_names = [
    clean_name(
        wnid_to_name[
            label_to_wnid[i]
        ]
    )
    for i in range(1000)
]


# ============================================================
# Dataset
# ============================================================

class ImageListDataset(Dataset):

    def __init__(self, manifest, transform):

        df = pd.read_csv(manifest)

        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]

        image = Image.open(path).convert("RGB")

        image = self.transform(image)

        label = self.labels[idx]

        return image, label


# ============================================================
# Text Encoder
# ============================================================

class TextEncoder(nn.Module):

    def __init__(self, clip_model):

        super().__init__()

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)

        # Gradient checkpointing:
        # avoid storing activations for all 1000 text prompts
        x = checkpoint_sequential(
            self.transformer.resblocks,
            len(self.transformer.resblocks),
            x,
            use_reentrant=False
        )

        x = x.permute(1, 0, 2)

        x = self.ln_final(x).type(self.dtype)

        x = (
            x[
                torch.arange(x.shape[0]),
                tokenized_prompts.argmax(dim=-1)
            ]
            @ self.text_projection
        )

        return x


# ============================================================
# Prompt Learner
# ============================================================

class PromptLearner(nn.Module):

    def __init__(
        self,
        class_names,
        clip_model,
        n_ctx=16
    ):

        super().__init__()

        dtype = clip_model.dtype

        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random continuous context
        ctx_vectors = torch.empty(
            n_ctx,
            ctx_dim,
            dtype=torch.float32
        )

        self.dtype = dtype

        nn.init.normal_(
            ctx_vectors,
            std=0.02
        )

        self.ctx = nn.Parameter(
            ctx_vectors
        )

        self.n_ctx = n_ctx

        class_names = [
            name.replace("_", " ")
            for name in class_names
        ]

        prompt_prefix = " ".join(
            ["X"] * n_ctx
        )

        prompts = [
            prompt_prefix + " " + name + "."
            for name in class_names
        ]

        tokenized_prompts = torch.cat(
            [
                clip.tokenize(p)
                for p in prompts
            ]
        )

        tokenized_prompts = tokenized_prompts.to(
            DEVICE
        )

        with torch.no_grad():

            embedding = clip_model.token_embedding(
                tokenized_prompts
            ).type(dtype)

        # SOS token
        self.register_buffer(
            "token_prefix",
            embedding[:, :1, :]
        )

        # everything after the context tokens
        self.register_buffer(
            "token_suffix",
            embedding[:, 1+n_ctx:, :]
        )

        self.tokenized_prompts = tokenized_prompts

        self.n_cls = len(class_names)


    def forward(self):

        ctx = self.ctx.to(self.dtype)

        if ctx.dim() == 2:

            ctx = ctx.unsqueeze(0)

            ctx = ctx.expand(
                self.n_cls,
                -1,
                -1
            )

        prompts = torch.cat(
            [
                self.token_prefix,
                ctx,
                self.token_suffix
            ],
            dim=1
        )

        return prompts


# ============================================================
# CoOp
# ============================================================

class CoOp(nn.Module):

    def __init__(
        self,
        clip_model,
        class_names
    ):

        super().__init__()

        self.clip_model = clip_model

        self.prompt_learner = PromptLearner(
            class_names,
            clip_model,
            N_CTX
        )

        self.text_encoder = TextEncoder(
            clip_model
        )

        self.logit_scale = (
            clip_model.logit_scale
        )


    def encode_text(self):

        prompts = self.prompt_learner()

        tokenized = (
            self.prompt_learner
            .tokenized_prompts
        )

        text_features = self.text_encoder(
            prompts,
            tokenized
        )

        text_features = (
            text_features /
            text_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        return text_features


# ============================================================
# Load CLIP
# ============================================================

print("=" * 70)
print("CoOp ImageNet-LT")
print("=" * 70)

print("Loading CLIP ViT-B/32...")

clip_model, preprocess = clip.load(
    "ViT-B/32",
    device=DEVICE
)

# Stability
# clip_model.float()  # keep CLIP FP16 to save GPU memory

# Freeze every CLIP parameter
for p in clip_model.parameters():
    p.requires_grad = False


model = CoOp(
    clip_model,
    class_names
).to(DEVICE)


# Only context vectors should be trainable

trainable = [
    (n, p.numel())
    for n, p in model.named_parameters()
    if p.requires_grad
]

print("\nTrainable parameters:")

for n, p in trainable:
    print(n, p)

print(
    "Total trainable:",
    sum(x[1] for x in trainable)
)


# ============================================================
# Data
# ============================================================

train_dataset = ImageListDataset(
    TRAIN_MANIFEST,
    preprocess
)

test_dataset = ImageListDataset(
    TEST_MANIFEST,
    preprocess
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.SGD(
    model.prompt_learner.parameters(),
    lr=LR,
    momentum=0.9
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

criterion = nn.CrossEntropyLoss()

# ============================================================
# Frequency-balanced loss for gradient probe
# ============================================================

GAMMA = 0.5
EPS = 1e-6

probe_train_frequency = {}

with open(TRAIN_LIST) as f:
    for line in f:
        _, label = line.strip().rsplit(" ", 1)
        label = int(label)
        probe_train_frequency[label] = (
            probe_train_frequency.get(label, 0) + 1
        )

class_weights = torch.tensor(
    [
        (probe_train_frequency[c] + EPS) ** (-GAMMA)
        for c in range(1000)
    ],
    dtype=torch.float32,
    device=DEVICE
)

class_weights = (
    class_weights /
    class_weights.mean()
)

balanced_criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

print("\nBalanced gradient probe weights:")
print("Gamma:", GAMMA)
print(
    "Weight range:",
    class_weights.min().item(),
    "~",
    class_weights.max().item()
)


# ============================================================
# Resume training
# ============================================================

RESUME_PATH = "/tmp/__disable_training_resume__.pth"

START_EPOCH = 1

if os.path.exists(RESUME_PATH):

    print("\nResuming from:")
    print(RESUME_PATH)

    checkpoint = torch.load(
        RESUME_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.prompt_learner.load_state_dict(
        checkpoint["prompt_learner"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    START_EPOCH = checkpoint["epoch"] + 1

    # Rebuild scheduler state corresponding to completed epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        last_epoch=checkpoint["epoch"] - 1
    )

    print(
        "Resume epoch:",
        START_EPOCH
    )

else:
    print("\nNo resume checkpoint found.")



# ============================================================
# Save zero-shot/reference text features
# ============================================================

with torch.no_grad():

    zs_prompts = [
        f"a photo of a {name}"
        for name in class_names
    ]

    zs_tokens = clip.tokenize(
        zs_prompts
    ).to(DEVICE)

    zs_features = clip_model.encode_text(
        zs_tokens
    )

    zs_features /= zs_features.norm(
        dim=-1,
        keepdim=True
    )

np.save(
    os.path.join(
        OUT_DIR,
        "zeroshot_text_features.npy"
    ),
    zs_features.cpu().numpy()
)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(epoch):

    model.eval()

    text_features = model.encode_text()

    total = 0
    correct = 0

    per_class_correct = np.zeros(1000)
    per_class_total = np.zeros(1000)

    rows = []

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        image_features = clip_model.encode_image(
            images
        )

        image_features /= image_features.norm(
            dim=-1,
            keepdim=True
        )

        logits = (
            clip_model.logit_scale.exp()
            * image_features
            @ text_features.T
        )

        pred = logits.argmax(dim=1)

        total += labels.numel()

        correct += (
            pred == labels
        ).sum().item()

        logits_np = logits.cpu().numpy()
        labels_np = labels.cpu().numpy()
        pred_np = pred.cpu().numpy()
        img_np = image_features.cpu().numpy()
        txt_np = text_features.cpu().numpy()

        for i, y in enumerate(labels_np):

            y = int(y)
            p = int(pred_np[i])

            per_class_total[y] += 1
            per_class_correct[y] += int(p == y)

            true_logit = float(
                logits_np[i, y]
            )

            true_sim = float(
                img_np[i] @ txt_np[y]
            )

            wrong = logits_np[i].copy()
            wrong[y] = -np.inf

            margin = float(
                true_logit - wrong.max()
            )

            rows.append([
                y,
                p,
                int(p == y),
                true_logit,
                true_sim,
                margin
            ])


    acc = correct / total

    print(
        f"\nEpoch {epoch} test accuracy: "
        f"{acc:.4f}"
    )

    # Save sample-level results
    result_path = os.path.join(
        OUT_DIR,
        f"epoch_{epoch}_predictions.csv"
    )

    with open(
        result_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "label",
            "prediction",
            "correct",
            "true_logit",
            "true_similarity",
            "margin"
        ])

        writer.writerows(rows)


    # Save learned text prototypes
    np.save(
        os.path.join(
            OUT_DIR,
            f"epoch_{epoch}_text_features.npy"
        ),
        text_features.cpu().numpy()
    )

    return acc


# ============================================================
# Training
# ============================================================

# ============================================================
# Gradient Contribution & Conflict Probe
# ============================================================

from collections import defaultdict
import math

# ------------------------------------------------------------
# Probe configuration
# ------------------------------------------------------------

# First analyze the best CoOp checkpoint.
# Later we can change this to None to inspect initialization.
PROBE_CHECKPOINT = os.path.join(
    OUT_DIR,
    "coop_epoch_9.pth"
)

# 500 natural long-tail batches gives a useful first estimate.
PROBE_STEPS = 500

PROBE_OUT = os.path.join(
    OUT_DIR,
    "gradient_probe_epoch9_balanced"
)

os.makedirs(PROBE_OUT, exist_ok=True)


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

if PROBE_CHECKPOINT is not None:

    print("\n" + "=" * 70)
    print("GRADIENT PROBE")
    print("=" * 70)

    print("Loading probe checkpoint:")
    print(PROBE_CHECKPOINT)

    ckpt = torch.load(
        PROBE_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.prompt_learner.load_state_dict(
        ckpt["prompt_learner"]
    )

    print(
        "Loaded checkpoint epoch:",
        ckpt.get("epoch", "unknown")
    )

else:

    print("\nUsing randomly initialized CoOp prompt.")


# ------------------------------------------------------------
# Make sure only prompt context requires gradient
# ------------------------------------------------------------

for p in clip_model.parameters():
    p.requires_grad = False

ctx_param = model.prompt_learner.ctx

ctx_param.requires_grad_(True)

print(
    "Prompt parameter shape:",
    tuple(ctx_param.shape)
)

print(
    "Prompt parameter count:",
    ctx_param.numel()
)


# ------------------------------------------------------------
# Build class -> frequency -> group mapping
# ------------------------------------------------------------

train_frequency = {}

with open(TRAIN_LIST) as f:

    for line in f:

        _, label = line.strip().rsplit(" ", 1)

        label = int(label)

        train_frequency[label] = (
            train_frequency.get(label, 0) + 1
        )


def get_group(freq):

    if freq > 100:
        return "Many"

    elif freq >= 20:
        return "Medium"

    else:
        return "Few"


group_names = [
    "Many",
    "Medium",
    "Few"
]

group_to_id = {
    "Many": 0,
    "Medium": 1,
    "Few": 2
}

class_group_id = torch.tensor(
    [
        group_to_id[
            get_group(
                train_frequency[c]
            )
        ]
        for c in range(1000)
    ],
    device=DEVICE,
    dtype=torch.long
)


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

stats = {}

for g in group_names:

    stats[g] = {

        # number of samples observed
        "sample_count": 0,

        # number of batches containing this group
        "batch_count": 0,

        # norm of gradient of GROUP-MEAN loss
        "raw_grad_norm_sum": 0.0,

        # norm of this group's contribution to actual batch gradient
        "weighted_grad_norm_sum": 0.0,

        # accumulate actual contribution vectors across natural batches
        "cumulative_vector":
            torch.zeros(
                ctx_param.numel(),
                dtype=torch.float64
            ),
    }


pairs = [
    ("Many", "Medium"),
    ("Many", "Few"),
    ("Medium", "Few")
]

pair_cosines = {
    pair: []
    for pair in pairs
}

decomposition_errors = []

probe_rows = []


# ------------------------------------------------------------
# Probe
# ------------------------------------------------------------

model.train()

print(
    "\nNatural long-tail gradient probing..."
)

print(
    "Probe steps:",
    PROBE_STEPS
)


for step, (images, labels) in enumerate(
    train_loader,
    start=1
):

    if step > PROBE_STEPS:
        break

    images = images.to(
        DEVICE,
        non_blocking=True
    )

    labels = labels.to(
        DEVICE,
        non_blocking=True
    )

    batch_size_actual = labels.numel()


    # --------------------------------------------------------
    # Frozen visual representation
    # --------------------------------------------------------

    with torch.no_grad():

        image_features = clip_model.encode_image(
            images
        )

        image_features /= image_features.norm(
            dim=-1,
            keepdim=True
        )


    # --------------------------------------------------------
    # Full 1000-class prompt encoding
    # --------------------------------------------------------

    text_features = model.encode_text()

    logits = (
        clip_model.logit_scale.exp()
        * image_features
        @ text_features.T
    )

    logits_fp32 = logits.float()


    # group identity of each sample in this batch
    sample_group_ids = class_group_id[
        labels
    ]


    # --------------------------------------------------------
    # Group-specific prompt gradients
    # --------------------------------------------------------

    group_grad = {}
    group_n = {}

    present_groups = []

    for g in group_names:

        gid = group_to_id[g]

        mask = (
            sample_group_ids == gid
        )

        n_g = int(
            mask.sum().item()
        )

        group_n[g] = n_g

        if n_g == 0:
            continue

        present_groups.append(g)

        # ----------------------------------------------------
        # Frequency-balanced group loss
        # ----------------------------------------------------

        group_logits = logits_fp32[mask]
        group_labels = labels[mask]

        per_sample_ce = nn.functional.cross_entropy(
            group_logits,
            group_labels,
            reduction="none"
        )

        sample_weights_g = class_weights[
            group_labels
        ]

        group_loss = (
            sample_weights_g * per_sample_ce
        ).sum() / sample_weights_g.sum()

        grad = torch.autograd.grad(
            group_loss,
            ctx_param,
            retain_graph=True,
            create_graph=False
        )[0]

        grad_flat = (
            grad.detach()
            .float()
            .reshape(-1)
        )

        group_grad[g] = grad_flat

        raw_norm = float(
            torch.linalg.vector_norm(
                grad_flat
            ).item()
        )

        # Actual contribution coefficient under weighted CE:
        #
        # L_batch = sum_g (W_g / W_batch) * L_g
        #
        group_weight_sum = float(
            sample_weights_g.sum().item()
        )

        batch_weight_sum = float(
            class_weights[labels].sum().item()
        )

        weight = (
            group_weight_sum /
            batch_weight_sum
        )

        contribution = (
            weight * grad_flat
        )

        contribution_norm = float(
            torch.linalg.vector_norm(
                contribution
            ).item()
        )

        stats[g]["sample_count"] += n_g

        stats[g]["batch_count"] += 1

        stats[g]["raw_grad_norm_sum"] += (
            raw_norm
        )

        stats[g]["weighted_grad_norm_sum"] += (
            contribution_norm
        )

        stats[g]["cumulative_vector"] += (
            contribution
            .double()
            .cpu()
        )


        probe_rows.append({
            "step": step,
            "group": g,
            "group_samples": n_g,
            "batch_size": batch_size_actual,
            "sample_fraction":
                weight,
            "raw_grad_norm":
                raw_norm,
            "weighted_contribution_norm":
                contribution_norm,
        })


    # --------------------------------------------------------
    # Gradient direction similarity
    # --------------------------------------------------------

    for g1, g2 in pairs:

        if (
            g1 not in group_grad
            or g2 not in group_grad
        ):
            continue

        a = group_grad[g1]
        b = group_grad[g2]

        denom = (
            torch.linalg.vector_norm(a)
            *
            torch.linalg.vector_norm(b)
            + 1e-12
        )

        cosine = float(
            torch.dot(a, b)
            / denom
        )

        pair_cosines[
            (g1, g2)
        ].append(cosine)


    # --------------------------------------------------------
    # Validate our gradient decomposition on first 10 batches
    # --------------------------------------------------------

    if step <= 10:

        full_loss = balanced_criterion(
            logits_fp32,
            labels
        )

        actual_grad = torch.autograd.grad(
            full_loss,
            ctx_param,
            retain_graph=False,
            create_graph=False
        )[0].detach().float().reshape(-1)

        reconstructed = torch.zeros_like(
            actual_grad
        )

        batch_weight_sum = float(
            class_weights[labels].sum().item()
        )

        for g in present_groups:

            gid = group_to_id[g]

            mask_g = (
                sample_group_ids == gid
            )

            group_weight_sum = float(
                class_weights[
                    labels[mask_g]
                ].sum().item()
            )

            reconstructed += (
                group_weight_sum /
                batch_weight_sum
            ) * group_grad[g]

        relative_error = float(
            torch.linalg.vector_norm(
                actual_grad - reconstructed
            )
            /
            (
                torch.linalg.vector_norm(
                    actual_grad
                )
                + 1e-12
            )
        )

        decomposition_errors.append(
            relative_error
        )

    else:

        # Release the retained computation graph.
        # Use a final cheap backward target.
        last_g = present_groups[-1]

        dummy = (
            group_grad[last_g]
        )

        # Graph is freed naturally when local tensors
        # go out of scope after this iteration.


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        step == 1
        or step % 50 == 0
    ):

        counts_text = " | ".join(
            [
                f"{g}:{stats[g]['sample_count']}"
                for g in group_names
            ]
        )

        print(
            f"Step {step:04d}/{PROBE_STEPS} | "
            + counts_text
        )


    # Explicitly remove graph-heavy tensors before next batch
    del logits
    del logits_fp32
    del text_features
    del image_features
    del group_grad


# ============================================================
# Summaries
# ============================================================

print("\n" + "=" * 100)
print("GRADIENT CONTRIBUTION SUMMARY")
print("=" * 100)


summary_rows = []

total_samples = sum(
    stats[g]["sample_count"]
    for g in group_names
)


for g in group_names:

    s = stats[g]

    batch_count = s["batch_count"]

    avg_raw = (
        s["raw_grad_norm_sum"]
        / batch_count
        if batch_count > 0
        else float("nan")
    )

    avg_weighted = (
        s["weighted_grad_norm_sum"]
        / batch_count
        if batch_count > 0
        else float("nan")
    )

    cumulative_norm = float(
        torch.linalg.vector_norm(
            s["cumulative_vector"]
        ).item()
    )

    sample_share = (
        s["sample_count"]
        / total_samples
        if total_samples > 0
        else float("nan")
    )

    summary_rows.append({
        "group": g,
        "samples_seen":
            s["sample_count"],
        "sample_share":
            sample_share,
        "batches_present":
            batch_count,
        "avg_group_mean_grad_norm":
            avg_raw,
        "avg_batch_contribution_norm":
            avg_weighted,
        "cumulative_gradient_vector_norm":
            cumulative_norm,
    })


summary_df = pd.DataFrame(
    summary_rows
)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# Pairwise gradient cosine
# ============================================================

print("\n" + "=" * 100)
print("GRADIENT DIRECTION COSINE")
print("=" * 100)


cosine_rows = []

for pair in pairs:

    values = np.array(
        pair_cosines[pair],
        dtype=np.float64
    )

    if len(values) == 0:

        print(
            f"{pair[0]} vs {pair[1]}: "
            "no co-occurring batches"
        )

        continue

    row = {
        "group_a": pair[0],
        "group_b": pair[1],
        "n_batches": len(values),
        "mean_cosine":
            values.mean(),
        "median_cosine":
            np.median(values),
        "std_cosine":
            values.std(),
        "negative_fraction":
            np.mean(values < 0),
        "near_zero_fraction":
            np.mean(
                np.abs(values) < 0.1
            ),
    }

    cosine_rows.append(row)

    print(
        f"{pair[0]:6s} vs {pair[1]:6s} | "
        f"n={len(values):4d} | "
        f"mean={values.mean(): .4f} | "
        f"median={np.median(values): .4f} | "
        f"negative={np.mean(values < 0):.2%}"
    )


cosine_df = pd.DataFrame(
    cosine_rows
)


# ============================================================
# Cumulative-vector direction
# ============================================================

print("\n" + "=" * 100)
print("CUMULATIVE GRADIENT VECTOR COSINE")
print("=" * 100)


for g1, g2 in pairs:

    a = stats[g1][
        "cumulative_vector"
    ]

    b = stats[g2][
        "cumulative_vector"
    ]

    denom = (
        torch.linalg.vector_norm(a)
        *
        torch.linalg.vector_norm(b)
        + 1e-12
    )

    cosine = float(
        torch.dot(a, b)
        /
        denom
    )

    print(
        f"{g1:6s} vs {g2:6s}: "
        f"{cosine: .6f}"
    )


# ============================================================
# Decomposition check
# ============================================================

if decomposition_errors:

    print("\n" + "=" * 100)
    print("GRADIENT DECOMPOSITION CHECK")
    print("=" * 100)

    print(
        "Mean relative error:",
        np.mean(
            decomposition_errors
        )
    )

    print(
        "Max relative error:",
        np.max(
            decomposition_errors
        )
    )


# ============================================================
# Save
# ============================================================

detail_df = pd.DataFrame(
    probe_rows
)

detail_path = os.path.join(
    PROBE_OUT,
    "gradient_probe_per_batch.csv"
)

summary_path = os.path.join(
    PROBE_OUT,
    "gradient_probe_summary.csv"
)

cosine_path = os.path.join(
    PROBE_OUT,
    "gradient_probe_cosines.csv"
)

detail_df.to_csv(
    detail_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

cosine_df.to_csv(
    cosine_path,
    index=False
)


print("\nSaved:")
print(detail_path)
print(summary_path)
print(cosine_path)

print("\nGRADIENT PROBE COMPLETE.")
