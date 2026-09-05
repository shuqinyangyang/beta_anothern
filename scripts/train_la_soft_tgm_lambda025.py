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
    "/results/la_soft_tgm_tau10_beta09_lambda025"
)

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda"

N_CTX = 16

BATCH_SIZE = 16

LR = 0.002

EPOCHS = 10

# ============================================================
# Logit Adjustment
# ============================================================

LA_TAU = 1.0
LA_EPS = 1e-12

# Conflict-aware optimization
EMA_BETA = 0.90
PROJ_STRENGTH = 0.25
GRAD_EPS = 1e-12

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

# ============================================================
# Train frequency + Logit Adjustment prior
# ============================================================

train_frequency = {}

with open(TRAIN_LIST) as f:

    for line in f:

        _, label = line.strip().rsplit(
            " ",
            1
        )

        label = int(label)

        train_frequency[label] = (
            train_frequency.get(
                label,
                0
            )
            + 1
        )


class_counts = torch.tensor(
    [
        train_frequency[c]
        for c in range(1000)
    ],
    dtype=torch.float32,
    device=DEVICE
)


class_prior = (
    class_counts
    /
    class_counts.sum()
)


log_prior = torch.log(
    class_prior + LA_EPS
)


# Train-time LA:
#
# z'_c = z_c + tau log(pi_c)

logit_adjustment = (
    LA_TAU * log_prior
)


print("\nLA + TGM configuration:")

print(
    "LA tau:",
    LA_TAU
)

print(
    "EMA beta:",
    EMA_BETA
)

print(
    "Class prior range:",
    class_prior.min().item(),
    "~",
    class_prior.max().item()
)

print(
    "Logit adjustment range:",
    logit_adjustment.min().item(),
    "~",
    logit_adjustment.max().item()
)


criterion = nn.CrossEntropyLoss()


# ============================================================
# Resume training
# ============================================================

# LA + TGM experiment:
# always start from a fresh random prompt.

START_EPOCH = 1
RESUMED_FEW_GRAD_EMA = None

print("\nLA + TGM experiment:")
print("Training from scratch.")
print("No resume checkpoint.")


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
# Training
# ============================================================

# ------------------------------------------------------------
# Class-frequency group mapping
# ------------------------------------------------------------

class_group = torch.zeros(
    1000,
    dtype=torch.long,
    device=DEVICE
)

for c in range(1000):
    n = train_frequency[c]

    if n > 100:
        class_group[c] = 0      # Many
    elif n >= 20:
        class_group[c] = 1      # Medium
    else:
        class_group[c] = 2      # Few


# ------------------------------------------------------------
# Few gradient memory
# ------------------------------------------------------------

few_grad_ema = RESUMED_FEW_GRAD_EMA

# Best result from completed epochs 1-2
best_acc = 0.0

global_conflict_count = 0
global_compare_count = 0


for epoch in range(START_EPOCH, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    epoch_conflict_count = 0
    epoch_compare_count = 0
    epoch_cos_sum = 0.0

    for step, (images, labels) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # ====================================================
        # Frozen image encoder
        # ====================================================

        with torch.no_grad():

            image_features = clip_model.encode_image(
                images
            )

            image_features /= image_features.norm(
                dim=-1,
                keepdim=True
            )

        # ====================================================
        # Text / prompt features
        # ====================================================

        text_features = model.encode_text()

        logits = (
            clip_model.logit_scale.exp()
            * image_features
            @ text_features.T
        )

        logits_fp32 = logits.float()

        # ====================================================
        # ====================================================

        # ====================================================
        # Train-time Logit Adjustment
        # z'_c = z_c + tau * log(pi_c)
        # ====================================================

        adjusted_logits = (
            logits_fp32
            + logit_adjustment.unsqueeze(0)
        )

        per_sample_ce = nn.functional.cross_entropy(
            adjusted_logits,
            labels,
            reduction="none"
        )

        # LA is the only frequency correction.
        loss = per_sample_ce.mean()

        ctx_param = model.prompt_learner.ctx

        # ====================================================
        # Base Balanced-CoOp gradient
        # ====================================================

        base_grad = torch.autograd.grad(
            loss,
            ctx_param,
            retain_graph=True,
            create_graph=False
        )[0]

        # ====================================================
        # Split current batch by frequency group
        # ====================================================

        batch_group = class_group[labels]

        many_mask = batch_group == 0
        few_mask = batch_group == 2

        n_many = int(many_mask.sum().item())
        n_few = int(few_mask.sum().item())

        # ====================================================
        # Update Few-gradient EMA when Few samples exist
        # ====================================================

        if n_few > 0:

            few_labels = labels[few_mask]
            few_ce = per_sample_ce[few_mask]

            # Few gradient under LA objective
            few_loss = few_ce.mean()

            few_grad = torch.autograd.grad(
                few_loss,
                ctx_param,
                retain_graph=True,
                create_graph=False
            )[0]

            few_flat = (
                few_grad.detach()
                .float()
                .reshape(-1)
            )

            few_norm = torch.linalg.vector_norm(
                few_flat
            )

            if few_norm.item() > GRAD_EPS:

                few_dir = (
                    few_flat /
                    (few_norm + GRAD_EPS)
                )

                if few_grad_ema is None:

                    few_grad_ema = (
                        few_dir.clone()
                    )

                else:

                    few_grad_ema = (
                        EMA_BETA * few_grad_ema
                        +
                        (1.0 - EMA_BETA) * few_dir
                    )

                    few_grad_ema = (
                        few_grad_ema /
                        (
                            torch.linalg.vector_norm(
                                few_grad_ema
                            )
                            + GRAD_EPS
                        )
                    )

        # ====================================================
        # Conflict-aware modification of Many gradient
        # ====================================================

        final_grad = (
            base_grad.detach()
            .float()
        )

        if (
            n_many > 0
            and few_grad_ema is not None
        ):

            many_labels = labels[many_mask]
            many_ce = per_sample_ce[many_mask]

            # Many gradient under LA objective
            many_loss = many_ce.mean()

            many_grad = torch.autograd.grad(
                many_loss,
                ctx_param,
                retain_graph=False,
                create_graph=False
            )[0]

            many_flat = (
                many_grad.detach()
                .float()
                .reshape(-1)
            )

            many_norm = (
                torch.linalg.vector_norm(
                    many_flat
                )
            )

            tail_ref = (
                few_grad_ema /
                (
                    torch.linalg.vector_norm(
                        few_grad_ema
                    )
                    + GRAD_EPS
                )
            )

            if many_norm.item() > GRAD_EPS:

                dot_value = torch.dot(
                    many_flat,
                    tail_ref
                )

                cosine = float(
                    dot_value /
                    (
                        many_norm
                        + GRAD_EPS
                    )
                )

                epoch_compare_count += 1
                global_compare_count += 1

                epoch_cos_sum += cosine

                if dot_value.item() < 0:

                    epoch_conflict_count += 1
                    global_conflict_count += 1

                    # ----------------------------------------
                    # Remove conflicting Many component
                    # ----------------------------------------

                    projected_many = (
                        many_flat
                        -
                        PROJ_STRENGTH
                        * dot_value
                        * tail_ref
                    )

                    # ----------------------------------------
                    #
                    # ----------------------------------------

                    # Under ordinary mean CE with LA:
                    # alpha_M = n_M / B
                    alpha_many = (
                        n_many
                        /
                        labels.numel()
                    )

                    correction = (
                        alpha_many
                        *
                        (
                            projected_many
                            -
                            many_flat
                        )
                    )

                    final_grad = (
                        final_grad.reshape(-1)
                        + correction
                    ).reshape_as(ctx_param)

        # ====================================================
        # Manual gradient update
        # ====================================================

        optimizer.zero_grad(
            set_to_none=True
        )

        ctx_param.grad = (
            final_grad.to(
                dtype=ctx_param.dtype
            )
        )

        optimizer.step()

        running_loss += loss.item()

        # ====================================================
        # Logging
        # ====================================================

        if step % 100 == 0:

            if epoch_compare_count > 0:

                conflict_rate = (
                    epoch_conflict_count
                    /
                    epoch_compare_count
                )

                mean_cos = (
                    epoch_cos_sum
                    /
                    epoch_compare_count
                )

            else:

                conflict_rate = 0.0
                mean_cos = float("nan")

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"step {step:04d}/{len(train_loader)} | "
                f"loss {running_loss/step:.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.6f} | "
                f"conflict {conflict_rate:.2%} | "
                f"cos {mean_cos:.4f} | "
                f"FewEMA {'Y' if few_grad_ema is not None else 'N'}"
            )

        del logits
        del logits_fp32
        del text_features
        del image_features
        del base_grad
        del final_grad

    # ========================================================
    # End epoch
    # ========================================================

    scheduler.step()

    if epoch_compare_count > 0:

        epoch_conflict_rate = (
            epoch_conflict_count /
            epoch_compare_count
        )

        epoch_mean_cos = (
            epoch_cos_sum /
            epoch_compare_count
        )

    else:

        epoch_conflict_rate = 0.0
        epoch_mean_cos = float("nan")

    print(
        f"\nEpoch {epoch} conflict stats | "
        f"rate={epoch_conflict_rate:.2%} | "
        f"mean cosine={epoch_mean_cos:.4f}"
    )

    # ========================================================
    # Save checkpoint
    # ========================================================

    checkpoint = {
        "epoch": epoch,

        "prompt_learner":
            model.prompt_learner.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "few_grad_ema":
            (
                few_grad_ema.cpu()
                if few_grad_ema is not None
                else None
            ),


        "ema_beta":
            EMA_BETA,

        "proj_strength":
            PROJ_STRENGTH,
    }

    torch.save(
        checkpoint,
        os.path.join(
            OUT_DIR,
            f"coop_epoch_{epoch}.pth"
        )
    )

    # ========================================================
    # Evaluate
    # ========================================================

    acc = evaluate(epoch)

    if acc > best_acc:

        best_acc = acc

        torch.save(
            checkpoint,
            os.path.join(
                OUT_DIR,
                "coop_best.pth"
            )
        )


# ============================================================
# Final report
# ============================================================

if global_compare_count > 0:

    overall_conflict_rate = (
        global_conflict_count
        /
        global_compare_count
    )

else:

    overall_conflict_rate = 0.0


print("\n" + "=" * 70)
print("CONFLICT-AWARE TRAINING COMPLETE")
print("=" * 70)

print(
    "Best accuracy:",
    f"{best_acc:.4f}"
)

print(
    "Overall conflict rate:",
    f"{overall_conflict_rate:.2%}"
)

print(
    "Results:",
    OUT_DIR
)
