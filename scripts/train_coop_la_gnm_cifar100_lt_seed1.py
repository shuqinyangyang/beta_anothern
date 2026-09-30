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


# ============================================================
# Config
# ============================================================

ROOT = "/root/autodl-tmp/clip-lt"

CIFAR_ROOT = ROOT + "/data/cifar-100-python"
LT_INDEX_FILE = ROOT + "/data/cifar100_lt_ir100/train_indices.npy"

OUT_DIR = (
    ROOT +
    "/results/cifar100_lt_coop_la_gnm_seed1_full"
)

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda"

N_CTX = 16

BATCH_SIZE = 16

LR = 0.002

EPOCHS = 10

# Logit Adjustment
LA_TAU = 1.0
LA_EPS = 1e-12

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
# CIFAR-100 class names
# ============================================================

import pickle

with open(
    os.path.join(CIFAR_ROOT, "meta"),
    "rb"
) as f:
    cifar_meta = pickle.load(f, encoding="bytes")

class_names = [
    x.decode("utf-8").replace("_", " ")
    for x in cifar_meta[b"fine_label_names"]
]

assert len(class_names) == 100

print("\nCIFAR-100 classes:", len(class_names))
print("First classes:", class_names[:10])


# ============================================================
# CIFAR100-LT Dataset
# ============================================================

class CIFAR100PickleDataset(Dataset):

    def __init__(
        self,
        split,
        transform,
        indices=None
    ):
        assert split in ["train", "test"]

        file_path = os.path.join(
            CIFAR_ROOT,
            split
        )

        with open(file_path, "rb") as f:
            data = pickle.load(
                f,
                encoding="bytes"
            )

        images = data[b"data"]
        labels = np.asarray(
            data[b"fine_labels"],
            dtype=np.int64
        )

        # CIFAR format:
        # [N, 3072] -> [N, 3, 32, 32]
        images = images.reshape(
            -1, 3, 32, 32
        ).transpose(
            0, 2, 3, 1
        )

        if indices is not None:
            indices = np.asarray(
                indices,
                dtype=np.int64
            )
            images = images[indices]
            labels = labels[indices]

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = Image.fromarray(
            self.images[idx]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(self.labels[idx])

        return image, label


lt_train_indices = np.load(
    LT_INDEX_FILE
)

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
        # avoid storing activations for all 100 text prompts
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
print("CoOp + LA CIFAR100-LT (IR=100)")
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

train_dataset = CIFAR100PickleDataset(
    split="train",
    transform=preprocess,
    indices=lt_train_indices
)

test_dataset = CIFAR100PickleDataset(
    split="test",
    transform=preprocess,
    indices=None
)

print("\nCIFAR100-LT dataset:")
print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

train_counts = np.bincount(
    train_dataset.labels,
    minlength=100
)

print("Classes:", len(train_counts))
print(
    "Train count range:",
    int(train_counts.min()),
    "->",
    int(train_counts.max())
)
print(
    "IR:",
    float(train_counts.max()) /
    float(train_counts.min())
)

# ============================================================
# Logit Adjustment
# ============================================================

class_counts = torch.tensor(
    train_counts,
    dtype=torch.float32,
    device=DEVICE
)

class_prior = (
    class_counts /
    class_counts.sum()
)

log_prior = torch.log(
    class_prior + LA_EPS
)

logit_adjustment = (
    LA_TAU * log_prior
)

print("\nLogit Adjustment:")
print("Tau:", LA_TAU)
print(
    "Class prior range:",
    class_prior.min().item(),
    "->",
    class_prior.max().item()
)
print(
    "Logit adjustment range:",
    logit_adjustment.min().item(),
    "->",
    logit_adjustment.max().item()
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
# Resume training
# ============================================================

# CIFAR100-LT experiment starts from scratch.
RESUME_PATH = "/tmp/no_cifar100_lt_la_resume.pth"

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

    per_class_correct = np.zeros(100)
    per_class_total = np.zeros(100)

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

    # --------------------------------------------------------
    # Long-tailed group evaluation
    # Many:  > 100 training samples
    # Medium: 20--100 training samples
    # Few:   < 20 training samples
    # --------------------------------------------------------

    per_class_acc = np.divide(
        per_class_correct,
        per_class_total,
        out=np.zeros_like(per_class_correct),
        where=per_class_total > 0
    )

    many_mask = train_counts > 100
    medium_mask = (
        (train_counts >= 20)
        & (train_counts <= 100)
    )
    few_mask = train_counts < 20

    many_acc = float(
        per_class_acc[many_mask].mean()
    )
    medium_acc = float(
        per_class_acc[medium_mask].mean()
    )
    few_acc = float(
        per_class_acc[few_mask].mean()
    )

    gap = many_acc - few_acc

    print(
        f"\nEpoch {epoch} test accuracy: "
        f"{acc:.4f}"
    )

    print(
        "Group classes | "
        f"Many {many_mask.sum()} | "
        f"Medium {medium_mask.sum()} | "
        f"Few {few_mask.sum()}"
    )

    print(
        "Group accuracy | "
        f"Many {many_acc:.4f} | "
        f"Medium {medium_acc:.4f} | "
        f"Few {few_acc:.4f} | "
        f"Many-Few gap {gap:.4f}"
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

best_acc = 0.0

for epoch in range(START_EPOCH, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    for step, (
        images,
        labels
    ) in enumerate(train_loader, start=1):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # Frozen image encoder:
        # no gradient graph needed.
        with torch.no_grad():

            image_features = (
                clip_model.encode_image(
                    images
                )
            )

            image_features /= (
                image_features.norm(
                    dim=-1,
                    keepdim=True
                )
            )


        # --------------------------------------------------
        # Matched GNM adaptation
        #
        # GNM neighborhood parameters:
        # rho = 0.05
        # dis_mul = 0.1
        #
        # Perturb only trainable CoOp prompt parameters,
        # evaluate LA loss at the perturbed prompt,
        # restore the original prompt, then apply SGD.
        # --------------------------------------------------
        optimizer.zero_grad()

        gnm_rho = 0.05
        gnm_dis_mul = 0.1

        old_params = {}

        with torch.no_grad():
            for p in model.parameters():
                if not p.requires_grad:
                    continue

                old_params[p] = p.data.clone()

                noise = (
                    torch.randn_like(p) / 3.0
                ).clamp_(-1.0, 1.0)

                perturb = (
                    gnm_rho
                    * gnm_dis_mul
                    * 0.5
                    * noise
                )

                p.add_(perturb)

        # Forward at the perturbed prompt.
        text_features = model.encode_text()

        logits = (
            clip_model.logit_scale.exp()
            * image_features
            @ text_features.T
        )

        # Same train-time Logit Adjustment as CoOp+LA.
        adjusted_logits = (
            logits.float()
            + logit_adjustment.unsqueeze(0)
        )

        loss = criterion(
            adjusted_logits,
            labels
        )

        # Gradient evaluated at perturbed prompt.
        loss.backward()

        # Restore original prompt parameters while retaining
        # the neighborhood gradient.
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.data.copy_(old_params[p])

        optimizer.step()


        running_loss += loss.item()


        if step % 100 == 0:

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"step {step:04d}/{len(train_loader)} | "
                f"loss {running_loss/step:.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.6f}"
            )


    scheduler.step()


    # Save checkpoint
    checkpoint = {
        "epoch": epoch,
        "prompt_learner":
            model.prompt_learner.state_dict(),
        "optimizer":
            optimizer.state_dict()
    }

    torch.save(
        checkpoint,
        os.path.join(
            OUT_DIR,
            f"coop_epoch_{epoch}.pth"
        )
    )


    # Test at each epoch
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


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Best accuracy:",
    f"{best_acc:.4f}"
)

print(
    "Results:",
    OUT_DIR
)
