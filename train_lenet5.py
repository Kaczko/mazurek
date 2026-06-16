"""Train LeNet-5 on the contour-shape data set.

Run (after generate_dataset.py, or it will build the data set on the fly):
    python train_lenet5.py

Outputs (in ``outputs/``):
    learning_curves.png          loss + accuracy vs. epoch (train & test)
    classification_clean.png     20 random test images + predictions
    classification_noisy.png     20 random test images + heavy noise + predictions
    lenet5_shapes.pt             trained model weights
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.dataset import (
    CLASS_NAMES, CLASS_GLYPHS, build_dataset, train_test_split, add_noise,
    augment_image,
)
from src.model import LeNet5

OUTPUT_DIR = "outputs"
EVAL_NOISE_SIGMA = 45.0  # heavy noise used for the "noisy" classification demo


# --------------------------------------------------------------------------- #
# Data helpers                                                                #
# --------------------------------------------------------------------------- #
def load_or_build(output_dir: str, per_class: int, seed: int):
    npz_path = os.path.join(output_dir, "dataset.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        print(f"Loaded data set from {npz_path}: {data['images'].shape[0]} images")
        return data["images"], data["labels"]
    print("dataset.npz not found - generating data set...")
    images, labels = build_dataset(per_class=per_class, seed=seed,
                                   progress=print)
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(npz_path, images=images, labels=labels,
                        class_names=np.array(CLASS_NAMES))
    return images, labels


def to_tensor(images: np.ndarray, labels: np.ndarray):
    """Normalise uint8 grayscale to [0,1] float tensors of shape (N,1,32,32)."""
    x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
    y = torch.from_numpy(labels.astype(np.int64))
    return x, y


class AugmentedDataset(Dataset):
    """Wraps the uint8 training images and augments them on the fly."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, seed: int = 0):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = augment_image(self.images[idx], self.rng)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return x, int(self.labels[idx])


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss_sum += criterion(out, yb).item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return loss_sum / total, correct / total


def train(model, train_loader, eval_train_loader, test_loader,
          device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    history = {"train_loss": [], "test_loss": [],
               "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        tr_loss, tr_acc = evaluate(model, eval_train_loader, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        print(f"epoch {epoch:3d}/{epochs}  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  |  "
              f"test_loss={te_loss:.4f} acc={te_acc:.3f}")
    return history


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def plot_learning_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(epochs, history["train_loss"], "o-", label="uczenie (train)")
    ax1.plot(epochs, history["test_loss"], "s-", label="test")
    ax1.set_xlabel("epoka"); ax1.set_ylabel("strata (cross-entropy)")
    ax1.set_title("Krzywa uczenia – strata"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="uczenie (train)")
    ax2.plot(epochs, history["test_acc"], "s-", label="test")
    ax2.set_xlabel("epoka"); ax2.set_ylabel("dokładność")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Krzywa uczenia – dokładność"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("LeNet-5 – krzywe uczenia", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved learning curves -> {path}")


def plot_classification(model, images, labels, device, path, title,
                        n: int = 20, noise_sigma: float = 0.0,
                        seed: int = 2024):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=n, replace=False)
    sel = images[idx]
    sel_labels = labels[idx]

    if noise_sigma > 0:
        noise_rng = np.random.default_rng(seed + 1)
        sel = np.stack([add_noise(im, noise_sigma, noise_rng) for im in sel])

    x, _ = to_tensor(sel, sel_labels)
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1).cpu().numpy()
        conf = probs.max(1).values.cpu().numpy()

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.3))
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        ax.set_xticks([]); ax.set_yticks([])
        if k >= n:
            ax.axis("off")
            continue
        ax.imshow(sel[k], cmap="gray", vmin=0, vmax=255)
        true_name = CLASS_NAMES[sel_labels[k]]
        pred_name = CLASS_NAMES[preds[k]]
        ok = preds[k] == sel_labels[k]
        color = "green" if ok else "red"
        ax.set_title(
            f"prawd: {CLASS_GLYPHS[true_name]} {true_name}\n"
            f"pred: {CLASS_GLYPHS[pred_name]} {pred_name} ({conf[k]*100:.0f}%)",
            fontsize=8, color=color)
        for spine in ax.spines.values():
            spine.set_color(color); spine.set_linewidth(2)

    acc = float(np.mean(preds == sel_labels))
    fig.suptitle(f"{title}  –  trafność na pokazanych: {acc*100:.0f}%",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved classification results -> {path}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    images, labels = load_or_build(args.output_dir, args.per_class, args.seed)

    # 50% train / 50% test, drawn randomly per class, no validation set.
    xtr, ytr, xte, yte = train_test_split(images, labels,
                                          train_fraction=0.5, seed=args.seed)
    print(f"Train: {len(xtr)} images | Test: {len(xte)} images "
          f"(per class ~{len(xtr)//len(CLASS_NAMES)} / "
          f"{len(xte)//len(CLASS_NAMES)})")

    xtr_t, ytr_t = to_tensor(xtr, ytr)
    xte_t, yte_t = to_tensor(xte, yte)
    # Training loader augments on the fly; the eval loaders use raw images so
    # the reported train/test metrics reflect the fixed data set.
    train_loader = DataLoader(AugmentedDataset(xtr, ytr, seed=args.seed),
                              batch_size=args.batch_size, shuffle=True)
    eval_train_loader = DataLoader(TensorDataset(xtr_t, ytr_t),
                                   batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(xte_t, yte_t),
                             batch_size=args.batch_size)

    model = LeNet5(num_classes=len(CLASS_NAMES)).to(device)
    history = train(model, train_loader, eval_train_loader, test_loader,
                    device, args.epochs, args.lr)

    # Outputs.
    plot_learning_curves(history,
                         os.path.join(args.output_dir, "learning_curves.png"))
    plot_classification(model, xte, yte, device,
                        os.path.join(args.output_dir, "classification_clean.png"),
                        "Klasyfikacja – 20 obrazów z bazy (bez szumu)",
                        n=20, noise_sigma=0.0, seed=2024)
    plot_classification(model, xte, yte, device,
                        os.path.join(args.output_dir, "classification_noisy.png"),
                        f"Klasyfikacja – 20 obrazów z bazy + szum (σ={EVAL_NOISE_SIGMA:.0f})",
                        n=20, noise_sigma=EVAL_NOISE_SIGMA, seed=2024)

    model_path = os.path.join(args.output_dir, "lenet5_shapes.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model -> {model_path}")
    print(f"\nFinal test accuracy: {history['test_acc'][-1]*100:.2f}%")


if __name__ == "__main__":
    main()
