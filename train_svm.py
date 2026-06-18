"""Train an SVM on the contour-shape data set.

This is the SVM counterpart of ``train_lenet5.py``: it uses the exact same data
set, the same 50% / 50% per-class split (no validation set), produces learning
curves and the same classification demos (20 clean test images + 20 noisy
ones).  The only difference is the classifier: instead of the LeNet-5 neural
network it uses a Support Vector Machine (scikit-learn ``SVC``) on the flattened
32x32 grayscale pixels.

To avoid over-fitting a high-capacity RBF kernel quickly memorises the small,
high-dimensional (1024-pixel) training set (train accuracy pinned at 100%), the
default classifier is a **linear** SVM with strong regularisation (small ``C``)
trained on a generously augmented training set.  This keeps the train/test gap
small (train and test accuracy stay close, both well below the memorising
regime).  The kernel and ``C`` are configurable via the CLI.

Run (after generate_dataset.py, or it will build the data set on the fly):
    python train_svm.py

Outputs (in ``outputs/``):
    learning_curves_svm.png        accuracy vs. training-set size (train & test)
    classification_clean_svm.png   20 random test images + predictions
    classification_noisy_svm.png   20 random test images + heavy noise + predictions
    svm_shapes.joblib              trained SVM pipeline
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.dataset import (
    CLASS_NAMES, CLASS_GLYPHS, build_dataset, train_test_split, add_noise,
    augment_image,
)

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


def to_features(images: np.ndarray) -> np.ndarray:
    """Flatten uint8 grayscale images to [0,1] feature vectors (N, 1024)."""
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def expand_with_augmentation(images: np.ndarray, labels: np.ndarray,
                             copies: int, seed: int):
    """Grow the training set with ``copies`` augmented versions per image.

    The SVM is not trained iteratively, so (unlike the network) it cannot see a
    fresh variant every epoch.  Instead we expand the training set once with
    lightly augmented copies, which plays the same regularising role.
    """
    rng = np.random.default_rng(seed)
    aug_imgs = [images]
    aug_lbls = [labels]
    for _ in range(copies):
        batch = np.stack([augment_image(im, rng) for im in images])
        aug_imgs.append(batch)
        aug_lbls.append(labels)
    return np.concatenate(aug_imgs), np.concatenate(aug_lbls)


# --------------------------------------------------------------------------- #
# Learning curve                                                              #
# --------------------------------------------------------------------------- #
def compute_learning_curve(make_clf, Xtr, ytr, Xte, yte,
                           fractions, seed: int):
    """Train on growing subsets and record train/test accuracy.

    Mirrors the neural-network learning curve, but the x-axis is the number of
    training samples instead of epochs (an SVM has no epochs).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Xtr))
    Xtr, ytr = Xtr[order], ytr[order]

    sizes, train_acc, test_acc = [], [], []
    for frac in fractions:
        n = max(len(np.unique(ytr)) * 2, int(round(frac * len(Xtr))))
        n = min(n, len(Xtr))
        clf = make_clf()
        clf.fit(Xtr[:n], ytr[:n])
        sizes.append(n)
        train_acc.append(clf.score(Xtr[:n], ytr[:n]))
        test_acc.append(clf.score(Xte, yte))
        print(f"  train_size={n:5d}  train_acc={train_acc[-1]:.3f}  "
              f"test_acc={test_acc[-1]:.3f}")
    return np.array(sizes), np.array(train_acc), np.array(test_acc)


def plot_learning_curves(sizes, train_acc, test_acc, path, kernel="linear"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(sizes, train_acc, "o-", label="uczenie (train)")
    ax1.plot(sizes, test_acc, "s-", label="test")
    ax1.set_xlabel("liczba próbek uczących")
    ax1.set_ylabel("dokładność"); ax1.set_ylim(0, 1.02)
    ax1.set_title("Krzywa uczenia – dokładność"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(sizes, 1.0 - train_acc, "o-", label="uczenie (train)")
    ax2.plot(sizes, 1.0 - test_acc, "s-", label="test")
    ax2.set_xlabel("liczba próbek uczących")
    ax2.set_ylabel("błąd (1 - dokładność)")
    ax2.set_title("Krzywa uczenia – błąd"); ax2.legend(); ax2.grid(alpha=0.3)

    gap = float(train_acc[-1] - test_acc[-1])
    fig.suptitle(f"SVM ({kernel}) – krzywe uczenia "
                 f"(luka train–test = {gap*100:.1f} pp)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved learning curves -> {path}")


# --------------------------------------------------------------------------- #
# Classification demo                                                         #
# --------------------------------------------------------------------------- #
def plot_classification(clf, images, labels, path, title,
                        n: int = 20, noise_sigma: float = 0.0,
                        seed: int = 2024):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=n, replace=False)
    sel = images[idx]
    sel_labels = labels[idx]

    if noise_sigma > 0:
        noise_rng = np.random.default_rng(seed + 1)
        sel = np.stack([add_noise(im, noise_sigma, noise_rng) for im in sel])

    feats = to_features(sel)
    preds = clf.predict(feats)
    # Pseudo-confidence from the one-vs-rest decision scores (softmax).
    scores = clf.decision_function(feats)
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    conf = (exp / exp.sum(axis=1, keepdims=True)).max(axis=1)

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
        conf_txt = f" ({conf[k]*100:.0f}%)"
        ax.set_title(
            f"prawd: {CLASS_GLYPHS[true_name]} {true_name}\n"
            f"pred: {CLASS_GLYPHS[pred_name]} {pred_name}{conf_txt}",
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
    parser.add_argument("--aug-copies", type=int, default=20,
                        help="augmented copies added per training image")
    parser.add_argument("--kernel", default="linear",
                        choices=["linear", "rbf", "poly", "sigmoid"],
                        help="SVM kernel (linear is the low-overfitting default)")
    parser.add_argument("--C", type=float, default=0.01,
                        help="SVM regularisation parameter (smaller = stronger "
                             "regularisation / less over-fitting)")
    parser.add_argument("--gamma", default="scale",
                        help="kernel coefficient for rbf/poly/sigmoid "
                             "(float or 'scale'/'auto')")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    images, labels = load_or_build(args.output_dir, args.per_class, args.seed)

    # 50% train / 50% test, drawn randomly per class, no validation set.
    xtr, ytr, xte, yte = train_test_split(images, labels,
                                          train_fraction=0.5, seed=args.seed)
    print(f"Train: {len(xtr)} images | Test: {len(xte)} images "
          f"(per class ~{len(xtr)//len(CLASS_NAMES)} / "
          f"{len(xte)//len(CLASS_NAMES)})")

    # Expand the training set with augmented copies (regularisation).
    if args.aug_copies > 0:
        xtr_aug, ytr_aug = expand_with_augmentation(
            xtr, ytr, args.aug_copies, seed=args.seed)
        print(f"Augmented training set: {len(xtr)} -> {len(xtr_aug)} images "
              f"(+{args.aug_copies} copies each)")
    else:
        xtr_aug, ytr_aug = xtr, ytr

    Xtr = to_features(xtr_aug)
    Xte = to_features(xte)

    try:
        gamma = float(args.gamma)
    except ValueError:
        gamma = args.gamma

    def make_clf():
        return make_pipeline(
            StandardScaler(),
            SVC(kernel=args.kernel, C=args.C, gamma=gamma,
                decision_function_shape="ovr", random_state=args.seed),
        )

    print(f"Training SVM ({args.kernel} kernel, C={args.C})...")
    clf = make_clf()
    clf.fit(Xtr, ytr_aug)
    train_acc = clf.score(Xtr, ytr_aug)
    test_acc = clf.score(Xte, yte)
    print(f"SVM trained.  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    # Learning curve (accuracy vs. number of training samples).
    print("Computing learning curve...")
    fractions = np.linspace(0.1, 1.0, 10)
    sizes, tr_curve, te_curve = compute_learning_curve(
        make_clf, Xtr, ytr_aug, Xte, yte, fractions, seed=args.seed)
    plot_learning_curves(
        sizes, tr_curve, te_curve,
        os.path.join(args.output_dir, "learning_curves_svm.png"),
        kernel=args.kernel)

    # Classification demos.
    plot_classification(
        clf, xte, yte,
        os.path.join(args.output_dir, "classification_clean_svm.png"),
        "SVM – klasyfikacja 20 obrazów z bazy (bez szumu)",
        n=20, noise_sigma=0.0, seed=2024)
    plot_classification(
        clf, xte, yte,
        os.path.join(args.output_dir, "classification_noisy_svm.png"),
        f"SVM – klasyfikacja 20 obrazów z bazy + szum (σ={EVAL_NOISE_SIGMA:.0f})",
        n=20, noise_sigma=EVAL_NOISE_SIGMA, seed=2024)

    model_path = os.path.join(args.output_dir, "svm_shapes.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved trained SVM -> {model_path}")
    print(f"\nFinal test accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
