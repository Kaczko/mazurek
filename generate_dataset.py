"""Generate the contour-shape data set and the input-image overview PDF.

Run:
    python generate_dataset.py

Outputs (in ``outputs/``):
    dataset.npz            all images + labels (uint8 grayscale, 32x32)
    input_images.pdf       one-page graphic overview of the input images
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import (
    CLASS_NAMES, CLASS_GLYPHS, build_dataset, clean_params, render_shape,
)

OUTPUT_DIR = "outputs"


def render_overview_pdf(images: np.ndarray, labels: np.ndarray,
                        path: str, per_class: int = 12) -> None:
    """One PDF page: for each class a clean glyph + a row of variants."""
    n_classes = len(CLASS_NAMES)
    cols = 1 + per_class  # first column = clean reference
    fig, axes = plt.subplots(n_classes, cols,
                             figsize=(cols * 0.85, n_classes * 0.95))
    fig.suptitle("Obrazy wejściowe – figury konturowe 32×32 (grayscale 0–255)",
                 fontsize=14, y=0.995)

    rng = np.random.default_rng(99)
    cp = clean_params()
    for row, name in enumerate(CLASS_NAMES):
        # Column 0: clean reference rendering.
        ref = render_shape(name, cp)
        ax = axes[row, 0]
        ax.imshow(ref, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("tab:blue"); spine.set_linewidth(1.5)
        ax.set_ylabel(f"{CLASS_GLYPHS[name]} {name}", fontsize=9,
                      rotation=0, ha="right", va="center", labelpad=28)
        if row == 0:
            ax.set_title("wzorzec", fontsize=8)

        # Remaining columns: random augmented variants of this class.
        idx = np.where(labels == row)[0]
        pick = rng.choice(idx, size=per_class, replace=False)
        for c, i in enumerate(pick, start=1):
            ax = axes[row, c]
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0 and c == 1:
                ax.set_title("warianty (zakłócenia / przesunięcie / rozmycie / pochylenie)",
                             fontsize=8, loc="left")

    plt.tight_layout(rect=[0.04, 0.0, 1.0, 0.97])
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=100,
                        help="number of variants generated per shape")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.per_class} variants for each of "
          f"{len(CLASS_NAMES)} classes...")
    images, labels = build_dataset(per_class=args.per_class, seed=args.seed,
                                   progress=print)

    npz_path = os.path.join(args.output_dir, "dataset.npz")
    np.savez_compressed(npz_path, images=images, labels=labels,
                        class_names=np.array(CLASS_NAMES))
    print(f"Saved data set: {images.shape[0]} images -> {npz_path}")
    print(f"  dtype={images.dtype}, range=[{images.min()}, {images.max()}]")

    pdf_path = os.path.join(args.output_dir, "input_images.pdf")
    render_overview_pdf(images, labels, pdf_path)
    print(f"Saved input-image overview (1 page) -> {pdf_path}")


if __name__ == "__main__":
    main()
