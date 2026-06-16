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

from src.dataset import CLASS_NAMES, CLASS_GLYPHS, build_dataset

OUTPUT_DIR = "outputs"


def render_overview_pdf(images: np.ndarray, labels: np.ndarray,
                        path: str, block_cols: int = 20) -> None:
    """One PDF page showing *all* generated input images.

    Every class is laid out as a contiguous block of all its variants
    (``block_cols`` images per row), and the blocks are stacked vertically so
    the full data set (e.g. 7 classes x 100 = 700 images) fits on a single page.
    """
    n_classes = len(CLASS_NAMES)
    per_class = int(np.bincount(labels).min())  # variants available per class
    block_rows = int(np.ceil(per_class / block_cols))
    total_rows = n_classes * block_rows

    fig, axes = plt.subplots(
        total_rows, block_cols,
        figsize=(block_cols * 0.42, total_rows * 0.42),
        squeeze=False,
    )
    fig.suptitle(
        f"Obrazy wejściowe – figury konturowe 32×32 (grayscale 0–255)\n"
        f"{n_classes} klas × {per_class} = {n_classes * per_class} obrazów "
        f"(szum / przesunięcie / rozmycie / pochylenie)",
        fontsize=11, y=0.997)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        ax.axis("off")

    for ci, name in enumerate(CLASS_NAMES):
        idx = np.where(labels == ci)[0][:per_class]
        for k, img_i in enumerate(idx):
            r = ci * block_rows + k // block_cols
            c = k % block_cols
            ax = axes[r, c]
            ax.axis("on")
            ax.imshow(images[img_i], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3); spine.set_color("0.8")
        # Class label to the left of the block's first row.
        label_ax = axes[ci * block_rows, 0]
        label_ax.set_ylabel(f"{CLASS_GLYPHS[name]}  {name}", fontsize=10,
                            rotation=0, ha="right", va="top", labelpad=12)

    plt.subplots_adjust(left=0.11, right=0.995, top=0.95, bottom=0.005,
                        wspace=0.08, hspace=0.08)
    fig.savefig(path, format="pdf")
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
