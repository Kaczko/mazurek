"""Generation of the contour-shape data set.

For every shape class we render a clean 32x32 grayscale contour figure and then
produce many randomly augmented variants (noise, shift, blur, skew/slant,
rotation, scaling, stroke-width and intensity changes).  The images are kept as
``uint8`` grayscale (0-255), i.e. real grayscale and *not* a 0/1 black & white
mask, exactly as requested.

The shapes correspond to the requested glyphs:

    ○  circle     □  square     △  triangle   ◇  diamond
    ☆  star       ⬠  pentagon   ⬡  hexagon

The figures are drawn geometrically (with anti-aliasing through super-sampling)
which is more robust than relying on a particular system font carrying these
Unicode glyphs.  The varying stroke width / size / slant plays the same role as
"using different fonts and rasterising them".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Canonical class order.  Index == label used everywhere else.
CLASS_NAMES = [
    "circle",
    "square",
    "triangle",
    "diamond",
    "star",
    "pentagon",
    "hexagon",
]

# Unicode glyph for each class (used only for nice plot titles).
CLASS_GLYPHS = {
    "circle": "○",
    "square": "□",
    "triangle": "△",
    "diamond": "◇",
    "star": "☆",
    "pentagon": "⬠",
    "hexagon": "⬡",
}

IMG_SIZE = 32          # final image side in pixels
SUPERSAMPLE = 8        # render at IMG_SIZE * SUPERSAMPLE then shrink (anti-alias)


# --------------------------------------------------------------------------- #
# Shape geometry                                                              #
# --------------------------------------------------------------------------- #
def _regular_polygon(cx: float, cy: float, r: float, n: int,
                     start_angle: float) -> list[tuple[float, float]]:
    """Vertices of a regular ``n``-gon centred at ``(cx, cy)``."""
    pts = []
    for k in range(n):
        a = start_angle + 2.0 * math.pi * k / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _star_polygon(cx: float, cy: float, r: float, n: int,
                  start_angle: float, inner_ratio: float = 0.42
                  ) -> list[tuple[float, float]]:
    """Vertices of an ``n``-pointed star."""
    pts = []
    for k in range(2 * n):
        a = start_angle + math.pi * k / n
        rr = r if k % 2 == 0 else r * inner_ratio
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _shape_points(name: str, cx: float, cy: float, r: float):
    """Return (points, is_circle) for the requested shape."""
    up = -math.pi / 2.0  # point straight up
    if name == "circle":
        return None, True
    if name == "square":
        return _regular_polygon(cx, cy, r, 4, math.pi / 4.0), False
    if name == "diamond":
        return _regular_polygon(cx, cy, r, 4, up), False
    if name == "triangle":
        return _regular_polygon(cx, cy, r, 3, up), False
    if name == "pentagon":
        return _regular_polygon(cx, cy, r, 5, up), False
    if name == "hexagon":
        return _regular_polygon(cx, cy, r, 6, up), False
    if name == "star":
        return _star_polygon(cx, cy, r, 5, up), False
    raise ValueError(f"unknown shape: {name!r}")


# --------------------------------------------------------------------------- #
# Augmentation parameters                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class AugParams:
    scale: float          # radius as a fraction of half the image side
    dx: float             # horizontal shift in final pixels
    dy: float             # vertical shift in final pixels
    rotation: float       # degrees
    shear: float          # slant factor (pochylenie)
    stroke: float         # contour width in final pixels
    blur: float           # gaussian blur radius in final pixels
    noise: float          # gaussian noise std (0-255 scale)
    fg: int               # contour intensity (dark)
    bg: int               # background intensity (light)


def random_params(rng: np.random.Generator) -> AugParams:
    return AugParams(
        scale=rng.uniform(0.60, 0.84),
        dx=rng.uniform(-4.0, 4.0),
        dy=rng.uniform(-4.0, 4.0),
        rotation=rng.uniform(-12.0, 12.0),
        shear=rng.uniform(-0.26, 0.26),
        stroke=rng.uniform(1.6, 3.2),
        blur=rng.uniform(0.0, 1.0),
        noise=rng.uniform(0.0, 20.0),
        fg=int(rng.integers(0, 50)),
        bg=int(rng.integers(210, 256)),
    )


def clean_params() -> AugParams:
    """A neutral, noise-free rendering used for the dataset overview."""
    return AugParams(scale=0.72, dx=0.0, dy=0.0, rotation=0.0, shear=0.0,
                     stroke=2.2, blur=0.4, noise=0.0, fg=20, bg=255)


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def render_shape(name: str, p: AugParams,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Render a single shape variant and return a ``uint8`` 32x32 array."""
    S = SUPERSAMPLE
    big = IMG_SIZE * S
    img = Image.new("L", (big, big), color=p.bg)
    draw = ImageDraw.Draw(img)

    cx = cy = big / 2.0
    r = p.scale * (big / 2.0)
    width = max(1, int(round(p.stroke * S)))

    pts, is_circle = _shape_points(name, cx, cy, r)
    if is_circle:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     outline=p.fg, width=width)
    else:
        closed = list(pts) + [pts[0]]
        draw.line(closed, fill=p.fg, width=width, joint="curve")

    # Geometric transforms applied on the high-resolution canvas.
    # 1) rotation around the centre
    if abs(p.rotation) > 1e-6:
        img = img.rotate(p.rotation, resample=Image.BICUBIC,
                         fillcolor=p.bg, center=(cx, cy))
    # 2) shear / slant (pochylenie) via an affine transform
    if abs(p.shear) > 1e-6:
        a, b = 1.0, p.shear
        c = -p.shear * cy  # keep the centre roughly fixed
        img = img.transform((big, big), Image.AFFINE,
                            (a, b, c, 0.0, 1.0, 0.0),
                            resample=Image.BICUBIC, fillcolor=p.bg)

    # Shrink to target size (this is where anti-aliasing -> grayscale happens).
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.LANCZOS)

    # 3) translation / shift (przesunięcie)
    if abs(p.dx) > 1e-6 or abs(p.dy) > 1e-6:
        img = img.transform((IMG_SIZE, IMG_SIZE), Image.AFFINE,
                            (1.0, 0.0, -p.dx, 0.0, 1.0, -p.dy),
                            resample=Image.BICUBIC, fillcolor=p.bg)

    # 4) blur (rozmycie)
    if p.blur > 1e-3:
        img = img.filter(ImageFilter.GaussianBlur(radius=p.blur))

    arr = np.asarray(img, dtype=np.float32)

    # 5) additive gaussian noise (zakłócenia / szum)
    if p.noise > 1e-3:
        assert rng is not None, "rng required when noise > 0"
        arr = arr + rng.normal(0.0, p.noise, size=arr.shape)

    return np.clip(arr, 0, 255).astype(np.uint8)


def add_noise(image: np.ndarray, sigma: float,
              rng: np.random.Generator) -> np.ndarray:
    """Add gaussian noise to an existing ``uint8`` image."""
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, size=image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _border_bg(image: np.ndarray) -> int:
    """Estimate the background intensity from the image border."""
    border = np.concatenate([image[0, :], image[-1, :],
                             image[:, 0], image[:, -1]])
    return int(np.median(border))


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light, on-the-fly augmentation used during training.

    Applies a small random rotation, translation, blur and additive noise so
    that the network sees a fresh variant of every training image at each epoch.
    This multiplies the effective amount of training data without changing the
    fixed 50/50 split.
    """
    bg = _border_bg(image)
    img = Image.fromarray(image, mode="L")

    angle = rng.uniform(-7.0, 7.0)
    if abs(angle) > 1e-3:
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=bg)

    dx = rng.uniform(-2.5, 2.5)
    dy = rng.uniform(-2.5, 2.5)
    if abs(dx) > 1e-3 or abs(dy) > 1e-3:
        img = img.transform(img.size, Image.AFFINE,
                            (1.0, 0.0, -dx, 0.0, 1.0, -dy),
                            resample=Image.BILINEAR, fillcolor=bg)

    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.7)))

    arr = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.6:
        arr = arr + rng.normal(0.0, rng.uniform(3.0, 13.0), size=arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Data set assembly                                                           #
# --------------------------------------------------------------------------- #
def build_dataset(per_class: int = 100, seed: int = 1234,
                  progress: Callable[[str], None] | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Create ``per_class`` augmented variants for every shape.

    Returns ``(images, labels)`` where ``images`` has shape
    ``(num_classes * per_class, 32, 32)`` as ``uint8`` and ``labels`` are the
    integer class indices.
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for label, name in enumerate(CLASS_NAMES):
        for _ in range(per_class):
            p = random_params(rng)
            images.append(render_shape(name, p, rng))
            labels.append(label)
        if progress:
            progress(f"  generated {per_class} variants of {name}")
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def train_test_split(images: np.ndarray, labels: np.ndarray,
                     train_fraction: float = 0.5, seed: int = 7
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random per-class split (no validation set)."""
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_fraction))
        tr_idx.extend(idx[:n_train].tolist())
        te_idx.extend(idx[n_train:].tolist())
    tr_idx = np.array(tr_idx)
    te_idx = np.array(te_idx)
    rng.shuffle(tr_idx)
    rng.shuffle(te_idx)
    return (images[tr_idx], labels[tr_idx], images[te_idx], labels[te_idx])
