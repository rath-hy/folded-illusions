"""Print pictures on paper so they appear on the folded model.

    python decal.py                                   # test pictures: a big F and B
    python decal.py --front cat.jpg --back dog.jpg    # your own pictures

Each picture is center-cropped (not distorted) to the folded model's silhouette as seen
from that side: front is Flat-Folder's folded view, back is the view with Flip. Every
visible point of the sheet gets the color of the picture where it lands when folded.
Only visible areas are printed, plus a thin bleed so slightly imperfect folds don't
reveal white slivers. Tiny letters A-D in hidden areas mark the corners, and the
preview shows where each corner ends up.

Writes:
  decal_side2.png   the sheet to print, as seen from side 2 (side 1 too, if it has visible areas)
  decal_preview.png the pictures, and the model simulated from the printed sheets
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from fold_map import FoldedModel, apply, inside
from directions import shown_side

HERE = os.path.dirname(os.path.abspath(__file__))
CORNERS = "ABCD"


def font(size):
    # Helvetica on a Mac, otherwise the DejaVu Sans that ships with matplotlib (e.g. on Colab).
    for path in ["/System/Library/Fonts/Helvetica.ttc", font_manager.findfont("DejaVu Sans")]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def test_picture(letter, color, size=(1300, 1000)):
    # A big letter over diagonal stripes, so rotation, mirroring and alignment errors are easy to spot.
    w, h = size
    im = Image.new("RGB", size, color)
    d = ImageDraw.Draw(im)
    light = tuple(int(c + (255 - c) * 0.45) for c in color)
    for x in range(-h, w, 90):
        d.line([(x, h), (x + h, 0)], fill=light, width=26)
    d.text((w / 2, h * 0.5), letter, fill="white", font=font(int(h * 0.42)), anchor="mm",
           stroke_width=12, stroke_fill=(30, 30, 30))
    return im


def picture_aspect(model):
    # Width / height of the folded silhouette's bounding box, which each picture is fitted to.
    lo, hi = model.folded.min(0), model.folded.max(0)
    return (hi[0] - lo[0]) / (hi[1] - lo[1])


def fit(picture, aspect):
    # Center-crop a picture to the given width/height aspect, as a float array in [0, 1].
    picture = ImageOps.exif_transpose(picture)
    w, h = picture.size
    if w / h > aspect:
        picture = picture.crop((round((w - h * aspect) / 2), 0, round((w + h * aspect) / 2), h))
    else:
        picture = picture.crop((0, round((h - w / aspect) / 2), w, round((h + w / aspect) / 2)))
    return np.asarray(picture.convert("RGB"), float) / 255


def sample(img, u, v):
    # Bilinear lookup of img at normalized coordinates (u right, v down), clamped to the edges.
    h, w = img.shape[:2]
    x, y = np.clip(u * w - 0.5, 0, w - 1), np.clip(v * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def view_coords(q, lo, hi, view):
    # Normalized picture coordinates of folded points q. Flat-Folder draws y down; the back view is
    # the model turned over left to right.
    u = (q[:, 0] - lo[0]) / (hi[0] - lo[0])
    v = (q[:, 1] - lo[1]) / (hi[1] - lo[1])
    return (1 - u if view == "back" else u), v


def render_side(model, pictures, side, res, bleed):
    # One side of the sheet in raw sheet coordinates (row = y, col = x): its colors, the pixels that
    # show on the folded model, and those grown by the bleed (what gets printed).
    lo, hi = model.folded.min(0), model.folded.max(0)
    slo, shi = model.sheet.min(0), model.sheet.max(0)
    g = np.stack(np.meshgrid(np.arange(res), np.arange(res)), -1).reshape(-1, 2)
    P = slo + (g + 0.5) / res * (shi - slo)
    color, visible = np.ones((len(P), 3)), np.zeros(len(P), bool)
    for f, poly in enumerate(model.faces):
        idx = np.flatnonzero(inside(P, model.sheet[poly]))
        if len(idx) == 0:
            continue
        # This side of face f faces one view. Every pixel gets that view's color, visible or not, so the
        # bleed continues the picture smoothly past the visible edge.
        view = "front" if shown_side(model, f, "front") == side else "back"
        q = apply(model.to_folded[f], P[idx])
        color[idx] = sample(pictures[view], *view_coords(q, lo, hi, view))
        top, bottom, _ = model.stack_ends(model.cover(q))
        visible[idx] = (top == f) if view == "back" else (bottom == f)
    visible = visible.reshape(res, res)
    grow = 2 * round(bleed * res) + 1
    printed = Image.fromarray(visible.astype(np.uint8) * 255)
    if grow > 1:
        printed = printed.filter(ImageFilter.MaxFilter(grow))
    printed = np.asarray(printed) > 0
    return color.reshape(res, res, 3), visible, printed


def corners(model):
    # The sheet's four corners, labeled A-D clockwise from Flat-Folder's top-left, with where each
    # lands on the folded model.
    slo, shi = model.sheet.min(0), model.sheet.max(0)
    pts = np.array([[slo[0], slo[1]], [shi[0], slo[1]], [shi[0], shi[1]], [slo[0], shi[1]]])
    folded = np.array([model.folded[np.argmin(np.linalg.norm(model.sheet - c, axis=1))] for c in pts])
    return pts, folded


def label_spots(model, printed, res):
    # For each corner, the unprinted spot nearest a point just inside it, with room for a small label.
    slo, shi = model.sheet.min(0), model.sheet.max(0)
    box = 2 * (res // 60) + 1
    free = Image.fromarray((~printed).astype(np.uint8) * 255).filter(ImageFilter.MinFilter(box))
    free = np.asarray(free) > 0
    free[:box], free[-box:], free[:, :box], free[:, -box:] = False, False, False, False   # not at the edge
    hidden = np.argwhere(free)
    spots = []
    for c in corners(model)[0]:
        target = (c - slo) / (shi - slo) * res
        target = target + (res / 2 - target) * 0.1
        spots.append(tuple(hidden[np.argmin(((hidden - target[::-1]) ** 2).sum(1))][::-1]) if len(hidden) else None)
    return spots


def make_sheets(model, pictures, res=2400, bleed=0.01):
    # Printable sheets for every side of the paper that shows on the folded model. Returns
    # {side: dict(image, color, visible, printed)}, where image is a PIL image as seen from that side.
    sheets = {}
    side1, side2, _, _ = model.sheet_visibility(200)       # quick check of which sides show at all
    for side in [s for s, vis in [(1, side1), (2, side2)] if vis.any()]:
        color, visible, printed = render_side(model, pictures, side, res, bleed)
        raw = np.where(printed[..., None], color, 1.0)
        turned = side == 2                       # side 2 is the sheet turned over left to right
        image = Image.fromarray(((raw[:, ::-1] if turned else raw) * 255).round().astype(np.uint8))
        d = ImageDraw.Draw(image)
        d.rectangle([0, 0, res - 1, res - 1], outline=(200, 200, 200), width=2)
        small = font(max(12, res // 90))
        for letter, spot in zip(CORNERS, label_spots(model, printed, res)):
            if spot is not None:
                col, row = spot
                d.text((res - 1 - col if turned else col, row), letter, fill=(120, 120, 120), font=small, anchor="mm")
        sheets[side] = dict(image=image, color=raw, visible=visible, printed=printed)
    return sheets


def simulate(model, sheets, res=700):
    # The folded model as it would look from the printed sheets: {view: rgb array}, plus bounds.
    views, bounds, _ = model.views(res)
    slo, shi = model.sheet.min(0), model.sheet.max(0)
    out = {}
    for view in ["front", "back"]:
        xy, side, _ = views[view]
        t = np.nan_to_num((xy.reshape(-1, 2) - slo) / (shi - slo))   # NaN where there's no paper
        rgb = np.full((len(t), 3), 0.85)
        for s, sheet in sheets.items():
            hit = side.ravel() == s
            rgb[hit] = sample(sheet["color"] * sheet["printed"][..., None] + (~sheet["printed"][..., None]),
                              t[hit, 0], t[hit, 1])
        out[view] = rgb.reshape(*side.shape, 3)
    return out, bounds


def plot_preview(model, pictures, simulated, bounds):
    # The pictures on top; the simulated folded model below, with the corners A-D marked.
    lo, hi = bounds
    _, folded = corners(model)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    for col, view in enumerate(["front", "back"]):
        axes[0, col].imshow(pictures[view]); axes[0, col].set_title(f"{view} picture")
        turned = view == "back"
        x0, x1 = (hi[0], lo[0]) if turned else (lo[0], hi[0])
        axes[1, col].imshow(simulated[view], extent=(x0, x1, hi[1], lo[1]))
        axes[1, col].set_title(f"folded, {view}" + (" (Flip)" if turned else ""))
        # Corners that land in the same place (like the crane's two wing tips) share one label.
        groups = {}
        for letter, p in zip(CORNERS, folded):
            groups.setdefault(tuple(np.round(p, 6)), []).append(letter)
        for p, letters in groups.items():
            axes[1, col].annotate("/".join(letters), p, color="#c22", fontsize=11, fontweight="bold",
                                  ha="center", va="center", xytext=(0, 14), textcoords="offset points")
            axes[1, col].plot(*p, "o", color="#c22", ms=4)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--front", help="picture to show on the front of the folded model")
    ap.add_argument("--back", help="picture to show on the back of the folded model")
    ap.add_argument("--cp", default=os.path.join(HERE, "004_traditional_Crane_cp.fold"))
    ap.add_argument("--state", default=os.path.join(HERE, "004_traditional_Crane_state.fold"))
    ap.add_argument("--res", type=int, default=2400, help="sheet size in pixels (2400 = 8 inches at 300 dpi)")
    ap.add_argument("--bleed", type=float, default=0.01, help="margin around visible areas, as a fraction of the sheet")
    args = ap.parse_args()

    model = FoldedModel(args.cp, args.state)
    aspect = picture_aspect(model)
    pictures = {"front": fit(Image.open(args.front) if args.front else test_picture("F", (196, 120, 60)), aspect),
                "back": fit(Image.open(args.back) if args.back else test_picture("B", (52, 98, 168)), aspect)}

    sheets = make_sheets(model, pictures, args.res, args.bleed)
    for side, sheet in sheets.items():
        sheet["image"].save(f"decal_side{side}.png", dpi=(args.res / 8, args.res / 8))
        print(f"side {side}: printed area {sheet['printed'].mean():.0%} of the sheet "
              f"({sheet['visible'].mean():.0%} visible + bleed) -> decal_side{side}.png")
    simulated, bounds = simulate(model, sheets)
    plot_preview(model, pictures, simulated, bounds).savefig("decal_preview.png", dpi=110)
    print("wrote decal_preview.png")


if __name__ == "__main__":
    main()
