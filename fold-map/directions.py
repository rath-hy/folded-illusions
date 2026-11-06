"""Which way is up on each visible part of the sheet?

    python directions.py crease_pattern.fold folded_state.fold

On every visible region, draws a letter F in the orientation a picture must be printed
in to appear upright on the folded model. A backwards F would mean that region shows a
mirror image. Draws one panel per side of the sheet that has visible regions. Writes
directions.png.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from fold_map import FoldedModel, PALETTE, apply, extent, inside

# Screen directions in folded coordinates. Flat-Folder draws y pointing down, so "up" is -y.
# The back view is turned over left to right, so its "right" is -x.
UP = np.array([0.0, -1.0])
RIGHT = {"front": np.array([1.0, 0.0]), "back": np.array([-1.0, 0.0])}
COLORS = {"front": "#b8864a", "back": "#2f63a8"}


def shown_side(model, f, view):
    # Which side of the sheet face f shows to a view. Unflipped faces keep side 2 (+z) facing +z, and
    # the back view looks from +z.
    unflipped = model.normal[f] > 0
    return 2 if unflipped == (view == "back") else 1


def visible_regions(model, res=500):
    # (face, view, side, sheet points) for every visible part of every face.
    lo, hi = model.sheet.min(0), model.sheet.max(0)
    g = lo + (np.stack(np.meshgrid(np.arange(res), np.arange(res)), -1).reshape(-1, 2) + 0.5) / res * (hi - lo)
    for f, poly in enumerate(model.faces):
        P = g[inside(g, model.sheet[poly])]
        if len(P) == 0:
            continue
        top, bottom, _ = model.stack_ends(model.cover(apply(model.to_folded[f], P)))
        for view, hit in [("back", top == f), ("front", bottom == f)]:
            if hit.sum() > 0.002 * res * res:
                yield f, view, shown_side(model, f, view), P[hit]


def plot_directions(model, side1, side2):
    # Returns the figure and whether any region would need a mirrored picture.
    lo, hi = model.sheet.min(0), model.sheet.max(0)
    regions = list(visible_regions(model))
    sides = [s for s in (1, 2) if any(r[2] == s for r in regions)]
    fig, axes = plt.subplots(1, len(sides), figsize=(8 * len(sides), 8.4), squeeze=False)
    light = PALETTE * 0.35 + 0.65
    mirrored = False
    for ax, s in zip(axes[0], sides):
        turned = s == 2                     # each side drawn as seen from that side
        ax.imshow(light[side2 if s == 2 else side1], extent=extent(lo, hi, turned), interpolation="nearest")
        for (a, b), kind in zip(model.edges, model.assignment):
            if kind in "MVB":
                ax.plot(*model.sheet[[a, b]].T, color="#999", lw=0.5 if kind != "B" else 1.2, zorder=1)
        for f, view, side, pts in regions:
            if side != s:
                continue
            # The face's linear map from folded to sheet coordinates turns screen directions into sheet directions.
            L = model.to_sheet[f][:2]
            up, right = UP @ L, RIGHT[view] @ L
            # Seen from side 2 the sheet is turned over, which flips handedness on the page.
            handed = (right[0] * up[1] - right[1] * up[0]) * (1 if turned else -1)
            mirrored |= handed < 0
            c = pts.mean(0)
            size = min(0.6 * np.sqrt(len(pts) / 500 ** 2) * (hi - lo).max(), 0.06)
            F = np.array([[-0.25, -0.5], [-0.25, 0.5], [0.3, 0.5], [-0.25, 0.5], [-0.25, 0.05], [0.2, 0.05]]) * size
            ax.plot(*(c + F[:, :1] * right + F[:, 1:] * up).T, color=COLORS[view], lw=2.2,
                    solid_capstyle="round", zorder=4)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"side {s}: each F shows which way is up on the folded model")
    handles = [plt.Line2D([], [], color=c, lw=3) for c in COLORS.values()]
    fig.legend(handles, ["visible from the front", "visible from the back"], loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig, mirrored


def main(cp_path, state_path):
    model = FoldedModel(cp_path, state_path)
    side1, side2, _, _ = model.sheet_visibility(800)
    fig, mirrored = plot_directions(model, side1, side2)
    fig.savefig("directions.png", dpi=150)
    print("mirrored regions:", "some" if mirrored else "none")
    print("wrote directions.png")


if __name__ == "__main__":
    main(*sys.argv[1:3])
