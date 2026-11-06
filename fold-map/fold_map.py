"""Which parts of an origami sheet are visible once it's folded flat?

    python fold_map.py crease_pattern.fold folded_state.fold

The crease pattern gives each vertex's position on the unfolded sheet. The folded
state (Flat-Folder: Export -> state) gives each vertex's folded position and the
stacking order of overlapping faces (faceOrders). Both files share vertex and face
numbering.

Conventions match Flat-Folder's display, which draws coordinates with y pointing down,
so the viewer looks at the paper from -z:
  side 1   the side of the sheet you see in Flat-Folder's crease pattern view
  side 2   the other side
  front    the folded model as Flat-Folder draws it
  back     the folded model with "Flip" checked, i.e. turned over left to right
All images and arrays are oriented as you'd see them from that side.

Writes:
  visibility_sheet.png   both sides of the sheet, colored by which view shows them
  folded_views.png       front and back of the folded model, textured by sheet position
  fold_map.npz           per-pixel fold maps for both views, for splicing images later
"""
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

EPS = 1e-9


def signed_area(P):
    return 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])


def affine(src, dst):
    # 3x2 matrix M with [src, 1] @ M = dst (exact for a rigid motion, possibly a reflection).
    M, *_ = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)
    return M


def apply(M, P):
    return np.c_[P, np.ones(len(P))] @ M


def inside(P, poly):
    # Points P (n, 2) inside the convex polygon poly (k, 2), boundary included.
    if signed_area(poly) < 0:
        poly = poly[::-1]
    a, b = poly, np.roll(poly, -1, axis=0)
    cross = ((b[:, 0] - a[:, 0])[None] * (P[:, 1, None] - a[None, :, 1])
             - (b[:, 1] - a[:, 1])[None] * (P[:, 0, None] - a[None, :, 0]))
    return (cross >= -EPS).all(axis=1)


class FoldedModel:
    def __init__(self, cp_path, state_path):
        cp, st = json.load(open(cp_path)), json.load(open(state_path))
        assert cp["edges_vertices"] == st["edges_vertices"], "files don't share vertex numbering"
        self.sheet = np.array(cp["vertices_coords"], float)
        self.folded = np.array(st["vertices_coords"], float)
        self.faces = [np.array(f) for f in st["faces_vertices"]]
        self.edges = np.array(st["edges_vertices"])
        self.assignment = st["edges_assignment"]
        n = len(self.faces)

        # normal: +1 if the face's side-2 (+z) normal still points +z after folding, -1 if it flipped over.
        area = lambda pts, f: signed_area(pts[f])
        self.normal = np.array([np.sign(area(self.folded, f)) * np.sign(area(self.sheet, f)) for f in self.faces])
        self.to_folded = [affine(self.sheet[f], self.folded[f]) for f in self.faces]
        self.to_sheet = [affine(self.folded[f], self.sheet[f]) for f in self.faces]

        # higher[f, g]: f is above g, i.e. farther toward +z. FOLD's [f, g, s] means f is on the side
        # g's normal points to when s = +1, so flip the sign when g's normal points -z.
        self.higher = np.zeros((n, n), bool)
        known = np.zeros((n, n), bool)
        for f, g, s in st["faceOrders"]:
            up = s * self.normal[g] > 0
            if known[f, g] and self.higher[f, g] != up:
                raise ValueError(f"contradictory order for faces {f}, {g}")
            self.higher[f, g], self.higher[g, f] = up, not up
            known[f, g] = known[g, f] = True
        self.known = known

    def cover(self, Q):
        # (len(Q), n_faces): which faces contain each folded-plane point.
        return np.stack([inside(Q, self.folded[f]) for f in self.faces], axis=1)

    def stack_ends(self, C):
        # Topmost and bottommost face at each point (-1 where no paper). Also counts points where
        # two covering faces have no recorded order, which would make the answer ambiguous.
        Cf = C.astype(float)
        above = Cf @ self.higher.astype(float)       # [p, f]: covering faces higher than f
        below = Cf @ self.higher.T.astype(float)     # [p, f]: covering faces lower than f
        top = np.where(C & (above == 0), np.arange(C.shape[1]), -1).max(axis=1)
        bottom = np.where(C & (below == 0), np.arange(C.shape[1]), -1).max(axis=1)
        unknown = ((Cf @ (~self.known).astype(float)) * Cf - Cf).max(axis=1) > 0
        return top, bottom, int(unknown.sum())

    def sheet_visibility(self, res):
        # For each side of the sheet, as seen from that side: 0 hidden, 1 visible from the front, 2 from the back.
        lo, hi = self.sheet.min(0), self.sheet.max(0)
        xs = lo[0] + (np.arange(res) + 0.5) / res * (hi[0] - lo[0])
        ys = lo[1] + (np.arange(res) + 0.5) / res * (hi[1] - lo[1])
        P = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        side1, side2 = np.zeros(len(P), np.uint8), np.zeros(len(P), np.uint8)
        unknown = 0
        for f, poly in enumerate(self.faces):
            idx = np.flatnonzero(inside(P, self.sheet[poly]))
            if len(idx) == 0:
                continue
            top, bottom, u = self.stack_ends(self.cover(apply(self.to_folded[f], P[idx])))
            unknown += u
            unflipped = self.normal[f] > 0
            # The front view looks from -z, so it sees the bottommost face. An unflipped face shows its
            # -z side (side 1) to the front view and its +z side (side 2) to the back view.
            (side1 if unflipped else side2)[idx[bottom == f]] = 1
            (side2 if unflipped else side1)[idx[top == f]] = 2
        side1, side2 = side1.reshape(res, res), side2.reshape(res, res)
        return side1, side2[:, ::-1], (lo, hi), unknown

    def views(self, res):
        # Fold maps for the front and back views: per folded-plane pixel, the sheet point shown and its side.
        lo, hi = self.folded.min(0), self.folded.max(0)
        pad = 0.03 * (hi - lo).max()
        lo, hi = lo - pad, hi + pad
        w = res
        h = int(round(res * (hi[1] - lo[1]) / (hi[0] - lo[0])))
        xs = lo[0] + (np.arange(w) + 0.5) / w * (hi[0] - lo[0])
        ys = lo[1] + (np.arange(h) + 0.5) / h * (hi[1] - lo[1])
        Q = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        top, bottom, unknown = self.stack_ends(self.cover(Q))
        out = {}
        # Front looks from -z and sees the bottommost face; an unflipped face shows it side 1.
        # Back looks from +z, sees the topmost face, and is mirrored left to right like turning the model over.
        for view, face, shows_side1, mirror in [("front", bottom, lambda f: self.normal[f] > 0, False),
                                                ("back", top, lambda f: self.normal[f] < 0, True)]:
            sheet_xy = np.full((len(Q), 2), np.nan)
            side = np.zeros(len(Q), np.uint8)              # 0 no paper, 1 side 1, 2 side 2
            for f in np.unique(face[face >= 0]):
                idx = face == f
                sheet_xy[idx] = apply(self.to_sheet[f], Q[idx])
                side[idx] = 1 if shows_side1(f) else 2
            arrays = [sheet_xy.reshape(h, w, 2), side.reshape(h, w), face.reshape(h, w)]
            out[view] = tuple(a[:, ::-1] if mirror else a for a in arrays)
        return out, (lo, hi), unknown


def texture(xy, side, sheet_lo, sheet_hi):
    # Color each sheet point by position (hue = angle around the center, checkerboard for scale).
    # Side 2 is drawn paler so the two sides are easy to tell apart.
    t = (xy - sheet_lo) / (sheet_hi - sheet_lo)
    hue = (np.arctan2(t[..., 1] - 0.5, t[..., 0] - 0.5) / (2 * np.pi)) % 1
    checker = ((np.floor(t[..., 0] * 8) + np.floor(t[..., 1] * 8)) % 2).astype(float)
    sat = np.where(side == 1, 0.75, 0.3)
    val = np.where(side == 1, 0.8 + 0.2 * checker, 0.75 + 0.25 * checker)
    rgb = hsv_to_rgb(np.stack([np.nan_to_num(hue), sat, val], -1))
    rgb[side == 0] = 0.92
    return rgb


def extent(lo, hi, mirrored):
    # imshow extent for an image drawn y-down (like Flat-Folder), optionally turned over left to right,
    # so the model's own coordinates can be plotted on top of it.
    x0, x1 = (hi[0], lo[0]) if mirrored else (lo[0], hi[0])
    return (x0, x1, hi[1], lo[1])


def draw_creases(ax, model, swap=False):
    # Mountain red, valley blue. The file labels them as seen from side 2 (+z); Flat-Folder shows side 1, swapped.
    colors = {"M": "#d22", "V": "#22d", "B": "#000"}
    if swap:
        colors["M"], colors["V"] = colors["V"], colors["M"]
    for (a, b), kind in zip(model.edges, model.assignment):
        if kind in colors:
            ax.plot(*model.sheet[[a, b]].T, color=colors[kind], lw=0.6 if kind != "B" else 1.2)


PALETTE = np.array([[0.82, 0.82, 0.82], [0.83, 0.69, 0.48], [0.36, 0.55, 0.80]])   # hidden, front, back


def plot_visibility(model, side1, side2, sheet_bounds):
    # Both sides of the sheet, each as seen from that side, colored by which view shows it.
    slo, shi = sheet_bounds
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    for ax, s, title, turned in [(axes[0], side1, "side 1 (as shown in Flat-Folder)", False),
                                 (axes[1], side2, "side 2 (sheet turned over left to right)", True)]:
        ax.imshow(PALETTE[s], extent=extent(slo, shi, turned), interpolation="nearest")
        draw_creases(ax, model, swap=not turned)
        ax.set_title(title); ax.set_aspect("equal"); ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PALETTE]
    fig.legend(handles, ["hidden", "visible from the front", "visible from the back"], loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_folded_views(model, views, folded_bounds, sheet_bounds):
    # The folded model from both sides, textured by where each visible point sits on the sheet.
    (flo, fhi), (slo, shi) = folded_bounds, sheet_bounds
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
    s = np.stack(np.meshgrid(np.linspace(slo[0], shi[0], 400), np.linspace(slo[1], shi[1], 400)), -1)
    axes[0].imshow(texture(s, np.ones(s.shape[:2], np.uint8), slo, shi), extent=extent(slo, shi, False))
    draw_creases(axes[0], model, swap=True)
    axes[0].set_title("test texture on side 1 (side 2 is paler)")
    for ax, view in zip(axes[1:], ["front", "back"]):
        xy, side, _ = views[view]
        ax.imshow(texture(xy, side, slo, shi), extent=extent(flo, fhi, view == "back"), interpolation="nearest")
        ax.set_title(f"folded, {view}" + (" (Flip)" if view == "back" else ""))
    for ax in axes:
        ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    return fig


def summarize(model, side1, side2, unknown):
    print(f"faces: {len(model.faces)}, flipped: {int((model.normal < 0).sum())}")
    for name, s in [("side 1", side1), ("side 2", side2)]:
        print(f"{name}: {np.mean(s == 1):6.1%} visible from the front, {np.mean(s == 2):6.1%} from the back, "
              f"{np.mean(s == 0):6.1%} hidden")
    print(f"sample points where touching faces have no recorded order: {unknown} "
          "(expected only on crease lines, where faces meet edge to edge)")


def main(cp_path, state_path):
    model = FoldedModel(cp_path, state_path)
    side1, side2, sheet_bounds, unk_sheet = model.sheet_visibility(800)
    views, folded_bounds, unk_view = model.views(900)
    summarize(model, side1, side2, unk_sheet + unk_view)

    plot_visibility(model, side1, side2, sheet_bounds).savefig("visibility_sheet.png", dpi=150)
    plot_folded_views(model, views, folded_bounds, sheet_bounds).savefig("folded_views.png", dpi=150)
    plt.close("all")

    np.savez_compressed("fold_map.npz", **{f"{v}_{k}": arr for v, (xy, side, face) in views.items()
                                           for k, arr in [("sheet_xy", xy), ("side", side), ("face", face)]},
                        folded_bounds=np.array(folded_bounds), sheet_bounds=np.array(sheet_bounds),
                        side1=side1, side2=side2)
    print("wrote visibility_sheet.png, folded_views.png, fold_map.npz")


if __name__ == "__main__":
    main(*sys.argv[1:3])
