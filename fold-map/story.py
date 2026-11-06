"""Folding story: a different picture at each stage of folding a crane.

    python story.py                  # 0 flat, 1 bird base, 2 crane
    python story.py --text ABC       # any three characters
    python story.py --stages "preliminary base" "bird base" crane    # 0, 1, 2, 3

The pictures are single characters, so no generative model is needed. Each later stage's
character is sized, placed and stretched to fit inside what that stage shows, choosing the
placements (and which way round each stage is held) so the characters compete for as few sheet
pixels as possible. The flat character then goes where the folded ones leave room. Everything
else stays paper white, so a slightly sloppy fold only shows a little more white.

Writes story_sheet.png (the printed side, 300 dpi at 20 cm) and story_preview.png (the sheet,
and each stage folded virtually from it).

story.ipynb uses the same stages with DeepFloyd IF instead, for pictures that hide each other.
"""
import argparse
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from fold_map import FoldedModel, apply, inside
from illusion import SHEET_MM, FoldViews

HERE = os.path.dirname(os.path.abspath(__file__))
O, A, B = (0.5, 0.5), (0.0, 0.0), (1.0, 0.0)

# name, crease pattern, folded state, and the sheet points that should be up and down when the
# stage is held the usual way (None: as Flat-Folder draws it, which for the crane is wings up).
STAGES = [
    ("preliminary base", "stages/preliminary_cp.fold", "stages/preliminary_state.fold", (O, A)),
    ("bird base", "stages/bird_cp.fold", "stages/bird_state.fold", (B, A)),
    ("crane", "004_traditional_Crane_cp.fold", "004_traditional_Crane_state.fold", None),
]
# The preliminary base is left out by default: each of its faces is one uncut quarter of the sheet,
# so its picture sits on the flat sheet intact. The bird base and crane cut their paper into 2 and
# about 10 pieces, and with the sides least_shared_sides picks, they share no paper.
DEFAULT_STAGES = ["bird base", "crane"]


def load_stages(names=DEFAULT_STAGES):
    """The stages to fold through, in folding order, picked by name (None: all of STAGES).
    Returns (stages, models): the STAGES entries and their FoldedModels."""
    stages = [st for st in STAGES if names is None or st[0] in names]
    if names is not None and len(stages) != len(names):
        raise ValueError(f"unknown stage in {names}; choose from {[st[0] for st in STAGES]}")
    return stages, [FoldedModel(os.path.join(HERE, cp), os.path.join(HERE, st)) for _, cp, st, _ in stages]


COLORS = [(20, 20, 20), (214, 40, 40), (29, 78, 216), (22, 128, 61)]   # one per character, in story order


def bold_font(size):
    for path, index in [("/System/Library/Fonts/Helvetica.ttc", 1),
                        (font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")), 0)]:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            pass
    return ImageFont.load_default()


def glyph(ch, size=400):
    # The character as a tightly cropped boolean bitmap.
    font = bold_font(size)
    im = Image.new("L", (size * 2, size * 2))
    ImageDraw.Draw(im).text((size, size), ch, fill=255, font=font, anchor="mm")
    a = np.asarray(im) > 127
    rows, cols = np.flatnonzero(a.any(1)), np.flatnonzero(a.any(0))
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def stamp(g, cx, cy, w, h, U, V):
    # Whether points (U, V) (canvas units) fall on glyph g drawn w x h, centered at (cx, cy).
    gu = ((U - cx) / w + 0.5) * g.shape[1]
    gv = ((V - cy) / h + 0.5) * g.shape[0]
    ok = (gu >= 0) & (gu < g.shape[1]) & (gv >= 0) & (gv < g.shape[0])
    out = np.zeros(U.shape, bool)
    out[ok] = g[gv[ok].astype(int), gu[ok].astype(int)]
    return out


class StageView:
    """One side of one stage, held upright, on a canvas the size of the sheet centered on the model.

    Canvas coordinates are in sheet widths, (0, 0) top-left to (1, 1) bottom-right.
      visible   (res, res) bool, raw sheet coordinates (row = y, col = x): sheet points seen
      uv        (res, res, 2): where each visible sheet point lands on the canvas
      piece     (res, res) int: which rigid piece of paper each sheet point belongs to (faces
                that move together); a character spread over several pieces is scattered when flat
      outline(n) (n, n) bool: canvas pixels covered by paper
    """

    def __init__(self, name, model, view, turn, res):
        self.name, self.model, self.view, self.turn = name, model, view, turn
        self.center = (model.folded.min(0) + model.folded.max(0)) / 2
        self.size = (model.sheet.max(0) - model.sheet.min(0)).max()
        slo, shi = model.sheet.min(0), model.sheet.max(0)
        g = np.stack(np.meshgrid(np.arange(res), np.arange(res)), -1).reshape(-1, 2)
        P = slo + (g + 0.5) / res * (shi - slo)
        visible, uv = np.zeros(len(P), bool), np.full((len(P), 2), np.nan)
        piece, pieces = np.full(len(P), -1), {}
        for f, poly in enumerate(model.faces):
            idx = np.flatnonzero(inside(P, model.sheet[poly]))
            if len(idx) == 0:
                continue
            piece[idx] = pieces.setdefault(tuple(np.round(np.asarray(model.to_folded[f]).ravel(), 4)), len(pieces))
            q = apply(model.to_folded[f], P[idx])
            top, bottom, _ = model.stack_ends(model.cover(q))
            seen = (bottom == f) if view == "front" else (top == f)
            visible[idx[seen]] = True
            uv[idx[seen]] = self.to_canvas(q[seen])
        self.visible, self.uv, self.piece = visible.reshape(res, res), uv.reshape(res, res, 2), piece.reshape(res, res)

    def to_canvas(self, q):
        # Inverse of FoldViews' canvas: mirror the back view, then undo the turn.
        x = (q[:, 0] - self.center[0]) * (-1 if self.view == "back" else 1)
        y = q[:, 1] - self.center[1]
        a = np.radians(self.turn)
        return np.stack([x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)], 1) / self.size + 0.5

    def outline(self, n):
        c = (np.arange(n) + 0.5) / n - 0.5
        u, v = (m.ravel() * self.size for m in np.meshgrid(c, c))
        a = np.radians(self.turn)
        u, v = u * np.cos(a) + v * np.sin(a), -u * np.sin(a) + v * np.cos(a)
        Q = np.stack([self.center[0] + (-u if self.view == "back" else u), self.center[1] + v], -1)
        top, bottom, _ = self.model.stack_ends(self.model.cover(Q))
        return ((bottom if self.view == "front" else top) >= 0).reshape(n, n)


def upright_turn(model, view, up):
    # Clockwise canvas turn (degrees) that puts sheet point up[0] straight above up[1].
    if up is None:
        return 0.0
    near = lambda p: model.folded[np.argmin(np.linalg.norm(model.sheet - np.array(p), axis=1))]
    d = near(up[0]) - near(up[1])
    dx = -d[0] if view == "back" else d[0]
    return float(np.degrees(np.arctan2(dx, d[1]))) + 180


def fit_candidates(g, region, n_keep=40, margin=0.02):
    # Placements (cx, cy, w, h) of glyph g that fit inside `region` (a canvas mask) with a margin,
    # largest first, spread out so they aren't all tiny shifts of one placement.
    n = region.shape[0]
    from PIL import ImageFilter
    k = 2 * max(1, round(margin * n)) + 1
    inner = np.asarray(Image.fromarray(region.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(k))) > 0
    rows, cols = np.flatnonzero(inner.any(1)), np.flatnonzero(inner.any(0))
    if len(rows) == 0:
        return []
    V, U = (np.mgrid[0:n, 0:n] + 0.5) / n
    span_h, span_w = (rows[-1] - rows[0] + 1) / n, (cols[-1] - cols[0] + 1) / n
    aspect = g.shape[1] / g.shape[0]
    found = []
    for h in np.linspace(span_h, span_h * 0.2, 16):
        for stretch in (1.0, 0.8, 1.25, 0.65, 1.5):
            w = h * aspect * stretch
            if w > span_w:
                continue
            for cy in np.linspace(rows[0] / n + h / 2, (rows[-1] + 1) / n - h / 2, 9):
                for cx in np.linspace(cols[0] / n + w / 2, (cols[-1] + 1) / n - w / 2, 9):
                    ink = stamp(g, cx, cy, w, h, U, V)
                    if ink.any() and (ink & ~inner).sum() <= 0.002 * ink.sum():
                        found.append((h * w * min(stretch, 1 / stretch), (cx, cy, w, h)))
        if len(found) >= n_keep * 3:
            break
    found.sort(key=lambda t: -t[0])
    keep = []
    for score, p in found:
        if all(np.hypot(p[0] - k[1][0], p[1] - k[1][1]) > 0.03 or abs(p[3] - k[1][3]) > 0.03 for k in keep):
            keep.append((score, p))
        if len(keep) == n_keep:
            break
    return keep


def ink_on_sheet(g, place, sv):
    # Sheet pixels (raw coordinates) where this stage's character should be.
    ink = np.zeros(sv.visible.shape, bool)
    uv = sv.uv[sv.visible]
    ink[sv.visible] = stamp(g, *place, uv[:, 0], uv[:, 1])
    return ink


def intact(ink, sv):
    # Share of a character's ink on its largest single piece of paper: 1 means it reads whole
    # when the sheet is flat, lower means folding scatters it.
    ids, counts = np.unique(sv.piece[ink], return_counts=True)
    return counts.max() / counts.sum() if len(counts) else 1.0


def plan(text, stages, models, res=160, canvas=160, weight=4.0, scatter=2.0):
    """Choose a side for each stage and a placement for every character.

    Returns (views, placements, report): the StageView for each folded stage, (cx, cy, w, h)
    per character (the flat one in printed-sheet coordinates), and the clashes and intactness of
    the choice. The score rewards big characters, charges for every sheet pixel two characters
    both claim, and (with `scatter`) charges for characters that stay whole on the flat sheet.
    """
    best = None
    for sides in itertools.product(["front", "back"], repeat=len(models)):
        views = [StageView(name, m, side, upright_turn(m, side, up), res)
                 for (name, _, _, up), m, side in zip(stages, models, sides)]
        glyphs = [glyph(ch) for ch in text[1:]]
        cands = [fit_candidates(g, v.outline(canvas)) for g, v in zip(glyphs, views)]
        inks = [[ink_on_sheet(g, p, v) for _, p in c] for g, v, c in zip(glyphs, views, cands)]
        # A pixel both stages show is fine only if neither puts its character there.
        clash = {}
        for s, t in itertools.combinations(range(len(views)), 2):
            both = views[s].visible & views[t].visible
            clash[s, t] = np.array([[(both & (a | b)).sum() for b in inks[t]] for a in inks[s]]) / res ** 2
        area = [np.array([sc for sc, _ in c]) for c in cands]
        whole = [np.array([intact(i, v) for i in ink]) for ink, v in zip(inks, views)]
        for pick in itertools.product(*[range(len(c)) for c in cands]):
            score = sum(np.log(area[s][i]) - scatter * whole[s][i] for s, i in enumerate(pick)) \
                - weight * 100 * sum(clash[s, t][pick[s], pick[t]] for s, t in clash)
            if best is None or score > best[0]:
                best = (score, views, [cands[s][i][1] for s, i in enumerate(pick)], clash, pick, whole)
    _, views, places, clash, pick, whole = best
    # The flat character goes where no folded stage shows paper, as big as fits.
    taken = np.zeros((res, res), bool)
    for v in views:
        taken |= v.visible
    free = ~taken[:, ::-1]                       # printed side is the sheet turned over
    flat = fit_candidates(glyph(text[0]), np.asarray(Image.fromarray(free).resize((canvas, canvas))), n_keep=1)
    if not flat:
        flat = fit_candidates(glyph(text[0]), np.ones((canvas, canvas), bool), n_keep=1)
    report = {f"clash {views[s].name} / {views[t].name}": f"{clash[s, t][pick[s], pick[t]]:.2%} of the sheet"
              for s, t in clash}
    report.update({f"{v.name} character on its largest piece": f"{whole[s][pick[s]]:.0%}" for s, v in enumerate(views)})
    return views, [flat[0][1]] + places, report


def render(text, models, views, places, res):
    # The printed side at res x res, plus the stage views rebuilt at this resolution.
    hi = [StageView(v.name, m, v.view, v.turn, res) for v, m in zip(views, models)]
    sheet = np.full((res, res, 3), 255, np.uint8)
    V, U = (np.mgrid[0:res, 0:res] + 0.5) / res
    flat_ink = stamp(glyph(text[0]), *places[0], U, V)[:, ::-1]           # into raw coordinates
    sheet[flat_ink] = COLORS[0]
    for k, (ch, place, v) in enumerate(zip(text[1:], places[1:], hi), 1):
        sheet[v.visible] = 255                                               # later stages win clashes
        sheet[ink_on_sheet(glyph(ch), place, v)] = COLORS[k % len(COLORS)]
    return sheet[:, ::-1].copy(), hi                                         # as seen printed side up


def shared_paper(models, sides, res=160):
    """For each stage, the share of its visible paper that another stage also shows. Those
    pixels are painted once for several pictures, so they must suit all of them."""
    seen = []
    for m, side in zip(models, sides):
        _, side2, _, _ = m.sheet_visibility(res)
        seen.append(side2 == (1 if side == "front" else 2))
    return [float((s & np.logical_or.reduce([t for j, t in enumerate(seen) if j != i] or [np.zeros_like(s)])).sum()
                  / s.sum()) for i, s in enumerate(seen)]


def least_shared_sides(models, res=160):
    """Which side of each stage to show. Each stage can be held either way round; this picks the
    sides where the stage that shares the most of its paper shares as little as possible (ties go
    to the least sharing in total). Consecutive stages always share a lot, because folding keeps
    much of the outside on the outside."""
    options = list(itertools.product(["front", "back"], repeat=len(models)))
    return list(min(options, key=lambda sides: (max(sh := shared_paper(models, sides, res)), sum(sh))))


def stage_views(stages, models, sides, res, seed=0):
    """FoldViews for each stage at `res` pixels, held upright, as (FoldViews, side) pairs for
    illusion.generate."""
    return [(FoldViews(m, res, seed, turn={side: upright_turn(m, side, up)}), side)
            for m, side, (_, _, _, up) in zip(models, sides, stages)]


def story_images(sheet, stages, models, sides, res=450):
    """The printed side flat, then each stage folded from it and held upright, as PIL images. The
    folding uses decal.simulate, independently of the pixel maps used to make the sheet, so this
    doubles as a check."""
    from decal import simulate
    flat = Image.fromarray((np.clip(sheet, 0, 1) * 255).round().astype(np.uint8)).resize((1024, 1024), Image.LANCZOS)
    raw = np.asarray(flat, float)[:, ::-1] / 255         # printed side 2 is the sheet turned over
    images = [flat]
    for m, side, (_, _, _, up) in zip(models, sides, stages):
        sim, _ = simulate(m, {2: dict(color=raw, printed=np.ones(raw.shape[:2], bool))}, res=res)
        img = Image.fromarray((sim[side] * 255).round().astype(np.uint8))
        images.append(img.rotate(-upright_turn(m, side, up), Image.BICUBIC, expand=True, fillcolor=(217, 217, 217)))
    return images


def plot_story(sheet, stages, models, sides, titles, res=450):
    """The printed side flat, then each stage folded from it (see story_images)."""
    images = story_images(sheet, stages, models, sides, res)
    fig, axes = plt.subplots(1, len(images), figsize=(4.2 * len(images), 4.6))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=DEFAULT_STAGES, metavar="STAGE",
                    help="stages to fold through, from: " + ", ".join(f'"{st[0]}"' for st in STAGES)
                    + " (default: " + ", ".join(f'"{n}"' for n in DEFAULT_STAGES) + ")")
    ap.add_argument("--text", default=None, help="one character per stage: flat, then each folded stage (default 0, 1, 2, ...)")
    ap.add_argument("--scatter", type=float, default=2.0,
                    help="how much to prefer characters split across pieces of paper, so they don't read when flat")
    ap.add_argument("--res", type=int, default=2362, help="printed sheet size in pixels (2362 = 20 cm at 300 dpi)")
    args = ap.parse_args()
    stages, models = load_stages(args.stages)
    args.text = args.text or "0123456789"[:len(stages) + 1]
    if len(args.text) != len(stages) + 1:
        ap.error(f"--text needs {len(stages) + 1} characters")
    views, places, report = plan(args.text, stages, models, scatter=args.scatter)
    for v in views:
        print(f"{v.name}: {v.view} side, turned {v.turn:.0f} degrees")
    for k, c in report.items():
        print(f"{k}: {c}")
    printed, _ = render(args.text, models, views, places, args.res)
    Image.fromarray(printed).save("story_sheet.png", dpi=(args.res / SHEET_MM * 25.4,) * 2)
    titles = [f"flat: {args.text[0]}"] + [f"{v.name} ({v.view}): {ch}" for v, ch in zip(views, args.text[1:])]
    plot_story(printed.astype(float) / 255, stages, models, [v.view for v in views], titles).savefig("story_preview.png", dpi=80)
    print("wrote story_sheet.png and story_preview.png")


if __name__ == "__main__":
    main()
