"""A smiling Bayon face on the cube tessellation, painted one face at a time.

Head-on, the model shows the face in a photo: each of its 25 faces takes the edges of the part of the
photo it covers, and Stable Diffusion XL with a canny ControlNet paints that face from its own prompt.
Faces you like are kept on disk, so trying a new prompt on one face never touches the others.

The geometry (faces, camera, bounds, render) is the same as in cubism.ipynb. No GPU is needed here;
the notebook brings the painter.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------------------------------
# The model (as in cubism.ipynb). Cubes at lattice points of x + y + z = 0, seen head-on along (1, 1, 1),
# minus the hollow corner at the bottom, plus a wall on each side.
CUBES = [(a, b, -(a + b)) for a in range(3) for b in range(3)]
DROP = {(0, (3, 2, -4)), (1, (2, 3, -4)), (0, (3, 1, -3)), (1, (1, 3, -3))}   # (axis, corner)
WALLS = [(1, (2, 0, -1)), (0, (0, 2, -1))]                                  # (axis, corner)
AXES = ("x", "y", "z")
E = np.eye(3)
HEAD_ON = np.ones(3) / np.sqrt(3)
HEAD_ON_UP = np.array([-1.0, -1.0, 2.0]) / np.sqrt(6)
LIT = (0.8, 0.9, 1.0)


def _face(k, corner):
    i, j = [m for m in range(3) if m != k]
    return (k, np.array(corner, float), E[i], E[j])


def faces():
    """Every visible face: (axis it faces, corner, edge 1, edge 2)."""
    out = [_face(k, np.add(p, E[k])) for p in CUBES for k in range(3)]
    out = [f for f in out if (f[0], tuple(f[1].astype(int))) not in DROP]
    return out + [_face(k, c) for k, c in WALLS]


def camera(t=0.0, axis=2):
    """Toward-viewer direction and screen right and up, tilted a fraction t from head-on toward an axis."""
    a, b = HEAD_ON, E[axis]
    ang = np.arccos(a @ b)
    d = (np.sin((1 - t) * ang) * a + np.sin(t * ang) * b) / np.sin(ang)
    up = HEAD_ON_UP - (HEAD_ON_UP @ d) * d
    up /= np.linalg.norm(up)
    return d, np.cross(up, d), up


def bounds(cam, fs):
    _, r, u = cam
    pts = np.array([c + s * e1 + q * e2 for _, c, e1, e2 in fs for s in (0, 1) for q in (0, 1)])
    return (pts @ r).min(), (pts @ r).max(), (pts @ u).min(), (pts @ u).max()


def render(textures, t=0.0, axis=2, size=512, margin=0.06, bg=1.0, light=None):
    """The model seen at a tilt t toward an axis. textures: one (n, n, 3) array in [0, 1] per face (in
    faces() order), rows along its edge 2. light: a brightness per axis, e.g. LIT."""
    fs = faces()
    d, r, u = cam = camera(t, axis)
    x0, x1, y0, y1 = bounds(cam, fs)
    span = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    X, Y = np.meshgrid((x0 + x1) / 2 + ((np.arange(size) + 0.5) / size - 0.5) * span,
                       (y0 + y1) / 2 - ((np.arange(size) + 0.5) / size - 0.5) * span)
    out = np.full((size, size, 3), float(bg))
    depth = np.full((size, size), -np.inf)
    for i, (k, c, e1, e2) in enumerate(fs):
        if E[k] @ d < 1e-6:
            continue
        inv = np.linalg.inv(np.array([[e1 @ r, e2 @ r], [e1 @ u, e2 @ u]]))
        qx, qy = X - c @ r, Y - c @ u
        s, q = inv[0, 0] * qx + inv[0, 1] * qy, inv[1, 0] * qx + inv[1, 1] * qy
        z = (c + s[..., None] * e1 + q[..., None] * e2) @ d
        hit = (s >= 0) & (s <= 1) & (q >= 0) & (q <= 1) & (z > depth)
        n = textures[i].shape[0]
        si, qi = (s * n).astype(int).clip(0, n - 1), (q * n).astype(int).clip(0, n - 1)
        out[hit] = textures[i][qi[hit], si[hit]] * (1 if light is None else light[k])
        depth[hit] = z[hit]
    return out


def save_gif(textures, path, frames=14, size=360, hold=6, light=LIT):
    """Head-on, tilt to x and back, then y, then z."""
    ease = lambda s: (1 - np.cos(np.pi * s)) / 2
    imgs = []
    for k in range(3):
        ts = [ease(i / frames) for i in range(frames + 1)]
        seq = [0.0] * 2 + ts + [1.0] * hold + ts[::-1]
        imgs += [Image.fromarray((render(textures, t, k, size, light=light) * 255).astype(np.uint8)) for t in seq]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=60, loop=0)


# ---------------------------------------------------------------------------------------------------
# Numbering. Faces are numbered 1 to 25 in reading order head-on: top to bottom, then left to right.
def _head_on_centers():
    _, r, u = camera(0.0)
    return np.array([(c + (e1 + e2) / 2) @ np.stack([r, u], 1) for _, c, e1, e2 in faces()])


def _reading_order():
    ctr = _head_on_centers()
    rows = np.round(-ctr[:, 1] / 0.2) * 0.2              # group faces whose centers share a height
    return sorted(range(len(ctr)), key=lambda i: (rows[i], ctr[i, 0]))


FACE_INDEX = {n + 1: i for n, i in enumerate(_reading_order())}   # face number -> index in faces()
NUMBERS = list(FACE_INDEX)                                        # 1 .. 25


def axis_of(n):
    return AXES[faces()[FACE_INDEX[n]][0]]


# ---------------------------------------------------------------------------------------------------
# Each face's picture frame. A face is painted as an upright square: as you see it when the model is
# tilted to look straight along that face's axis, turned by a multiple of 90 degrees so its top is as
# close to up as it can be. Its right edge R and down edge D are cube edges, never mirrored.
def frame(n):
    """(origin corner, R, D) of face n's picture: pixel (col, row) of a size-S picture sits at
    origin + (col + 0.5) / S * R + (row + 0.5) / S * D."""
    k, c, e1, e2 = faces()[FACE_INDEX[n]]
    _, r, u = camera(1.0, k)
    best = None
    for R in (e1, -e1, e2, -e2):
        for D in (e1, -e1, e2, -e2):
            if abs(R @ D) > 0.5 or not np.allclose(np.cross(R, D), -E[k]):
                continue
            score = R @ r - D @ u
            if best is None or score > best[0]:
                best = (score, R, D)
    _, R, D = best
    origin = c + sum(e for e in (e1, e2) if (R @ e) < 0 or (D @ e) < 0)
    return origin, R, D


def to_texture(n, picture, res=256):
    """Face n's upright picture, resampled into render()'s layout for that face (rows along edge 2)."""
    pic = np.asarray(picture, float)
    pic = pic / 255 if pic.max() > 1 else pic
    if pic.ndim == 2:
        pic = np.repeat(pic[..., None], 3, 2)
    S = pic.shape[0]
    _, c, e1, e2 = faces()[FACE_INDEX[n]]
    origin, R, D = frame(n)
    g = (np.arange(res) + 0.5) / res
    P = c + g[None, :, None] * e1 + g[:, None, None] * e2 - origin
    col = ((P @ R) * S).astype(int).clip(0, S - 1)
    row = ((P @ D) * S).astype(int).clip(0, S - 1)
    return pic[row, col]


# ---------------------------------------------------------------------------------------------------
# Placing the photo. FIT says where the model sits in the photo, head-on: its center as fractions of
# the photo's width and height, and its width (the model's widest extent) as a fraction of the photo's.
DEFAULT_FIT = dict(x=0.5, y=0.5, width=0.9)


def _photo_map(photo_shape, fit):
    """Screen coordinates head-on -> photo pixels: returns a function of (sx, sy)."""
    H, W = photo_shape[:2]
    x0, x1, y0, y1 = bounds(camera(0.0), faces())
    scale = fit["width"] * W / (x1 - x0)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    return lambda sx, sy: (fit["x"] * W + (sx - mx) * scale, fit["y"] * H - (sy - my) * scale)


def face_photo(photo, n, fit=DEFAULT_FIT, size=1024, pad=0):
    """The part of the photo face n covers head-on, unsheared into its upright square. Parts of the
    face that fall outside the photo are black. pad: extra pixels of the surroundings on every side."""
    photo = np.asarray(photo)
    _, r, u = camera(0.0)
    to_px = _photo_map(photo.shape, fit)
    origin, R, D = frame(n)

    def px(P):
        return to_px(P @ r, P @ u)

    p0, pr, pd = (np.array(px(origin + a * R + b * D)) for a, b in ((0, 0), (1, 0), (0, 1)))
    # Picture pixel (col, row) -> photo pixel, as an affine map on pixel centers.
    A = np.stack([(pr - p0) / size, (pd - p0) / size], 1)
    t = p0 + A @ (np.array([0.5, 0.5]) - pad)
    M = np.hstack([A, t[:, None]]).astype(np.float32)
    return cv2.warpAffine(photo, M, (size + 2 * pad,) * 2, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------------------------------------------
# The guide: one line drawing of the whole photo, white lines on black, at the photo's size. Every face
# cuts its ControlNet guide from it, so lines run on across face borders. It's an ordinary image: save
# it, clean it up in any paint program (or draw your own), and load it back.
#
# method "lineart": a learned line detector (the lineart annotator from ControlNet), run on the photo
#   after edge-preserving smoothing. It draws like an artist: contours of the face, eyes, nose, lips and
#   crown, and far less of the stone's texture than canny. Needs PyTorch and controlnet_aux (Colab).
#   smooth: passes of edge-preserving blur first; more drops more texture.
#   resolution: the size the detector works at; lower gives simpler, bolder lines.
#   threshold: 0-255; higher keeps only the stronger lines.
# method "canny": plain canny, no downloads. On weathered stone it mostly finds texture.
#   blur (photo pixels), low, high: canny's settings.
# Both: min_area drops line fragments smaller than this many photo pixels.
DEFAULT_GUIDE = dict(method="lineart", smooth=4, resolution=384, threshold=80, min_area=40,
                     blur=2.5, low=40, high=110)
_detector = None


def make_guide(photo, guide=DEFAULT_GUIDE):
    """The photo's line drawing: a uint8 array the size of the photo, 255 on lines."""
    photo = np.asarray(photo)[..., :3]
    if guide["method"] == "lineart":
        global _detector
        if _detector is None:
            from controlnet_aux import LineartDetector
            _detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        smooth = photo.copy()
        for _ in range(guide["smooth"]):
            smooth = cv2.bilateralFilter(smooth, 9, 50, 7)
        lines = _detector(Image.fromarray(smooth), detect_resolution=guide["resolution"],
                          image_resolution=max(photo.shape[:2]))
        lines = cv2.resize(np.asarray(lines.convert("L")), (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_LINEAR)
        lines = (lines > guide["threshold"]).astype(np.uint8) * 255
    elif guide["method"] == "canny":
        gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        if guide["blur"] > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), guide["blur"])
        lines = cv2.Canny(gray, guide["low"], guide["high"])
    else:
        raise ValueError(f"unknown guide method {guide['method']!r}: use 'lineart' or 'canny'")
    return drop_specks(lines, guide["min_area"])


def drop_specks(lines, min_area):
    """Remove connected bits of line smaller than min_area pixels."""
    if min_area <= 0:
        return lines
    count, labels, stats, _ = cv2.connectedComponentsWithStats((lines > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def load_guide(path, photo_shape):
    """A saved or hand-edited guide, as white lines on black at the photo's size. Black lines on white
    (as you'd draw on paper) are flipped automatically."""
    g = np.asarray(Image.open(path).convert("L"))
    if g.shape != tuple(photo_shape[:2]):
        g = cv2.resize(g, (photo_shape[1], photo_shape[0]), interpolation=cv2.INTER_AREA)
    if g.mean() > 127:
        g = 255 - g
    return np.where(g > 127, 255, 0).astype(np.uint8)


# thickness: the line width the ControlNet sees, on a 1024-pixel face.
LINE_WIDTH = 6


def thin(lines):
    """Lines thinned to one pixel wide (Zhang and Suen's thinning), keeping their shape and joins."""
    img = (np.asarray(lines) > 0).astype(np.uint8)
    img = np.pad(img, 1)
    while True:
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(img, -dy, 0), -dx, 1) for dy, dx in
                 ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))]   # P2 .. P9
            count = sum(p)
            seq = p + [p[0]]
            turns = sum((seq[i] == 0) & (seq[i + 1] == 1) for i in range(8))
            if step == 0:
                c1, c2 = p[0] * p[2] * p[4], p[2] * p[4] * p[6]
            else:
                c1, c2 = p[0] * p[2] * p[6], p[0] * p[4] * p[6]
            drop = (img == 1) & (count >= 2) & (count <= 6) & (turns == 1) & (c1 == 0) & (c2 == 0)
            if drop.any():
                img[drop] = 0
                changed = True
        if not changed:
            break
    return (img[1:-1, 1:-1] * 255).astype(np.uint8)


def face_edges(guide, n, fit=DEFAULT_FIT, size=1024, width=None):
    """White lines on black: the ControlNet's guide for face n, cut from the whole guide and unsheared
    into the face's upright square. Lines are redrawn at the same width on every face, whatever the
    photo's resolution: width pixels on a 1024-pixel face (LINE_WIDTH by default)."""
    width = LINE_WIDTH if width is None else width
    # Cut out with a margin and thin before cropping, so lines run straight off the face's edges
    # instead of bending toward its corners.
    pad = size // 8
    e = face_photo(cv2.GaussianBlur(guide.astype(np.float32), (0, 0), 0.7), n, fit, size, pad)
    if size >= 400:
        e = cv2.GaussianBlur(e, (0, 0), size / 400)       # smooth the enlarged pixel steps
    e = thin(e > 40)[pad:pad + size, pad:pad + size]
    w = max(1, int(round(width * size / 1024)))
    return cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w))) if w > 1 else e


def head_on_photo(photo, fit=DEFAULT_FIT, size=768, margin=0.06):
    """What the model shows head-on if each face is printed with its piece of the photo."""
    return render([to_texture(n, face_photo(photo, n, fit, 256), 256) for n in _in_faces_order()],
                  size=size, margin=margin)


def _in_faces_order():
    by_index = {i: n for n, i in FACE_INDEX.items()}
    return [by_index[i] for i in range(len(faces()))]


# ---------------------------------------------------------------------------------------------------
# The numbered map: face outlines and numbers over the head-on view.
def face_polygons(size=768, margin=0.06, t=0.0, axis=2):
    """Face number -> its four corners in pixels of a render() of that size, margin and tilt."""
    cam = camera(t, axis)
    fs = faces()
    _, r, u = cam
    x0, x1, y0, y1 = bounds(cam, fs)
    span = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    out = {}
    for n, i in FACE_INDEX.items():
        _, c, e1, e2 = fs[i]
        pts = [c, c + e1, c + e1 + e2, c + e2]
        out[n] = [(((p @ r) - (x0 + x1) / 2) / span * size + size / 2,
                   ((y0 + y1) / 2 - (p @ u)) / span * size + size / 2) for p in pts]
    return out


def outlined_photo(photo, fit=DEFAULT_FIT, color=(255, 60, 60)):
    """The photo (or guide) with every face's outline and number drawn where it falls, head-on."""
    photo = np.asarray(photo)
    img = Image.fromarray(np.repeat(photo[..., None], 3, 2) if photo.ndim == 2 else photo[..., :3].copy())
    draw = ImageDraw.Draw(img)
    to_px, (_, r, u) = _photo_map(photo.shape, fit), camera(0.0)
    font = _font(max(12, photo.shape[1] // 30))
    for n, i in FACE_INDEX.items():
        _, c, e1, e2 = faces()[i]
        poly = [to_px(p @ r, p @ u) for p in (c, c + e1, c + e1 + e2, c + e2)]
        draw.polygon(poly, outline=color, width=max(2, photo.shape[1] // 300))
        draw.text(tuple(np.mean(poly, 0)), str(n), fill=(255, 230, 0), font=font, anchor="mm",
                  stroke_width=2, stroke_fill="black")
    return img


def _font(px):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    return ImageFont.load_default(px)


def numbered(image, size=768, margin=0.06, color=(255, 60, 60), labels=None):
    """The image (a head-on render of that size and margin) with face outlines and numbers on top.
    labels: optional text per face number, shown under its number."""
    img = Image.fromarray((np.asarray(image) * (255 if np.asarray(image).max() <= 1 else 1)).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    font, small = _font(size // 28), _font(size // 52)
    for n, poly in face_polygons(size, margin).items():
        draw.polygon(poly, outline=color, width=max(1, size // 400))
        cx, cy = np.mean(poly, 0)
        draw.text((cx, cy), str(n), fill=color, font=font, anchor="mm", stroke_width=2, stroke_fill="black")
        if labels and labels.get(n):
            draw.text((cx, cy + size / 30), labels[n], fill="white", font=small, anchor="mm",
                      stroke_width=2, stroke_fill="black")
    return img


# ---------------------------------------------------------------------------------------------------
# The store, a folder (on Google Drive in Colab, so it survives disconnects):
#
#   WORK/guide.png                         the line drawing every face is cut from; edit it freely
#   WORK/candidates/face_07/003.png .json  every painting of face 7, numbered, never overwritten,
#                                          with the prompt, seed and settings that made it
#   WORK/kept/face_07.png .json            the candidate you kept for face 7
#
# Painting only ever adds candidates, and keeping only copies one into kept/. So nothing you keep is
# repainted, and nothing you painted is lost.
class Store:
    def __init__(self, root):
        self.root = Path(root)
        for d in ("candidates", "kept"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    @property
    def guide_path(self):
        return self.root / "guide.png"

    def _cdir(self, n):
        return self.root / "candidates" / f"face_{n:02d}"

    def candidates(self, n):
        """Candidate numbers for face n, oldest first."""
        d = self._cdir(n)
        return sorted(int(p.stem) for p in d.glob("*.png")) if d.exists() else []

    def add_candidate(self, n, picture, info):
        """Save a new painting of face n; returns its candidate number."""
        _check(n)
        d = self._cdir(n)
        d.mkdir(parents=True, exist_ok=True)
        c = max(self.candidates(n), default=0) + 1
        picture.save(d / f"{c:03d}.png")
        info = dict(info, face=n, candidate=c, made=time.strftime("%Y-%m-%d %H:%M"))
        (d / f"{c:03d}.json").write_text(json.dumps(info, indent=1))
        return c

    def candidate(self, n, c):
        """(picture, info) of candidate c of face n."""
        d = self._cdir(n)
        if not (d / f"{c:03d}.png").exists():
            raise KeyError(f"face {n} has no candidate {c}; it has {self.candidates(n) or 'none'}")
        return Image.open(d / f"{c:03d}.png").convert("RGB"), json.loads((d / f"{c:03d}.json").read_text())

    def keep(self, n, c):
        """Keep candidate c as face n's painting, replacing any earlier choice."""
        pic, info = self.candidate(n, c)
        pic.save(self.root / "kept" / f"face_{n:02d}.png")
        (self.root / "kept" / f"face_{n:02d}.json").write_text(json.dumps(info, indent=1))
        return info

    def unkeep(self, n):
        """Face n goes back to unpainted. Its candidates stay."""
        for ext in ("png", "json"):
            (self.root / "kept" / f"face_{n:02d}.{ext}").unlink(missing_ok=True)

    def kept(self):
        """Face number -> (picture, info) for every kept face."""
        out = {}
        for p in sorted((self.root / "kept").glob("face_*.png")):
            out[int(p.stem.split("_")[1])] = (Image.open(p).convert("RGB"),
                                              json.loads(p.with_suffix(".json").read_text()))
        return out


def _check(n):
    if n not in FACE_INDEX:
        raise ValueError(f"faces are numbered 1 to {len(FACE_INDEX)}, not {n}")


def todo(store, faces_wanted="unkept"):
    """The faces to paint: "unkept", "all", or a list of face numbers."""
    if faces_wanted == "unkept":
        kept = store.kept()
        return [n for n in NUMBERS if n not in kept]
    if faces_wanted == "all":
        return list(NUMBERS)
    for n in faces_wanted:
        _check(n)
    return list(faces_wanted)


def board(store, guide, fit=DEFAULT_FIT, res=256):
    """One texture per face, in faces() order, for render(): the kept painting, or for a face not yet
    kept, its guide lines in gray, so you can see what's left."""
    kept = store.kept()
    out = []
    for n in _in_faces_order():
        if n in kept:
            out.append(to_texture(n, kept[n][0].resize((res, res), Image.LANCZOS), res))
        else:
            e = face_edges(guide, n, fit, res * 2)
            g = 0.3 + 0.45 * (cv2.resize(e, (res, res), interpolation=cv2.INTER_AREA) / 255.0)
            out.append(to_texture(n, g, res))
    return out


def contact_sheet(store, n, columns=4, thumb=256):
    """Every candidate of face n in a grid, labeled with its number and seed, the kept one outlined."""
    nums = store.candidates(n)
    if not nums:
        return None
    kept = store.kept().get(n, (None, {}))[1].get("candidate")
    rows = (len(nums) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    for k, c in enumerate(nums):
        pic, info = store.candidate(n, c)
        x, y = (k % columns) * thumb, (k // columns) * (thumb + 22)
        sheet.paste(pic.resize((thumb, thumb), Image.LANCZOS), (x, y + 22))
        mark = "  KEPT" if c == kept else ""
        draw.text((x + 4, y + 3), f"{c}  seed {info.get('seed')}{mark}", fill="black", font=font)
        if c == kept:
            draw.rectangle([x, y + 22, x + thumb - 1, y + 22 + thumb - 1], outline=(220, 30, 30), width=5)
    return sheet


# ---------------------------------------------------------------------------------------------------
# Painting (GPU). Stable Diffusion XL with a canny ControlNet; the canny ControlNet follows any white
# lines on black, so it takes the guide as is.
DEFAULT_PAINT = dict(control_scale=0.6, guidance=7.0, steps=30, size=1024)


def load_painter():
    import gc

    import torch
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline
    # Each piece goes to the GPU as soon as it loads, so Colab's RAM never holds them all at once.
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16).to("cuda")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
    gc.collect()
    painter = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae,
        torch_dtype=torch.float16, variant="fp16").to("cuda")
    painter.vae.enable_tiling()
    gc.collect()
    return painter


def guide_id(guide):
    """A short fingerprint of a guide, saved with each candidate, to tell which guide made it."""
    import hashlib
    return hashlib.md5(np.ascontiguousarray(guide).tobytes()).hexdigest()[:8]


def paint(painter, store, guide, n, prompt, seeds, style="", negative="", fit=DEFAULT_FIT,
          settings=DEFAULT_PAINT):
    """Paint face n once per seed, and save each as a new candidate. Returns the candidate numbers."""
    import torch
    _check(n)
    size = settings["size"]
    lines = face_edges(guide, n, fit, size)
    control = Image.fromarray(lines).convert("RGB")
    full = ", ".join(p for p in (prompt, style) if p)
    made = []
    for seed in seeds:
        pic = painter(full, negative_prompt=negative or None, image=control,
                      controlnet_conditioning_scale=settings["control_scale"], guidance_scale=settings["guidance"],
                      num_inference_steps=settings["steps"], width=size, height=size,
                      generator=torch.Generator("cuda").manual_seed(int(seed))).images[0]
        made.append(store.add_candidate(n, pic, dict(prompt=prompt, style=style, negative=negative, seed=int(seed),
                                                     settings=dict(settings), fit=dict(fit), guide=guide_id(guide))))
    return made
