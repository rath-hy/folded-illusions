"""A smiling Bayon face on the cube tessellation, painted one face at a time.

Head-on, the model shows the face in a study: a flat geometric painting made of polygons (an SVG). Each
of its 25 faces takes the part of the study it covers, and Stable Diffusion XL paints it from its own
prompt, starting from the study's shades and following its edges. Finishing then pins each polygon's
lightness close to the study's and draws the study's edges on top, thick and black, so the face reads
whatever colors each painting uses. Faces you like are kept on disk, so trying a new prompt on one face
never touches the others.

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
# Placing a picture. FIT says where the model sits on a picture (the study, or any image), head-on: its
# center as fractions of the picture's width and height, and its width (the model's widest extent) as a
# fraction of the picture's.
DEFAULT_FIT = dict(x=0.5, y=0.5, width=1.0)


def _photo_map(photo_shape, fit):
    """Screen coordinates head-on -> picture pixels: returns a function of (sx, sy)."""
    H, W = photo_shape[:2]
    x0, x1, y0, y1 = bounds(camera(0.0), faces())
    scale = fit["width"] * W / (x1 - x0)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    return lambda sx, sy: (fit["x"] * W + (sx - mx) * scale, fit["y"] * H - (sy - my) * scale)


def _face_affine(photo_shape, n, fit, size):
    """(p0, A): face n's upright picture pixel (col, row) sits at picture point p0 + A @ (col + 0.5, row + 0.5)."""
    _, r, u = camera(0.0)
    to_px = _photo_map(photo_shape, fit)
    origin, R, D = frame(n)
    p0, pr, pd = (np.array(to_px(P @ r, P @ u)) for P in (origin, origin + R, origin + D))
    return p0, np.stack([(pr - p0) / size, (pd - p0) / size], 1)


def face_photo(photo, n, fit=DEFAULT_FIT, size=1024, pad=0):
    """The part of a picture face n covers head-on, unsheared into its upright square. Parts of the face
    that fall outside the picture are black. pad: extra pixels of the surroundings on every side."""
    photo = np.asarray(photo)
    p0, A = _face_affine(photo.shape, n, fit, size)
    t = p0 + A @ (np.array([0.5, 0.5]) - pad)
    M = np.hstack([A, t[:, None]]).astype(np.float32)
    return cv2.warpAffine(photo, M, (size + 2 * pad,) * 2, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def head_on_photo(photo, fit=DEFAULT_FIT, size=768, margin=0.06):
    """What the model shows head-on if each face is printed with its piece of a picture."""
    return render([to_texture(n, face_photo(photo, n, fit, 256), 256) for n in _in_faces_order()],
                  size=size, margin=margin)


def _in_faces_order():
    by_index = {i: n for n, i in FACE_INDEX.items()}
    return [by_index[i] for i in range(len(faces()))]


# ---------------------------------------------------------------------------------------------------
# The study: a flat geometric painting of the face (an SVG of filled polygons), which sets the whole
# composition. It gives every face two things, drawn straight from the polygons at full resolution:
#
#   edges   every polygon's outline and every line in the SVG, all one color and one width. The painter
#           follows them, and they're drawn again on top of the finished face, so the face stays legible
#           whatever gets painted inside.
#   shades  each polygon (a "chunk") has a target lightness, from its fill. The painter starts from them,
#           and finishing moves each chunk's average lightness toward its target while keeping the
#           painting's own colors and texture. A chunk can be any color, as long as it's that light.
#
# Fills are measured as CIELAB lightness L* (0 black to 100 white), and the study's darkest and lightest
# fills are stretched to SHADES, so the darkest chunks are dark but still colored, and black edges stand
# out against them.
SHADES = (15.0, 92.0)
STUDY_CROP = dict(left=1 / 8, right=1 / 8, top=0.0, bottom=1 / 8)


class Study:
    """The polygons of a flat SVG painting, cropped. Coordinates are SVG units with the crop's top-left
    at (0, 0). chunks: (points, (r, g, b)) in painting order. lines: (points, closed)."""

    def __init__(self, width, height, chunks, lines, source=""):
        self.width, self.height = width, height
        self.chunks, self.lines = chunks, lines
        self.source = source

    @property
    def shape(self):
        return (self.height, self.width)

    def lightness(self, shades=SHADES):
        """Target L* of each chunk, the study's range stretched to shades."""
        L = np.array([_lightness(c) for _, c in self.chunks])
        lo, hi = L.min(), L.max()
        return shades[0] + (L - lo) / max(hi - lo, 1e-6) * (shades[1] - shades[0])

    def edges(self):
        """Every outline and line, as (points, closed)."""
        return [(p, True) for p, _ in self.chunks] + self.lines

    def id(self):
        import hashlib
        return hashlib.md5(self.source.encode()).hexdigest()[:8]


def _lightness(rgb):
    px = np.array([[rgb]], np.float32) / 255
    return float(cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0, 0])


def _points(s):
    v = [float(x) for x in s.replace(",", " ").split()]
    return np.array(v).reshape(-1, 2)


def load_study(path, crop=STUDY_CROP):
    """Read the study's SVG: its filled polygons and rectangles become chunks, its lines and polylines
    become extra edges. Hatching patterns, filters and translucent overlays are ignored. crop: fractions
    of the SVG's width or height to cut from each side."""
    import re
    text = Path(path).read_text()
    shapes = re.sub(r"<defs\b.*?</defs>", "", text, flags=re.S)          # pattern tiles aren't shapes
    vb = re.search(r'viewBox="([^"]+)"', text)
    vx, vy, vw, vh = (float(x) for x in vb.group(1).split()) if vb else (0, 0, float(re.search(
        r'width="([\d.]+)', text).group(1)), float(re.search(r'height="([\d.]+)', text).group(1)))
    x0, y0 = vx + crop["left"] * vw, vy + crop["top"] * vh
    W, H = vw * (1 - crop["left"] - crop["right"]), vh * (1 - crop["top"] - crop["bottom"])
    shift = np.array([x0, y0])

    def attr(tag, name):
        m = re.search(rf'\b{name}="([^"]*)"', tag)
        return m.group(1) if m else None

    chunks, lines = [], []
    for tag in re.findall(r"<(?:rect|polygon|polyline|line)\b[^>]*>", shapes):
        kind = re.match(r"<(\w+)", tag).group(1)
        fill = attr(tag, "fill") or ""
        if kind in ("rect", "polygon"):
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", fill) or attr(tag, "fill-opacity") or attr(tag, "filter"):
                continue
            if kind == "rect":
                x, y, w, h = (float(attr(tag, k) or 0) for k in ("x", "y", "width", "height"))
                pts = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            else:
                pts = _points(attr(tag, "points"))
            chunks.append((pts - shift, tuple(int(fill[i:i + 2], 16) for i in (1, 3, 5))))
        elif kind == "polyline":
            lines.append((_points(attr(tag, "points")) - shift, False))
        else:
            pts = np.array([[float(attr(tag, k)) for k in ("x1", "y1")], [float(attr(tag, k)) for k in ("x2", "y2")]])
            lines.append((pts - shift, False))
    if not chunks:
        raise ValueError(f"{path}: no filled polygons found")
    return Study(W, H, chunks, lines, source=text + repr(sorted(crop.items())))


def _face_points(study, n, fit, size):
    """A function taking study points to face n's picture pixels (continuous, pixel centers at +0.5)."""
    p0, A = _face_affine(study.shape, n, fit, size)
    Ainv = np.linalg.inv(A)
    return lambda P: (np.asarray(P, float) - p0) @ Ainv.T


_SUB = 4   # subpixel bits for drawing


def _fx(P):
    return np.round(P * (1 << _SUB)).astype(np.int32)


def face_chunks(study, n, fit=DEFAULT_FIT, size=1024):
    """Which chunk covers each pixel of face n's picture (int32; -1 outside the study)."""
    to_face = _face_points(study, n, fit, size)
    ids = np.full((size, size), -1, np.int32)
    for k, (pts, _) in enumerate(study.chunks):
        cv2.fillPoly(ids, [_fx(to_face(pts))], k, lineType=cv2.LINE_8, shift=_SUB)
    return ids


def _edge_px(width, size):
    return max(1, int(round(width * size / 1024)))


def face_lines(study, n, fit=DEFAULT_FIT, size=1024, width=24):
    """Face n's edges, white on black (uint8, antialiased): width pixels on a 1024-pixel face."""
    to_face = _face_points(study, n, fit, size)
    out = np.zeros((size, size), np.uint8)
    w = _edge_px(width, size)
    for pts, closed in study.edges():
        cv2.polylines(out, [_fx(to_face(pts))], closed, 255, w, cv2.LINE_AA, shift=_SUB)
    return out


def _gray(L):
    """L* -> sRGB gray in [0, 1]."""
    L = np.asarray(L, np.float32)
    lab = np.zeros(np.atleast_2d(L).shape + (3,), np.float32)
    lab[..., 0] = np.atleast_2d(L)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1).reshape(L.shape + (3,))


def face_shades(study, n, fit=DEFAULT_FIT, size=1024, shades=SHADES):
    """Face n's target lightness as a gray picture in [0, 1] (black outside the study)."""
    ids = face_chunks(study, n, fit, size)
    L = np.where(ids >= 0, study.lightness(shades)[ids], 0.0)
    return _gray(L)


# How finished faces look: how far each chunk's lightness moves to its target (0 none, 1 all the way),
# the edges' width on a 1024-pixel face and their color, and the lightness range.
DEFAULT_FINISH = dict(shade=0.8, edge_width=24, edge_color=(0, 0, 0), shades=SHADES)


def finish(picture, study, n, fit=DEFAULT_FIT, settings=DEFAULT_FINISH):
    """A painting of face n, finished: each chunk's lightness moved toward its target, its texture and
    colors kept, and the study's edges drawn on top. Returns an RGB uint8 array."""
    pic = np.asarray(Image.fromarray(np.asarray(picture)).convert("RGB") if not isinstance(picture, Image.Image)
                     else picture.convert("RGB"), np.float32) / 255
    S = pic.shape[0]
    ids = face_chunks(study, n, fit, S)
    lines = face_lines(study, n, fit, S, settings["edge_width"]).astype(np.float32) / 255
    target = study.lightness(settings["shades"])
    lab = cv2.cvtColor(pic, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    out = L.copy()
    visible = lines < 0.5
    for k in np.unique(ids):
        if k < 0:
            continue
        m = ids == k
        seen = m & visible if (m & visible).any() else m
        mean = L[seen].mean()
        goal = mean + settings["shade"] * (target[k] - mean)
        dev = L[m] - mean
        # Keep the texture, squeezed only as much as it takes to fit between black and white.
        spread = np.percentile(np.abs(dev), 95)
        room = min(goal, 100 - goal)
        out[m] = goal + dev * (min(1.0, room / spread) if spread > 1e-6 else 1.0)
    lab[..., 0] = out
    rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    color = np.array(settings["edge_color"], np.float32) / 255
    rgb = rgb * (1 - lines[..., None]) + color * lines[..., None]
    return (rgb * 255).round().astype(np.uint8)


def study_face(study, n, fit=DEFAULT_FIT, size=1024, settings=DEFAULT_FINISH):
    """Face n as the study itself: its target shades in gray, with edges. What an unpainted face shows."""
    gray = (face_shades(study, n, fit, size, settings["shades"]) * 255).astype(np.uint8)
    return finish(gray, study, n, fit, dict(settings, shade=1.0))


def draw_study(study, scale=2.0, settings=DEFAULT_FINISH, fit=DEFAULT_FIT, original=False):
    """The whole study as a picture (scale pixels per SVG unit): target shades in gray with edges, or with
    original=True its own fill colors. Edges are as wide as they'll be on the faces."""
    W, H = int(round(study.width * scale)), int(round(study.height * scale))
    target = study.lightness(settings["shades"])
    img = np.zeros((H, W, 3), np.uint8)
    for k, (pts, color) in enumerate(study.chunks):
        c = color if original else tuple(int(v) for v in (_gray(target[k]) * 255).round())
        cv2.fillPoly(img, [_fx(pts * scale)], c, cv2.LINE_AA, shift=_SUB)
    _, A = _face_affine((H, W), 13, fit, 1024)                # pixels per face pixel, for the edge width
    w = max(1, int(round(settings["edge_width"] * np.linalg.norm(A[:, 0]))))
    for pts, closed in study.edges():
        cv2.polylines(img, [_fx(pts * scale)], closed, tuple(int(v) for v in settings["edge_color"]), w,
                      cv2.LINE_AA, shift=_SUB)
    return img


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
    """A picture (the study, or a photo) with every face's outline and number drawn where it falls, head-on."""
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
#   WORK/candidates/face_07/003.png .json  every painting of face 7, numbered, never overwritten,
#                                          with the prompt, seed and settings that made it
#   WORK/kept/face_07.png .json            the candidate you kept for face 7
#   WORK/finished/face_07.png              face 7 finished: shades matched, edges drawn
#
# Candidates are saved raw, before finishing, so finishing settings can change without repainting.
# Painting only ever adds candidates, and keeping only copies one into kept/. So nothing you keep is
# repainted, and nothing you painted is lost.
class Store:
    def __init__(self, root):
        self.root = Path(root)
        for d in ("candidates", "kept"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

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


def board(store, study, fit=DEFAULT_FIT, res=256, settings=DEFAULT_FINISH):
    """One texture per face, in faces() order, for render(): the kept painting, finished, or for a face
    not yet kept, the study itself (gray shades and edges), so you can see what's left."""
    kept = store.kept()
    out = []
    for n in _in_faces_order():
        if n in kept:
            pic = finish(kept[n][0].resize((res * 2, res * 2), Image.LANCZOS), study, n, fit, settings)
        else:
            pic = study_face(study, n, fit, res * 2, settings)
        out.append(to_texture(n, cv2.resize(pic, (res, res), interpolation=cv2.INTER_AREA), res))
    return out


def save_finished(store, study, folder, fit=DEFAULT_FIT, settings=DEFAULT_FINISH):
    """Every kept face, finished at full size, saved as folder/face_NN.png. Returns the face numbers."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for n, (pic, _) in store.kept().items():
        Image.fromarray(finish(pic, study, n, fit, settings)).save(folder / f"face_{n:02d}.png")
    return sorted(store.kept())


def contact_sheet(store, n, columns=4, thumb=256, finished=None):
    """Every candidate of face n in a grid, labeled with its number and seed, the kept one outlined.
    finished: optional function picture -> finished picture, to show candidates as they'd end up."""
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
        if finished is not None:
            pic = Image.fromarray(finished(pic))
        x, y = (k % columns) * thumb, (k // columns) * (thumb + 22)
        sheet.paste(pic.resize((thumb, thumb), Image.LANCZOS), (x, y + 22))
        mark = "  KEPT" if c == kept else ""
        draw.text((x + 4, y + 3), f"{c}  seed {info.get('seed')}{mark}", fill="black", font=font)
        if c == kept:
            draw.rectangle([x, y + 22, x + thumb - 1, y + 22 + thumb - 1], outline=(220, 30, 30), width=5)
    return sheet


# ---------------------------------------------------------------------------------------------------
# Painting (GPU). Stable Diffusion XL with a canny ControlNet, starting from the face's target shades
# (img2img) and following its edges. The ControlNet sees the edges thin (control_width on a 1024-pixel
# face), as canny would draw them; the thick edges are drawn on afterwards by finish().
#   strength: how far the painter may move from the shades it starts from (1 ignores them).
DEFAULT_PAINT = dict(control_scale=0.6, guidance=7.0, steps=30, size=1024, strength=0.8, control_width=4)


def load_painter():
    import gc

    import torch
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
    # Each piece goes to the GPU as soon as it loads, so Colab's RAM never holds them all at once.
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16).to("cuda")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
    gc.collect()
    painter = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae,
        torch_dtype=torch.float16, variant="fp16").to("cuda")
    painter.vae.enable_tiling()
    gc.collect()
    return painter


def paint(painter, store, study, n, prompt, seeds, style="", negative="", fit=DEFAULT_FIT,
          settings=DEFAULT_PAINT, shades=SHADES):
    """Paint face n once per seed, and save each raw painting as a new candidate (finishing happens when
    it's shown, so it can change later). Returns the candidate numbers."""
    import torch
    _check(n)
    size = settings["size"]
    control = Image.fromarray(face_lines(study, n, fit, size, settings["control_width"])).convert("RGB")
    start = Image.fromarray((face_shades(study, n, fit, size, shades) * 255).astype(np.uint8))
    full = ", ".join(p for p in (prompt, style) if p)
    made = []
    for seed in seeds:
        pic = painter(full, negative_prompt=negative or None, image=start, control_image=control,
                      strength=settings["strength"], controlnet_conditioning_scale=settings["control_scale"],
                      guidance_scale=settings["guidance"], num_inference_steps=settings["steps"],
                      width=size, height=size, generator=torch.Generator("cuda").manual_seed(int(seed))).images[0]
        made.append(store.add_candidate(n, pic, dict(prompt=prompt, style=style, negative=negative, seed=int(seed),
                                                     settings=dict(settings), fit=dict(fit), study=study.id())))
    return made
