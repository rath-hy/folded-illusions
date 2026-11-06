"""Fold maps from photos: fold a printed ChArUco board, photograph a view, and recover where every
visible point of the view came from on the sheet.

A ChArUco board is OpenCV's camera calibration target: a checkerboard with a small ArUco marker in
each white square. Each marker is a black-and-white code with its own number, so a marker found in
a photo says which part of the sheet that patch of paper is, and its four corners say how the
patch is turned and placed. Folding moves paper in rigid pieces, so:

  1. detect the markers in a photo of a folded view,
  2. group them into rigid pieces (markers on one piece agree on one transform, found by RANSAC),
  3. assign every pixel of paper in the photo to the piece whose transform best predicts the
     board's pattern there (local normalized cross-correlation).

The result is, for each photo pixel, the point of the printed sheet it shows: the same kind of map
fold_map.py computes from a crease pattern, but measured from real paper, for any stage of any
model, folded flat or not.

simulate_photo() folds the board virtually with a FoldedModel, adds blur, noise, uneven light and
shadows along the layer edges, and returns the true map alongside, to test the whole pipeline.

Coordinates: sheet points are in millimeters on the printed side as you look at it lying flat,
x to the right and y down from the top-left corner. Needs opencv-python 4.7 or later.
"""
from dataclasses import dataclass

import numpy as np
import cv2

from fold_map import apply
from illusion import SHEET_MM


# ---------------------------------------------------------------------------------------------
# The board


@dataclass
class Board:
    squares: int            # squares per side
    sheet_mm: float         # side of the printed square
    image: np.ndarray       # (R, R) uint8, the printed side, 0 black, 255 white
    corners: dict           # marker id -> (4, 2) corners on the sheet in mm
    dictionary: object
    charuco: object

    @property
    def px_mm(self):        # board image pixels per mm
        return self.image.shape[0] / self.sheet_mm

    @property
    def square_mm(self):
        return self.sheet_mm / self.squares


def make_board(squares=44, sheet_mm=SHEET_MM, px_per_square=40, marker_ratio=0.75):
    """A ChArUco board filling a square sheet. Half the squares hold a marker, and OpenCV's
    4x4 dictionary has 1,000 markers, so squares can go up to 44 per side: on 200 mm (the square
    the story notebooks print), 4.5 mm squares with 3.4 mm markers. More squares is better, because
    a marker cut by a crease or covered by another layer can't be read. (Folding scales with the
    paper, so what matters is squares per sheet, not their size in mm.)"""
    if squares * squares // 2 > 1000:
        raise ValueError("at most 44 squares per side: the dictionary has 1,000 markers")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    square = sheet_mm / squares
    charuco = cv2.aruco.CharucoBoard((squares, squares), square, marker_ratio * square, dictionary)
    size = squares * px_per_square
    image = charuco.generateImage((size, size), marginSize=0, borderBits=1)
    corners = {int(i): np.asarray(p, float)[:, :2] for i, p in zip(np.ravel(charuco.getIds()), charuco.getObjPoints())}
    return Board(squares, sheet_mm, image, corners, dictionary, charuco)


def save_board_pdf(board, path, page_mm=(210, 297), top_mm=10):
    """An A4 PDF with the board at actual size, centered across the page, a thin cut line around
    it, and a 100 mm scale bar to check the printer didn't rescale it. Print at 100% (not "fit to
    page")."""
    from PIL import Image, ImageDraw
    dpi = board.px_mm * 25.4
    W, H = (round(m * board.px_mm) for m in page_mm)
    page = Image.new("L", (W, H), 255)
    R = board.image.shape[0]
    x0, y0 = (W - R) // 2, round(top_mm * board.px_mm)
    page.paste(Image.fromarray(board.image), (x0, y0))
    draw = ImageDraw.Draw(page)
    draw.rectangle([x0 - 1, y0 - 1, x0 + R, y0 + R], outline=160)
    bar_y = y0 + R + round(10 * board.px_mm)
    draw.line([x0, bar_y, x0 + round(100 * board.px_mm), bar_y], fill=0, width=3)
    for x in (x0, x0 + round(100 * board.px_mm)):
        draw.line([x, bar_y - 12, x, bar_y + 12], fill=0, width=3)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=round(3 * board.px_mm))        # about 3 mm tall
    except TypeError:                                                   # Pillow before 10.1
        font = None
    draw.text((x0, bar_y + 18), f"this bar is 100 mm: print at 100%, not fit to page\n"
                                f"ChArUco board, {board.squares} x {board.squares} squares of "
                                f"{board.square_mm:.2f} mm, 4x4_1000 markers. Cut along the gray line.",
              fill=0, font=font, spacing=round(board.px_mm))
    page.save(path, resolution=dpi)


def sample_board(board, xy):
    """Board brightness in [0, 1] at sheet points xy (..., 2) in mm, bilinear; NaN off the sheet."""
    img = board.image.astype(np.float32) / 255
    s = board.px_mm
    mx = (xy[..., 0] * s - 0.5).astype(np.float32)
    my = (xy[..., 1] * s - 0.5).astype(np.float32)
    out = cv2.remap(img, np.nan_to_num(mx, nan=-10), np.nan_to_num(my, nan=-10), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    out[~np.isfinite(mx) | (xy[..., 0] < 0) | (xy[..., 1] < 0) | (xy[..., 0] > board.sheet_mm)
        | (xy[..., 1] > board.sheet_mm)] = np.nan
    return out


# ---------------------------------------------------------------------------------------------
# Simulated photos


BACKGROUND = (0.25, 0.45, 0.8)     # a plain colored backdrop: paper is black and white, so it separates easily


def simulate_photo(model, view, board, px_mm=15, printed_side=2, blur=1.0, noise=0.02, light=0.3,
                   shadow=0.5, background=BACKGROUND, seed=0):
    """A photo of the folded board, looking straight at one view ("front" or "back"), at px_mm
    pixels per mm. Adds blur (Gaussian sigma in pixels), noise, a light falloff across the frame
    (light: how much darker the dimmest corner is), and dark lines where one layer's edge sits on
    another (shadow: how dark). Returns (photo (H, W, 3) in [0, 1], truth), where truth holds
    sheet_mm (H, W, 2), the printed-sheet point each pixel shows (NaN off paper or on the
    unprinted side), and face (H, W), the crease-pattern face shown (-1 off paper)."""
    rng = np.random.default_rng(seed)
    lo, hi = model.folded.min(0), model.folded.max(0)
    mm = board.sheet_mm / (model.sheet.max(0) - model.sheet.min(0)).max()    # model units -> mm
    res = int(round((hi - lo).max() * 1.06 * mm * px_mm))
    views, _, _ = model.views(res)
    xy, side, face = views[view]
    slo, shi = model.sheet.min(0), model.sheet.max(0)
    t = (xy - slo) / (shi - slo)                         # 0..1 across the sheet, raw (side 1) orientation
    u = (1 - t[..., 0]) if printed_side == 2 else t[..., 0]   # side 2 is the sheet turned over
    sheet_mm = np.stack([u, t[..., 1]], -1) * board.sheet_mm
    printed = side == printed_side
    sheet_mm[~printed] = np.nan

    gray = np.where(printed, np.nan_to_num(sample_board(board, sheet_mm), nan=1.0), 1.0)
    paper = side > 0
    # Shadows along layer edges: wherever the visible face changes between neighbors.
    edge = np.zeros_like(paper)
    for axis in (0, 1):
        d = np.diff(face, axis=axis) != 0
        if axis == 0:
            edge[1:] |= d; edge[:-1] |= d
        else:
            edge[:, 1:] |= d; edge[:, :-1] |= d
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool) & paper
    gray = np.where(edge, gray * (1 - shadow), gray)

    rgb = np.where(paper[..., None], gray[..., None] * np.array([1.0, 0.98, 0.95]), np.array(background))
    H, W = paper.shape
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    a = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(a) * xx + np.sin(a) * yy)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    rgb = rgb * (1 - light * ramp)[..., None]
    if blur:
        rgb = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), blur)
    rgb = np.clip(rgb + rng.normal(0, noise, rgb.shape), 0, 1).astype(np.float32)
    return rgb, dict(sheet_mm=sheet_mm, face=np.where(paper, face, -1))


# ---------------------------------------------------------------------------------------------
# Recovering the map from a photo


def paper_mask(photo, min_saturation=0.3, min_value=0.15):
    """Paper pixels: the board is black and white and the backdrop is colored, so the backdrop is
    whatever is clearly colored (black ink has no reliable hue, so dark pixels count as paper)."""
    hsv = cv2.cvtColor((np.clip(photo, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    backdrop = (hsv[..., 1] > min_saturation * 255) & (hsv[..., 2] > min_value * 255)
    backdrop = cv2.morphologyEx(backdrop.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return ~backdrop.astype(bool)


def detect(photo, board):
    """Markers in a photo: {id: (4, 2) corners in photo pixels}."""
    gray = (np.clip(photo.mean(-1) if photo.ndim == 3 else photo, 0, 1) * 255).astype(np.uint8)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(board.dictionary, params).detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): np.asarray(c, float).reshape(4, 2) for i, c in zip(np.ravel(ids), corners)}


def _fit(src, dst, kind):
    if kind == "homography":
        M, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), 0)
        return M
    # Rotation, scale and shift, by least squares: x' = a x - b y + tx, y' = b x + a y + ty.
    x, y = src[:, 0], src[:, 1]
    one, zero = np.ones_like(x), np.zeros_like(x)
    A = np.concatenate([np.stack([x, -y, one, zero], 1), np.stack([y, x, zero, one], 1)])
    a, b, tx, ty = np.linalg.lstsq(A, np.concatenate([dst[:, 0], dst[:, 1]]), rcond=None)[0]
    return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1]])


def _project(M, P):
    q = np.c_[P, np.ones(len(P))] @ M.T
    return q[:, :2] / q[:, 2:]


def _px_per_mm(M, board, ids):
    # Photo pixels per sheet mm around a piece's markers. A homography's scale changes across the
    # sheet, so this measures it where the piece is.
    P = np.concatenate([board.corners[j] for j in ids])
    c = P.mean(0)
    a, b, d = _project(M, np.array([c, c + [1, 0], c + [0, 1]]))
    u, v = b - a, d - a
    return np.sqrt(abs(u[0] * v[1] - u[1] * v[0]))


def fit_pieces(markers, board, kind="similarity", tol=0.15):
    """Group detected markers into rigid pieces of paper. Repeatedly takes the transform (sheet mm
    -> photo pixels) of one marker that the most other markers agree with, to within tol marker
    sides, refits it to all of them, and sets them aside. kind: "similarity" (photo taken straight
    on) or "homography" (at an angle). Returns a list of (3x3 matrix, [marker ids]), largest first."""
    side_px = {i: np.linalg.norm(c[1] - c[0]) for i, c in markers.items()}
    # Skip markers whose own corners don't fit the transform (misread, or bent across a crease).
    left = {i: c for i, c in markers.items() if i in board.corners and
            np.abs(_project(_fit(board.corners[i], c, kind), board.corners[i]) - c).max() < tol * side_px[i]}
    pieces = []
    while left:
        ids = list(left)
        best = None
        for i in ids:
            M = _fit(board.corners[i], left[i], kind)
            agree = [j for j in ids
                     if np.abs(_project(M, board.corners[j]) - left[j]).max() < tol * side_px[j]]
            if best is None or len(agree) > len(best[1]):
                best = (M, agree)
        M, agree = best
        for _ in range(3):              # refit to everything that agrees, then look again
            M = _fit(np.concatenate([board.corners[j] for j in agree]), np.concatenate([left[j] for j in agree]), kind)
            agree = [j for j in ids
                     if np.abs(_project(M, board.corners[j]) - left[j]).max() < tol * side_px[j]] or agree
        pieces.append((M, agree))
        for j in agree:
            del left[j]
    return pieces


def _local_ncc(a, b, valid, win):
    # Normalized cross-correlation of a and b over a win x win window around each pixel, using only
    # pixels where valid.
    k = (win, win)
    w = cv2.boxFilter(valid.astype(np.float32), -1, k, normalize=False)
    a = np.where(valid, a, 0).astype(np.float32)
    b = np.where(valid, b, 0).astype(np.float32)
    s = lambda x: cv2.boxFilter(x, -1, k, normalize=False)
    n = np.maximum(w, 1)
    ma, mb = s(a) / n, s(b) / n
    cov = s(a * b) / n - ma * mb
    va = s(a * a) / n - ma * ma
    vb = s(b * b) / n - mb * mb
    ncc = cov / np.sqrt(np.maximum(va * vb, 1e-8))
    return np.where(w > 0.5 * win * win, ncc, -1)


def dense_map(photo, board, pieces, mask=None, window_mm=None, min_ncc=0.5):
    """For every paper pixel of the photo, the sheet point it shows: each piece's transform predicts
    the board's pattern across the photo, and each pixel goes to the piece whose prediction matches
    best nearby (normalized cross-correlation over a window about one checker square wide). Returns
    (sheet_mm (H, W, 2) with NaN where no piece matches, label (H, W) piece index or -1)."""
    gray = photo.mean(-1) if photo.ndim == 3 else photo
    H, W = gray.shape
    mask = paper_mask(photo) if mask is None else mask
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    best = np.full((H, W), -np.inf, np.float32)
    label = np.full((H, W), -1)
    sheet_mm = np.full((H, W, 2), np.nan)
    for k, (M, ids) in enumerate(pieces):
        win = int(round((window_mm or board.square_mm) * _px_per_mm(M, board, ids))) | 1
        xy = _project(np.linalg.inv(M), pts).reshape(H, W, 2)
        pred = sample_board(board, xy)
        valid = mask & np.isfinite(pred)
        pred = cv2.GaussianBlur(np.nan_to_num(pred, nan=0.5).astype(np.float32), (0, 0), 1.0)
        ncc = _local_ncc(gray, pred, valid, win)
        take = valid & (ncc > best) & (ncc > min_ncc)
        best[take] = ncc[take]
        label[take] = k
        sheet_mm[take] = xy[take]
    return sheet_mm, label


def recover(photo, board, kind="similarity"):
    """The whole pipeline: markers, pieces, and the dense map. Returns a dict with markers,
    pieces, sheet_mm and label."""
    markers = detect(photo, board)
    pieces = fit_pieces(markers, board, kind)
    sheet_mm, label = dense_map(photo, board, pieces)
    return dict(markers=markers, pieces=pieces, sheet_mm=sheet_mm, label=label)


def _seam_lines(points, tol, min_length, gap, rng, tries=300):
    # Straight runs of points, by sequential RANSAC. Each line must be supported along min_length,
    # and a stretch with no support for over gap ends it. Returns [(center, direction, t0, t1)].
    lines = []
    left = points.astype(np.float32)
    while len(left) >= min_length:
        best = None
        for _ in range(tries):
            a, b = left[rng.choice(len(left), 2, replace=False)]
            d = b - a
            n = np.hypot(*d)
            if n < tol:
                continue
            count = (np.abs((left - a) @ (np.array([-d[1], d[0]]) / n)) < tol).sum()
            if best is None or count > best[0]:
                best = (count, a, d / n)
        if best is None or best[0] < min_length:
            break
        _, a, d = best
        for _ in range(3):              # refit by total least squares
            near = np.abs((left - a) @ np.array([-d[1], d[0]])) < tol
            a = left[near].mean(0)
            d = np.linalg.svd(left[near] - a, full_matrices=False)[2][0]
        near = np.abs((left - a) @ np.array([-d[1], d[0]])) < tol
        t = (left[near] - a) @ d
        order = np.sort(t)
        cuts = np.nonzero(np.diff(order) > gap)[0]
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, len(order) - 1]
        for i, j in zip(starts, ends):
            if order[j] - order[i] >= min_length:
                lines.append((a, d, order[i], order[j]))
        left = left[~near]
    return lines


def straighten(result, photo, board, mask=None, check_squares=0.5, min_ncc=0.5, seed=0):
    """Straighten the seams of a recovered map. Pieces of flat-folded paper meet along straight
    lines, but the matching window straddles them, which makes seams ragged. This drops specks,
    fits lines to the boundary between each pair of neighboring pieces, redraws the pixels near
    each line to its correct side, and fills narrow gaps where the nearest piece's pattern matches
    the photo. It can't recover paper with no piece, like a tip with no whole marker.
    Returns a copy of result with new label and sheet_mm, plus seams: [(a, b, end1, end2)]."""
    rng = np.random.default_rng(seed)
    pieces = result["pieces"]
    if not pieces:
        return {**result, "seams": []}
    label = result["label"].copy()
    mask = paper_mask(photo) if mask is None else mask
    H, W = label.shape
    win = float(np.median([board.square_mm * _px_per_mm(M, board, ids) for M, ids in pieces]))

    # Specks: fragments of a piece smaller than about one square.
    for k in np.unique(label[label >= 0]):
        n, comps, stats, _ = cv2.connectedComponentsWithStats((label == k).astype(np.uint8), connectivity=8)
        for c in range(1, n):
            if stats[c, cv2.CC_STAT_AREA] < win * win:
                label[comps == c] = -1

    # Seam points: where two pieces meet, or face each other across a narrow gap.
    reach = 0.75 * win
    present = [int(k) for k in np.unique(label[label >= 0])]
    dist = {k: cv2.distanceTransform((label != k).astype(np.uint8), cv2.DIST_L2, 5) for k in present}
    seams = []
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            seam = (dist[a] < reach) & (dist[b] < reach) & (np.abs(dist[a] - dist[b]) <= 1.5) & mask
            if seam.sum() < win:
                continue
            rows, cols = np.nonzero(seam)
            for c, d, t0, t1 in _seam_lines(np.c_[cols, rows], 0.15 * win, win, 0.5 * win, rng):
                along = (xx - c[0]) * d[0] + (yy - c[1]) * d[1]
                across = -(xx - c[0]) * d[1] + (yy - c[1]) * d[0]
                band = (np.abs(across) < reach) & (along >= t0) & (along <= t1) & mask
                band &= (label == a) | (label == b)     # unmatched pixels wait for the checked fill below
                # Which side is which comes from the pixels just outside the band's middle.
                if (across[label == a].mean() if (label == a).any() else 0) >= (across[label == b].mean() if (label == b).any() else 0):
                    label[band & (across >= 0)], label[band & (across < 0)] = a, b
                else:
                    label[band & (across >= 0)], label[band & (across < 0)] = b, a
                seams.append((a, b, c + t0 * d, c + t1 * d))

    # Narrow gaps left over go to the nearest piece, if its predicted pattern matches the photo there.
    if (label >= 0).any():
        dist_any, nearest = cv2.distanceTransformWithLabels((label < 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                            labelType=cv2.DIST_LABEL_PIXEL)
        values = np.zeros(nearest.max() + 1, label.dtype)
        values[nearest[label >= 0]] = label[label >= 0]
        fill = (label < 0) & mask & (dist_any < 0.5 * win)
        gray = photo.mean(-1) if photo.ndim == 3 else photo
        pts = np.stack([xx.ravel(), yy.ravel()], 1)
        for k in present:
            region = fill & (values[nearest] == k)
            if not region.any():
                continue
            M, ids = pieces[k]
            xy = _project(np.linalg.inv(M), pts).reshape(H, W, 2)
            pred = sample_board(board, xy)
            valid = mask & np.isfinite(pred)
            pred = cv2.GaussianBlur(np.nan_to_num(pred, nan=0.5).astype(np.float32), (0, 0), 1.0)
            size = int(round(check_squares * board.square_mm * _px_per_mm(M, board, ids))) | 1
            ncc = _local_ncc(gray, pred, valid, size)
            label[region & (ncc > min_ncc)] = k

    sheet_mm = np.full((H, W, 2), np.nan)
    for k, (M, _) in enumerate(pieces):
        rows, cols = np.nonzero(label == k)
        if len(rows):
            sheet_mm[rows, cols] = _project(np.linalg.inv(M), np.c_[cols, rows].astype(float))
    return {**result, "label": label, "sheet_mm": sheet_mm, "seams": seams}


def evaluate(result, truth, tol_mm=0.5):
    """Compare a recovered map to the true one, over the paper that shows the printed side:
    the share of it recovered to within tol_mm, wrong by over 2 tol_mm, and not recovered."""
    shown = np.isfinite(truth["sheet_mm"][..., 0])
    got = np.isfinite(result["sheet_mm"][..., 0])
    err = np.linalg.norm(result["sheet_mm"] - truth["sheet_mm"], axis=-1)
    both = shown & got
    return dict(markers=len(result["markers"]), pieces=len(result["pieces"]),
                accurate=float((both & (err < tol_mm)).sum() / shown.sum()),
                wrong=float((both & (err >= 2 * tol_mm)).sum() / shown.sum()),
                missing=float((shown & ~got).sum() / shown.sum()),
                median_error_mm=float(np.median(err[both])) if both.any() else np.nan,
                error_mm=np.where(both, err, np.nan))


# ---------------------------------------------------------------------------------------------
# Putting a picture on a fold: mark an area in a photo, get the sheet to print
#
# The photo says what the viewer will see from that spot. So for each sheet point visible inside
# the marked area, print the picture's color at the place that point lands in the photo. Fold the
# print the same way, look from the same spot, and the picture lines up.


def load_photo(path):
    """A photo as RGB floats in [0, 1], turned upright by its EXIF orientation."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def marked_area(marked, min_share=0.0005):
    """The area marked in magenta on a copy of the photo, outlined or filled in, as a boolean
    mask. Several separate marks are fine, but an outline has to be closed."""
    hsv = cv2.cvtColor((np.clip(marked, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    ink = ((hsv[..., 0] > 135) & (hsv[..., 0] < 165) & (hsv[..., 1] > 90) & (hsv[..., 2] > 130)).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    outlines, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines = [c for c in outlines if cv2.contourArea(c) > min_share * ink.size]
    if not outlines:
        raise ValueError("no magenta area found in the marked photo: is the outline closed?")
    area = np.zeros(ink.shape, np.uint8)
    cv2.drawContours(area, outlines, -1, 1, cv2.FILLED)
    if cv2.distanceTransform(area, cv2.DIST_L2, 5).max() < 0.01 * max(ink.shape):
        raise ValueError("the magenta mark is just a thin line: close the outline, or fill the area in")
    return area.astype(bool)


def map_area(photo, board, area, kind="homography", min_markers=4, min_share=0.02):
    """recover() for a marked area of a photo. Pieces need min_markers markers, since one small
    marker's transform extrapolates badly. Stray matches under min_share of the area are dropped,
    and pixels that matched no piece (glare, curved edges) go to the nearest piece that did.
    Returns a dict like recover()'s, with label and sheet_mm set inside the area, plus matched:
    the share of the area that matched without that help."""
    markers = detect(photo, board)
    pieces = [p for p in fit_pieces(markers, board, kind) if len(p[1]) >= min_markers]
    if not pieces:
        raise ValueError(f"only {len(markers)} markers found, too few to form a piece of paper")
    _, label = dense_map(photo, board, pieces, mask=area)
    total = area.sum()
    for k in np.unique(label[label >= 0]):
        if (label == k).sum() < min_share * total:
            label[label == k] = -1
    matched = float((label[area] >= 0).mean())
    if matched == 0:
        raise ValueError("no piece of paper with enough markers lies inside the marked area")
    missing = area & (label < 0)
    if missing.any():                   # copy each missing pixel's label from its nearest labeled pixel
        _, nearest = cv2.distanceTransformWithLabels((label < 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
        values = np.zeros(nearest.max() + 1, label.dtype)
        values[nearest[label >= 0]] = label[label >= 0]
        label[missing] = values[nearest[missing]]
    H, W = label.shape
    sheet_mm = np.full((H, W, 2), np.nan)
    for k, (M, _) in enumerate(pieces):
        rows, cols = np.nonzero(label == k)
        if len(rows):
            sheet_mm[rows, cols] = _project(np.linalg.inv(M), np.c_[cols, rows].astype(float))
    return dict(markers=markers, pieces=pieces, sheet_mm=sheet_mm, label=label, area=area, matched=matched)


def place_picture(picture, area, fit="contain"):
    """The picture laid over the photo, upright in the frame and centered on the marked area.
    fit: "contain" shows the whole picture inside the area's bounding box, and "cover" fills the
    box, cropping what sticks out. Returns (image (H, W, 3), mask (H, W): where the picture is)."""
    x, y, w, h = cv2.boundingRect(area.astype(np.uint8))
    ph, pw = picture.shape[:2]
    scale = (min if fit == "contain" else max)(w / pw, h / ph)
    nw, nh = max(1, round(pw * scale)), max(1, round(ph * scale))
    small = cv2.resize(picture.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    small = np.clip(small, 0, 1)
    x0, y0 = x + (w - nw) // 2, y + (h - nh) // 2
    # Crop whatever falls outside the photo or, for "cover", the box.
    left, top = max(x - x0, 0), max(y - y0, 0)
    right, bottom = min(x + w - x0, nw), min(y + h - y0, nh)
    image = np.ones(area.shape + (3,), np.float32)
    mask = np.zeros(area.shape, bool)
    image[y0 + top:y0 + bottom, x0 + left:x0 + right] = small[top:bottom, left:right]
    mask[y0 + top:y0 + bottom, x0 + left:x0 + right] = True
    return image, mask


def project_picture(picture, mapped, board, fit="contain", dpi=300):
    """The sheet to print: for each sheet point visible in the marked area, the picture's color
    where that point lands in the photo. Returns (sheet (R, R, 3), printed (R, R): where it has
    ink). Everything else stays white."""
    overlay, inside = place_picture(picture, mapped["area"], fit)
    R = round(board.sheet_mm / 25.4 * dpi)
    u = (np.arange(R) + 0.5) * board.sheet_mm / R
    xx, yy = np.meshgrid(u, u)
    grid = np.stack([xx.ravel(), yy.ravel()], 1)
    sheet = np.ones((R, R, 3), np.float32)
    printed = np.zeros((R, R), bool)
    H, W = inside.shape
    for k, (M, ids) in enumerate(mapped["pieces"]):
        if not (mapped["label"] == k).any():
            continue
        q = np.c_[grid, np.ones(len(grid))] @ M.T
        # A homography also maps points from behind the camera into the frame: keep only those on
        # the same side as the piece's own markers.
        c = np.concatenate([board.corners[j] for j in ids]).mean(0)
        front = (q[:, 2] * (M[2] @ [c[0], c[1], 1])) > 0
        p = (q[:, :2] / np.where(front, q[:, 2:].T, 1).T).reshape(R, R, 2)
        col, row = np.rint(p[..., 0]).astype(int), np.rint(p[..., 1]).astype(int)
        ok = front.reshape(R, R) & (col >= 0) & (col < W) & (row >= 0) & (row < H)
        take = np.zeros((R, R), bool)
        take[ok] = (mapped["label"][row[ok], col[ok]] == k) & inside[row[ok], col[ok]]
        take &= ~printed
        colors = cv2.remap(overlay, p[..., 0].astype(np.float32), p[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        sheet[take] = colors[take]
        printed |= take
    return sheet, printed


def predicted_view(photo, sheet, mapped, board):
    """The photo with the marked area repainted from the printed sheet, folded by the recovered
    map: what you should see from that spot if the print is folded like the board."""
    out = photo.copy()
    s = sheet.shape[0] / board.sheet_mm
    xy = np.nan_to_num(mapped["sheet_mm"], nan=-10.0) * s - 0.5
    painted = cv2.remap(sheet, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
    out[mapped["area"]] = painted[mapped["area"]]
    return np.clip(out, 0, 1)


def save_print(sheet, path, board, dpi=300, page_mm=(210, 297), top_mm=10, note=None):
    """An A4 PDF of the sheet at the board's size and position. The sheet stays clean inside a dark
    cut line, and everything outside it is hatched, to throw away. Print at 100%. note: extra lines
    of text for the label."""
    from PIL import Image, ImageDraw, ImageFont
    px = lambda mm: round(mm / 25.4 * dpi)
    W, H = px(page_mm[0]), px(page_mm[1])
    R = px(board.sheet_mm)
    img = Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).resize((R, R), Image.LANCZOS)
    page = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(page)
    for d in range(-H, W, px(6)):
        draw.line([d, H, d + H, 0], fill=(200, 200, 200), width=2)
    x0, y0 = (W - R) // 2, px(top_mm)
    page.paste(img, (x0, y0))
    draw.rectangle([x0 - 3, y0 - 3, x0 + R + 2, y0 + R + 2], outline=(0, 0, 0), width=3)
    bx, by = x0, y0 + R + px(10)              # the label and scale bar, on a white box in the waste
    extra = f"\n{note}" if note else ""
    bottom = by + px(15 + 4.6 * extra.count("\n"))
    draw.rectangle([bx - px(3), by - px(4), bx + px(125), bottom], fill="white", outline=(120, 120, 120))
    draw.line([bx, by, bx + px(100), by], fill=0, width=3)
    for x in (bx, bx + px(100)):
        draw.line([x, by - 12, x, by + 12], fill=0, width=3)
    try:
        font = ImageFont.load_default(size=px(3))
    except TypeError:                         # Pillow before 10.1
        font = None
    draw.text((bx, by + px(3)), f"This bar is 100 mm: print at 100%, not fit to page.\n"
                                f"Cut along the black line. Keep the {board.sheet_mm:g} mm square, "
                                f"throw away the hatched paper.{extra}", fill=0, font=font, spacing=px(1.5))
    page.save(path, resolution=dpi)
    return page


def plot_projection(photo, mapped, sheet, board):
    """The marked area and markers found, the sheet to print, and the predicted view."""
    import matplotlib.pyplot as plt
    area = mapped["area"]
    x, y, w, h = cv2.boundingRect(area.astype(np.uint8))
    m = max(w, h) // 3
    zoom = (slice(max(y - m, 0), y + h + m), slice(max(x - m, 0), x + w + m))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    shown = photo.copy()
    shown[area] = 0.6 * shown[area] + 0.4 * np.array([1, 0, 1])
    axes[0].imshow(shown)
    for c in mapped["markers"].values():
        axes[0].add_patch(plt.Polygon(c, fill=False, ec="#f33", lw=0.5))
    axes[0].set_title(f"{len(mapped['markers'])} markers; {mapped['matched']:.0%} of the area matched directly")
    axes[1].imshow(sheet)
    axes[1].add_patch(plt.Rectangle((0, 0), sheet.shape[1] - 1, sheet.shape[0] - 1, fill=False, ec="k"))
    axes[1].set_title("the sheet to print, unfolded")
    axes[2].imshow(predicted_view(photo, sheet, mapped, board)[zoom])
    axes[2].set_title("predicted view from the camera")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# Pictures across the faces of a cube
#
# A cube has an advantage over a free-form fold: its faces are known squares, so nothing has to be
# marked by hand. Its visible faces are found from the markers alone, and their outline is the
# canvas. Real paper bows, dents and rounds over at its edges, so each face gets a homography plus a
# thin-plate spline, fitted to the markers and to the checker corners between them.


def fit_warp(P, Q, smoothing=None, step_mm=0.25):
    """A map from sheet points P (mm) to photo points Q (px): a homography, plus, unless smoothing is
    None, a thin-plate spline through what the homography misses. A spline can follow a local dent,
    and smoothing sets how closely it chases individual points. Beyond the outermost measured points
    the correction stays frozen at its last value. It's stored on a fine grid, so applying it to
    millions of pixels stays fast."""
    H = _fit(P, Q, "homography")
    warp = dict(H=H, grid=None)
    if smoothing is not None:
        from scipy.interpolate import RBFInterpolator
        spline = RBFInterpolator(P, Q - _project(H, P), kernel="thin_plate_spline", smoothing=smoothing, degree=1)
        lo, hi = P.min(0), P.max(0)
        n = np.ceil((hi - lo) / step_mm).astype(int) + 1
        xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], n[0]), np.linspace(lo[1], hi[1], n[1]))
        warp.update(lo=lo, hi=hi, grid=spline(np.c_[xx.ravel(), yy.ravel()]).reshape(n[1], n[0], 2))
    return warp


def apply_warp(warp, xy):
    q = _project(warp["H"], xy)
    if warp["grid"] is None:
        return q
    grid = warp["grid"]                 # bilinear lookup, with points clipped to the measured area
    t = (np.clip(xy, warp["lo"], warp["hi"]) - warp["lo"]) / (warp["hi"] - warp["lo"]) * (np.array(grid.shape[1::-1]) - 1)
    i = np.minimum(t.astype(int), np.array(grid.shape[1::-1]) - 2)
    f = t - i
    fx, fy = f[:, 0:1], f[:, 1:2]
    return q + ((1 - fy) * ((1 - fx) * grid[i[:, 1], i[:, 0]] + fx * grid[i[:, 1], i[:, 0] + 1])
                + fy * ((1 - fx) * grid[i[:, 1] + 1, i[:, 0]] + fx * grid[i[:, 1] + 1, i[:, 0] + 1]))


def holdout_px(P, Q, smoothing, folds=10, seed=0):
    """How well a warp predicts points it wasn't fitted to: rms and largest error in photo pixels,
    over random folds."""
    order = np.random.default_rng(seed).permutation(len(P))
    errors = []
    for f in range(folds):
        test = order[f::folds]
        train = np.setdiff1d(order, test)
        errors.append(np.linalg.norm(apply_warp(fit_warp(P[train], Q[train], smoothing), P[test]) - Q[test], axis=1))
    errors = np.concatenate(errors)
    return float(np.sqrt((errors ** 2).mean())), float(errors.max())


def rim_px(P, Q, box, smoothing, rim_mm=None):
    """Harder than holdout_px: fit to the middle of a face and predict its rim. rms and largest error."""
    x0, y0, side = box
    rim_mm = side / 11 if rim_mm is None else rim_mm
    near = np.minimum.reduce([P[:, 0] - x0, P[:, 1] - y0, x0 + side - P[:, 0], y0 + side - P[:, 1]]) < rim_mm
    errors = np.linalg.norm(apply_warp(fit_warp(P[~near], Q[~near], smoothing), P[near]) - Q[near], axis=1)
    return float(np.sqrt((errors ** 2).mean())), float(errors.max())


def find_faces(markers, board, n=3, tol=0.8):
    """The n visible flat faces, from the markers alone. A generous tol keeps a bowed face together
    as one piece. Each face's markers are snapped outward to the board's squares, and the result has
    to be square. Returns {"top" | "left" | "right": dict(box=(x0, y0, side), ids=[...])}, named by
    where each face is in the photo."""
    pieces = sorted(fit_pieces(markers, board, "homography", tol), key=lambda p: -len(p[1]))[:n]
    if len(pieces) < n or len(pieces[-1][1]) < 8:
        raise ValueError(f"expected {n} faces, found pieces of {[len(p[1]) for p in pieces]} markers")
    s = board.square_mm
    faces = []
    for _, ids in pieces:
        P = np.concatenate([board.corners[j] for j in ids])
        lo, hi = np.floor(P.min(0) / s + 1e-6), np.ceil(P.max(0) / s - 1e-6)
        if hi[0] - lo[0] != hi[1] - lo[1]:
            raise ValueError(f"a face spans {hi - lo} squares, so it isn't square: is a face partly hidden?")
        faces.append(dict(box=tuple(round(float(v), 6) for v in (lo[0] * s, lo[1] * s, (hi[0] - lo[0]) * s))))
    for f in faces:                         # every marker inside the square, even if RANSAC set it aside
        x0, y0, side = f["box"]
        f["ids"] = [j for j in markers if j in board.corners and x0 < board.corners[j][:, 0].mean() < x0 + side
                    and y0 < board.corners[j][:, 1].mean() < y0 + side]
        P = np.concatenate([board.corners[j] for j in f["ids"]])
        Q = np.concatenate([markers[j] for j in f["ids"]])
        f["center"] = _project(_fit(P, Q, "homography"), np.array([[x0 + side / 2, y0 + side / 2]]))[0]
    faces.sort(key=lambda f: f["center"][1])  # highest in the photo is the top; the other two by x
    top, others = faces[0], sorted(faces[1:], key=lambda f: f["center"][0])
    return dict(top=top, left=others[0], right=others[1])


def checker_corners(photo, board, box, warp, reach_px=14):
    """The checkerboard corners inside a face, located in the photo near where warp predicts them.
    These need no readable code, so they cover markers the detector missed. Returns (sheet mm, photo
    px, share found). Corners that moved more than reach_px are dropped as unreliable."""
    x0, y0, side = box
    s = board.square_mm
    n = int(round(side / s))
    xx, yy = np.meshgrid(x0 + s * np.arange(1, n), y0 + s * np.arange(1, n))
    P = np.c_[xx.ravel(), yy.ravel()]
    q = apply_warp(warp, P).astype(np.float32)
    gray = (np.clip(photo.mean(-1), 0, 1) * 255).astype(np.uint8)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    found = cv2.cornerSubPix(gray, q.reshape(-1, 1, 2).copy(), (13, 13), (-1, -1), crit).reshape(-1, 2)
    keep = np.linalg.norm(found - q, axis=1) < reach_px
    return P[keep], found[keep].astype(float), float(keep.mean())


def fit_faces(photo, markers, board, faces, smoothing=30, rounds=2):
    """Adds to each face its measured points P (mm) and Q (px) and its warp. Markers first, then the
    checker corners found with that fit, then a fit to both."""
    for f in faces.values():
        Pm = np.concatenate([board.corners[j] for j in f["ids"]])
        Qm = np.concatenate([markers[j] for j in f["ids"]])
        warp = fit_warp(Pm, Qm, 300)    # stiffer until the corners add data
        for r in range(rounds):
            Pc, Qc, share = checker_corners(photo, board, f["box"], warp, reach_px=14 - 3 * r)
            f.update(P=np.r_[Pm, Pc], Q=np.r_[Qm, Qc], Pm=Pm, Qm=Qm, corners_found=share)
            warp = fit_warp(f["P"], f["Q"], smoothing)
        f["warp"] = warp
    return faces


def face_outline(face, samples=60):
    """A face's square outline in the photo, as an (N, 2) array."""
    x0, y0, side = face["box"]
    t = np.linspace(0, side, samples, endpoint=False)
    o = np.zeros_like(t)
    square = np.r_[np.c_[x0 + t, y0 + o], np.c_[x0 + side + o, y0 + t],
                   np.c_[x0 + side - t, y0 + side + o], np.c_[x0 + o, y0 + side - t]]
    return apply_warp(face["warp"], square)


def rectified_face(photo, face, px_mm=10):
    """What a face looks like in the photo, flattened back to a square by its warp. Returns
    (image, sheet points (n, n, 2) in mm)."""
    x0, y0, side = face["box"]
    n = int(round(side * px_mm))
    u = (np.arange(n) + 0.5) / px_mm
    xx, yy = np.meshgrid(x0 + u, y0 + u)
    q = apply_warp(face["warp"], np.c_[xx.ravel(), yy.ravel()]).reshape(n, n, 2).astype(np.float32)
    return cv2.remap(photo, q[..., 0], q[..., 1], cv2.INTER_LINEAR), np.stack([xx, yy], -1)


def picture_frame(picture, faces, fit="cover"):
    """Where the picture sits in the photo: upright and centered on the faces' bounding box. "cover"
    scales it to fill the box. Returns (left, top, photo pixels per picture pixel)."""
    all_points = np.concatenate([face_outline(f) for f in faces.values()])
    lo, hi = all_points.min(0), all_points.max(0)
    ph, pw = picture.shape[:2]
    scale = (max if fit == "cover" else min)((hi[0] - lo[0]) / pw, (hi[1] - lo[1]) / ph)
    center = (lo + hi) / 2
    return center[0] - pw * scale / 2, center[1] - ph * scale / 2, scale


def project_onto_faces(picture, faces, board, fit="cover", dpi=300, bleed_mm=4):
    """The sheet to print: each point of a face's square, plus bleed_mm of the paper around it, gets
    the picture's color at the place it lands in the photo. Past a face's edge, the paper goes around
    a corner out of sight, and the bleed there covers small folding errors with color instead of
    white. Returns (sheet (R, R, 3), printed (R, R), frame)."""
    R = round(board.sheet_mm / 25.4 * dpi)
    u = (np.arange(R) + 0.5) * board.sheet_mm / R
    xx, yy = np.meshgrid(u, u)
    names = list(faces)
    dist = []
    for name in names:
        x0, y0, side = faces[name]["box"]
        dx = np.maximum(np.maximum(x0 - xx, xx - x0 - side), 0)
        dy = np.maximum(np.maximum(y0 - yy, yy - y0 - side), 0)
        dist.append(np.hypot(dx, dy))
    dist = np.stack(dist)
    label = dist.argmin(0)
    label[dist.min(0) > bleed_mm] = -1
    left, top, scale = picture_frame(picture, faces, fit)
    mx, my = np.full((R, R), -1, np.float32), np.full((R, R), -1, np.float32)
    for k, name in enumerate(names):
        rows, cols = np.nonzero(label == k)
        q = apply_warp(faces[name]["warp"], np.c_[xx[rows, cols], yy[rows, cols]])
        mx[rows, cols] = (q[:, 0] - left) / scale - 0.5
        my[rows, cols] = (q[:, 1] - top) / scale - 0.5
    colors = cv2.remap(picture.astype(np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    sheet = np.ones((R, R, 3), np.float32)
    printed = label >= 0
    sheet[printed] = np.clip(colors[printed], 0, 1)
    return sheet, printed, (left, top, scale)


def add_fold_ticks(sheet, faces, board, length_mm=3, shade=0.55):
    """Short gray ticks on the sheet's rim, where the faces' edge lines would reach it, as a guide
    for folding. The rim isn't on any of the three visible faces."""
    out = sheet.copy()
    s = out.shape[0] / board.sheet_mm
    n = round(length_mm * s)
    xs, ys = set(), set()
    for f in faces.values():
        x0, y0, side = f["box"]
        xs |= {x0, x0 + side}
        ys |= {y0, y0 + side}
    for x in xs:
        c = round(x * s)
        out[:n, c - 1:c + 2] = shade
        out[-n:, c - 1:c + 2] = shade
    for y in ys:
        c = round(y * s)
        out[c - 1:c + 2, :n] = shade
        out[c - 1:c + 2, -n:] = shade
    return out


def white_levels(photo, faces, percentile=92):
    """How bright white paper is on each face, from the photo. Ink can only darken paper, so these
    differences show in the result as a different shade on each face."""
    return {name: float(np.percentile(rectified_face(photo, f, px_mm=8)[0].mean(-1), percentile))
            for name, f in faces.items()}


def faces_view(photo, sheet, faces, board, shade=True):
    """The photo with the cube repainted from the printed sheet, folded by the measured maps: the
    predicted view from the camera. Pixels in the narrow grooves between faces go to the nearest
    face. shade: keep each face's measured brightness, as real paper would. Returns (view, cube mask)."""
    names = list(faces)
    H, W = photo.shape[:2]
    outlines = {n: face_outline(faces[n]).astype(np.int32) for n in names}
    label = np.full((H, W), -1, np.int32)
    for k, n in enumerate(names):
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [outlines[n]], 1)
        label[(mask > 0) & (label < 0)] = k
    hull = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.concatenate(list(outlines.values()))), 1)
    hull = cv2.dilate(hull, np.ones((17, 17), np.uint8))     # the rounded edges pass the ideal outline a little
    missing = (hull > 0) & (label < 0)
    if missing.any():
        _, nearest = cv2.distanceTransformWithLabels((label < 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
        values = np.zeros(nearest.max() + 1, label.dtype)
        values[nearest[label >= 0]] = label[label >= 0]
        label[missing] = values[nearest[missing]]
    out = photo.copy()
    levels = white_levels(photo, faces) if shade else {n: 1.0 for n in names}
    s = sheet.shape[0] / board.sheet_mm
    for k, n in enumerate(names):
        rows, cols = np.nonzero(label == k)
        p = np.c_[cols, rows].astype(float)
        warp = faces[n]["warp"]
        Hi = np.linalg.inv(warp["H"])
        target = _project(Hi, p)
        xy = target.copy()
        for _ in range(6):              # undo the correction by fixed-point iteration
            xy = xy + target - _project(Hi, apply_warp(warp, xy))
        mx, my = np.full((H, W), -10, np.float32), np.full((H, W), -10, np.float32)
        mx[rows, cols], my[rows, cols] = xy[:, 0] * s - 0.5, xy[:, 1] * s - 0.5
        painted = cv2.remap(sheet, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
        out[rows, cols] = painted[rows, cols] * levels[n]
    return np.clip(out, 0, 1), hull > 0


def _cube_points(faces):
    # Each face's sheet -> 3D map. The corner the three faces share is the origin; the cube lies in
    # the negative octant, with z up. Needs the top face to share a sheet edge with left and right.
    top, left, right = faces["top"]["box"], faces["left"]["box"], faces["right"]["box"]
    corners = lambda b: {(b[0], b[1]), (b[0] + b[2], b[1]), (b[0], b[1] + b[2]), (b[0] + b[2], b[1] + b[2])}
    shared = corners(top) & corners(left) & corners(right)
    if len(shared) != 1:
        raise ValueError("the three faces don't meet at one point of the sheet")
    c = np.array(shared.pop(), float)
    center = lambda b: np.array([b[0] + b[2] / 2, b[1] + b[2] / 2]) - c
    ax, ay = np.array([np.sign(center(top)[0]), 0.0]), np.array([0.0, np.sign(center(top)[1])])
    # The top's edge shared with left runs along the axis where both lie on the same side.
    to_left, to_right = (ax, ay) if center(left) @ ax > 0 else (ay, ax)
    other = lambda v, along: np.sign(v - (v @ along) * along)   # a face's edge that isn't shared with the top
    A, B, D = np.array([0, -1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, -1.0])
    maps = dict(top=(to_left, A, to_right, B), left=(to_left, A, other(center(left), to_left), D),
                right=(to_right, B, other(center(right), to_right), D))
    return c, top[2], maps


def cube_camera(faces, focal_35mm, shape):
    """Where the camera was, from the measured points and an ideal rigid cube. focal_35mm: the lens's
    35 mm equivalent focal length, from the photo's EXIF. Returns a dict with K, position (mm from
    the shared corner, z up, cube below and behind it), distance, height above the cube's base, rms
    (px, what the ideal cube misses of the real paper), and tilt: each face's angle away from the line
    of sight."""
    c, side, maps = _cube_points(faces)
    H, W = shape[:2]
    f = focal_35mm / np.hypot(36, 24) * np.hypot(W, H)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    obj, img = [], []
    for name, face in faces.items():
        a, A, b, B = maps[name]
        d = face["P"] - c
        obj.append((d @ a)[:, None] * A + (d @ b)[:, None] * B)
        img.append(face["Q"])
    obj, img = np.concatenate(obj).astype(np.float32), np.concatenate(img).astype(np.float32)
    _, rvec, tvec = cv2.solvePnP(obj, img, K, None)
    q = cv2.projectPoints(obj, rvec, tvec, K, None)[0][:, 0]
    R_, _ = cv2.Rodrigues(rvec)
    position = (-R_.T @ tvec).ravel()
    d = position / np.linalg.norm(position)
    normals = dict(top=[0, 0, 1.0], left=[1.0, 0, 0], right=[0, 1.0, 0])
    return dict(K=K, position=position, distance=float(np.linalg.norm(position)),
                height=float(position[2] + side), rms=float(np.sqrt(((q - img) ** 2).sum(1).mean())),
                tilt={n: float(np.degrees(np.arccos(np.dot(v, d)))) for n, v in normals.items()})


def render_cube(sheet, faces, board, position, K, size=700, zoom=1.0, background=0.82):
    """The folded print as a perfect cube, seen by a pinhole camera at position, aimed at the cube's
    center. It uses none of the photo's bowing, so it's a second opinion on the sheet, and it can show
    the view from places the photo wasn't taken."""
    position = np.asarray(position, float)
    c, side, maps = _cube_points(faces)
    target = np.full(3, -side / 2.0)
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R_ = np.stack([right, down, forward])
    t = -R_ @ position
    f = K[0, 0] * size / (2 * K[0, 2]) * zoom
    Kv = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    out = np.full((size, size, 3), background, np.float32)
    s = sheet.shape[0] / board.sheet_mm
    normals = dict(top=[0, 0, 1.0], left=[1.0, 0, 0], right=[0, 1.0, 0])
    for name, face in faces.items():
        if np.dot(normals[name], position) <= 0:
            continue                    # facing away from this camera
        x0, y0, sd = face["box"]
        sq = np.array([[x0, y0], [x0 + sd, y0], [x0 + sd, y0 + sd], [x0, y0 + sd]])
        a, A, b, B = maps[name]
        pts3 = ((sq - c) @ a)[:, None] * A + ((sq - c) @ b)[:, None] * B
        px = (Kv @ ((R_ @ pts3.T).T + t).T).T
        px = px[:, :2] / px[:, 2:]
        H = cv2.getPerspectiveTransform((sq * s - 0.5).astype(np.float32), px.astype(np.float32))
        layer = cv2.warpPerspective(sheet, H, (size, size), flags=cv2.INTER_LINEAR)
        mask = np.zeros((size, size), np.uint8)
        cv2.fillConvexPoly(mask, px.astype(np.int32), 1)
        out[mask > 0] = layer[mask > 0]
    return out


# ---------------------------------------------------------------------------------------------
# Pictures


def piece_colors(n):
    import matplotlib
    return matplotlib.colormaps["tab20"](np.arange(n) % 20)[:, :3]


def on_sheet(result, board, res=600):
    """Where the recovered view comes from on the printed sheet: the board faded, with each piece's
    paper tinted in its color. This is the visibility map you'd otherwise shade by hand."""
    img = cv2.resize(board.image, (res, res), interpolation=cv2.INTER_AREA).astype(float) / 255
    out = np.repeat((0.75 + 0.25 * img)[..., None], 3, -1)
    lab = result["label"]
    xy = result["sheet_mm"][lab >= 0] / board.sheet_mm * res
    rc = np.clip(xy[:, ::-1].astype(int), 0, res - 1)
    colors = piece_colors(len(result["pieces"]))[lab[lab >= 0]]
    tint = np.zeros((res, res, 3)); count = np.zeros((res, res))
    np.add.at(tint, (rc[:, 0], rc[:, 1]), colors)
    np.add.at(count, (rc[:, 0], rc[:, 1]), 1)
    seen = count > 0
    out[seen] = 0.35 * out[seen] * img[seen][:, None] + 0.65 * tint[seen] / count[seen][:, None]
    return out


def plot_recovery(photo, result, board, truth=None):
    """The photo with detected markers, the pieces found, where they come from on the sheet, and
    (with the truth from simulate_photo) the error."""
    import matplotlib.pyplot as plt
    n = 4 if truth is not None else 3
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5))
    axes[0].imshow(np.clip(photo, 0, 1))
    for c in result["markers"].values():
        axes[0].add_patch(plt.Polygon(c, fill=False, ec="#f33", lw=0.8))
    axes[0].set_title(f"photo: {len(result['markers'])} markers found")
    lab = result["label"]
    colors = piece_colors(len(result["pieces"]))
    pieces = np.ones(lab.shape + (3,))
    pieces[lab >= 0] = colors[lab[lab >= 0]]
    axes[1].imshow(pieces)
    axes[1].set_title(f"{len(result['pieces'])} rigid pieces")
    axes[2].imshow(on_sheet(result, board))
    axes[2].set_title("where they are on the printed sheet")
    if truth is not None:
        ev = evaluate(result, truth)
        im = axes[3].imshow(ev["error_mm"], vmin=0, vmax=2, cmap="magma")
        axes[3].imshow(np.where(np.isfinite(truth["sheet_mm"][..., 0]) & ~np.isfinite(result["sheet_mm"][..., 0]),
                                1.0, np.nan), cmap="cool", vmin=0, vmax=1)
        axes[3].set_title(f"error (mm), missing in cyan\n{ev['accurate']:.1%} within 0.5 mm, "
                          f"{ev['wrong']:.1%} off by over 1 mm")
        fig.colorbar(im, ax=axes[3], fraction=0.035)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
