"""Fold illusions: a sheet that shows one picture flat and another once folded.

The printed side of the sheet is a single image seen two ways: whole when the paper is
flat, and cut up and rearranged by the fold map when it's folded. Folding moves paper
rigidly, so each folded view is (almost exactly) a rearrangement of a subset of the
sheet's pixels. That makes it possible to generate both pictures at once with a pixel-
space diffusion model, as in Visual Anagrams (Geng, Park, Owens, CVPR 2024): at every
denoising step, show the model the sheet flat and folded, each with its own prompt, map
the predictions back onto the sheet, and combine them.

This module has two parts:
  - FoldViews: the flat view and the folded front/back views as pixel index maps (NumPy).
  - composite() and generate(): a no-AI baseline, and the DeepFloyd IF sampler (PyTorch).

Works for any flat-folded model that shows only one side of the paper (like the crane).
"""
import numpy as np

from fold_map import apply

VIEWS = ("flat", "front", "back")


class FoldViews:
    """Pixel maps between the printed side of an R x R sheet and the folded views.

    The sheet image is stored as you see the printed side when the paper lies flat, printed
    side up. Each folded view is an R x R canvas at the same scale (one sheet pixel per canvas
    pixel), centered on the folded model, oriented like Flat-Folder's display (front) or turned
    over left to right (back).

    For each folded view:
      paper[v]    (R, R) bool: canvas pixels covered by paper
      source[v]   (R, R) int: for paper pixels, the sheet pixel (flat index) shown there
      pairs[v]    (sheet pixels, canvas pixels): which canvas pixel each visible sheet pixel
                  takes its value from when going back to the sheet. Rotating a pixel grid
                  can't be exactly one-to-one, so a few canvas pixels show a sheet pixel that
                  another canvas pixel owns, and a few sheet pixels borrow a neighbor's value.
      unique[v]   fraction of paper pixels that own their sheet pixel
      visible[v]  (R, R) bool on the sheet: pixels shown in this view
    """

    def __init__(self, model, res, seed=0, turn=None):
        # turn: {view: degrees} to rotate a view's canvas clockwise, e.g. to hold a base point up.
        side1, side2, _, _ = model.sheet_visibility(res)       # each as seen from its own side
        if side1.any() and side2.any():
            raise NotImplementedError("this model shows both sides of the paper; only one-sided models are supported")
        self.printed_side = 1 if side1.any() else 2
        self.res = res
        self.model = model
        vis = side1 if self.printed_side == 1 else side2
        self.visible = {"front": vis == 1, "back": vis == 2}

        slo, shi = model.sheet.min(0), model.sheet.max(0)
        self.pixel = (shi - slo).max() / res
        self.sheet_lo = slo
        center = (model.folded.min(0) + model.folded.max(0)) / 2
        rng = np.random.default_rng(seed)
        self.paper, self.source, self.pairs, self.unique = {}, {}, {}, {}
        for view in ("front", "back"):
            # Folded-plane position of every canvas pixel center. The back view is mirrored.
            c = (np.arange(res) - res / 2 + 0.5) * self.pixel
            u, v = (a.ravel() for a in np.meshgrid(c, c))
            a = np.radians((turn or {}).get(view, 0))
            u, v = u * np.cos(a) + v * np.sin(a), -u * np.sin(a) + v * np.cos(a)
            Q = np.stack([center[0] + (-u if view == "back" else u), center[1] + v], -1)
            top, bottom, _ = model.stack_ends(model.cover(Q))
            face = bottom if view == "front" else top
            paper = face >= 0
            # Where each paper pixel comes from on the sheet, in printed-side pixel coordinates.
            target = np.full((len(Q), 2), np.nan)
            for f in np.unique(face[paper]):
                idx = face == f
                target[idx] = self._sheet_pixel(apply(model.to_sheet[f], Q[idx]))
            self.paper[view] = paper.reshape(res, res)
            source, self.pairs[view], self.unique[view] = self._assign(target, paper, self.visible[view].ravel(), rng)
            self.source[view] = source.reshape(res, res)

    def _sheet_pixel(self, P):
        # Fractional (row, col) of sheet points in the printed-side image.
        col = (P[:, 0] - self.sheet_lo[0]) / self.pixel - 0.5
        row = (P[:, 1] - self.sheet_lo[1]) / self.pixel - 0.5
        if self.printed_side == 2:                               # side 2 is the sheet turned over
            col = self.res - 1 - col
        return np.stack([row, col], 1)

    def _assign(self, target, paper, visible, rng):
        # Give each paper pixel its own visible sheet pixel near where it should come from, so the
        # view is as close to a pure rearrangement as possible. Paper pixels left without one reuse
        # the nearest sheet pixel; sheet pixels left without one take the nearest paper pixel's value.
        R = self.res
        source = np.zeros(R * R, int)
        primary = np.zeros(R * R, bool)
        used = np.zeros(R * R, bool)
        offsets = np.array([(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)])
        for k in rng.permutation(np.flatnonzero(paper)):
            cand = np.round(target[k]).astype(int) + offsets
            cand = cand[(cand >= 0).all(1) & (cand < R).all(1)]
            flat = cand[:, 0] * R + cand[:, 1]
            d = ((cand - target[k]) ** 2).sum(1)
            free = visible[flat] & ~used[flat]
            if free.any():
                best = flat[free][np.argmin(d[free])]
                used[best] = primary[k] = True
            else:
                best = flat[np.argmin(d)]
            source[k] = best
        owners = np.flatnonzero(primary)
        sheet_px, canvas_px = list(source[owners]), list(owners)
        paper_idx = np.flatnonzero(paper)
        for s in np.flatnonzero(visible & ~used):
            rc = np.array(divmod(s, R))
            canvas_px.append(paper_idx[np.argmin(((target[paper_idx] - rc) ** 2).sum(1))])
            sheet_px.append(s)
        return source, (np.array(sheet_px), np.array(canvas_px)), primary.sum() / max(paper.sum(), 1)

    def stats(self):
        return {v: dict(paper_pixels=int(self.paper[v].sum()), unique=round(float(self.unique[v]), 3),
                        visible_sheet_pixels_covered=float(np.isin(np.flatnonzero(self.visible[v]),
                                                                   self.pairs[v][0]).mean()))
                for v in ("front", "back")}

    def to_view(self, sheet, view, background):
        # sheet: (..., R, R) array; returns the view, with background where there's no paper.
        if view == "flat":
            return sheet
        flat = sheet.reshape(*sheet.shape[:-2], -1)
        out = flat[..., self.source[view].ravel()].reshape(sheet.shape)
        return np.where(self.paper[view], out, background)

    def corner_positions(self):
        # Where corners A-D (clockwise from Flat-Folder's top-left of the sheet) appear on the printed
        # side, as (row, col) pixel positions, for orienting the print.
        slo, shi = self.model.sheet.min(0), self.model.sheet.max(0)
        pts = np.array([[slo[0], slo[1]], [shi[0], slo[1]], [shi[0], shi[1]], [slo[0], shi[1]]])
        return np.clip(self._sheet_pixel(pts), 0, self.res - 1)


def composite(views, flat_picture, front_picture, back_picture):
    """No-AI baseline: the flat picture everywhere, except the visible regions, which carry the
    folded pictures (like a decal). All pictures are (R, R, 3) arrays in [0, 1], and the folded
    ones are drawn on the folded canvas. Returns the printed side as an (R, R, 3) array."""
    sheet = flat_picture.copy().reshape(-1, 3)
    for view, picture in (("front", front_picture), ("back", back_picture)):
        sheet_px, canvas_px = views.pairs[view]
        sheet[sheet_px] = picture.reshape(-1, 3)[canvas_px]
    return sheet.reshape(flat_picture.shape)


def canvas_picture(views, picture, view, background=0.85):
    """Place a picture on a folded view's canvas, center-cropped to cover the folded model's
    bounding box, with a plain background around it. picture: PIL image."""
    from PIL import Image
    from decal import fit
    rows, cols = np.nonzero(views.paper[view])
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    pic = fit(picture, (c1 - c0) / (r1 - r0))
    pic = np.asarray(Image.fromarray((pic * 255).astype(np.uint8)).resize((c1 - c0, r1 - r0), Image.LANCZOS), float) / 255
    out = np.full((views.res, views.res, 3), background)
    out[r0:r1, c0:c1] = pic
    return out


def square_picture(picture, res):
    from PIL import Image
    from decal import fit
    return np.asarray(Image.fromarray((fit(picture, 1.0) * 255).astype(np.uint8)).resize((res, res), Image.LANCZOS), float) / 255


def printable(sheet, pixels=2400):
    """The printed side as a PIL image at print size (2400 px = 8 inches at 300 dpi)."""
    from PIL import Image
    return Image.fromarray((np.clip(sheet, 0, 1) * 255).round().astype(np.uint8)).resize((pixels, pixels), Image.LANCZOS)


SHEET_MM = 200          # printed square: about as big as fits on A4 inside a home printer's margins


def print_page(sheet, path, sheet_mm=SHEET_MM, dpi=300, page_mm=(210, 297), top_mm=10):
    """An A4 PDF with the sheet at actual size, centered across the page, with a thin gray cut line
    around it. Print at 100% (not "fit to page"). Returns the sheet as printed (a PIL image)."""
    from PIL import Image, ImageDraw
    px = lambda mm: round(mm / 25.4 * dpi)
    img = printable(sheet, px(sheet_mm))
    page = Image.new("RGB", (px(page_mm[0]), px(page_mm[1])), "white")
    x0, y0 = (page.width - img.width) // 2, px(top_mm)
    page.paste(img, (x0, y0))
    ImageDraw.Draw(page).rectangle([x0 - 2, y0 - 2, x0 + img.width + 1, y0 + img.height + 1], outline=(170, 170, 170))
    page.save(path, resolution=dpi)
    return img


def plot_result(views, sheet, title="", sim_res=500):
    """The sheet flat (corners A-D marked), and the folded front and back simulated from it with
    decal.simulate, which folds the sheet independently of the views used to make it."""
    import matplotlib.pyplot as plt
    from decal import CORNERS, corners, simulate
    big = np.asarray(printable(sheet, 1024), float) / 255
    raw = big[:, ::-1] if views.printed_side == 2 else big
    sim, (lo, hi) = simulate(views.model, {views.printed_side: dict(color=raw, printed=np.ones(raw.shape[:2], bool))},
                             res=sim_res)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
    axes[0].imshow(np.clip(sheet, 0, 1))
    for letter, (r, c) in zip(CORNERS, views.corner_positions()):
        axes[0].annotate(letter, (c, r), color="#c22", fontsize=12, fontweight="bold", ha="center", va="center",
                         xytext=(-12 if c > views.res / 2 else 12, -12 if r > views.res / 2 else 12),
                         textcoords="offset points")
    axes[0].set_title("unfolded (printed side)")
    _, folded = corners(views.model)
    for ax, view in zip(axes[1:], ("front", "back")):
        turned = view == "back"
        x0, x1 = (hi[0], lo[0]) if turned else (lo[0], hi[0])
        ax.imshow(sim[view], extent=(x0, x1, hi[1], lo[1]))
        groups = {}
        for letter, p in zip(CORNERS, folded):
            groups.setdefault(tuple(np.round(p, 6)), []).append(letter)
        for p, letters in groups.items():
            ax.annotate("/".join(letters), p, color="#c22", fontsize=10, fontweight="bold", ha="center",
                        xytext=(0, 12), textcoords="offset points")
        ax.set_title(f"folded, {view}" + (" (Flip)" if turned else ""))
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# DeepFloyd IF sampler (needs PyTorch and a GPU)


def _entries(views):
    # Folded views as a list of (FoldViews, "front" or "back"); a single FoldViews means both its sides.
    return [(views, "front"), (views, "back")] if isinstance(views, FoldViews) else list(views)


def _torch_maps(views, device):
    import torch
    return [dict(source=torch.as_tensor(fv.source[v].ravel(), device=device),
                 paper=torch.as_tensor(fv.paper[v], device=device),
                 sheet_px=torch.as_tensor(fv.pairs[v][0], device=device),
                 canvas_px=torch.as_tensor(fv.pairs[v][1], device=device))
            for fv, v in _entries(views)]


def _gather(x, m, background):
    # x: (B, C, R, R) sheet -> folded view, with background (B, C, R, R) where there's no paper.
    B, C, R, _ = x.shape
    out = x.reshape(B, C, R * R)[:, :, m["source"]].reshape(B, C, R, R)
    return background.where(~m["paper"], out) if background is not None else out


def _combine(preds, maps, fold_weight, R):
    # preds: [flat prediction, then one per folded view], each (C, R, R) in its view's space. Map
    # the folded predictions back to the sheet and blend per pixel: hidden pixels follow the flat
    # view alone; a pixel seen in folded views gets weight fold_weight from each of them and
    # 1 - fold_weight from the flat view, normalized.
    flat = preds[0]
    C = flat.shape[0]
    total = (1 - fold_weight) * flat.reshape(C, -1).float()
    weight = total.new_full((R * R,), 1 - fold_weight)
    for pred, m in zip(preds[1:], maps):
        total.index_add_(1, m["sheet_px"], fold_weight * pred.reshape(C, -1)[:, m["canvas_px"]].float())
        weight.index_add_(0, m["sheet_px"], weight.new_full((len(m["sheet_px"]),), fold_weight))
    return (total / weight).to(flat.dtype).reshape(C, R, R)


def _step(unet, x, cond, t, embeds, maps, guidance, fold_weight, bg_t, class_labels=None):
    # One denoising step's combined prediction. x: (1, 3, R, R) sheet at timestep t; cond: optional
    # per-view list of (1, 3, R, R) upscaled stage-1 sheets (stage 2), each already noised to its
    # view's noise level in class_labels (one per view); bg_t: (1, 3, R, R) background at timestep t,
    # packed in a tuple with the per-view list of noised backgrounds for the conditioning.
    import torch
    R = x.shape[-1]
    bg_x, bg_c = bg_t
    inputs = []
    for k, m in enumerate([None] + maps):                         # the flat view, then each folded view
        xv = x if m is None else _gather(x, m, bg_x)
        if cond is not None:
            cv = cond[k] if m is None else _gather(cond[k], m, bg_c[k])
            xv = torch.cat([xv, cv], 1)
        inputs.append(xv)
    n = len(inputs)
    batch = torch.cat(inputs * 2)                                 # uncond for every view, then cond
    neg, pos = embeds
    kwargs = {} if class_labels is None else {"class_labels": class_labels.repeat(2)}
    out = unet(batch, t, encoder_hidden_states=torch.cat([neg, pos]), return_dict=False, **kwargs)[0]
    uncond, text = out.chunk(2)
    noise = uncond[:, :3] + guidance * (text[:, :3] - uncond[:, :3])
    pred = torch.cat([noise, text[:, 3:]], 1)                     # guided noise + predicted variance
    return _combine(list(pred), maps, fold_weight, R)[None]


def generate(stage_1, stage_2, views64, views256, embeds, guidance=(7.0, 4.0), steps=(50, 40),
             fold_weight=0.5, background=0.85, noise_level=250, seed=0, progress=None):
    """Sample the printed side of a fold illusion with DeepFloyd IF.

    stage_1, stage_2: IFPipeline and IFSuperResolutionPipeline. views64, views256: FoldViews for
    stage 1 (usually 64 pixels, the size IF's stage 1 was trained at; 128 gives small folded views
    four times the pixels) and stage 2 (256 pixels), for the front and back of one model, or
    matching lists of (FoldViews, "front" or "back") for any folded views, e.g. several stages of
    folding. embeds: (negative, positive) prompt embeddings for the flat view and then each folded
    view in order, each of shape (1 + number of folded views, tokens, dim). fold_weight: how much
    visible pixels follow the folded pictures (vs. the flat one). background: brightness of the
    plain backdrop the folded views are drawn on, in [0, 1]. noise_level: how much stage 2 may
    depart from stage 1 (0 to 1000; IF's default is 250), one number for every view or one per
    view (flat first). A high level on a folded view lets stage 2 redraw that picture at 256
    pixels instead of sharpening stage 1's. Returns (stage-1 sheet, final sheet) as (R, R, 3)
    arrays in [0, 1].
    """
    import torch
    import torch.nn.functional as F
    device = stage_1._execution_device
    dtype = stage_1.unet.dtype
    gen = torch.Generator(device).manual_seed(seed)
    neg, pos = (e.to(device, dtype) for e in embeds)

    def run(pipe, views, R, n_steps, scale, cond=None, class_labels=None):
        maps = _torch_maps(views, device)
        pipe.scheduler.set_timesteps(n_steps, device=device)
        x = torch.randn((1, 3, R, R), generator=gen, device=device, dtype=dtype) * pipe.scheduler.init_noise_sigma
        bg = torch.full((1, 3, R, R), background * 2 - 1, device=device, dtype=dtype)
        bg_noise = torch.randn((1, 3, R, R), generator=gen, device=device, dtype=dtype)
        bg_c = None
        if cond is not None:
            # The conditioning image and the backdrop around each folded view, noised to each view's level.
            noise = torch.randn_like(bg)
            bg_c = [pipe.image_noising_scheduler.add_noise(bg, noise, level.reshape(1)) for level in class_labels]
            cond = [pipe.image_noising_scheduler.add_noise(cond, noise, level.reshape(1)) for level in class_labels]
        for i, t in enumerate(pipe.scheduler.timesteps):
            # The backdrop is known, so it's re-noised to the current timestep, as in inpainting.
            bg_t = pipe.scheduler.add_noise(bg, bg_noise, t.reshape(1))
            pred = _step(pipe.unet, pipe.scheduler.scale_model_input(x, t), cond, t, (neg, pos), maps,
                         scale, fold_weight, (bg_t, bg_c), class_labels)
            x = pipe.scheduler.step(pred, t, x, generator=gen, return_dict=False)[0]
            if progress:
                progress(i, len(pipe.scheduler.timesteps))
        return x

    entries1, entries2 = _entries(views64), _entries(views256)
    R1, R2 = entries1[0][0].res, entries2[0][0].res
    levels = [noise_level] * (1 + len(entries2)) if np.isscalar(noise_level) else list(noise_level)
    if len(levels) != 1 + len(entries2):
        raise ValueError(f"noise_level needs one number, or one per view (flat first): {1 + len(entries2)}")
    with torch.no_grad():
        x1 = run(stage_1, entries1, R1, steps[0], guidance[0])
        # Stage 2 sees the stage-1 sheet upscaled and noised, like IFSuperResolutionPipeline.
        up = F.interpolate(x1.float(), (R2, R2), mode="bilinear", align_corners=True).to(dtype)
        x2 = run(stage_2, entries2, R2, steps[1], guidance[1], cond=up,
                 class_labels=torch.tensor(levels, device=device))

    to_img = lambda x: ((x[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    return to_img(x1), to_img(x2)
