"""Ambiguous fold art: pictures that read one way on the flat sheet, and resolve into something else
once folding cuts a piece out of its context.

The goal is no longer to hide the folded pictures. A wing's paper may look like anything on the
flat sheet, as long as it reads as part of the scene there, and as the wing's subject once folded.
Dalí's double images work this way: context decides what a shape is.

On the crane, each folded view is a mosaic of shards from all over the sheet. Each wing is an
upright triangle joined from two shards that lie along two different edges of the flat sheet,
touching only at a corner tip, each rotated differently by the fold. So folding does three things to a piece of the picture at once: it cuts
the piece out of its surroundings, turns it, and joins it to another piece.

Every method here is a set of views of the sheet, each with its own prompt, sampled together as in
MultiDiffusion (Bar-Tal et al., ICML 2023): at every denoising step, each view's noise prediction is
mapped back onto the sheet and the predictions are averaged per pixel, weighted. Optionally, as in
Factorized Diffusion (Geng, Park, Owens, ECCV 2024), some views control only the low frequencies and
others only the high ones. The methods (see METHODS):

  average     the baseline: the flat view and both folded views, equally weighted, which is the
              blend illusion.generate uses with fold_weight 0.5
  context     a local anagram inside a global scene: the baseline, plus windows tiled over the sheet
              that each ask for a close-up detail of the scene, so every neighborhood of the sheet
              reads as part of the scene while the folded views read as their subjects
  arcimboldo  the flat prompt controls only the low frequencies, as the large gestalt; the folded
              views and the context windows control the high frequencies, as many small objects
  silhouette  the folded views show only the wings, not the whole crane, and their prompts describe
              the wing's shape, so the paper's own outline does part of the work
  fold-in     like a MAD fold-in: each wing's two shards, in place on the flat sheet with a margin
              of surroundings, ask for a scene detail, and joined on the wing they ask for its subject

The sampler has the same structure as illusion.generate (stage 1, then stage 2 conditioned on the
upscaled stage 1, with a re-noised plain backdrop around every view), but takes any list of views,
each with its own prompt and weight, and batches the model calls so that many views fit in memory.
illusion.py is unchanged.
"""
from dataclasses import dataclass

import numpy as np

import illusion
from factorized import split

METHODS = ("average", "context", "arcimboldo", "silhouette", "fold-in")

# The blur width each method splits frequencies at, as a fraction of the sheet's side (0: no split).
SIGMA_FRAC = {"arcimboldo": 1 / 8}

# How far every view fades out from an edge where its paper meets the gray backdrop, as a fraction
# of the sheet's side: 1 pixel at stage 1, 4 at stage 2. The model draws an edge wherever paper
# meets backdrop. Without the fade, the folded crane's outline left straight lines across the flat
# sheet, and windows drew a frame along the sheet's border, which shows as a line down the middle of
# every folded wing (the two wing shards join along the sheet's border).
EDGE_FADE = 1 / 64


def _ramp(dist, width, R):
    # Weight from dist, a pixel's distance to the nearest backdrop pixel (1 right next to it): rises
    # linearly and reaches 1 once dist is more than width * R.
    dist = np.asarray(dist, float)
    return np.ones_like(dist) if width <= 0 else np.clip(dist / (width * R + 1), 0, 1)


def _border_fade(sheet_px, R, width):
    # Weight of each sheet pixel by its distance to the sheet's own border (1 on the border row).
    r, c = sheet_px // R, sheet_px % R
    return _ramp(np.minimum.reduce([r, c, R - 1 - r, R - 1 - c]) + 1, width, R)


@dataclass
class View:
    """One way of looking at the sheet, with its own prompt.

    name: label for plots. prompt: key into the prompt embeddings. source: (R * R,) sheet pixel
    (flat index) each canvas pixel shows. paper: (R, R) bool, canvas pixels that show the sheet (the
    rest is plain backdrop). sheet_px, canvas_px: one-to-one pairs saying which canvas pixel each
    covered sheet pixel takes its prediction from. weight: its share in the per-pixel average.
    part: "all", or "low" / "high" to control only those frequencies. pixel_weight: optional
    (len(sheet_px),) factors in [0, 1] on weight, per covered sheet pixel, to fade a view out
    toward its edges.
    """
    name: str
    prompt: str
    source: np.ndarray
    paper: np.ndarray
    sheet_px: np.ndarray
    canvas_px: np.ndarray
    weight: float = 1.0
    part: str = "all"
    pixel_weight: np.ndarray = None

    def weights(self):
        """(len(sheet_px),) weight of each covered sheet pixel."""
        w = np.full(len(self.sheet_px), float(self.weight))
        return w if self.pixel_weight is None else w * self.pixel_weight

    @property
    def res(self):
        return self.paper.shape[0]

    def covers(self):
        """(R, R) bool: the sheet pixels this view controls."""
        out = np.zeros(self.res * self.res, bool)
        out[self.sheet_px] = True
        return out.reshape(self.res, self.res)

    def show(self, sheet, background=1.0, faded=False):
        """The view's canvas for an (R, R, 3) sheet. faded: blend each pixel into the background
        by how much say it has (its pixel_weight), to show where the view fades out."""
        R = self.res
        out = np.asarray(sheet, float).reshape(R * R, -1)[self.source].reshape(R, R, -1)
        out = np.where(self.paper[..., None], out, background)
        if faded and self.pixel_weight is not None:
            alpha = np.zeros(R * R)
            alpha[self.canvas_px] = self.pixel_weight
            alpha = alpha.reshape(R, R)[..., None]
            out = np.where(self.paper[..., None], background + alpha * (out - background), background)
        return out


# ---------------------------------------------------------------------------------------------
# Views (NumPy)


def flat_view(fv, prompt="flat", **kw):
    """The whole sheet as it lies flat."""
    R = fv.res
    every = np.arange(R * R)
    return View("flat", prompt, every, np.ones((R, R), bool), every, every, **kw)


def fold_view(fv, side, prompt, keep=None, name=None, edge_fade=0.0, **kw):
    """A folded view of the crane ("front" or "back"), optionally limited to the paper whose sheet
    pixels are in keep ((R, R) bool), with the rest of the canvas left as backdrop. edge_fade: fade
    the view's weight to zero over this distance (a fraction of the sheet's side) toward the edge of
    its paper, where the model draws the paper's outline (see EDGE_FADE)."""
    from scipy import ndimage
    R = fv.res
    source = fv.source[side].ravel()
    paper = fv.paper[side].copy()
    sheet_px, canvas_px = fv.pairs[side]
    if keep is not None:
        keep = keep.ravel()
        paper &= keep[source].reshape(R, R)
        ok = keep[sheet_px] & paper.ravel()[canvas_px]
        sheet_px, canvas_px = sheet_px[ok], canvas_px[ok]
    if edge_fade > 0 and "pixel_weight" not in kw:
        dist = ndimage.distance_transform_edt(np.pad(paper, 1))[1:-1, 1:-1].ravel()
        kw["pixel_weight"] = _ramp(dist[canvas_px], edge_fade, R)
    return View(name or side, prompt, source, paper, sheet_px, canvas_px, **kw)


def shards(fv, side):
    """The connected pieces of the sheet that a folded view shows, as (R, R) bool masks, largest first."""
    from scipy import ndimage
    labels, n = ndimage.label(fv.visible[side])
    masks = [labels == k for k in range(1, n + 1)]
    return sorted(masks, key=lambda m: -m.sum())


def wing_shards(fv, side, frac=0.8):
    """The shards that make up the wing: on the crane, the two largest, of equal size, which join
    into the upright triangle on the folded view."""
    pieces = shards(fv, side)
    return [m for m in pieces if m.sum() >= frac * pieces[0].sum()]


def wing(fv, side):
    """(R, R) bool: the sheet pixels of a view's wing."""
    return np.any(wing_shards(fv, side), axis=0)


def in_place_view(fv, mask, prompt, margin=0.0, feather=0.0, edge_fade=0.0, name="in place", **kw):
    """Part of the sheet as it lies flat, unrotated, moved to the middle of the canvas: the pixels in
    mask ((R, R) bool), grown by margin (a fraction of the sheet's side) to include surroundings.
    feather: fade the view's weight to zero over this distance (a fraction of the sheet's side)
    toward the region's edges inside the sheet. The model sees plain backdrop past the region's edge
    and draws that edge, so a fade keeps it from leaving a seam. edge_fade: the same, but usually
    shorter, toward the sheet's own border, where the model would draw a frame (see EDGE_FADE)."""
    from scipy import ndimage
    R = fv.res
    grow = round(margin * R)
    region = ndimage.distance_transform_edt(~mask) <= grow if grow else mask   # grown by true distance
    rows, cols = np.nonzero(region)
    dr = round((rows.min() + rows.max()) / 2 - (R - 1) / 2)
    dc = round((cols.min() + cols.max()) / 2 - (R - 1) / 2)
    r, c = np.mgrid[:R, :R]
    sr, sc = r + dr, c + dc                                      # canvas (r, c) shows sheet (sr, sc)
    inside = (sr >= 0) & (sr < R) & (sc >= 0) & (sc < R)
    sr, sc = sr.clip(0, R - 1), sc.clip(0, R - 1)
    paper = inside & region[sr, sc]
    source = (sr * R + sc).ravel()
    canvas_px = np.flatnonzero(paper)
    sheet_px = source[canvas_px]
    if feather > 0 and "pixel_weight" not in kw:
        # Distance to the nearest pixel outside the region, with the sheet's edge not counting as outside.
        inside = np.pad(region, 1, mode="edge")
        dist = ndimage.distance_transform_edt(inside)[1:-1, 1:-1]
        # Pixels more than feather from the edge (all of mask, when feather is the margin) keep full weight.
        kw["pixel_weight"] = np.clip((dist.ravel()[sheet_px] - 0.5) / max(feather * R - 0.5, 0.5), 0, 1)
    if edge_fade > 0:
        border = _border_fade(sheet_px, R, edge_fade)
        kw["pixel_weight"] = border if kw.get("pixel_weight") is None else np.minimum(kw["pixel_weight"], border)
    return View(name, prompt, source, paper, sheet_px, canvas_px, **kw)


def windows(fv, prompt, size=0.5, stride=0.25, feather=True, edge_fade=0.0, **kw):
    """Square windows tiled over the sheet, overlapping, each a view of its own: MultiDiffusion's
    sliding windows. size and stride are fractions of the sheet's side.

    feather: fade each window's weight linearly to zero across its overlap with its neighbors, on
    every side except the sheet's own edge (as Mixture of Diffusers fades its tiles). With size twice
    the stride, the faded windows add up to exactly 1 at every pixel, so weight is the windows'
    total say anywhere on the sheet. Without fading, each window's edge meets plain backdrop, the
    model draws that edge, and the windows leave straight seams on the sheet. edge_fade: also fade
    every window over this short distance toward the sheet's own border (see EDGE_FADE); the windows
    then add up to less than 1 there."""
    R = fv.res
    s, step = round(size * R), max(1, round(stride * R))
    starts = sorted(set(list(range(0, R - s + 1, step)) + [R - s]))
    ramp = max(1, s - step)                                      # the overlap with the next window

    def fade(x, a):
        # Weight along one axis at sheet positions x, for the window starting at a.
        w = np.ones(len(x))
        if a > 0:
            w = np.minimum(w, (x - a + 0.5) / ramp)
        if a + s < R:
            w = np.minimum(w, (a + s - x - 0.5) / ramp)
        return np.clip(w, 0, 1)

    out = []
    for r0 in starts:
        for c0 in starts:
            box = np.zeros((R, R), bool)
            box[r0:r0 + s, c0:c0 + s] = True
            v = in_place_view(fv, box, prompt, name=f"window {r0},{c0}", **kw)
            if feather:
                v.pixel_weight = fade(v.sheet_px // R, r0) * fade(v.sheet_px % R, c0)
            if edge_fade > 0:
                border = _border_fade(v.sheet_px, R, edge_fade)
                v.pixel_weight = border if v.pixel_weight is None else v.pixel_weight * border
            out.append(v)
    return out


def method_views(method, fv, context_weight=1.0, margin=1 / 16, edge_fade=EDGE_FADE):
    """The views a method samples with, for one FoldViews. Prompt keys: "flat", "front", "back",
    "context" (a close-up detail of the flat scene), and "front_shape", "back_shape" (the wing
    subjects described with the wing's shape). context_weight: the context windows' total weight at
    every pixel (they're faded to add up to one). margin: surroundings around each shard in the
    fold-in method, faded out across that margin. edge_fade: how far every view but the flat one
    fades out where its paper meets the backdrop (EDGE_FADE; 0 turns it off)."""
    e = edge_fade
    if method == "average":
        return [flat_view(fv), fold_view(fv, "front", "front", edge_fade=e), fold_view(fv, "back", "back", edge_fade=e)]
    if method == "context":
        return method_views("average", fv, edge_fade=e) + windows(fv, "context", weight=context_weight, edge_fade=e)
    if method == "arcimboldo":
        return ([flat_view(fv, part="low"), fold_view(fv, "front", "front", part="high", edge_fade=e),
                 fold_view(fv, "back", "back", part="high", edge_fade=e)]
                + windows(fv, "context", weight=context_weight, part="high", edge_fade=e))
    if method == "silhouette":
        return [flat_view(fv)] + [fold_view(fv, side, f"{side}_shape", keep=wing(fv, side), name=f"{side} wing",
                                            edge_fade=e) for side in ("front", "back")]
    if method == "fold-in":
        views = [flat_view(fv)]
        for side in ("front", "back"):
            views.append(fold_view(fv, side, side, keep=wing(fv, side), name=f"{side} wing", edge_fade=e))
            views += [in_place_view(fv, m, "context", margin, feather=margin, edge_fade=e,
                                    name=f"{side} shard {k + 1} in place")
                      for k, m in enumerate(wing_shards(fv, side))]
        return views
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


# ---------------------------------------------------------------------------------------------
# Sampler (PyTorch)


def combine(preds, views, maps, R, sigma=0.0):
    """Map each view's prediction back onto the sheet and average per pixel, weighted. If any view
    has part "high" and sigma > 0, the noise channels take the "low" and "all" views' average
    blurred, plus the "high" views' detail (factorized.split); variance channels always take the
    plain average. With sigma 0, parts are ignored and every view counts at every frequency.
    preds: one (C, R, R) prediction per view, in its own canvas. maps: from torch_maps."""
    import torch
    C = preds[0].shape[0]

    def average(members, fallback=None):
        total = torch.zeros((C, R * R), device=preds[0].device)
        weight = torch.zeros(R * R, device=preds[0].device)
        for k in members:
            m = maps[k]
            total.index_add_(1, m["sheet_px"], m["weight"] * preds[k].reshape(C, -1)[:, m["canvas_px"]].float())
            weight.index_add_(0, m["sheet_px"], m["weight"])
        out = total / weight.clamp(min=1e-12)
        if fallback is not None:
            out = torch.where(weight > 0, out, fallback.reshape(C, -1))
        return out.reshape(C, R, R)

    everything = average(range(len(views)))
    high = [k for k, v in enumerate(views) if v.part == "high"]
    if not high or sigma <= 0:                   # no split: every view counts at every frequency
        return everything.to(preds[0].dtype)
    low = average([k for k, v in enumerate(views) if v.part != "high"])
    detail = average(high, fallback=low)
    noise = split(low[:3], detail[:3], sigma)
    return torch.cat([noise, everything[3:]]).to(preds[0].dtype)


def torch_maps(views, device="cpu"):
    """Each view's pixel maps and per-pixel weights as torch tensors, for combine."""
    import torch
    return [dict(source=torch.as_tensor(v.source, device=device), paper=torch.as_tensor(v.paper, device=device),
                 sheet_px=torch.as_tensor(v.sheet_px, device=device),
                 canvas_px=torch.as_tensor(v.canvas_px, device=device),
                 weight=torch.as_tensor(v.weights(), dtype=torch.float32, device=device)) for v in views]


def encode_prompts(encoder, prompts):
    """{key: text} -> {key: (negative, positive)} embeddings, each (1, tokens, dim), from an IFPipeline."""
    keys = list(prompts)
    pos, neg = encoder.encode_prompt([prompts[k] for k in keys])
    return {k: (neg[i:i + 1], pos[i:i + 1]) for i, k in enumerate(keys)}


def _per_stage(value):
    # A setting given once for both stages, or as a (stage 1, stage 2) pair.
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


def generate(stage_1, stage_2, views1, views2, embeds, guidance=(7.0, 4.0), steps=(50, 40), sigma_frac=0.0,
             background=0.85, noise_level=250, seed=0, batch_size=8, progress=None, stage1=None):
    """Sample a sheet with DeepFloyd IF from any set of views.

    views1, views2: lists of View for stage 1 (64 pixels) and stage 2 (256 pixels), usually
    method_views(method, views64) and method_views(method, views256). embeds: {prompt key:
    (negative, positive)} from encode_prompts. sigma_frac: the frequency split's blur width as a
    fraction of the sheet's side, used if any view has part "low" or "high". sigma_frac and
    batch_size (views per model call; lower it if the GPU runs out of memory) take one value for
    both stages or a (stage 1, stage 2) pair. noise_level: how much noise stage 2 adds to the
    upscaled stage-1 sheet it's guided by; lower keeps stage 2 closer to stage 1. Other arguments as
    in illusion.generate.

    Run one stage at a time: with stage_2 (or views2) None, only stage 1 runs. With stage1, an
    (R, R, 3) stage-1 sheet in [0, 1] from an earlier run, stage 1 is skipped. Each stage draws its
    noise from its own generator seeded by seed, so stage 2 comes out the same either way.
    Returns (stage-1 sheet, final sheet) as (R, R, 3) arrays in [0, 1]; the final sheet is None if
    only stage 1 ran."""
    import torch
    import torch.nn.functional as F
    pipe0 = stage_1 if stage_1 is not None else stage_2
    device = pipe0._execution_device
    dtype = pipe0.unet.dtype
    sigma_frac, batch_size = _per_stage(sigma_frac), _per_stage(batch_size)
    E = {k: tuple(e.to(device, dtype).reshape(1, *e.shape[-2:]) for e in pair) for k, pair in embeds.items()}

    def run(pipe, views, n_steps, scale, gen, sigma, batch_size, cond=None, level=None):
        R = views[0].res
        maps = torch_maps(views, device)
        pipe.scheduler.set_timesteps(n_steps, device=device)
        x = torch.randn((1, 3, R, R), generator=gen, device=device, dtype=dtype) * pipe.scheduler.init_noise_sigma
        bg = torch.full((1, 3, R, R), background * 2 - 1, device=device, dtype=dtype)
        bg_noise = torch.randn((1, 3, R, R), generator=gen, device=device, dtype=dtype)
        bg_c = None
        if cond is not None:
            noise = torch.randn(bg.shape, generator=gen, device=device, dtype=dtype)   # seeded, unlike randn_like
            bg_c = pipe.image_noising_scheduler.add_noise(bg, noise, level.reshape(1))
            cond = pipe.image_noising_scheduler.add_noise(cond, noise, level.reshape(1))
        for i, t in enumerate(pipe.scheduler.timesteps):
            bg_t = pipe.scheduler.add_noise(bg, bg_noise, t.reshape(1))
            xs = pipe.scheduler.scale_model_input(x, t)
            inputs = []
            for m in maps:
                xv = illusion._gather(xs, m, bg_t)
                if cond is not None:
                    xv = torch.cat([xv, illusion._gather(cond, m, bg_c)], 1)
                inputs.append(xv)
            preds = []
            for j in range(0, len(views), batch_size):
                chunk, vs = inputs[j:j + batch_size], views[j:j + batch_size]
                neg = torch.cat([E[v.prompt][0] for v in vs])
                pos = torch.cat([E[v.prompt][1] for v in vs])
                kw = {} if level is None else {"class_labels": level.reshape(1).repeat(2 * len(vs))}
                out = pipe.unet(torch.cat(chunk * 2), t, encoder_hidden_states=torch.cat([neg, pos]),
                                return_dict=False, **kw)[0]
                uncond, text = out.chunk(2)
                guided = uncond[:, :3] + scale * (text[:, :3] - uncond[:, :3])
                preds += list(torch.cat([guided, text[:, 3:]], 1))
            pred = combine(preds, views, maps, R, sigma * R)[None]
            x = pipe.scheduler.step(pred, t, x, generator=gen, return_dict=False)[0]
            if progress:
                progress(i, len(pipe.scheduler.timesteps))
        return x

    to_img = lambda x: ((x[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    with torch.no_grad():
        if stage1 is None:
            gen1 = torch.Generator(device).manual_seed(seed)
            x1 = run(stage_1, views1, steps[0], guidance[0], gen1, sigma_frac[0], batch_size[0]).clamp(-1, 1)
        else:
            img = torch.as_tensor(np.asarray(stage1, np.float32), device=device)
            x1 = (img * 2 - 1).permute(2, 0, 1)[None].to(dtype)
        if stage_2 is None or views2 is None:
            return to_img(x1), None
        gen2 = torch.Generator(device).manual_seed(seed + 1_000_000)
        R2 = views2[0].res
        up = F.interpolate(x1.float(), (R2, R2), mode="bilinear", align_corners=True).to(dtype)
        x2 = run(stage_2, views2, steps[1], guidance[1], gen2, sigma_frac[1], batch_size[1],
                 cond=up, level=torch.tensor(noise_level, device=device))
    return to_img(x1), to_img(x2)


# ---------------------------------------------------------------------------------------------
# Looking at results


def shard_crops(fv, sheet, side, margin=1 / 16, background=1.0):
    """Each wing shard as it appears on the flat sheet: its bounding box plus margin (a fraction of
    the sheet's side) of surroundings, padded to a square with background. Padding instead of
    widening the crop keeps each crop about its own shard: the crane's two wing shards share a
    corner of the sheet, and wide square crops of the two came out the same."""
    R = fv.res
    sheet = np.asarray(sheet, float)
    g = round(margin * R)
    out = []
    for m in wing_shards(fv, side):
        rows, cols = np.nonzero(m)
        r0, r1 = max(rows.min() - g, 0), min(rows.max() + 1 + g, R)
        c0, c1 = max(cols.min() - g, 0), min(cols.max() + 1 + g, R)
        crop = sheet[r0:r1, c0:c1]
        h, w = crop.shape[:2]
        n = max(h, w)
        square = np.full((n, n, sheet.shape[2]), background)
        square[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = crop
        out.append(square)
    return out


def score(fv, sheets, embed_images, embed_texts, texts, margin=1 / 16):
    """Rank sheets with an image-text model such as CLIP (factorized.clip_embedders).

    texts: (flat, front, back) descriptions. For each sheet: flat, front, back are the cosine
    similarities of the flat sheet and the two folded views to their own text. front_switch and
    back_switch measure the reveal: the folded view's similarity to its text, minus the mean
    similarity to the same text of its wing shards seen in place on the flat sheet (shard_crops). Positive means
    folding turns those pixels into the subject. score = mean of flat, front, back, plus the mean
    switch. Returns rows, best first. A rough guide, not a judge of beauty."""
    t = np.asarray(embed_texts(list(texts)), float)
    rows = []
    for label, sheet in sheets.items():
        whole = [sheet] + [fold_view(fv, s, s).show(sheet) for s in ("front", "back")]
        crops = {s: shard_crops(fv, sheet, s, margin) for s in ("front", "back")}
        images = whole + crops["front"] + crops["back"]
        v = np.asarray(embed_images(images), float)
        own = [float(v[k] @ t[k]) for k in range(3)]
        row = dict(label=label, flat=own[0], front=own[1], back=own[2])
        start = 3
        for k, s in ((1, "front"), (2, "back")):
            n = len(crops[s])
            in_place = float(np.mean(v[start:start + n] @ t[k]))
            row[f"{s}_switch"] = own[k] - in_place
            start += n
        row["score"] = float(np.mean(own) + (row["front_switch"] + row["back_switch"]) / 2)
        rows.append(row)
    return sorted(rows, key=lambda r: -r["score"])


def plot_views(views, sheet, cols=6):
    """Each view's canvas for a sheet: what the model is shown for that view at every step. Faded
    pixels have less say in the blend (pixel_weight)."""
    import matplotlib.pyplot as plt
    rows = int(np.ceil(len(views) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, v in zip(axes.ravel(), views):
        ax.imshow(np.clip(v.show(sheet, 0.85, faded=True), 0, 1))
        extra = "" if v.part == "all" else f", {v.part}"
        ax.set_title(f"{v.name}\n{v.prompt} ×{v.weight:g}{extra}", fontsize=8)
    fig.tight_layout()
    return fig


def plot_screen(fv, sheets, titles=("flat", "front wing", "back wing"), cols=4):
    """A compact grid for picking seeds: for each sheet, the flat sheet and each wing on its own, as
    folded. sheets: {label: (R, R, 3) sheet}."""
    import matplotlib.pyplot as plt
    rows = int(np.ceil(len(sheets) / cols))
    fig, axes = plt.subplots(rows, 3 * cols, figsize=(3.3 * cols, 1.25 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (label, sheet) in enumerate(sheets.items()):
        r, c = divmod(k, cols)
        ax_flat, *ax_wings = axes[r, 3 * c:3 * c + 3]
        ax_flat.imshow(np.clip(sheet, 0, 1))
        ax_flat.set_title(label, fontsize=8)
        for ax, side, title in zip(ax_wings, ("front", "back"), titles[1:]):
            v = fold_view(fv, side, side, keep=wing(fv, side))
            rows_, cols_ = np.nonzero(v.paper)
            crop = v.show(sheet)[rows_.min():rows_.max() + 1, cols_.min():cols_.max() + 1]
            ax.imshow(np.clip(crop, 0, 1))
            ax.set_title(title, fontsize=7)
    fig.tight_layout()
    return fig


def plot_reveal(fv, sheets, titles=("flat", "front wing", "back wing"), margin=1 / 16):
    """One row per sheet: the flat sheet, then for each wing its shards in place on the flat sheet
    (in context) and the folded view (isolated and joined)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(sheets), 5, figsize=(17, 3.6 * len(sheets)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.2, 1, 1.2, 1]})
    for row, (label, sheet) in zip(axes, sheets.items()):
        row[0].imshow(np.clip(sheet, 0, 1))
        row[0].set_title(f"{label}: {titles[0]}", fontsize=9)
        for k, side in enumerate(("front", "back")):
            crops = shard_crops(fv, sheet, side, margin)
            h = max(c.shape[0] for c in crops)
            gap = np.ones((h, max(2, h // 12), 3))
            pad = lambda c: np.concatenate([c, np.ones((h - c.shape[0], c.shape[1], 3))]) if c.shape[0] < h else c
            strip = np.concatenate(sum([[pad(c), gap] for c in crops], [])[:-1], 1)
            row[1 + 2 * k].imshow(np.clip(strip, 0, 1))
            row[1 + 2 * k].set_title(f"{titles[1 + k]}: shards in place", fontsize=9)
            row[2 + 2 * k].imshow(np.clip(fold_view(fv, side, side).show(sheet), 0, 1))
            row[2 + 2 * k].set_title(f"{titles[1 + k]}: folded", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    return fig
