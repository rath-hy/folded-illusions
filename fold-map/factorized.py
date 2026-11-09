"""Factorized blending for fold illusions: a different way to share visible pixels.

illusion.generate averages, per pixel, the noise predicted through the flat view and through the
folded views, so a visible pixel ends up as a compromise neither prompt asked for. Factorized
Diffusion (Geng, Park, Owens, ECCV 2024) suggests another way to share a pixel: split the image
into components that add back up to it, and let each prompt control one component. Here the
split is by frequency, on the sheet:

    combined = G(flat) + (folded - G(folded))

G is a Gaussian blur over the whole sheet, flat is the noise predicted through the flat view, and
folded is the folded views' predictions mapped back onto the sheet (the flat prediction where no
folded view shows). So the flat picture controls the sheet's broad shapes, and the folded pictures
control the fine detail where their paper shows. A folded view shows a small patch of the sheet,
so its whole picture is fine detail at the sheet's scale. low="folded" swaps the roles.

Both terms are linear, and they're applied to noise predictions rather than to the noisy image,
which is what keeps the noise statistics valid in Factorized Diffusion.

Everything else (prompts, backdrop, both stages, seeds) is illusion.generate's own code: this
module only swaps its blending step while it runs.
"""
import contextlib

import numpy as np

import illusion

_average = illusion._combine        # the original blend, kept before _blending swaps it out


def blur(x, sigma):
    """Gaussian blur of a (C, R, R) torch tensor over its last two axes, edges reflected."""
    import torch
    import torch.nn.functional as F
    if sigma <= 0:
        return x
    R = x.shape[-1]
    r = min(int(np.ceil(3 * sigma)), R - 1)
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    C = x.shape[0]
    y = x.float()[None]
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return y[0]


def folded_on_sheet(preds, maps, R):
    """The folded views' predictions mapped back onto the sheet and averaged where several land on
    one pixel, with the flat prediction wherever no folded view shows. preds: [flat, then one per
    folded view], each (C, R, R) in its own view's space."""
    import torch
    flat = preds[0].float()
    C = flat.shape[0]
    total = flat.new_zeros((C, R * R))
    count = flat.new_zeros(R * R)
    for pred, m in zip(preds[1:], maps):
        total.index_add_(1, m["sheet_px"], pred.reshape(C, -1)[:, m["canvas_px"]].float())
        count.index_add_(0, m["sheet_px"], count.new_ones(len(m["sheet_px"])))
    seen = count > 0
    mixed = torch.where(seen, total / count.clamp(min=1), flat.reshape(C, -1))
    return mixed.reshape(C, R, R)


def split(low_src, high_src, sigma):
    """low_src's low frequencies plus high_src's high frequencies."""
    return blur(low_src, sigma) + high_src - blur(high_src, sigma)


def factorized_combine(preds, maps, fold_weight, R, sigma, low="flat"):
    """The blend: low frequencies from one source, high frequencies from the other, on the noise
    channels. IF also predicts a variance in its last channels, which keeps illusion's own blend."""
    import torch
    averaged = _average(preds, maps, fold_weight, R)                  # used for the variance channels
    flat = preds[0].float()[:3]
    folded = folded_on_sheet(preds, maps, R)[:3]
    noise = split(flat, folded, sigma) if low == "flat" else split(folded, flat, sigma)
    return torch.cat([noise.to(averaged.dtype), averaged[3:]], 0)


@contextlib.contextmanager
def _blending(sigma_frac, low, stages):
    # Swap illusion's blending step for the factorized one, for the sheet sizes in stages.
    def combine(preds, maps, fold_weight, R):
        if R not in stages:
            return _average(preds, maps, fold_weight, R)
        return factorized_combine(preds, maps, fold_weight, R, sigma_frac * R, low)

    illusion._combine = combine
    try:
        yield
    finally:
        illusion._combine = _average


def generate(stage_1, stage_2, views64, views256, embeds, sigma_frac=1 / 32, low="flat",
             factorize=(True, True), **kwargs):
    """illusion.generate with the factorized blend. sigma_frac: the blur's width as a fraction of
    the sheet's side (1/32 is 2 pixels at stage 1 and 8 at stage 2), so both stages split at the
    same physical scale. low: "flat" gives the flat picture the low frequencies; "folded" swaps
    them. factorize: whether to use it in (stage 1, stage 2); elsewhere the original average is
    used. Other arguments, including fold_weight (which still blends the variance channels), go
    to illusion.generate. Returns (stage-1 sheet, final sheet)."""
    sizes = {views.res for on, views in zip(factorize, (_first(views64), _first(views256))) if on}
    with _blending(sigma_frac, low, sizes):
        return illusion.generate(stage_1, stage_2, views64, views256, embeds, **kwargs)


def _first(views):
    return illusion._entries(views)[0][0]


def preview(views, flat_picture, front_picture, back_picture, sigma_frac=1 / 32, low="flat"):
    """No-AI preview of the same split on finished pictures: illusion.composite's sheet (folded
    pictures where they show, flat elsewhere) with its low frequencies replaced by the flat
    picture's. Shows what sigma_frac does before spending GPU time. Returns an (R, R, 3) sheet."""
    from scipy.ndimage import gaussian_filter
    mixed = illusion.composite(views, flat_picture, front_picture, back_picture)
    flat = np.asarray(flat_picture, float)
    g = lambda x: gaussian_filter(x, (sigma_frac * views.res, sigma_frac * views.res, 0), mode="reflect")
    low_src, high_src = (flat, mixed) if low == "flat" else (mixed, flat)
    return np.clip(g(low_src) + high_src - g(high_src), 0, 1)


def views_of(views, sheet, background=1.0):
    """The flat sheet and its two folded views, as (R, R, 3) images."""
    s = np.moveaxis(np.asarray(sheet, float), -1, 0)
    return [sheet] + [np.moveaxis(views.to_view(s, v, background), 0, -1) for v in ("front", "back")]


def plot_comparison(views, sheets, titles=("flat", "fox wing (front view)", "owl wing (back view)")):
    """One row per sheet in sheets ({label: (R, R, 3) sheet}): flat, then each folded view."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(sheets), 3, figsize=(12, 4 * len(sheets)), squeeze=False)
    for row, (label, sheet) in zip(axes, sheets.items()):
        for ax, image, title in zip(row, views_of(views, sheet), titles):
            ax.imshow(np.clip(image, 0, 1))
            ax.set_title(f"{label}: {title}", fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    return fig
