"""The ambiguous-fold methods: their views, the blend, and a run of the sampler on a stand-in model."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "randn"):                          # an empty namespace package, not PyTorch
    pytest.skip("PyTorch isn't installed", allow_module_level=True)

FOLD_MAP = Path(__file__).resolve().parent.parent / "fold-map"
sys.path.insert(0, str(FOLD_MAP))

import ambiguous  # noqa: E402
import illusion  # noqa: E402
from fold_map import FoldedModel  # noqa: E402
from test_factorized import _Pipe  # noqa: E402

PROMPTS = ("flat", "front", "back", "context", "front_shape", "back_shape")


@pytest.fixture(scope="module")
def fv():
    model = FoldedModel(str(FOLD_MAP / "004_traditional_Crane_cp.fold"), str(FOLD_MAP / "004_traditional_Crane_state.fold"))
    return illusion.FoldViews(model, 32)


def preds(views, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(6, v.res, v.res, generator=g) for v in views]


@pytest.mark.parametrize("method", ambiguous.METHODS)
def test_every_sheet_pixel_a_view_covers_takes_one_paper_pixel(fv, method):
    # Folded views may let a few sheet pixels share a neighbor's canvas pixel (see FoldViews).
    for v in ambiguous.method_views(method, fv):
        assert len(np.unique(v.sheet_px)) == len(v.sheet_px) == len(v.canvas_px)
        assert v.paper.ravel()[v.canvas_px].all()
        assert v.prompt in PROMPTS


def touching(a, b):
    # Pixel pairs, one in a and one in b, that are side by side.
    return int((a[:, :-1] & b[:, 1:]).sum() + (b[:, :-1] & a[:, 1:]).sum()
               + (a[:-1] & b[1:]).sum() + (b[:-1] & a[1:]).sum())


def test_each_crane_wing_joins_two_equal_shards_that_barely_touch_on_the_sheet(fv):
    R = fv.res
    for side in ("front", "back"):
        a, b = ambiguous.wing_shards(fv, side)
        assert abs(int(a.sum()) - int(b.sum())) <= 2
        assert touching(a, b) <= 2                                   # at most at a corner tip
        # On the folded view, the same two shards meet along a whole edge.
        src, paper = fv.source[side], fv.paper[side]
        on = lambda m: paper & m.ravel()[src.ravel()].reshape(R, R)
        assert touching(on(a), on(b)) > R / 4


def test_in_place_views_keep_the_sheet_orientation(fv):
    mask = ambiguous.wing_shards(fv, "front")[0]
    v = ambiguous.in_place_view(fv, mask, "context")
    sheet = np.random.default_rng(0).random((fv.res, fv.res, 3))
    # A pure shift keeps row-major order, so the canvas lists the same pixels in the same order.
    assert v.paper.sum() == mask.sum()
    assert np.array_equal(v.show(sheet)[v.paper], sheet[mask])


def test_faded_windows_add_up_to_one_everywhere(fv):
    total = np.zeros(fv.res * fv.res)
    for v in ambiguous.windows(fv, "context"):
        np.add.at(total, v.sheet_px, v.weights())
    assert np.allclose(total, 1)


def test_windows_fade_out_at_inner_edges_but_not_at_the_sheets_edge(fv):
    R = fv.res
    for v in ambiguous.windows(fv, "context"):
        w = np.full(R * R, np.nan)
        w[v.sheet_px] = v.weights()
        w = w.reshape(R, R)
        rows, cols = np.nonzero(v.covers())
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
        for edge, on_sheet_edge in ((w[r0, c0:c1 + 1], r0 == 0), (w[r1, c0:c1 + 1], r1 == R - 1),
                                    (w[r0:r1 + 1, c0], c0 == 0), (w[r0:r1 + 1, c1], c1 == R - 1)):
            if not on_sheet_edge:
                assert edge.max() < 0.15
        if r0 == 0 and c0 == 0:
            assert w[0, 0] == 1


def test_fold_in_shards_keep_full_weight_and_fade_across_their_margin(fv):
    views = [v for v in ambiguous.method_views("fold-in", fv) if "in place" in v.name]
    assert len(views) == 4
    for v, shard in zip(views, ambiguous.wing_shards(fv, "front") + ambiguous.wing_shards(fv, "back")):
        on_shard = shard.ravel()[v.sheet_px]
        assert on_shard.sum() == shard.sum()
        assert np.allclose(v.weights()[on_shard], 1)
        assert v.weights()[~on_shard].min() < 0.5


def test_shard_crops_show_each_shard_on_its_own(fv):
    R = fv.res
    sheet = np.random.default_rng(0).random((R, R, 3))
    for side in ("front", "back"):
        crops = ambiguous.shard_crops(fv, sheet, side)
        assert len(crops) == 2 and not np.array_equal(crops[0], crops[1])
        for crop, shard in zip(crops, ambiguous.wing_shards(fv, side)):
            rows, cols = np.nonzero(shard)
            g = round(R / 16)
            assert crop.shape[0] == crop.shape[1] <= max(np.ptp(rows), np.ptp(cols)) + 1 + 2 * g
            # Every pixel of the shard is in its crop.
            assert np.isin(sheet[shard].round(9).view([("", float)] * 3), crop.reshape(-1, 3).round(9).view([("", float)] * 3)).all()


def test_the_average_method_is_illusions_own_blend(fv):
    views = ambiguous.method_views("average", fv)
    p = preds(views)
    ours = ambiguous.combine(p, views, ambiguous.torch_maps(views), fv.res)
    theirs = illusion._combine(p, illusion._torch_maps(fv, "cpu"), 0.5, fv.res)
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_the_arcimboldo_split_keeps_the_flat_lows_and_the_others_highs(fv):
    from factorized import blur
    views = ambiguous.method_views("arcimboldo", fv)
    p = preds(views)
    out = ambiguous.combine(p, views, ambiguous.torch_maps(views), fv.res, sigma=3.0)
    high = [k for k, v in enumerate(views) if v.part == "high"]
    plain = [dataclasses.replace(views[k], part="all") for k in high]       # their own weighted average
    detail = ambiguous.combine([p[k] for k in high], plain, ambiguous.torch_maps(plain), fv.res)
    assert torch.allclose(out[:3], blur(p[0][:3], 3.0) + detail[:3] - blur(detail[:3], 3.0), atol=1e-4)


def test_the_sampler_runs_every_method(fv):
    embeds = {k: (torch.zeros(1, 4, 8), torch.full((1, 4, 8), float(i))) for i, k in enumerate(PROMPTS)}
    for method in ambiguous.METHODS:
        views = ambiguous.method_views(method, fv)
        s1, s2 = ambiguous.generate(_Pipe(), _Pipe(), views, views, embeds, steps=(3, 2), batch_size=4,
                                    sigma_frac=ambiguous.SIGMA_FRAC.get(method, 0))
        assert s1.shape == s2.shape == (32, 32, 3) and np.isfinite(s2).all()


def test_scores_reward_a_wing_that_reads_as_its_subject_only_once_folded(fv):
    # A stand-in for CLIP: an image's embedding is the mean color of everything but its white
    # backdrop, and a text's is a named color.
    colors = {"red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1]}
    unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)

    def embed_images(images):
        px = [np.asarray(im).reshape(-1, 3) for im in images]
        return [unit(p[~(p == 1).all(1)].mean(0) + 1e-6) for p in px]

    embed_texts = lambda texts: [unit(colors[t]) for t in texts]
    R = fv.res
    plain = np.zeros((R, R, 3)); plain[..., 0] = 1                  # all red: nothing to reveal
    reveal = plain.copy()
    for side, color in (("front", 1), ("back", 2)):                 # wing paper green / blue
        reveal[ambiguous.wing(fv, side)] = np.eye(3)[color]
    rows = {r["label"]: r for r in ambiguous.score(fv, {"plain": plain, "reveal": reveal}, embed_images,
                                                     embed_texts, ("red", "green", "blue"))}
    assert rows["plain"]["front_switch"] == pytest.approx(0) and rows["plain"]["back_switch"] == pytest.approx(0)
    # In place, each wing shard sits among red; folded, the two join into a mostly green wing.
    assert rows["reveal"]["front_switch"] > 0.2 and rows["reveal"]["back_switch"] > 0.2
