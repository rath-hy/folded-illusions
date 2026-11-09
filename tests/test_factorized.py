"""The factorized blend's math, and a run of the whole sampler on a stand-in model (no GPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "randn"):                          # an empty namespace package, not PyTorch
    pytest.skip("PyTorch isn't installed", allow_module_level=True)

FOLD_MAP = Path(__file__).resolve().parent.parent / "fold-map"
sys.path.insert(0, str(FOLD_MAP))

import factorized  # noqa: E402
import illusion  # noqa: E402
from fold_map import FoldedModel  # noqa: E402


@pytest.fixture(scope="module")
def views():
    model = FoldedModel(str(FOLD_MAP / "004_traditional_Crane_cp.fold"), str(FOLD_MAP / "004_traditional_Crane_state.fold"))
    return illusion.FoldViews(model, 32)


@pytest.fixture(scope="module")
def maps(views):
    return illusion._torch_maps(views, "cpu")


def preds(views, seed=0, channels=6):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(channels, views.res, views.res, generator=g) for _ in range(3)]


def test_splitting_one_image_gives_it_back():
    x = torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(factorized.split(x, x, 3.0), x, atol=1e-5)


def test_the_split_takes_low_and_high_frequencies_from_each_source():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(3, 32, 32, generator=g), torch.randn(3, 32, 32, generator=g)
    out = factorized.split(a, b, 2.0)
    assert torch.allclose(out - b, factorized.blur(a, 2.0) - factorized.blur(b, 2.0), atol=1e-5)


def test_no_blur_gives_the_flat_prediction(views, maps):
    p = preds(views)
    out = factorized.factorized_combine(p, maps, 0.5, views.res, sigma=0)
    assert torch.allclose(out[:3], p[0][:3], atol=1e-5)


def test_folded_predictions_land_where_their_paper_shows(views, maps):
    p = preds(views)
    folded = factorized.folded_on_sheet(p, maps, views.res)
    visible = torch.as_tensor(views.visible["front"] | views.visible["back"])
    assert torch.equal(folded[:, ~visible], p[0][:, ~visible])
    assert not torch.allclose(folded[:, visible], p[0][:, visible])


def test_hidden_pixels_far_from_visible_ones_follow_the_flat_view(views, maps):
    from scipy.ndimage import distance_transform_edt
    p = preds(views)
    out = factorized.factorized_combine(p, maps, 0.5, views.res, sigma=0.5)
    visible = views.visible["front"] | views.visible["back"]
    far = torch.as_tensor(distance_transform_edt(~visible) > 4)
    assert far.any()
    assert torch.allclose(out[:3][:, far], p[0][:3][:, far], atol=1e-4)


def test_variance_channels_keep_the_original_blend(views, maps):
    p = preds(views)
    out = factorized.factorized_combine(p, maps, 0.5, views.res, sigma=2.0)
    assert torch.allclose(out[3:], illusion._combine(p, maps, 0.5, views.res)[3:])


class _Scheduler:
    init_noise_sigma = 1.0

    def set_timesteps(self, n, device=None):
        self.timesteps = torch.arange(n - 1, -1, -1)

    def scale_model_input(self, x, t):
        return x

    def add_noise(self, x, noise, t):
        return x + 0 * noise

    def step(self, pred, t, x, generator=None, return_dict=False):
        return (x - 0.1 * pred[:, :3],)


class _Unet:
    dtype = torch.float32

    def __call__(self, batch, t, encoder_hidden_states=None, return_dict=False, class_labels=None):
        return (torch.cat([0.5 * batch[:, :3], 0 * batch[:, :3]], 1),)


class _Pipe:
    _execution_device = "cpu"

    def __init__(self):
        self.unet, self.scheduler, self.image_noising_scheduler = _Unet(), _Scheduler(), _Scheduler()


def test_the_sampler_runs_and_restores_the_original_blend(views):
    embeds = (torch.zeros(3, 4, 8), torch.zeros(3, 4, 8))
    original = illusion._combine
    s1, s2 = factorized.generate(_Pipe(), _Pipe(), views, views, embeds, steps=(3, 2), sigma_frac=1 / 16)
    assert illusion._combine is original
    assert s1.shape == s2.shape == (32, 32, 3) and np.isfinite(s2).all()
    b1, b2 = illusion.generate(_Pipe(), _Pipe(), views, views, embeds, steps=(3, 2))
    assert not np.allclose(s2, b2)                       # the blend really changed


def test_preview_matches_composite_where_blur_is_off(views):
    flat = np.random.default_rng(0).random((32, 32, 3))
    front = np.random.default_rng(1).random((32, 32, 3))
    back = np.random.default_rng(2).random((32, 32, 3))
    out = factorized.preview(views, flat, front, back, sigma_frac=0)
    assert np.allclose(out, flat)
