"""The Bayon cubism helpers: face numbering and frames, cutting a guide into faces, and the store."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cubism"))

import bayon as b  # noqa: E402


def test_faces_are_numbered_1_to_25_top_to_bottom():
    assert b.NUMBERS == list(range(1, 26))
    assert sorted(b.FACE_INDEX.values()) == list(range(25))
    heights = [b._head_on_centers()[b.FACE_INDEX[n], 1] for n in b.NUMBERS]
    assert heights[0] == max(heights) and heights[-1] == min(heights)


@pytest.mark.parametrize("n", b.NUMBERS)
def test_face_pictures_are_upright_and_not_mirrored(n):
    k = b.faces()[b.FACE_INDEX[n]][0]
    _, R, D = b.frame(n)
    assert np.allclose(np.cross(R, D), -b.E[k])          # turned, never flipped
    _, r, u = b.camera(1.0, k)
    assert R @ r - D @ u > 1.2                              # within 45 degrees of upright, tilted to its axis


def test_faces_reassemble_the_photo_head_on():
    """Each face's piece of the photo, put back on the model, shows the photo again head-on."""
    rng = np.random.default_rng(0)
    photo = (rng.random((60, 50, 3)) * 255).astype(np.uint8)
    photo = np.asarray(Image.fromarray(photo).resize((500, 600), Image.BICUBIC))
    fit = dict(x=0.5, y=0.5, width=0.8)
    size, margin = 300, 0.06
    seen = b.head_on_photo(photo, fit, size, margin)
    # The same view straight from the photo: where each output pixel lands in it.
    x0, x1, y0, y1 = b.bounds(b.camera(0.0), b.faces())
    span = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    g = (np.arange(size) + 0.5) / size - 0.5
    sx, sy = (x0 + x1) / 2 + g[None, :] * span, (y0 + y1) / 2 - g[:, None] * span
    px, py = b._photo_map(photo.shape, fit)(sx, sy)
    expect = photo[py.astype(int).clip(0, 599), px.astype(int).clip(0, 499)] / 255
    on_model = (seen != 1.0).any(2)
    err = np.abs(seen[on_model] - expect[on_model]).mean()
    assert on_model.mean() > 0.3 and err < 0.05


def test_a_line_in_the_guide_shows_on_the_faces_it_crosses():
    guide = np.zeros((600, 500), np.uint8)
    guide[319:322, :] = 255                                  # a horizontal line across the model
    fit = dict(x=0.5, y=0.5, width=0.8)
    lit = [n for n in b.NUMBERS if b.face_edges(guide, n, fit, 128).any()]
    # Faces are convex, so the line crosses a face exactly when it passes between its top and bottom.
    to_px, (_, r, u) = b._photo_map(guide.shape, fit), b.camera(0.0)
    crossed = []
    for n in b.NUMBERS:
        _, c, e1, e2 = b.faces()[b.FACE_INDEX[n]]
        ys = [to_px(p @ r, p @ u)[1] for p in (c, c + e1, c + e2, c + e1 + e2)]
        if min(ys) + 2 < 320 < max(ys) - 2:
            crossed.append(n)
    assert len(crossed) >= 3 and lit == crossed


def test_canny_guide_and_specks():
    img = np.zeros((200, 200, 3), np.uint8)
    img[50:150, 50:150] = 255
    img[10, 10] = 255                                        # a speck
    g = b.make_guide(img, dict(b.DEFAULT_GUIDE, method="canny", blur=1.0, min_area=20))
    assert g.dtype == np.uint8 and g.shape == (200, 200)
    assert g[45:155, 45:155].any() and not g[5:15, 5:15].any()


def test_a_drawn_guide_is_flipped_to_white_on_black(tmp_path):
    drawn = np.full((100, 80), 255, np.uint8)
    drawn[50, :] = 0                                         # black ink on white paper
    Image.fromarray(drawn).save(tmp_path / "guide.png")
    g = b.load_guide(tmp_path / "guide.png", (200, 160))
    assert g.shape == (200, 160) and g.mean() < 20 and g[95:105].any()


def test_store_keeps_choices_and_never_overwrites(tmp_path):
    store = b.Store(tmp_path)
    red, blue = Image.new("RGB", (32, 32), "red"), Image.new("RGB", (32, 32), "blue")
    assert store.add_candidate(7, red, dict(prompt="lotus", seed=1)) == 1
    assert store.add_candidate(7, blue, dict(prompt="naga", seed=2)) == 2
    assert store.candidates(7) == [1, 2]
    assert 7 in b.todo(store)

    store.keep(7, 1)
    assert store.kept()[7][1]["prompt"] == "lotus"
    assert 7 not in b.todo(store) and len(b.todo(store)) == 24
    assert b.todo(store, [7]) == [7]                         # asked for by number, even though kept

    store.keep(7, 2)                                         # changing your mind replaces the choice
    assert store.kept()[7][1]["prompt"] == "naga"
    assert store.candidates(7) == [1, 2]                     # and both candidates are still there

    store.unkeep(7)
    assert 7 not in store.kept() and store.candidates(7) == [1, 2]
    with pytest.raises(KeyError):
        store.keep(7, 3)
    with pytest.raises(ValueError):
        store.add_candidate(26, red, {})


def test_board_mixes_kept_paintings_and_guide_lines(tmp_path):
    store = b.Store(tmp_path)
    store.add_candidate(13, Image.new("RGB", (64, 64), (200, 30, 30)), dict(seed=0))
    store.keep(13, 1)
    guide = np.zeros((600, 500), np.uint8)
    tex = b.board(store, guide, dict(x=0.5, y=0.5, width=0.8), res=32)
    assert len(tex) == 25
    red = tex[b.FACE_INDEX[13]]
    assert np.allclose(red, (200 / 255, 30 / 255, 30 / 255))
    assert np.allclose(tex[b.FACE_INDEX[1]], 0.3)            # unkept, no lines: plain gray
    sheet = b.contact_sheet(store, 13)
    assert sheet is not None and b.contact_sheet(store, 1) is None


def test_lines_are_redrawn_at_one_width_whatever_the_photo():
    bar = np.zeros((40, 60), np.uint8)
    bar[15:26, 5:55] = 255                                   # an 11-pixel-thick bar
    t = b.thin(bar)
    cols = (t > 0).sum(0)[15:45]
    assert cols.max() == 1 and (cols == 1).all()            # one pixel thick along its length
    fit = dict(x=0.5, y=0.5, width=0.8)
    for thickness in (2, 8):
        guide = np.zeros((600, 500), np.uint8)
        guide[318:318 + thickness, :] = 255
        e = b.face_edges(guide, 13, fit, 1024, width=6)
        width = 2 * cv2.distanceTransform(e, cv2.DIST_L2, 5).max()   # across the line, at any slant
        assert 4 <= width <= 8                                # about 6 pixels, however thick in the photo
