"""The Bayon cubism helpers: face numbering and frames, the study cut into faces, finishing, and the store."""
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


SVG = """<svg viewBox="0 0 400 600" xmlns="http://www.w3.org/2000/svg">
<defs><pattern id="h"><line x1="0" y1="0" x2="0" y2="9" stroke="#fff"/></pattern></defs>
<rect x="0" y="0" width="400" height="600" fill="#101010"/>
<polygon points="0,0 400,0 400,300 0,300" fill="#F0F0F0" stroke="#000"/>
<polygon points="0,0 400,0 400,300 0,300" fill="url(#h)"/>
<polyline points="0,450 400,450" fill="none" stroke="#fff"/>
<rect x="0" y="0" width="400" height="600" fill="#FFFFFF" fill-opacity="0.03"/>
</svg>"""


def _study(tmp_path, crop=None):
    p = tmp_path / "study.svg"
    p.write_text(SVG)
    return b.load_study(p, crop or dict(left=0, right=0, top=0, bottom=0))


def test_the_study_is_read_and_cropped(tmp_path):
    st = _study(tmp_path, dict(left=1 / 8, right=1 / 8, top=0, bottom=1 / 8))
    assert (st.width, st.height) == (300, 525)
    assert len(st.chunks) == 2 and len(st.lines) == 1          # hatching and the translucent veil skipped
    assert np.allclose(st.chunks[1][0][0], (-50, 0))          # shifted by the crop
    assert np.allclose(st.lightness(), [15, 92])               # darkest and lightest stretched to SHADES


def test_the_real_study_loads():
    st = b.load_study(Path(__file__).resolve().parent.parent / "cubism" / "bayon-study.svg")
    assert (st.width, st.height) == (600, 875) and len(st.chunks) > 80
    L = st.lightness()
    assert L.min() == 15 and L.max() == 92


def test_a_line_in_the_study_shows_on_the_faces_it_crosses(tmp_path):
    st = _study(tmp_path)
    fit = dict(x=0.5, y=0.5, width=0.8)
    lit = [n for n in b.NUMBERS if b.face_lines(st, n, fit, 128, width=8)[4:-4, 4:-4].any()]
    # The chunk border at y = 300 and the line at y = 450 are the only edges inside the model.
    to_px, (_, r, u) = b._photo_map(st.shape, fit), b.camera(0.0)
    crossed = []
    for n in b.NUMBERS:
        _, c, e1, e2 = b.faces()[b.FACE_INDEX[n]]
        ys = [to_px(p @ r, p @ u)[1] for p in (c, c + e1, c + e2, c + e1 + e2)]
        if any(min(ys) + 3 < y < max(ys) - 3 for y in (300, 450)):
            crossed.append(n)
    assert len(crossed) >= 4 and lit == crossed


def test_chunks_and_shades_follow_the_study(tmp_path):
    st = _study(tmp_path)
    fit = dict(x=0.5, y=0.5, width=0.8)
    for n in b.NUMBERS:
        ids = b.face_chunks(st, n, fit, 64)
        assert (ids >= 0).all()                                # the model sits inside the study
    top, bottom = b.NUMBERS[0], b.NUMBERS[-1]
    assert (b.face_chunks(st, top, fit, 64) == 1).all() and (b.face_chunks(st, bottom, fit, 64) == 0).all()
    assert b.face_shades(st, top, fit, 64).mean() > 0.8 > 0.3 > b.face_shades(st, bottom, fit, 64).mean()


def test_finishing_moves_each_chunk_toward_its_shade_and_draws_black_edges(tmp_path):
    import cv2
    st = _study(tmp_path)
    fit = dict(x=0.5, y=0.5, width=0.8)
    n = 13
    ids = b.face_chunks(st, n, fit, 256)
    assert set(np.unique(ids)) == {0, 1}                       # face 13 straddles the chunk border
    rng = np.random.default_rng(0)
    pic = np.clip(rng.normal(128, 25, (256, 256, 3)), 0, 255).astype(np.uint8)   # in gamut at any lightness
    L0 = cv2.cvtColor(pic.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]
    out = b.finish(Image.fromarray(pic), st, n, fit, dict(b.DEFAULT_FINISH, edge_width=0.001))
    L1 = cv2.cvtColor(out.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]
    lines = b.face_lines(st, n, fit, 256, 0.001) > 0
    target = st.lightness()
    for k in (0, 1):
        m = (ids == k) & ~lines
        moved = (L1[m].mean() - L0[m].mean()) / (target[k] - L0[m].mean())
        assert abs(moved - 0.8) < 0.05                         # 80% of the way
        assert L1[m].std() > 0.5 * min(L0[m].std(), 10)       # texture kept
    edged = b.finish(Image.fromarray(pic), st, n, fit, dict(b.DEFAULT_FINISH, edge_width=24))
    e = b.face_lines(st, n, fit, 256, 24) == 255
    assert e.any() and (edged[e] == 0).all()                   # edges solid black
    one = b.Study(400, 600, [(np.array([(0, 0), (400, 0), (400, 600), (0, 600)]), (0, 0, 0))],
                  [(np.array([(0, 280), (400, 330)]), False)])  # a single slanted line across face 13
    width = 2 * cv2.distanceTransform((b.face_lines(one, n, fit, 1024, 24) > 127).astype(np.uint8), cv2.DIST_L2, 5).max()
    assert 20 <= width <= 28


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


def test_board_mixes_finished_paintings_and_the_study(tmp_path):
    st = _study(tmp_path)
    fit = dict(x=0.5, y=0.5, width=0.8)
    store = b.Store(tmp_path / "work")
    store.add_candidate(13, Image.new("RGB", (64, 64), (200, 30, 30)), dict(seed=0))
    store.keep(13, 1)
    tex = b.board(store, st, fit, res=32)
    assert len(tex) == 25
    red = tex[b.FACE_INDEX[13]]
    assert (red[..., 0] > red[..., 2] + 0.2).mean() > 0.5     # still red, after finishing
    top = tex[b.FACE_INDEX[1]]                                 # unkept: the study's light gray
    assert np.allclose(top[..., 0], top[..., 2], atol=0.02) and top.mean() > 0.8
    assert b.contact_sheet(store, 13) is not None and b.contact_sheet(store, 1) is None
    assert b.save_finished(store, st, tmp_path / "fin", fit) == [13]
    assert (tmp_path / "fin" / "face_13.png").exists()
