"""Build the data for the provenance explorer (explorer.html) from cubism.ipynb and a painting.

    python build_explorer.py OUT_DIR [cubism-painting.zip]

Runs the notebook's own geometry and composition code with the settings in the zip, so the composition
matches the one the painting was made from. Without a zip, it uses the default settings and a stand-in
painting (bold colors laid flat over the composition), marked as a stand-in on the page.

Writes to OUT_DIR: explorer.html, data.json, and pictures: one atlas per layer (painting, edges,
composition; a 5 x 5 grid of face textures) and the head-on painting, edges and composition.
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parent / "cubism.ipynb"
TILE, GRID, HEAD_ON_PX = 128, 5, 512


def notebook_code():
    """The notebook's geometry, composition and diffusion helpers, without its plotting."""
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]
    pick = lambda marker: next(c for c in cells if marker in c)
    helpers = pick("def control_image")
    return "\n".join([pick("def render("), pick("def cubist_textures("), helpers[:helpers.index("\ncontrol, head_on")]])


def body_outline(guitar, n=160):
    """The body's outline in the guitar's (u, v) plane, by bisecting along rays from inside the waist."""
    cu = -0.55
    pts = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        lo, hi = 0.0, 3.0
        for _ in range(30):
            mid = (lo + hi) / 2
            inside = guitar(np.array([cu + mid * np.cos(a)]), np.array([mid * np.sin(a)]), np.array([0.0]))[0] != 0
            lo, hi = (mid, hi) if inside else (lo, mid)
        pts.append([round(cu + lo * np.cos(a), 4), round(lo * np.sin(a), 4)])
    return pts


def stand_in(head_on, control, face_id):
    """Bold flat colors over the composition: each face its own rotation of a synthetic-cubist palette,
    by the composition's brightness, with the control lines in black. Only until the real painting."""
    palette = np.array([[214, 48, 32], [28, 70, 190], [242, 196, 30], [24, 140, 90], [245, 242, 232], [30, 28, 30]])
    lum = head_on.mean(-1) / 255
    level = np.digitize(lum, [0.3, 0.45, 0.6, 0.75])
    out = palette[(level + face_id * 2) % len(palette)]
    out[face_id < 0] = 255
    out[np.asarray(control)[..., 0] > 0] = palette[5]
    return out.astype(np.uint8)


def atlas(textures):
    out = np.zeros((GRID * TILE, GRID * TILE, 3), np.uint8)
    for i, t in enumerate(textures):
        img = Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8)).resize((TILE, TILE), Image.LANCZOS)
        r, c = divmod(i, GRID)
        out[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = np.asarray(img)
    return Image.fromarray(out)


def main(out_dir, painting_zip=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ns = {}
    exec(notebook_code(), ns)
    settings = dict(JUMBLE=(35, 35, 35), ZOOM=(0.5, 1.1), ROLL=25, SEED=0)
    painting = None
    if painting_zip:
        with zipfile.ZipFile(painting_zip) as z:
            settings.update(json.loads(z.read("settings.json")))
            painting = Image.open(z.open("painting.png")).convert("RGB")
    recipes = []
    comp = ns["cubist_textures"](jumble=tuple(settings["JUMBLE"]), zoom=tuple(settings["ZOOM"]), roll=settings["ROLL"],
                                 seed=settings["SEED"], recipes=recipes)
    control, head_on = ns["control_image"](comp)
    size = ns["CONTROL_SIZE"]
    fs = ns["faces"]()
    ids = [np.full((2, 2, 3), (i + 1) / 64) for i in range(len(fs))]
    face_id = np.rint(ns["render"](textures=ids, size=size, margin=ns["CONTROL_MARGIN"], bg=0.0)[..., 0] * 64).astype(int) - 1
    if painting is None:
        painting = Image.fromarray(stand_in(head_on, control, face_id))
    painting = painting.resize((size, size), Image.LANCZOS)

    to_tex = ns["textures_from_head_on"]
    layers = {"painting": to_tex(painting), "edges": to_tex(control), "composition": comp}
    for name, tex in layers.items():
        atlas(tex).save(out / f"atlas_{name}.png", optimize=True)
    for name, img in {"painting": painting, "edges": control, "composition": Image.fromarray(head_on)}.items():
        img.resize((HEAD_ON_PX, HEAD_ON_PX), Image.LANCZOS).save(out / f"head_on_{name}.png", optimize=True)

    # Where each face sits in the head-on pictures, in their pixels.
    _, r, u = cam = ns["camera"](0.0)
    x0, x1, y0, y1 = ns["bounds"](cam, fs)
    span = max(x1 - x0, y1 - y0) * (1 + 2 * ns["CONTROL_MARGIN"])
    to_px = lambda p: [round((((p @ r) - (x0 + x1) / 2) / span + 0.5) * HEAD_ON_PX, 2),
                       round((((y0 + y1) / 2 - (p @ u)) / span + 0.5) * HEAD_ON_PX, 2)]
    faces_out = []
    for i, ((k, c, e1, e2), rec) in enumerate(zip(fs, recipes)):
        corners = [c, c + e1, c + e1 + e2, c + e2]
        faces_out.append(dict(id=i, corners=[p.tolist() for p in corners], head_on=[to_px(p) for p in corners], **rec))

    U, V, W = ns["frame_from"]((1, 0, 0), (0, -1, 2))
    data = dict(
        stand_in=painting_zip is None,
        settings={k: v for k, v in settings.items()},
        faces=faces_out, tile=TILE, grid=GRID, head_on_px=HEAD_ON_PX,
        guitar=dict(frame=[U.tolist(), V.tolist(), W.tolist()], body=body_outline(ns["guitar"]), body_w=[-0.25, 0.25],
                    boxes=[dict(name="neck", u=[0.7, 3.0], v=[-0.14, 0.14], w=[-0.05, 0.25]),
                           dict(name="headstock", u=[3.0, 3.7], v=[-0.24, 0.24], w=[0.0, 0.15]),
                           dict(name="bridge", u=[-1.33, -1.17], v=[-0.42, 0.42], w=[0.2, 0.32])],
                    soundhole=dict(u=-0.3, v=0.0, r=0.3, w=0.25),
                    features={k: list(v) for k, v in ns["FEATURES"].items()}),
        lit=list(ns["LIT"]), head_on_up=ns["HEAD_ON_UP"].tolist())
    (out / "data.json").write_text(json.dumps(data))
    shutil.copy(HERE / "explorer.html", out / "explorer.html")
    print(f"wrote {out}: {len(faces_out)} faces, {'stand-in' if painting_zip is None else 'real'} painting")


if __name__ == "__main__":
    main(*sys.argv[1:])
