# Cubism on a cube tessellation

**The idea.** Tristan Miller's Cube Tessellation is folded from an 8 × 8 grid into a relief of cubes, each showing three faces.
- **Tilted:** look straight along one cube axis, about 55° off the paper's normal. Two of every cube's faces turn edge-on, and the third tiles one continuous picture.
- **The piece:** one subject from three viewpoints, one per axis. Each tilt resolves one viewpoint.
- **Head-on:** all three viewpoints are cut into planes and interleaved, like a cubist painting.

**What sparked it.** After the crane, we wanted a fold where the views don't fight over the same paper. The visible parts of a cube tessellation split cleanly into three pictures, and seeing one subject from several sides at once is what cubism set out to do.

**[`cubism.ipynb`](cubism.ipynb)** is a virtual demo on the finished model's 25 faces (step 18 of the diagrams) as ideal squares, with no paper and no visibility map.

- **Head-on is the artwork and the key:** the guitar's outline is clearest there. Each face shows guitar only inside the outline.
- **Each face is its own fragment:** a close-up of one part of a 3D guitar from roughly that face's axis, with its own angle, framing and roll: front views on x, side views on y, views down the neck on z. Tilted toward any axis, you see a jumble of fragments.
- **The composition runs without a GPU.** A simple 3D guitar, built from shapes, decides the outline and the fragments, since prompts can't reliably ask for a viewpoint.
- **Diffusion paints it (GPU):** Stable Diffusion XL with a canny ControlNet paints the head-on view in bold colors and sharp geometric figures, following the composition's edges and the cube edges. The painting is then mapped back onto the faces.

**[`explorer/`](explorer/)** is the provenance explorer: a page where you click any face of the painted model to trace it back through the edge lines it followed, the composition fragment it came from, its window onto the 3D guitar, and where it sits in the head-on outline. The notebook's last cell downloads `cubism-painting.zip`; then, from this folder, `python explorer/build_explorer.py OUT_DIR cubism-painting.zip` writes the page and its data to `OUT_DIR`. Without a zip it uses a stand-in painting.

## The Bayon face (Nov 17)

**[`bayon.ipynb`](bayon.ipynb)** turns the piece into an homage to Cambodia. Head-on, the model shows a smiling face of the Bayon, Angkor's temple of faces. Each of the 25 faces is painted separately from its own prompt, so tilting the model breaks the face into scenes of home.

- **The photo sets the composition.** You place the model on it with `FIT`. Faces are numbered 1 to 25, top to bottom and left to right.
- **The guide is one line drawing of the whole photo,** saved to the working folder as `guide.png`. Each face cuts out the part it covers and unshears it into an upright square, so lines run on across face borders. Lines are thinned and redrawn at one width, so they look the same on every face whatever the photo's resolution.
- **Canny doesn't work on this photo.** On weathered stone it finds lichen and block joints, not the face. The default is ControlNet's learned lineart detector, run after edge-preserving smoothing (`smooth=4`, `resolution=384`, `threshold=80`). That gives the crown, eyes, nose and smile, plus some block joints. The guide is a plain image, so the plan is to clean it up by hand.
- **Each face is painted upright,** as you'd see it with the model tilted straight toward that face's axis, turned so its top is as close to up as possible and never mirrored.
- **Painting only adds candidates,** numbered per face, each saved with its prompt, seed and settings. Keeping one copies it aside. Kept faces are never repainted, so you can keep trying prompts on the rest. Everything lives in a Google Drive folder, so it survives Colab disconnects.
- **`bayon.py`** holds the code; [`tests/test_bayon.py`](../tests/test_bayon.py) covers the numbering, the face frames, cutting the guide into faces, and the store.

**How we know:** software only. The whole notebook ran on CPU with the real photo and the real lineart guide, with a stand-in painter (it fills each face with a flat color behind its guide lines): painting, keeping, repainting a kept face (its choice stayed) and the board all work. SDXL itself hasn't run on these guides yet.

The photo isn't committed; we need its source and license recorded before we publish anything made from it.

## The Bayon study (Nov 17, later)

**The photo guide didn't work.** Rath tried it in Colab, and the paintings were bad, mostly because the line drawing was far too detailed. Two attempts at simpler line drawings also failed. Straight lines found automatically still traced the stone. A hand-placed set of lines read as a face but looked like a totem, not cubism.

**So the composition is now a study:** [`bayon-study.svg`](bayon-study.svg), a flat geometric painting of the face in about 90 gray polygons. Rath made it with Google AI Studio. It replaces the photo, so the photo-license question no longer applies to this piece.

- **Cropped** by 1/8 on the left, right and bottom to trim the empty border (`CROP`). The model spans the crop's full width (`FIT` width 1.0), so about 30 units at the top and bottom fall off it.
- **Read as shapes, not pixels.** The polygons are drawn straight into each face's 1024-pixel square, so edges stay crisp. Hatching and the canvas texture are ignored.
- **Each polygon is a chunk with a target shade** from its fill, measured as CIELAB lightness. The study's range is stretched to L* 15–92, so the darkest chunks are dark but can still hold color, and black edges stand out on them.
- **Painting** uses SDXL img2img plus the canny ControlNet. It starts from the face's shades (`strength=0.8`) and follows its edges, drawn thin (4 px) as canny would draw them.
- **Finishing** moves each chunk's average lightness 80% of the way to its target, keeping the painting's colors and texture. Then it draws every edge on top in black, 24 px on a 1024-pixel face. So a chunk can be any color, but it's about as light as the study says. Candidates are saved raw and finished when shown, so `FINISH` can change without repainting.
- **The working folder moved** to `MyDrive/folded-illusions/bayon-study`, so the photo-guide candidates don't mix in.

**How we know:** software only. Tests cover reading and cropping the SVG, which faces an edge lands on, the shades, finishing (80% of the way, texture kept, solid black edges about 24 px wide) and the board. The whole notebook ran on CPU with a stand-in painter (smooth random color), including the paint, keep, repaint and board loop. Head-on, the face reads clearly even with random color in every face. SDXL hasn't run on it yet.

## Obama, as a Hockney-style joiner (Nov 21–22)

**We dropped the Bayon face** and removed its files (`bayon.py`, `bayon.ipynb`, `bayon-study.svg` and its tests). The sections above describe that work as it was; it lives on in the history.

**The subject is now Barack Obama, from a 3D scan.** After several tries (per-face fragments, matching a blurred head-on portrait), the version that works takes after David Hockney's photo joiners. Each of the 25 faces is its own photograph of the scan, with its own angle, zoom and rotation:

- **Loose geography:** each face shows roughly the part of the head that belongs where it sits head-on, so the brow and eyes are up top, the nose is in the middle, and mouths and chin are below.
- **Variety:** faces along one axis share a family of nearby angles (front, side, from below), so tilting shows a joiner from roughly one direction, but no two repeat an angle, and zooms run from an eye filling the face to a small whole head.
- **An oval outline head-on:** each face searches its rotation (any), a small slide and a zoom nudge so that head shows inside an oval and not outside it.
- **A gentle geography term:** features a face shows earn a little for landing near where they belong head-on and cost a little for landing far off.

We picked seed 1. **[`obama_paint.ipynb`](obama_paint.ipynb)** paints it: SDXL with a depth ControlNet paints the scan from seven angles in charcoal with bright presidential colors, after Kehinde Wiley's official portrait, plus a green floral backdrop (jasmine, chrysanthemums, blue lilies). The paintings are wrapped back onto the 3D head, and the joiner photographs the painted head; the backdrop fills wherever a face shows no head, continuous head-on.

**How we know:** software only. The joiner, the oval search and the wrap-back run locally; the wrap-back reproduced the scan's own look when fed the scan's renders as stand-in paintings. The notebook's code parses but hasn't run on a GPU yet.

The joiner code isn't committed yet.

**The scan** is "BARACK OBAMA" by louis on Sketchfab (https://sketchfab.com/3d-models/barack-obama-3010d6b0fdd843b397fa9e98437bc22a), licensed CC BY 4.0, so we can use it with credit. We'd guessed it was the Smithsonian's 2014 scan; it isn't. The author rebuilt it by photogrammetry from 125 frames of a video. The painting guides rendered from it are in [`obama_kit/`](obama_kit), with the credit. The scan itself stays out of the repo (its OBJ is 32 MB, over our limit).

## Painting the Obama joiner (Nov 22)

- **Paintings.** The first run of [`obama_paint.ipynb`](obama_paint.ipynb) gave seven consistent paintings, but realist and smooth rather than charcoal. Rather than repaint, we keep them and geometrize afterwards.
- **Facets.** The head's surface is cut into about 600 straight-edged triangles (denser around the eyes, nose and mouth), each taking the paintings' average color, mixed in at 0.85. Facet outlines are left off. Cutting in 3D keeps every tile and tilt in agreement; cutting each painting in 2D wouldn't.
- **Oil, first try.** A filter-based oil look (wobble, Kuwahara, dragged strokes, canvas) looked computer-made. Dropped.
- **Pointillist oil.** Each of the 25 face textures gets a dot underpainting first, made without a GPU, with dot size set by detail. SDXL then turns the dots into oil daubs. The first run looked good, but kept hard facet outlines and flat fills that a painter working only in dots couldn't make. The edge guide is now weaker and stops after 40% of the steps, and the negative prompt rules out outlines and fills; not yet rerun.
- **Backdrop.** Only about 10% of the backdrop ever shows: a band around the face. Text-to-image twice failed to give the right species, so [`obama_backdrop.ipynb`](obama_backdrop.ipynb) now builds a leaf wall (SDXL and FLUX.1-schnell compared) and pastes real flower photos with open licenses exactly into that band (credits in [`obama_kit/flowers/`](obama_kit/flowers)), then blends with one light pass. Kehinde Wiley's portrait gives the direction (species, colors, density) but is never fed to a model or stored here.

**How we know:** software only. The facets, the dots and the flower collage ran locally and were judged by eye. The oil-step fix and the backdrop notebook haven't run on a GPU yet; their code parses.

## Obama in the cubism notebook (Nov 24)

[`cubism.ipynb`](cubism.ipynb) now shows the Obama joiner instead of the guitar: it fetches the 25 finished faces from [`obama_faces/`](obama_faces/) (oil, clean or dots) and renders head-on, each axis, the tilt grid and `cubism_obama.gif`. It needs no GPU. The guitar version, which the provenance explorer in [`explorer/`](explorer/) reads, is at commit `acd474b` on `main`.

The oil faces are the third run of the pointillist oil section, at `OIL_STRENGTH` 0.42. The first run, at 0.5 with Seurat and Signac named in the prompt, invented long-haired women in profile and lost Obama; the second, at 0.32 with no artist names and figures in the negative prompt, kept Obama but muted the dots. 0.42 keeps him with a little more paint.

[`fold_map/`](fold_map/) has a label sheet for finding the real visibility map: an 8 × 8 grid on A4 with every cell and triangle labeled, to fold once and photograph.
