# Fold map

**Started:** 2026-10-31

## What sparked it

A possible CS180 project: printing images on paper so they show up on a folded origami model, or hiding images in places only visible when it's unfolded. Before any of that, I need to know which parts of the sheet are visible once it's folded, and where each visible point ends up.

## How it works

[`fold_map.py`](fold_map.py) takes a crease pattern and a folded state from [Flat-Folder](https://github.com/origamimagiro/flat-folder) (Jason Ku's solver for flat-folded states):

```
python fold_map.py 004_traditional_Crane_cp.fold 004_traditional_Crane_state.fold
```

1. **Positions:** each face of the crease pattern is a rigid piece. Comparing its corners on the sheet and in the folded state gives the map between the two, and tells you whether the piece was flipped over.
2. **Stacking:** the folded state's `faceOrders` say which of two overlapping faces is on top. In the FOLD spec, `[f, g, s]` is relative to **g's** normal, so the sign flips for faces that were turned over.
3. **Visibility:** a point of the sheet is visible from the front if its face is the bottommost layer where it lands (Flat-Folder's view looks from −z) and from the back if it's the topmost.

Outputs:

- `visibility_sheet.png`: both sides of the sheet, colored hidden / visible from the front / visible from the back.
- `folded_views.png`: the folded model from both sides, textured by where each visible point sits on the sheet.
- `fold_map.npz`: for each pixel of each folded view, the sheet point it shows and which side, for splicing images later.

Conventions follow Flat-Folder's display: side 1 is the side you see in its crease pattern view, the front is its folded view, and the back is its folded view with Flip checked.

## Crane results

The traditional crane (Flat-Folder example `004`; 72 faces and 5 valid folded states; this uses the state that was exported):

- **Side 1 is entirely hidden.** Fold from one-sided paper with the color on side 2 and no white shows, which matches real cranes.
- **Side 2:** 11.8% of it is visible from the front, 11.8% from the back, and 76.4% is hidden.
- **Visible areas don't follow the creases.** Some boundaries cut across faces, wherever another layer's edge covers part of a face.

## Checking it against a real crane

I folded a crane from notebook paper following [Rachel Katz's diagrams](https://origamiwithrachelkatz.oriland.com/folding/crane.php), stopping before opening the wings so it stayed flat. I shaded everything visible on one face with straight pencil lines and on the other with circles, then unfolded it. The shading matches the predicted regions in `visibility_sheet.png` (side 2). Photos are in [`crane_pics/`](crane_pics/), with location metadata removed.

## Which way is up

[`directions.py`](directions.py) draws a letter F on each visible region of side 2, oriented the way a picture must be printed there to appear upright on the folded crane (`directions.png`). For the crane, all 18 visible regions only need rotating (by multiples of 22.5°), never mirroring, since you see the same side of the paper you print on.

## Decal: print a picture that appears on the folded crane

```
python decal.py                                   # test pictures: a big F and B
python decal.py --front cat.jpg --back dog.jpg    # your own pictures
```

Each picture is center-cropped to the folded silhouette (about 1.3:1, landscape, for the crane) and mapped onto the sheet point by point: every visible point gets the color of the picture at the place it lands when folded. Only visible regions are printed, plus a 1% bleed (`--bleed`) so imperfect folds don't show white slivers. Tiny letters A–D in hidden areas mark the corners, and the preview shows where each lands (crane: A tail, C head, B and D wings). Writes `decal_side2.png` (8 × 8 inches at 300 dpi; also `decal_side1.png` for models that show both sides) and `decal_preview.png` (the model simulated from the printed sheets).

## Notebook

[`fold_map.ipynb`](fold_map.ipynb) walks through the whole workflow for any flat-foldable model from Flat-Folder: where each face lands, the layer stacking, the visibility map, which way is up, and the decal, with uploads for your own model and pictures on Colab. It imports the code from this folder (cloning the repo on Colab), and ends with a list of building blocks for further experiments.

## Illusions: one picture flat, another folded

[`illusions.ipynb`](illusions.ipynb) and [`illusion.py`](illusion.py) make a sheet that shows one picture when flat and other pictures on the folded crane's front and back. `FoldViews` turns the fold map into pixel maps between the printed side and the folded views (98–99% one-to-one at 256 × 256, since folding is rigid). On top of that:

- **Composite (CPU):** the flat picture in hidden areas and the folded pictures in visible ones. The baseline; the flat picture ends up shattered.
- **Generated (GPU):** Visual Anagrams (Geng, Park, Owens, CVPR 2024) adapted to folding, using DeepFloyd IF. At every denoising step the model sees the sheet flat and folded, each with its own prompt, and the predictions are mapped back and blended per pixel. The folded views are drawn on a plain backdrop held at the current noise level.

The geometry and the sampler's bookkeeping were tested on CPU with a stand-in model, and the notebook runs on a free Colab GPU.

### Trial 1

A city street at night (oil painting) flat, a red fox on one wing and a white owl on the other, which the code calls the front and back views ([`prints/`](prints/): the generated sheet, the A4 print file with two copies, and the virtual fold). Folded on the computer, both animals appear. Printed on a home printer and folded by hand ([`crane_pics/trial_1/`](crane_pics/trial_1/), including a timelapse), the fox comes through but the owl is barely visible: the print is darker and muddier than the screen, white crease lines cut across the picture, and the folds aren't exact. Bolder, simpler, higher-contrast pictures should survive better.

## Folding story: a picture at every stage

[`story.ipynb`](story.ipynb) and [`story.py`](story.py) make one sheet that shows a different picture at each stage of folding. By default the story has three parts: flat, bird base, crane (for example leaves, a chrysalis, a butterfly). The preliminary base is available too.

- **The stages.** [`make_stages.py`](make_stages.py) writes crease patterns for the two bases on the crane's sheet, with creases that match the crane's, and solves them with Flat-Folder's own solver, run outside the browser by [`ff_solve.mjs`](ff_solve.mjs). That script reproduces the crane state exported from the website exactly. The results are in `stages/`. Both bases leave the BD diagonal unfolded; folding it doubles each base over.
- **Digits (CPU).** `python story.py` places 0, 1, 2 (or 0 to 3 with `--stages "preliminary base" "bird base" crane`) so they compete for as few pixels as possible. It shows the core difficulty: each face of the preliminary base is an uncut quarter of the sheet, so its picture is readable flat. The bird base shows 2 pieces of paper, the crane about 10.
- **Generated (GPU).** The DeepFloyd sampler in `illusion.py` now takes any list of folded views, from any models. The notebook runs it with the flat sheet plus each stage, each with its own prompt, so later pictures can hide in the texture of earlier ones.
- **First GPU run.** With the digits 1, 2, 3 (pencil drawings on a "geometric mess"), the 1 vanished into the flat scene, and the 2 and 3 blurred into each other. The reason: consecutive stages show a lot of the same paper, because folding keeps much of the outside on the outside. With the sides used, 53% of the crane's visible paper also showed on the bird base. The notebook now picks sides by how much of each stage's own paper is shared (`least_shared_sides`; for the crane, the finished crane shares none and the two bases share 41% and 50%), and the next default story was a bud opening into a flower.
- **Second GPU run.** The flowers came out as roughly the same pink flower at every stage, so the story had no distinct beats; the flat sheet showed the preliminary base's flower in its corner; and the bird base drew a bud on a stem whatever it was asked for, because that fits its tall diamond. Lessons: give each stage a distinct subject and color, and fit each subject to its outline.
- **Three parts by default.** The preliminary base is now left out: its picture can't hide (one uncut piece), and it shared half its paper with the bird base. Without it, the bird base and crane share no paper, and the flat picture has 67% of the sheet to itself (53% before). The default story is leaves, a hanging chrysalis (the bird base's tall diamond), and a monarch butterfly (the crane's wings). What makes a stage good is scrambling the paper, not being small.

### More pixels for the crane

At stage 1 (64 × 64), the crane shows only 489 pixels of the sheet (its body is about 16 × 22), so its picture is composed in a tiny patch, and printing bigger doesn't change that. [`story_more_pixels.ipynb`](story_more_pixels.ipynb) tests two fixes on a new story (an aerial view of a home planet, a rocket on the bird base, a ringed planet on the crane):

- **Stage 2 redraws the folded pictures.** `generate` now takes a stage-2 noise level per view, so the folded views can be redrawn at 256 pixels while the flat view keeps stage 1's composition.
- **Stage 1 at 128.** `generate` takes stage-1 views at any resolution. The crane gets 1,925 pixels instead of 489, but the model was trained at 64.

Round 1: letting stage 2 redraw changed almost nothing, because every pixel a folded view shows is also in the flat view, which holds it at stage 1's picture. Stage 1 at 128 made the planet a clear round shape on the crane and the landscape richer, but split the rocket into two small ones (the model draws objects at the size it learned at 64). And asking for subjects "on a plain light background" backfired: the folded views came out mostly pale, which printed white patches framing the rockets on the flat sheet. Round 2 drops that phrase and compares prompts where the subject fills the paper, including surroundings made of the flat scene's colors (orange exhaust, blue clouds), plus 96 pixels and a mid-gray backdrop.

### Other models

To compare models, I measured Flat-Folder examples by how much paper each finished model shows per view and how scrambled it is. The crane shows 12% of the sheet per view, in 4 sizable pieces, the largest a quarter of the view. Yakko shows 17% in 8, the largest an eighth, and simple models like the pinwheel show 30–40% but in a few big pieces, which would be readable flat. Yakko can't be used for the folding story yet: keeping each crease's mountain or valley, no subset of Flat-Folder's Yakko creases folds flat except the whole model (and the sheet with its corner tips folded in), so its intermediate stages would have to be built from its actual folding sequence.

## Fold maps from photos

The bird base turned out to be a poor story stage: its two pieces are big uncut halves of the sheet. To choose stages by folding rather than by what a crease pattern can describe, [`photo_map.py`](photo_map.py) and [`photo_map.ipynb`](photo_map.ipynb) measure fold maps from photos instead:

1. Print a black-and-white ChArUco board ([`prints/charuco_board_a4.pdf`](prints/charuco_board_a4.pdf)): OpenCV's camera calibration target, a checkerboard with a numbered marker in each white square. 44 × 44 squares of 4.5 mm on a 200 mm sheet, 968 markers.
2. Fold it, and photograph each view of each stage you want, straight on, against a colored backdrop.
3. Each marker found says which part of the sheet that patch is and how it's turned. Markers that agree on one transform form a rigid piece (RANSAC), and every paper pixel goes to the piece whose transform best predicts the board's pattern there.

Tested on photos simulated from the known crane maps (with blur, noise, uneven light and shadows along layer edges): about 95% of the crane's visible paper is recovered to within 0.5 mm, about 1% is wrong, and the rest is lost at the neck and tail tips and along seams. The bases come out at 99%. The number of squares per sheet matters most: with 24 a side, 44% of the crane is lost. The camera needs at least 6 pixels per mm of paper, which a phone gives easily.

**Straight seams.** Pieces of a flat fold meet along straight lines, but the matching window straddles them, so seams come out ragged. `straighten` fits lines to the boundaries between neighboring pieces with RANSAC, redraws pixels to the correct side, and fills narrow gaps where the predicted pattern matches the photo. On 7 Flat-Folder models it wasn't developed on (11 views), it took paper within 0.5 mm from 96.5% to 97.6%, off by over 1 mm from 1.17% to 0.61%, and missing from 2.37% to 1.82%. The crane front goes from 95.1 / 1.32 / 3.56 to 96.1 / 0.98 / 2.90. Tips with no whole marker stay missing, and it's simulation only so far. My first idea was to compute seams from the two pieces' transforms, as creases, but a crease flips the paper, so two pieces showing the same side of a flat fold aren't joined by one, and the transforms disagree all along the seam.

**Print size.** The board and the story prints are 200 mm squares, one per A4 page (`illusion.print_page`), about as big as fits inside a home printer's margins. Folding scales with the paper, so a bigger sheet doesn't give a piece more markers, but it makes a millimeter of folding error a smaller share of each piece. Calibrate with the same size you print.

## Pictures on a fold, from a photo

The Ambassadors idea, with a first real photo: I folded the board into a sanbo (a little box), photographed it, and marked its floor in magenta on a copy. Section 6 of [`photo_map.ipynb`](photo_map.ipynb) works out which sheet points show inside the marked area and prints each one with the picture's color at the place it lands in the photo. The output is an A4 PDF with the 200 mm sheet inside a black cut line and the waste hatched ([`prints/box_holbein_a4.pdf`](prints/box_holbein_a4.pdf), Holbein's self-portrait on the floor).

On the real photo, 149 markers were found, about 60 pixels per side. The floor is one rigid piece, fitted to about 0.15 mm, and 88% of the marked area matched directly. The rest (glare, and paper curving up into the walls) borrows the floor's transform. The photo is taken from straight above a flat floor, so the picture lands undistorted in the middle of the sheet: this tests the pipeline, and isn't anamorphic yet. That needs an angled view, or an area spanning several facets. On simulated photos with a known camera tilt, bases with big pieces stay within 0.5 mm up to 60°. The crane loses about 10% of a marked area to slivers up to 45°, and 43% at 60°. The print itself hasn't been folded yet.

## A portrait across three faces of a cube

The first real anamorphosis: [`cube_anamorphosis.ipynb`](cube_anamorphosis.ipynb) traces it step by step. I folded the board into Kunihiko Kasahara's seamless cube and photographed one corner. No marking this time, because the three visible faces are the canvas. They turn out to be three 50 mm squares of the sheet that meet at its center, so the middle of the sheet becomes the corner facing the camera.

Real faces aren't flat. A flat fit per face misses by 7 to 13 pixels, so each face gets a homography plus a thin-plate spline fitted to its markers and the checker corners between them, which brings points it wasn't fitted to down to about 0.1 to 0.3 mm. The output, [`prints/cube_holbein_a4.pdf`](prints/cube_holbein_a4.pdf), is Holbein's self-portrait skewed into an L, with 4 mm of bleed past each edge. Fold it, look from about 18 cm with one eye, and it should straighten out. I printed and folded it, and the result looks pretty satisfactory by eye. The bleed gave useful wiggle room. The error hasn't been measured yet, and one fold can't show how well the board's bow repeats. The right face was also at least a third dimmer in the photo's light, so it needs even lighting.

## Next

- Splice an image onto the visible regions using `fold_map.npz`.
- Compare a fold map measured from photos to the computed map.
