# The four sights

**Started:** 2026-11-26

## What sparked it

The [same-pixels](../same-pixels/) method paints a new picture from a prompt using exactly the pixels of another image. Pointed at a thangka of the Buddha, it can paint the four sights that set Siddhartha on his path, **an old man, a sick man, a dead man and an ascetic**, each out of the Buddha's own pixels. Four paintings, one set of pixels, rearranged: impermanence in arithmetic.

## How it works

[`four_sights.ipynb`](four_sights.ipynb) (Colab, GPU) is the same-pixels notebook adapted:

- **Square** (512 × 512). The 2:3 thangka is squeezed to the square by default rather than cropped: the sights only use its colors, so the distortion doesn't matter, and squeezing keeps the trees, sky and offerings in the palette. `FIT = "crop"` uses the middle square instead.
- **The source is the thangka**, downloaded from where it was found, so nothing is uploaded and the painting isn't stored here.
- **One prompt per sight plus a shared thangka style**, and a negative prompt against photographs and 3D renders, so the four are representational, like the reference. The palette is forced to the thangka's anyway, which does much of the work.
- Every result is snapped to the thangka's exact pixels and checked: every exact RGB color appears exactly as many times as in the source.

## The source

`starting_image.avif` is a thangka of Shakyamuni Buddha with an alms bowl, two monks at his sides. It was downloaded from `cdn.sanity.io/images/caw6lfob/production/2bea4574…-2202x3259.jpg` (a Sanity CMS image; the site and painter aren't recorded yet). Its license is unknown, so it stays out of the repo until we know who painted it and whether we can use it.

## Status

- **Nov 26:** notebook written. The source download, the resize to 512 × 512, the projection and the exact snap were tested locally without a GPU (the snap gives an identical histogram). The diffusion itself hasn't run yet.
