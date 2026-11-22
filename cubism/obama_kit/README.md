# Guides for painting the Obama head

For each of seven views (front, two three-quarters, profile, below, below-side, above): a render of the scan to start painting from (`*_photo.png`), a depth map for the depth ControlNet (`*_depth.png`, white is near) and a mask of the head (`*_mask.png`). `cameras.json` holds each view's angle and the shared framing, so paintings can be wrapped back onto the head. [`../obama_paint.ipynb`](../obama_paint.ipynb) loads them from here.

## Source and license

Rendered from "BARACK OBAMA" by louis (https://sketchfab.com/louis), published on Sketchfab on June 8, 2015: https://sketchfab.com/3d-models/barack-obama-3010d6b0fdd843b397fa9e98437bc22a. Licensed under Creative Commons Attribution 4.0 (https://creativecommons.org/licenses/by/4.0/).

Changes: we cut away the base and back of the head, reoriented and rescaled it, and rendered these views with our own orthographic renderer.

Per the author, the model is a photogrammetry reconstruction from 125 frames of a video, not the Smithsonian's official 3D portrait.
