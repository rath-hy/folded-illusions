# The 25 face textures

The panels of the cube model for the Obama joiner (seed 1), numbered in the order the model's faces are listed, 1024 pixels square.

- **`clean/`**: the geometric faces. The SDXL paintings are wrapped onto the head and cut into about 600 flat facets (0.85 strength), and the floral backdrop is cut into triangles too.
- **`dots/`**: the same faces as a pointillist underpainting, made without a GPU. Dots run from 2.5 to 20 pixels in radius: small where the face is busy, large on flat facets and leaves. They overlap densely (each spot is covered about 4.5 times), so no flat ground is left to read as a fill.

[`../obama_paint.ipynb`](../obama_paint.ipynb) restyles `dots/` into pointillist oil, guided by the edges of `clean/`.

The head comes from "BARACK OBAMA" by louis on Sketchfab, CC BY 4.0; see [`../obama_kit/README.md`](../obama_kit/README.md). The paintings and backdrop are our own SDXL output.

These were made with the backdrop from before the flower step. Once the flowers are painted, these get remade.

**Nov 22, remade.** The faces now use the backdrop from [`../obama_backdrop.ipynb`](../obama_backdrop.ipynb): an SDXL leaf wall with the photographed chrysanthemums, jasmine and agapanthus placed on it and blended in. The backdrop is faceted lightly (2,500 small triangles mixed 0.75 over the painting, against the head's 0.85), so it echoes the head while the small jasmine flowers stay white. Their photo credits are in [`../obama_kit/flowers/README.md`](../obama_kit/flowers/README.md).
