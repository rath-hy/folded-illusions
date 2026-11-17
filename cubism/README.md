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
