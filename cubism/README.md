# Cubism on a cube tessellation

**The idea.** Tristan Miller's Cube Tessellation is folded from an 8 × 8 grid into a relief of cubes, each showing three faces.
- **Tilted:** look straight along one cube axis, about 55° off the paper's normal. Two of every cube's faces turn edge-on, and the third tiles one continuous picture.
- **The piece:** one subject from three viewpoints, one per axis. Each tilt resolves one viewpoint.
- **Head-on:** all three viewpoints are cut into planes and interleaved, like a cubist painting.

**What sparked it.** After the crane, we wanted a fold where the views don't fight over the same paper. The visible parts of a cube tessellation split cleanly into three pictures, and seeing one subject from several sides at once is what cubism set out to do.

**[`cubism.ipynb`](cubism.ipynb)** is a quick virtual demo. It paints the finished model's 25 faces (step 18 of the diagrams) as ideal squares, with no paper and no visibility map. Each viewpoint is generated alone with plain DeepFloyd, painted onto the faces and rendered from head-on to each tilt. The geometry runs without a GPU.

Since prompts can't reliably ask for a viewpoint, the notebook also builds a simple 3D guitar. Head-on is the key: every face shows guitar only inside the guitar's head-on outline, so head-on the outline reads clearly. Inside, each face is its own fragment, a close-up of one part of the guitar from roughly that face's axis, with its own angle, framing and roll: front views on x, side views on y, views down the neck on z. Tilted toward any axis, you see a jumble of fragments. It's painted in flat Braque tones, with no diffusion and no GPU, to test the composition first.
