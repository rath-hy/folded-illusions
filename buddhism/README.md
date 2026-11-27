# The four sights

**Started:** 2026-11-26

## What sparked it

The [same-pixels](../same-pixels/) method paints a new picture from a prompt using exactly the pixels of another image. Pointed at a painting of Manjushri, the bodhisattva of wisdom, it can paint the four sights that set Siddhartha on his path, **an old man, a sick man, a dead man and an ascetic**, each out of the same pixels. Four paintings, one set of pixels, rearranged: impermanence in arithmetic.

## How it works

[`four_sights.ipynb`](four_sights.ipynb) (Colab, GPU) is the same-pixels notebook adapted:

- **Square**, 1440 × 1440 by default.
- **The source is [`manjushri.jpeg`](manjushri.jpeg)**, fetched from this repo.
- **One prompt per sight plus a shared thangka style.** A shared negative prompt steers away from photographs and deity iconography, and each sight has its own negative for its particular failure. The palette is forced to Manjushri's anyway, which does much of the work of matching the style.
- Every result is snapped to the source's exact pixels and checked: every exact RGB color appears exactly as many times as in the source.

## The source

Manjushri seated on a lotus, Rubin Museum of Himalayan Art, object F1997.40.6 (https://rubinmuseum.org/collection/f1997-40-6/). [`manjushri.jpeg`](manjushri.jpeg) is a square crop of the museum's web image, the box (130, 467)–(1708, 2045): from the halo and the flaming sword down to the lotus and leaves, leaving out the brocade border and most of the water. The Rubin's terms for reusing its images haven't been checked yet.

## Status

- **Nov 26:** notebook written; the loading, the projection and the exact snap were tested locally without a GPU (the snap gives an identical histogram).
- **Nov 26, first run:** it worked, but the sights weren't harrowing enough. The style prompt (lotuses, haloes, ornament) pulled every figure toward a serene deity: the old man looked untroubled, the sick man sat calmly, and the dead man was a standing angel in white. The ascetic stood holding a bowl.
- **Nov 26, new prompts**, after a set of Thai temple murals of the four sights: the old man bent over a cane, the sick man curled up in pain on a mat, the dead man a skeleton on the bare ground, the ascetic sitting cross-legged in meditation under a tree. The style prompt no longer mentions lotuses, haloes or ornament; the shared negative rules out halo, throne, crown, deity and angel; each sight's own negative names its failure (upright and serene for the old man, calm and sitting up for the sick man, any living person or clothing for the dead man, standing or walking for the ascetic). Three seeds each. Not yet run.
- **Nov 27, second run:** the old man (seed 1), the dead man (seed 2) and the ascetic (seed 2) worked. The sick man didn't in any seed: "lying on a mat" gave a young, smooth, calm figure reclining as if asleep, and words for pain and sweat don't show in this flat style. The new sick man prompt names things a painter can show, after the Thai mural: a feverish man tossing on a bed, wrapped in a blanket and shivering, two relatives kneeling beside him wiping his brow. His negative rules out beautiful, elegant, young, sleeping peacefully and reclining nude. `PICK` holds the three picks.
- **Nov 27, bigger:** the default size is now 1440 × 1440, about the most SD3.5 Medium handles. At 512 the model was well below its home size of 1024, weak on faces, hands and small details, and the Manjushri crop (1578 across) lost most of its silk and pigment texture when shrunk. On GPUs under 30 GB the image encoder works in tiles. Seeds give different pictures at each size, so the 512 picks (old man 1, dead man 2, ascetic 2) don't carry over and `PICK` is reset.
- **Nov 27, third run (1440):** the old man, the dead man and the ascetic all worked in every seed. The sick man still didn't: the relatives took over the picture, and the sick man ended up standing or sitting up. The new prompt goes back to one person, after Rath's idea: a gaunt man lying alone on a straw mat, his face covered in swollen red boils and sores, coughing up blood. His negative now rules out sitting up, standing and other people. A new setting, `ONLY`, paints only the sights named (now just the sick man, four seeds); the others are read back from the earlier run's `four_sights.zip` if it's uploaded, so the pick cell can still put all four together.
