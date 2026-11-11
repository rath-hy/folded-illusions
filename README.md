# Folded Illusions

Our CS180 final project: origami as a series of illusions, each using a different trick revealed by folding. We're aiming for art, a research paper, and an exhibition.

Team onboarding doc: [Folded Illusions: Team Onboarding](https://claude.ai/artifact/STZNhmimTjpEWrkHPqANdj). Start there for the ideas and open questions. This repo holds the code, photos and print files.

## Tricks

| Trick | Demo | Code | Status |
| --- | --- | --- | --- |
| Multi-view diffusion | Crane: one picture flat, another on each wing | `fold-map/illusions.ipynb`, `fold-map/factorized_illusions.ipynb`, `fold-map/ambiguous_illusions.ipynb` | One print trial: the fox read well, the owl barely showed. Hiding the wings' pictures failed. Ambiguous pictures that folding resolves work better: arcimboldo gave a cicada that belongs to the flat scene, and with the XL model the plain average gave the final print: a koi and a red sun over Tokyo |
| Folding story | A new picture at each folding stage | `fold-map/story.ipynb`, `fold-map/story_more_pixels.ipynb` | Pipeline runs, no satisfying result yet |
| Anamorphosis | Holbein's face centered on a cube corner | `fold-map/cube_anamorphosis.ipynb` | Printed and folded; looks good by eye, error not measured |
| Same pixels | Buddhist imagery on a fortune teller | `same-pixels/same_pixels.ipynb` | Method works on flat images, nothing folded yet |
| Visual cryptography | Schrödinger's box on the seamless cube | `schrodingers-box/schrodingers_box.ipynb` | Notebook covers an older hinged-box design; needs rebuilding |

## Layout

```text
fold-map/           visibility maps, diffusion on folds, anamorphosis
  README.md         the project's full history, oldest first
  photo_map.py      visibility maps from photos of a folded ChArUco board
  fold_map.py       visibility maps from crease patterns
  illusion.py       views as pixel maps, and the DeepFloyd sampler
  story.py          multi-stage views
  charuco pics/     input photos and pictures
  crane_pics/       photos of real folds
  prints/           A4 PDFs, including the ChArUco board
same-pixels/        same-pixels method (Stable Diffusion 3.5)
schrodingers-box/   visual cryptography
tests/              smoke tests and repo rules, run on every pull request
```

The three project folders don't share code. Run notebooks from inside their own folder.

## Get started

```bash
git clone git@github.com:rath-hy/folded-illusions.git
cd folded-illusions
pip install -r requirements.txt
pytest
cd fold-map
jupyter notebook cube_anamorphosis.ipynb
```

`cube_anamorphosis.ipynb` is the best place to start: it needs no GPU, its outputs are saved, and it traces one trick from photo to print file. `schrodingers-box/schrodingers_box.ipynb` also has saved outputs. The other notebooks open blank until you run them.

## Diffusion notebooks on Colab

These need a GPU, and they use Hugging Face models you have to accept first.

1. Open a notebook in Colab: [illusions](https://colab.research.google.com/github/rath-hy/folded-illusions/blob/main/fold-map/illusions.ipynb), [factorized_illusions](https://colab.research.google.com/github/rath-hy/folded-illusions/blob/main/fold-map/factorized_illusions.ipynb), [ambiguous_illusions](https://colab.research.google.com/github/rath-hy/folded-illusions/blob/main/fold-map/ambiguous_illusions.ipynb), [story](https://colab.research.google.com/github/rath-hy/folded-illusions/blob/main/fold-map/story.ipynb), [story_more_pixels](https://colab.research.google.com/github/rath-hy/folded-illusions/blob/main/fold-map/story_more_pixels.ipynb), or [same_pixels](https://colab.research.google.com/github/rath-hy/folded-illusions/blob/main/same-pixels/same_pixels.ipynb). For another branch, replace `main` in the link. The setup cell clones `main`, so a notebook relying on unmerged code needs its branch cloned instead.
2. Choose Runtime → Change runtime type → T4 GPU.
3. On Hugging Face, accept the licenses for [IF-I-M](https://huggingface.co/DeepFloyd/IF-I-M-v1.0) and [IF-II-M](https://huggingface.co/DeepFloyd/IF-II-M-v1.0), or [SD 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) for `same_pixels`.
4. Create a Hugging Face read token, then choose Runtime → Run all and paste it when asked.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## History

This code started in [rath-hy/doodles](https://github.com/rath-hy/doodles), which keeps the earlier commit history. This repo is a snapshot of its three project folders.

## License

The code is under the [MIT License](LICENSE). Images by others keep their own terms; see CONTRIBUTING.md on recording sources.
