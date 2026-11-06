# Contributing

## Workflow

1. Pick or open an issue, and assign yourself so we don't double up.
2. Branch from `main`, named after the trick and the work: `anamorphosis/measure-cube-error`, `story/hiding-loss`.
3. Open a pull request early, as a draft if you like. Link the issue with `Closes #12`.
4. Merge once tests pass and a teammate has looked at it. We squash-merge.

Small fixes to docs can skip the review, but still go through a pull request so tests run.

## Say how you know

Every result should say what it was tested on, because these are very different claims:

- **Real paper:** printed, folded, and photographed.
- **Software:** the actual pipeline, with no physical fold.
- **Simulation:** a simulated photo or fold.

Say how many folds or photos it's based on, and whether it's measured or judged by eye. Report what failed too.

## Experiments

Record each experiment in an issue or pull request: what you changed, what you expected, what happened, and a picture. When it changes what the team should do next, add a few lines to the folder's README. READMEs are diaries, so add to them rather than rewriting, and correct anything that has become false.

## Notebooks

- Save outputs only in notebooks meant to be read on GitHub. Clear them everywhere else, because outputs make diffs unreadable.
- Two people editing one notebook will conflict, so say so on the issue first. Move anything reusable into a `.py` module.
- Keep each notebook under 10 MB. Tests enforce this.

## Files

- **Strip location data from photos.** Phone photos carry GPS. Tests fail on JPEG, PNG or TIFF files with GPS tags, but they can't check other formats such as HEIC or video. To strip them, run `exiftool -gps:all= -overwrite_original photo.jpg`.
- **Keep files under 30 MB.** Tests enforce this. History is permanent, so think before committing large files.
- **Note where images come from.** We may publish this work, so anything we didn't make needs its source and license recorded.
- **Keep generated prints if they matter.** Commit the PDF for anything you physically folded, so results can be traced to what was printed.

## Tests

Run `pytest` from the repo root before pushing. Tests also run on every pull request, without a GPU. They cover the cube pipeline on the real photo, plus the file rules above.
