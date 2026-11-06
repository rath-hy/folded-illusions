"""The file rules from CONTRIBUTING.md."""
import subprocess
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_MB = 30
MAX_NOTEBOOK_MB = 10
CHECKED_IMAGES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [ROOT / name for name in out.decode().split("\0") if name]


def names(paths):
    return [str(p.relative_to(ROOT)) for p in paths]


def test_files_are_small_enough():
    big = [p for p in tracked_files() if p.exists() and p.stat().st_size > MAX_FILE_MB * 10 ** 6]
    assert not big, f"over {MAX_FILE_MB} MB: {names(big)}"


def test_notebooks_are_small_enough():
    big = [p for p in tracked_files() if p.suffix == ".ipynb" and p.exists()
           and p.stat().st_size > MAX_NOTEBOOK_MB * 10 ** 6]
    assert not big, f"over {MAX_NOTEBOOK_MB} MB, clear some outputs: {names(big)}"


def test_photos_have_no_location_data():
    found = []
    for p in tracked_files():
        if p.suffix.lower() in CHECKED_IMAGES and p.exists():
            with Image.open(p) as image:
                if image.getexif().get_ifd(0x8825):
                    found.append(p)
    assert not found, f"strip GPS with exiftool -gps:all= (see CONTRIBUTING.md): {names(found)}"


@pytest.mark.parametrize("name", ["fold-map/prints/charuco_board_a4.pdf", "fold-map/charuco pics/holbein_cube.JPG"])
def test_key_files_exist(name):
    assert (ROOT / name).exists()
