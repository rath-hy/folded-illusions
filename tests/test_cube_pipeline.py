"""The cube anamorphosis pipeline on the real photo, from markers to a print file."""
import sys
from pathlib import Path

import numpy as np
import pytest

FOLD_MAP = Path(__file__).resolve().parent.parent / "fold-map"
sys.path.insert(0, str(FOLD_MAP))

import photo_map as pm  # noqa: E402


@pytest.fixture(scope="module")
def cube():
    board = pm.make_board()
    photo = pm.load_photo(FOLD_MAP / "charuco pics" / "holbein_cube.JPG")
    markers = pm.detect(photo, board)
    faces = pm.fit_faces(photo, markers, board, pm.find_faces(markers, board))
    return board, photo, markers, faces


def test_markers_are_found(cube):
    _, _, markers, _ = cube
    assert len(markers) >= 150


def test_faces_are_three_squares_around_the_sheet_center(cube):
    _, _, _, faces = cube
    assert {name: f["box"] for name, f in faces.items()} == {
        "top": (100.0, 50.0, 50.0), "left": (50.0, 50.0, 50.0), "right": (100.0, 100.0, 50.0)}


def test_maps_predict_points_they_were_not_fitted_to(cube):
    board, _, _, faces = cube
    for name, f in faces.items():
        scale = pm._px_per_mm(f["warp"]["H"], board, f["ids"])
        rms_px, _ = pm.holdout_px(f["P"], f["Q"], 30)
        assert rms_px / scale < 0.4, f"{name}: {rms_px / scale:.2f} mm"


def test_sheet_is_printed_only_on_the_faces_and_bleed(cube):
    board, _, _, faces = cube
    picture = np.full((8, 8, 3), 0.5, np.float32)
    sheet, printed, _ = pm.project_onto_faces(picture, faces, board, dpi=100, bleed_mm=4)
    area = printed.mean() * board.sheet_mm ** 2
    assert 3 * 50 ** 2 < area < 3 * 58 ** 2
    assert np.all(sheet[~printed] == 1)


def test_print_file_is_written(cube, tmp_path):
    board, _, _, faces = cube
    picture = np.full((8, 8, 3), 0.5, np.float32)
    sheet, _, _ = pm.project_onto_faces(picture, faces, board, dpi=100)
    path = tmp_path / "print.pdf"
    pm.save_print(pm.add_fold_ticks(sheet, faces, board), path, board, note="test")
    assert path.stat().st_size > 1000
