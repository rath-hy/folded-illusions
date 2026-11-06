"""Crease patterns and folded states for the crane's intermediate stages.

    python make_stages.py path/to/flat-folder

Writes stages/<name>.cp and, by running Flat-Folder's solver (ff_solve.mjs, needs node),
stages/<name>_cp.fold and stages/<name>_state.fold for:

  preliminary  the preliminary (square) base: the AC diagonal and both midlines
  bird         the bird base: the preliminary base's creases near the center, the kite folds
               from all four corners, and the petal folds on the B and D corners (the wings)

The sheet is the same unit square as 004_traditional_Crane_cp.fold, and every crease's mountain
or valley matches the crane where the crane has that crease. Both bases leave the BD diagonal
unfolded, as the crane does: it runs down the middle of each face of the preliminary base and of
each petal of the bird base. With it folded, each base folds in half again.
"""
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stages")

t = 0.5 * math.tan(math.radians(22.5))        # where the 22.5-degree corner lines meet the midlines
A, B, C, D, O = (0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)
S1, S2, S3, S4 = (0.5, t), (1 - t, 0.5), (0.5, 1 - t), (t, 0.5)

# Mountain and valley as seen from side 1 (the crane's hidden side).
STAGES = {
    "preliminary": [("V", A, C), ("M", (0.5, 0), (0.5, 1)), ("M", (0, 0.5), (1, 0.5))],
    "bird": [("V", A, C)] + [("M", O, s) for s in (S1, S2, S3, S4)]
            + [("M", p, q) for p, q in [(A, S1), (A, S4), (B, S1), (B, S2), (C, S2), (C, S3), (D, S4), (D, S3)]]
            + [("V", S1, S2), ("V", S4, S3)],
}
# Flat-Folder reads line formats from the other side of the paper (it swaps mountain and valley),
# so write them swapped.
CP_TYPE = {"B": 1, "M": 3, "V": 2}


def write_cp(name, creases):
    border = [("B", A, B), ("B", B, C), ("B", C, D), ("B", D, A)]
    path = os.path.join(OUT, name + ".cp")
    with open(path, "w") as f:
        for a, p, q in border + creases:
            f.write(f"{CP_TYPE[a]} {p[0]} {p[1]} {q[0]} {q[1]}\n")
    return path


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    os.makedirs(OUT, exist_ok=True)
    for name, creases in STAGES.items():
        cp = write_cp(name, creases)
        subprocess.run(["node", os.path.join(HERE, "ff_solve.mjs"), sys.argv[1], cp, os.path.join(OUT, name)],
                       check=True)


if __name__ == "__main__":
    main()
