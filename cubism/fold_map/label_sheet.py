"""A label sheet for finding the fold map of the cube tessellation (8x8 grid).

Every fold lies on the precrease grid, so every visible face should be made of whole grid cells or triangles of
them. Each cell gets an ID (row A-H from the top, column 1-8 from the left). The two diagonals split it into four
triangles, tinted by side (top red, right yellow, bottom blue, left green), each carrying the cell ID upright on the
sheet. Fold the sheet, photograph it, and the IDs, tints and text direction on each face give where its paper came
from and how it turned.

Page 1 is the front. Page 2 is the back, mirrored, for printing double-sided (flip on long edge), to catch faces
that show the back of the paper. Back labels start with a ' (e.g. 'C4) and are in gray tints.

usage: python label_sheet.py OUT.pdf
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon, Rectangle

MM = 1 / 25.4
PAGE = (210, 297)
SIDE = 190                    # the square, in mm
N = 8
X0, Y0 = (PAGE[0] - SIDE) / 2, 30     # bottom-left of the square on the page, in mm (y up)
CELL = SIDE / N
ROWS = "ABCDEFGH"
FRONT = {"top": "#f6c3bd", "right": "#f7e7a6", "bottom": "#bcd3f2", "left": "#c4e6c0"}
BACK = {"top": "#e4e4e4", "right": "#cfcfcf", "bottom": "#b9b9b9", "left": "#f2f2f2"}


def page(pdf, back):
    fig = plt.figure(figsize=(PAGE[0] * MM, PAGE[1] * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE[0]); ax.set_ylim(0, PAGE[1]); ax.set_aspect("equal"); ax.axis("off")
    tint = BACK if back else FRONT
    for r in range(N):
        for c in range(N):
            # On the back, the sheet is flipped left-right, so column c sits where column N-1-c is on the front.
            cc = N - 1 - c if back else c
            x, y = X0 + cc * CELL, Y0 + SIDE - (r + 1) * CELL
            cx, cy = x + CELL / 2, y + CELL / 2
            corners = {"tl": (x, y + CELL), "tr": (x + CELL, y + CELL), "br": (x + CELL, y), "bl": (x, y)}
            tris = {"top": ("tl", "tr"), "right": ("tr", "br"), "bottom": ("br", "bl"), "left": ("bl", "tl")}
            name = ("'" if back else "") + f"{ROWS[r]}{c + 1}"
            for side, (a, b) in tris.items():
                # On the back, left and right swap too, so a face's tint still names its side of the front cell.
                s = {"left": "right", "right": "left"}.get(side, side) if back else side
                ax.add_patch(Polygon([corners[a], corners[b], (cx, cy)], closed=True, fc=tint[s], ec="none"))
                px, py = (cx + (corners[a][0] + corners[b][0]) / 2 * 2) / 3, (cy + (corners[a][1] + corners[b][1]) / 2 * 2) / 3
                ax.text(px, py, name, ha="center", va="center", fontsize=6.5, color="#333", family="DejaVu Sans")
            ax.text(cx, cy, name, ha="center", va="center", fontsize=11, weight="bold", color="black",
                    family="DejaVu Sans", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            for (a, b) in (("tl", "br"), ("tr", "bl")):     # the diagonals, faint
                ax.plot(*zip(corners[a], corners[b]), color="#999", lw=0.3, ls=(0, (2, 2)))
    for i in range(1, N):                                    # the grid, faint
        ax.plot([X0 + i * CELL] * 2, [Y0, Y0 + SIDE], color="#888", lw=0.4)
        ax.plot([X0, X0 + SIDE], [Y0 + i * CELL] * 2, color="#888", lw=0.4)
    ax.add_patch(Rectangle((X0, Y0), SIDE, SIDE, fill=False, ec="black", lw=0.8))
    if back:
        ax.text(PAGE[0] / 2, 16, "Back of the sheet (mirrored). Labels with ' are the back of the paper.",
                ha="center", fontsize=8, color="#555")
    else:
        ax.text(X0, Y0 + SIDE + 12, "Cube tessellation: fold-map label sheet", fontsize=12, weight="bold")
        ax.text(X0, Y0 + SIDE + 6, "Cut along the black square. Fold with this side as the front (the side you see first "
                "in step 4). Rows A-H top to bottom, columns 1-8 left to right.", fontsize=6.5, color="#333")
        ax.text(X0, 22, "Tints: top red, right yellow, bottom blue, left green. Small labels are upright on the flat sheet, "
                "so their direction on a folded face shows how it turned.", fontsize=6.5, color="#333")
        ax.text(X0, 17, "Print at 100% (actual size). The square is 190 mm; each cell is 23.75 mm. "
                "Page 2 is optional: print double-sided, flip on long edge.", fontsize=6.5, color="#333")
    pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    with PdfPages(sys.argv[1]) as pdf:
        page(pdf, back=False)
        page(pdf, back=True)
