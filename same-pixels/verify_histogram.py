"""Count every pixel color in two images and list every color whose count differs.

    python verify_histogram.py seed0_source.png seed0_snapped.png

Writes color_counts.csv (every color in either image, with both counts) and
color_differences.csv (only the colors whose counts differ). If the two images have
identical RGB histograms, the second file is empty apart from its header.
"""
import sys
import numpy as np
from PIL import Image


def load_rgb(path):
    im = Image.open(path)
    if im.mode == "RGBA" and np.asarray(im)[..., 3].min() < 255:
        print(f"warning: {path} has transparent pixels; ignoring alpha")
    return np.asarray(im.convert("RGB")).reshape(-1, 3)


def color_table(a, b):
    # One row per distinct (R, G, B) found in either image: color, count in a, count in b.
    keys_a = (a[:, 0].astype(np.int64) << 16) | (a[:, 1].astype(np.int64) << 8) | a[:, 2]
    keys_b = (b[:, 0].astype(np.int64) << 16) | (b[:, 1].astype(np.int64) << 8) | b[:, 2]
    colors, inverse = np.unique(np.r_[keys_a, keys_b], return_inverse=True)
    count_a = np.bincount(inverse[:len(keys_a)], minlength=len(colors))
    count_b = np.bincount(inverse[len(keys_a):], minlength=len(colors))
    rgb = np.stack([colors >> 16, (colors >> 8) & 255, colors & 255], axis=1)
    return rgb, count_a, count_b


def save_csv(path, rgb, count_a, count_b, name_a, name_b):
    with open(path, "w") as f:
        f.write(f"R,G,B,hex,count_{name_a},count_{name_b},difference\n")
        for (r, g, bl), ca, cb in zip(rgb, count_a, count_b):
            f.write(f"{r},{g},{bl},#{r:02x}{g:02x}{bl:02x},{ca},{cb},{cb - ca}\n")


def main(path_a, path_b):
    a, b = load_rgb(path_a), load_rgb(path_b)
    name_a, name_b = "a", "b"
    print(f"a = {path_a}: {len(a):,} pixels")
    print(f"b = {path_b}: {len(b):,} pixels")
    if len(a) != len(b):
        print("different pixel counts, so the histograms can't be identical")

    rgb, count_a, count_b = color_table(a, b)
    diff = count_b != count_a
    only_a = (count_a > 0) & (count_b == 0)
    only_b = (count_b > 0) & (count_a == 0)

    save_csv("color_counts.csv", rgb, count_a, count_b, name_a, name_b)
    save_csv("color_differences.csv", rgb[diff], count_a[diff], count_b[diff], name_a, name_b)

    print(f"\ndistinct colors:          a {int((count_a > 0).sum()):,}   b {int((count_b > 0).sum()):,}   either {len(rgb):,}")
    print(f"colors only in a:         {int(only_a.sum()):,}  ({int(count_a[only_a].sum()):,} pixels)")
    print(f"colors only in b:         {int(only_b.sum()):,}  ({int(count_b[only_b].sum()):,} pixels)")
    print(f"colors in both, different counts: {int((diff & ~only_a & ~only_b).sum()):,}")
    print(f"pixels that would have to change: {int(np.abs(count_b - count_a).sum()) // 2:,}")

    if diff.any():
        print("\nlargest differences (full list in color_differences.csv):")
        print(f"{'R':>4} {'G':>4} {'B':>4}  {'hex':>8} {'count a':>8} {'count b':>8} {'b - a':>7}")
        for k in np.argsort(-np.abs(count_b - count_a)[diff], kind="stable")[:20]:
            (r, g, bl), ca, cb = rgb[diff][k], count_a[diff][k], count_b[diff][k]
            print(f"{r:>4} {g:>4} {bl:>4}  #{r:02x}{g:02x}{bl:02x} {ca:>8} {cb:>8} {cb - ca:>+7}")
        print("\nNOT identical: the RGB histograms differ.")
    else:
        print("\nIDENTICAL: every color appears exactly as many times in both images.")
    print("wrote color_counts.csv and color_differences.csv")


if __name__ == "__main__":
    main(*sys.argv[1:3])
