"""
download_and_split.py
Run this INSIDE Google Colab (needs internet access to the VOC mirrors).

What it does:
1. Downloads & extracts VOC2007 (trainval+test) and VOC2012 (trainval) using
   torchvision's built-in downloader (handles the official host + fallback mirror).
2. Builds three splits as plain text list files (one "image_id year" per line):
      splits/train.txt  -> VOC2007-trainval + VOC2012-trainval, minus the val holdout
      splits/val.txt    -> 10% stratified holdout carved out of the pool above
      splits/test.txt   -> VOC2007-test (kept untouched, used only for final evaluation)
3. Prints dataset summary.

Usage (Colab cell):
    !python download_and_split.py --root /content/data --out_root /content/data/splits
"""
from __future__ import annotations
import argparse
import random
from collections import defaultdict
from pathlib import Path

from xml_utils import parse_voc_xml, filter_to_selected_classes, VOC_CLASSES


def download_voc(root: str) -> dict:
    """Download VOC07 (trainval+test) and VOC12 (trainval) via torchvision.
    Returns dict of {tag: VOCDetection dataset root path}.
    """
    from torchvision.datasets import VOCDetection

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    print("Downloading VOC2007 trainval ...")
    VOCDetection(root=str(root), year="2007", image_set="trainval", download=True)
    print("Downloading VOC2007 test ...")
    VOCDetection(root=str(root), year="2007", image_set="test", download=True)
    print("Downloading VOC2012 trainval ...")
    VOCDetection(root=str(root), year="2012", image_set="trainval", download=True)

    return {
        "2007_trainval": root / "VOCdevkit" / "VOC2007",
        "2007_test": root / "VOCdevkit" / "VOC2007",
        "2012_trainval": root / "VOCdevkit" / "VOC2012",
    }


def read_image_set(voc_dir: Path, image_set: str) -> list[str]:
    """Read an ImageSets/Main/<image_set>.txt file -> list of image ids."""
    list_file = voc_dir / "ImageSets" / "Main" / f"{image_set}.txt"
    with open(list_file) as f:
        ids = [line.strip().split()[0] for line in f if line.strip()]
    return ids


def build_pool_entries(voc_dir: Path, image_set: str, year: str) -> list[dict]:
    """Return list of {id, year, img_path, xml_path} for every image in a set
    that contains AT LEAST ONE object from our 6-class traffic/security subset
    (VOC_CLASSES in xml_utils.py). Images that only contain irrelevant classes
    (e.g. a photo of just a cat or a bottle) are dropped entirely -- this
    project's topic only targets person/bicycle/car/motorbike/bus/train.
    """
    ids = read_image_set(voc_dir, image_set)
    entries = []
    n_dropped_irrelevant = 0
    for img_id in ids:
        xml_path = voc_dir / "Annotations" / f"{img_id}.xml"
        img_path = voc_dir / "JPEGImages" / f"{img_id}.jpg"
        if not xml_path.exists() or not img_path.exists():
            continue  # skip missing pairs defensively
        ann = filter_to_selected_classes(parse_voc_xml(xml_path))
        if not ann.objects:
            n_dropped_irrelevant += 1
            continue
        entries.append({"id": img_id, "year": year, "img_path": str(img_path), "xml_path": str(xml_path)})
    print(f"  [{voc_dir.name}/{image_set}] kept {len(entries)} images with >=1 of "
          f"{VOC_CLASSES}, dropped {n_dropped_irrelevant} irrelevant images")
    return entries


def stratified_holdout(entries: list[dict], frac: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Carve out `frac` of entries as a validation holdout, stratified by the
    *dominant* (most frequent) class present in each image so rare classes
    aren't accidentally starved from either split.
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for e in entries:
        ann = filter_to_selected_classes(parse_voc_xml(e["xml_path"]))
        if not ann.objects:
            dominant = "__empty__"
        else:
            counts = defaultdict(int)
            for o in ann.objects:
                counts[o.name] += 1
            dominant = max(counts, key=counts.get)
        buckets[dominant].append(e)

    val, train = [], []
    for cls, items in buckets.items():
        rng.shuffle(items)
        # Plain proportional rounding per bucket. With real VOC pool sizes
        # (hundreds of images per dominant class) this converges to ~`frac`
        # overall; no artificial "at least 1" floor, which would otherwise
        # over-allocate val for small/rare buckets.
        n_val = round(len(items) * frac)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(val)
    rng.shuffle(train)
    return train, val


def write_split(entries: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for e in entries:
            f.write(f"{e['id']} {e['year']} {e['img_path']} {e['xml_path']}\n")
    print(f"  wrote {len(entries):5d} entries -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/data", help="where VOC gets downloaded")
    ap.add_argument("--out_root", default="/content/data/splits", help="where split .txt files go")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_download", action="store_true", help="assume VOC already downloaded under --root")
    args = ap.parse_args()

    if not args.skip_download:
        download_voc(args.root)

    root = Path(args.root).resolve()  # always store ABSOLUTE paths downstream (COCO json "abs_path",
                                       # YOLO symlinks), regardless of the cwd this script is run from
    voc07 = root / "VOCdevkit" / "VOC2007"
    voc12 = root / "VOCdevkit" / "VOC2012"

    pool = []
    pool += build_pool_entries(voc07, "trainval", "2007")
    pool += build_pool_entries(voc12, "trainval", "2012")
    test = build_pool_entries(voc07, "test", "2007")

    train, val = stratified_holdout(pool, frac=args.val_frac, seed=args.seed)

    out_root = Path(args.out_root)
    print(f"Pool (07trainval+12trainval) = {len(pool)} images -> train {len(train)} / val {len(val)}")
    print(f"Test (07test, untouched)     = {len(test)} images")
    write_split(train, out_root / "train.txt")
    write_split(val, out_root / "val.txt")
    write_split(test, out_root / "test.txt")


if __name__ == "__main__":
    main()
