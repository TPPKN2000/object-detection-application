"""
xml_utils.py
Parse PASCAL VOC XML annotation files into plain Python dicts.

VOC annotation XML structure (relevant fields):
<annotation>
  <filename>000001.jpg</filename>
  <size><width>W</width><height>H</height><depth>3</depth></size>
  <object>
    <name>dog</name>
    <difficult>0</difficult>
    <bndbox><xmin></xmin><ymin></ymin><xmax></xmax><ymax></ymax></bndbox>
  </object>
  ...
</annotation>
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VocObject:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    difficult: bool = False


@dataclass
class VocAnnotation:
    filename: str
    width: int
    height: int
    objects: list = field(default_factory=list)
    source_xml: str = ""


def parse_voc_xml(xml_path: str | Path) -> VocAnnotation:
    """Parse a single VOC XML file into a VocAnnotation."""
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size_node = root.find("size")
    width = int(float(size_node.findtext("width")))
    height = int(float(size_node.findtext("height")))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name").strip()
        difficult_text = obj.findtext("difficult", default="0")
        difficult = str(difficult_text).strip() == "1"
        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        # Guard against degenerate boxes (some VOC files have off-by-one errors)
        xmin, xmax = sorted((xmin, xmax))
        ymin, ymax = sorted((ymin, ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append(
            VocObject(name=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, difficult=difficult)
        )

    return VocAnnotation(
        filename=filename,
        width=width,
        height=height,
        objects=objects,
        source_xml=str(xml_path),
    )


# All 20 standard PASCAL VOC classes (kept for reference / sanity checks only).
VOC_ALL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# ---------------------------------------------------------------------------
# Project topic: "Phat hien nguoi va phuong tien giao thong phuc vu giam sat
# an ninh do thi" (Person & vehicle detection for urban traffic/security
# surveillance). We subset VOC down to the 6 traffic/urban-security-relevant
# classes below. This is the SINGLE SOURCE OF TRUTH for class list + class-id
# mapping -- every other script (download_and_split/voc_to_coco/voc_to_yolo/
# dataset_stats) imports VOC_CLASSES from here, so changing the topic later
# only requires editing this one list.
# ---------------------------------------------------------------------------
VOC_CLASSES = ["person", "bicycle", "car", "motorbike", "bus", "train"]

CLASS_TO_ID = {name: idx for idx, name in enumerate(VOC_CLASSES)}
SELECTED_CLASS_SET = set(VOC_CLASSES)


def filter_to_selected_classes(ann: "VocAnnotation") -> "VocAnnotation":
    """Return a copy of `ann` keeping only objects whose class is in VOC_CLASSES.
    Objects belonging to VOC classes outside our 6-class subset (e.g. cat,
    sofa, bottle...) are dropped -- they are simply not a detection target
    for this project's topic.
    """
    kept = [o for o in ann.objects if o.name in SELECTED_CLASS_SET]
    return VocAnnotation(
        filename=ann.filename,
        width=ann.width,
        height=ann.height,
        objects=kept,
        source_xml=ann.source_xml,
    )
