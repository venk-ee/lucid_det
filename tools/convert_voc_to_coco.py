"""
Pascal VOC XML to COCO JSON format converter.
Converts VOC bounding box annotations into standardized COCO detection format.
"""

import argparse
import json
import os
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def convert_voc_to_coco(
    voc_dir: str,
    output_json: str,
    split_file: Optional[str] = None,
    include_difficult: bool = False,
) -> None:
    """
    Convert a Pascal VOC directory containing Annotations/ and JPEGImages/ to COCO JSON format.

    Args:
        voc_dir: Path to VOC directory (e.g. VOCdevkit/VOC2007).
        output_json: Path to output COCO JSON file.
        split_file: Optional path to split txt file (e.g. ImageSets/Main/trainval.txt).
        include_difficult: Whether to include difficult bounding boxes.
    """
    ann_dir = os.path.join(voc_dir, "Annotations")
    img_dir = os.path.join(voc_dir, "JPEGImages")

    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Annotations directory not found in: {voc_dir}")

    if split_file and os.path.isfile(split_file):
        with open(split_file, "r") as f:
            valid_ids = set(line.strip() for line in f if line.strip())
    else:
        valid_ids = None

    categories = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(VOC_CLASSES)]
    class_to_id = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    xml_files = sorted(os.listdir(ann_dir))

    for xml_fn in xml_files:
        if not xml_fn.endswith(".xml"):
            continue
        stem = os.path.splitext(xml_fn)[0]
        if valid_ids is not None and stem not in valid_ids:
            continue

        xml_path = os.path.join(ann_dir, xml_fn)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        if size is not None:
            width = int(size.findtext("width", "0"))
            height = int(size.findtext("height", "0"))
        else:
            width, height = 0, 0

        fn = root.findtext("filename", f"{stem}.jpg")
        if not os.path.splitext(fn)[1]:
            fn = f"{fn}.jpg"

        images.append({
            "id": img_id,
            "file_name": fn,
            "width": width,
            "height": height,
        })

        for obj in root.findall("object"):
            cls_name = obj.findtext("name")
            if cls_name not in class_to_id:
                continue

            difficult = int(obj.findtext("difficult", "0"))
            if difficult and not include_difficult:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))

            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)

            if w <= 0 or h <= 0:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_to_id[cls_name],
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": difficult,
            })
            ann_id += 1

        img_id += 1

    parent = os.path.dirname(output_json)
    if parent:
        os.makedirs(parent, exist_ok=True)

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(output_json, "w") as f:
        json.dump(coco_dict, f)

    print(f"Converted {len(images)} images and {len(annotations)} annotations to {output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC annotations to COCO JSON format")
    parser.add_argument("--voc-dir", type=str, required=True, help="Path to VOC directory (e.g. VOCdevkit/VOC2007)")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    parser.add_argument("--split", type=str, default=None, help="Optional ImageSets split txt file")
    parser.add_argument("--include-difficult", action="store_true", help="Include difficult annotations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_voc_to_coco(
        voc_dir=args.voc_dir,
        output_json=args.out,
        split_file=args.split,
        include_difficult=args.include_difficult,
    )
