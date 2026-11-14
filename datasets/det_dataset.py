"""
Detection dataset - COCO-style JSON annotations, pluggable transforms.
Provides DetDataset and det_collate for training and evaluation.
"""

import json
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import torch
from torch.utils.data import Dataset


class DetDataset(Dataset):
    """
    COCO format object detection dataset.

    Args:
        root: Directory containing image files.
        ann_file: Path to COCO format JSON annotation file.
        transform: Callable taking (img, bboxes_coco, labels) and returning
            (img_tensor, gt_boxes_xyxy, gt_labels).
        num_classes: Number of detection classes.
        cat_ids: Optional list of category IDs to keep/map to 0..num_classes-1.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        num_classes: int = 80,
        cat_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.num_classes = num_classes

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images: Dict[int, Dict[str, Any]] = {im["id"]: im for im in coco.get("images", [])}
        cats = sorted(coco.get("categories", []), key=lambda c: c["id"])

        if cat_ids is None:
            self.cat_ids = [c["id"] for c in cats][:num_classes] if cats else [1]
        else:
            self.cat_ids = list(cat_ids)
        self.num_classes = len(self.cat_ids)

        self.cat2label: Dict[int, int] = {cid: i for i, cid in enumerate(self.cat_ids)}
        self.label2cat: Dict[int, int] = {i: cid for cid, i in self.cat2label.items()}

        anns = defaultdict(list)
        for a in coco.get("annotations", []):
            if a.get("iscrowd", 0):
                continue
            if a["category_id"] not in self.cat2label:
                continue
            x, y, w, h = a["bbox"]
            if w < 1 or h < 1:
                continue
            # Store in COCO bbox format [x, y, w, h] and 0-indexed category label
            anns[a["image_id"]].append(
                ([float(x), float(y), float(w), float(h)], self.cat2label[a["category_id"]])
            )

        self.ids: List[int] = sorted(self.images.keys())
        self.anns = anns

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_id = self.ids[index]
        info = self.images[img_id]
        path = os.path.join(self.root, info["file_name"])

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Record original dimensions directly from loaded image array
        orig_h, orig_w = img.shape[:2]
        if info.get("height") and info.get("width"):
            orig_h = int(info["height"])
            orig_w = int(info["width"])

        items = self.anns.get(img_id, [])
        bboxes = [b for b, _ in items]
        labels = [lbl for _, lbl in items]

        if self.transform is not None:
            img, gt_boxes, gt_labels = self.transform(img, bboxes, labels)
        else:
            # Default fallback when no transform is provided
            if len(bboxes) > 0:
                boxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
                gt_boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
                gt_boxes[:, [0, 2]].clamp_(0.0, float(orig_w))
                gt_boxes[:, [1, 3]].clamp_(0.0, float(orig_h))
                gt_labels = torch.tensor(labels, dtype=torch.int64)
            else:
                gt_boxes = torch.empty((0, 4), dtype=torch.float32)
                gt_labels = torch.empty((0,), dtype=torch.int64)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return {
            "img": img,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "img_id": img_id,
            "ori_shape": (orig_h, orig_w),
        }


def det_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for detection batches.
    Stacks images into a single Tensor [B, 3, H, W] while preserving variable-length
    per-image ground truth boxes and labels as lists of Tensors.
    """
    imgs = torch.stack([b["img"] for b in batch])
    return {
        "img": imgs,
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "gt_labels": [b["gt_labels"] for b in batch],
        "img_ids": [b["img_id"] for b in batch],
        "ori_shapes": [b["ori_shape"] for b in batch],
    }
