"""
Generates synthetic mini-COCO dataset for fast local smoke testing and pipeline verification.
Creates 10 images and annotations in data/dummy/.
"""

import json
import os

import cv2
import numpy as np


def make_dummy_coco_data(output_dir: str = "data/dummy", num_images: int = 10) -> str:
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    images = []
    annotations = []
    ann_id = 1

    categories = [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "person"},
        {"id": 3, "name": "bicycle"},
    ]

    for i in range(num_images):
        fn = f"dummy_{i:04d}.jpg"
        h, w = 480, 640
        # Create patterned synthetic image
        img = rng.integers(50, 220, (h, w, 3), dtype=np.uint8)

        # Draw 1-3 colored rectangles as visual objects
        num_boxes = int(rng.integers(1, 4))
        for _ in range(num_boxes):
            bx = int(rng.integers(20, 400))
            by = int(rng.integers(20, 300))
            bw = int(rng.integers(40, 180))
            bh = int(rng.integers(40, 180))
            cat_id = int(rng.choice([1, 2, 3]))

            color = (
                int(rng.integers(0, 255)),
                int(rng.integers(0, 255)),
                int(rng.integers(0, 255)),
            )
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, -1)

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": cat_id,
                    "bbox": [bx, by, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        cv2.imwrite(os.path.join(img_dir, fn), img)
        images.append(
            {
                "id": i,
                "file_name": fn,
                "height": h,
                "width": w,
            }
        )

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    ann_file = os.path.join(output_dir, "det.json")
    with open(ann_file, "w") as f:
        json.dump(coco_dict, f, indent=2)

    print(
        f"Created dummy dataset in {output_dir}: {len(images)} images, {len(annotations)} annotations."
    )
    return ann_file


if __name__ == "__main__":
    make_dummy_coco_data()
