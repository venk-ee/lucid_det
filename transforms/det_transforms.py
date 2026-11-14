"""
Vision augmentations and preprocessing transforms.
Provides Letterbox, Albumentations pipeline builder, and Mosaic / MixUp primitives.
"""

from typing import Any, List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def letterbox_image(
    image: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, int, int, int, int]:
    """
    Resize while maintaining aspect ratio, then pad to target shape.

    Args:
        image: Input image array (H, W, 3).
        new_shape: Target (H, W) or single integer for square.
        color: Padding fill color (default: (114, 114, 114)).

    Returns:
        letterboxed: Padded image array.
        scale: Resize scale factor.
        pad_left: Horizontal padding offset.
        pad_top: Vertical padding offset.
        resized_w: Scaled width before padding.
        resized_h: Scaled height before padding.
    """
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape

    scale = min(target_h / float(orig_h), target_w / float(orig_w))
    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))

    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    letterboxed = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    return letterboxed, scale, pad_left, pad_top, resized_w, resized_h


def letterbox_boxes_coco(
    boxes: Sequence[Sequence[float]],
    scale: float,
    pad_left: int,
    pad_top: int,
) -> List[List[float]]:
    """Transform COCO [x, y, w, h] boxes according to letterbox scale and padding."""
    return [
        [x * scale + pad_left, y * scale + pad_top, w * scale, h * scale] for x, y, w, h in boxes
    ]


def unletterbox_boxes_xyxy(
    boxes: torch.Tensor,
    scale: float,
    pad_left: int,
    pad_top: int,
    orig_w: int,
    orig_h: int,
) -> torch.Tensor:
    """Reverse letterbox transform to map predictions back to original image space."""
    if boxes.numel() == 0:
        return boxes
    out = boxes.clone()
    out[:, 0] = (out[:, 0] - pad_left) / scale
    out[:, 2] = (out[:, 2] - pad_left) / scale
    out[:, 1] = (out[:, 1] - pad_top) / scale
    out[:, 3] = (out[:, 3] - pad_top) / scale

    out[:, 0].clamp_(0.0, float(orig_w))
    out[:, 2].clamp_(0.0, float(orig_w))
    out[:, 1].clamp_(0.0, float(orig_h))
    out[:, 3].clamp_(0.0, float(orig_h))
    return out


def mosaic_augment(
    images: Sequence[np.ndarray],
    boxes_list: Sequence[Sequence[Sequence[float]]],
    labels_list: Sequence[Sequence[int]],
    output_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    Mosaic 4-image augmentation. Combines 4 images into one mosaic.

    Args:
        images: 4 images in RGB uint8.
        boxes_list: 4 lists of COCO [x, y, w, h] boxes.
        labels_list: 4 lists of class labels.
        output_size: Target mosaic (H, W).

    Returns:
        mosaic_img: (H, W, 3) mosaic array.
        mosaic_boxes: Combined list of COCO [x, y, w, h] boxes.
        mosaic_labels: Combined list of class labels.
    """
    target_h, target_w = output_size
    xc = int(np.random.uniform(target_w * 0.25, target_w * 0.75))
    yc = int(np.random.uniform(target_h * 0.25, target_h * 0.75))

    mosaic_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    mosaic_boxes: List[List[float]] = []
    mosaic_labels: List[int] = []

    for i in range(4):
        img = images[i % len(images)]
        boxes = boxes_list[i % len(boxes_list)]
        labels = labels_list[i % len(labels_list)]

        h, w = img.shape[:2]

        if i == 0:  # top-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_w), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_h, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_w), min(target_h, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        pad_x = x1a - x1b
        pad_y = y1a - y1b

        for b, lbl in zip(boxes, labels):
            x, y, bw, bh = b
            nx = x + pad_x
            ny = y + pad_y
            # Clip box to valid patch
            nx1 = max(nx, x1a)
            ny1 = max(ny, y1a)
            nx2 = min(nx + bw, x2a)
            ny2 = min(ny + bh, y2a)
            if nx2 > nx1 + 2 and ny2 > ny1 + 2:
                mosaic_boxes.append([nx1, ny1, nx2 - nx1, ny2 - ny1])
                mosaic_labels.append(lbl)

    return mosaic_img, mosaic_boxes, mosaic_labels


def mixup_augment(
    img1: np.ndarray,
    boxes1: Sequence[Sequence[float]],
    labels1: Sequence[int],
    img2: np.ndarray,
    boxes2: Sequence[Sequence[float]],
    labels2: Sequence[int],
    alpha: float = 0.5,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """Alpha-blend two images and combine their bounding boxes."""
    mixed_img = (img1.astype(np.float32) * alpha + img2.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8
    )
    mixed_boxes = list(boxes1) + list(boxes2)
    mixed_labels = list(labels1) + list(labels2)
    return mixed_img, mixed_boxes, mixed_labels


class DetectionTransform:
    """
    Wrapper around an Albumentations Compose pipeline for object detection.

    Expects:
        - image: np.ndarray (H, W, 3) in RGB uint8
        - bboxes: Sequence of COCO format [x, y, w, h]
        - labels: Sequence of class labels (0-indexed int)

    Returns:
        - img: torch.Tensor [3, H, W]
        - gt_boxes: torch.Tensor [N, 4] in xyxy format
        - gt_labels: torch.Tensor [N] int64
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def __call__(
        self,
        image: np.ndarray,
        bboxes: Sequence[Sequence[float]],
        labels: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        res = self.pipeline(image=image, bboxes=bboxes, labels=labels)

        # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in res["bboxes"]]
        if len(boxes_xyxy) > 0:
            gt_boxes = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
            gt_labels = torch.as_tensor([int(lbl) for lbl in res["labels"]], dtype=torch.int64)
        else:
            gt_boxes = torch.empty((0, 4), dtype=torch.float32)
            gt_labels = torch.empty((0,), dtype=torch.int64)

        img_tensor = res["image"]
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).float() / 255.0

        if gt_boxes.numel() > 0:
            h, w = img_tensor.shape[-2:]
            gt_boxes[:, [0, 2]].clamp_(0.0, float(w))
            gt_boxes[:, [1, 3]].clamp_(0.0, float(h))

        return img_tensor, gt_boxes, gt_labels


def build_det_transform(
    img_size: Union[int, Tuple[int, int], Sequence[int]] = 640,
    pad_val: int = 114,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    is_train: bool = True,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    affine_prob: float = 0.5,
    color_jitter_prob: float = 0.5,
    brightness_contrast_prob: float = 0.5,
    clahe_prob: float = 0.3,
    motion_blur_prob: float = 0.2,
) -> DetectionTransform:
    """
    Build an Albumentations detection pipeline based on training/evaluation hyperparameters.

    Args:
        img_size: Target resolution (int or tuple/sequence).
        pad_val: Padding value for letterboxing.
        mean: Normalization RGB mean.
        std: Normalization RGB standard deviation.
        is_train: Whether to include data augmentations.
        hflip_prob: Probability of horizontal flip.
        vflip_prob: Probability of vertical flip.
        affine_prob: Probability of random affine transformation.
        color_jitter_prob: Probability of color jitter.
        brightness_contrast_prob: Probability of random brightness/contrast.
        clahe_prob: Probability of CLAHE contrast enhancement.
        motion_blur_prob: Probability of motion blur.

    Returns:
        DetectionTransform instance.
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for build_det_transform.")

    transforms_list = []

    if is_train:
        train_augs = [
            A.HorizontalFlip(p=hflip_prob),
            A.VerticalFlip(p=vflip_prob),
            A.Affine(
                scale=(0.75, 1.25),
                rotate=(-15, 15),
                translate_percent=(-0.0625, 0.0625),
                keep_ratio=True,
                border_mode=0,
                fill=(pad_val, pad_val, pad_val),
                p=affine_prob,
            ),
            A.ColorJitter(p=color_jitter_prob),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=brightness_contrast_prob,
            ),
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=clahe_prob,
            ),
            A.MotionBlur(
                blur_limit=5,
                p=motion_blur_prob,
            ),
        ]
        transforms_list.extend([t for t in train_augs if t.p > 0])

    h, w = (
        (img_size, img_size) if isinstance(img_size, int) else (int(img_size[0]), int(img_size[1]))
    )

    # Resize & Letterbox (PadIfNeeded)
    transforms_list.extend(
        [
            A.LongestMaxSize(max_size=max(h, w)),
            A.PadIfNeeded(
                min_height=h,
                min_width=w,
                border_mode=0,
                fill=(pad_val, pad_val, pad_val),
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    pipeline = A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_visibility=0.1,
            min_area=1.0,
            clip=True,
        ),
    )

    return DetectionTransform(pipeline)
