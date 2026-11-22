# Architectural Scalability Evaluation for LucidDet

## Executive Summary

The current LucidDet architecture follows a clean KISS Dependency Injection pattern: explicit small-scale maps (HEAD_MAP, NECK_MAP, ASSIGNER_MAP, CLS_LOSS_MAP, BOX_LOSS_MAP), callable-based DI for loss components, a well-defined Universal Assigner Protocol, and a DetectionOutput dataclass as the output contract. The backbone->neck->head pipeline in SingleStageDetector is modular and composable.

However, the architecture contains several hardcoded assumptions about axis-aligned 4-dimensional bounding boxes that permeate every layer. Extending to rotated detection, instance segmentation, and other paradigms requires systematic extensions at multiple levels. The good news is that the DI pattern itself scales well -- the problem is not the DI mechanism, but the tight coupling to a single task schema (axis-aligned bbox detection) baked into the output contract, loss dispatcher, assigner protocol, NMS, and evaluation pipeline.

Below is a layer-by-layer assessment.

---

## 1. Abstractions That Hold Cleanly

### 1.1. SingleStageDetector Composition Pattern

The backbone->neck->head pipeline is paradigm-agnostic. The `extract_feat()` method, the `forward()` dispatch between training/inference/export, and the checkpoint loading all work independently of what the head predicts. This pattern extends naturally to:

- **Rotated detection**: No changes needed in the detector base class.
- **Instance segmentation**: The pipeline structure remains valid; only the head output and postprocess change.
- **Two-stage detectors**: A new `TwoStageDetector` base class can follow the same composition pattern (backbone->neck->RPN->roi-head).
- **Query-based transformers**: A `TransformerDetector` can use backbone->neck->transformer_decoder with the same forward structure.

**Verdict: Holds cleanly. No changes required.**

### 1.2. Explicit Build Maps (HEAD_MAP, NECK_MAP, etc.)

The small explicit dictionaries for building components avoid global registries and dynamic imports. This pattern is KISS and scales well:

- New heads (e.g., `RotatedSepBNHead`, `MaskSepBNHead`) can be added by extending HEAD_MAP.
- New necks can be added by extending NECK_MAP.
- No metaprogramming, no plugin discovery, no configuration-driven class loading.

**Verdict: Holds cleanly. The explicit map pattern is the right choice and should be preserved.**

### 1.3. Loss DI via resolve_* Functions

The `resolve_cls_loss()`, `resolve_box_loss()`, and `resolve_assigner()` functions accept strings, classes, instances, or callables. This is a flexible DI pattern that naturally supports:

- New loss types (angle loss, mask loss) by extending the maps or passing instances directly.
- Custom assigners via callable injection.

**Verdict: Holds cleanly. The DI mechanism itself is extensible.**

### 1.4. generic_train / det_runner Training Loop

The `generic_train()` function accepts a `loss_fn` callable and a `model` nn.Module. The training loop in `train_one_epoch()` is generic:

```python
outputs = model(images, training_override=True)
loss, loss_dict = loss_fn(outputs, batch_targets)
```

This contract is task-agnostic. Any model that returns outputs consumable by a loss function fits. The validation loop is more tightly coupled to detection-specific output formats (see Section 2).

**Verdict: The training loop holds cleanly. The validation loop needs extension.**

---

## 2. Areas Requiring Minimal Extensions

### 2.1. DetectionOutput Dataclass -- The Central Bottleneck

`DetectionOutput` is the most critical extension point. Currently it has exactly four fields:

```python
@dataclass
class DetectionOutput:
    pred_cls: torch.Tensor  # (B, A, num_classes)
    pred_boxes: torch.Tensor  # (B, A, 4) -- hardcoded to xyxy
    anchor_centers: torch.Tensor  # (A, 2)
    anchor_strides: torch.Tensor  # (A, 1)
    detections: Optional[...] = None
```

**For rotated detection:**
- `pred_boxes` must become `(B, A, 5)` for xywha format (x, y, w, h, angle).
- A new field `pred_angles` could be added, or `pred_boxes` last dimension becomes configurable.
- The dataclass approach supports adding optional fields cleanly.

**For instance segmentation:**
- A new field `pred_masks` is needed: `(B, A, M)` where M is the number of mask prototypes (for mask-decoding heads like YOLACT/YOLOv8-seg) or `(B, A, H, W)` for direct mask prediction.
- Additional fields like `mask_prototypes` and `mask_coeffs` may be needed for prototype-based approaches.

**For keypoint detection:**
- A new field `pred_keypoints` is needed: `(B, A, K, 3)` for K keypoints with (x, y, visibility).

**Recommended approach:** Extend DetectionOutput with optional fields using a union or sealed subclass pattern:

```python
@dataclass
class DetectionOutput:
    pred_cls: torch.Tensor
    pred_boxes: torch.Tensor  # (B, A, D) where D in {4, 5} for bbox/rotated
    anchor_centers: torch.Tensor
    anchor_strides: torch.Tensor
    detections: Optional[...] = None
    # Optional extensions:
    pred_masks: Optional[torch.Tensor] = None  # (B, A, M) or (B, A, H, W)
    pred_keypoints: Optional[torch.Tensor] = None  # (B, A, K, 3)
```

This keeps the dataclass simple and backward-compatible. The `__getitem__`/`__setitem__`/`get()` methods already support dynamic field access.

**Verdict: Minimal extension. Add optional fields; no structural redesign needed.**

### 2.2. Head Output Dimensionality

Both `SepBNHead` and `YOLOXHead` hardcode the regression head output to 4 channels:

```python
self.rtm_reg.append(nn.Conv2d(feat_channels, 4, ...))  # SepBNHead
self.reg_preds.append(nn.Conv2d(feat_channels, 4, ...))  # YOLOXHead
```

**For rotated detection:**
- A `RotatedSepBNHead` subclass or a `reg_dims` parameter can change the 4 to 5.
- The box decoding logic inside `forward()` must change from ltrb->xyxy to a rotated box decoder (e.g., center, w, h, angle -> rotated box representation).

**For instance segmentation:**
- A `MaskSepBNHead` adds a mask branch: prototype prediction at one level and coefficient prediction at all levels.
- Example: `self.rtm_mask = nn.Conv2d(feat_channels, num_mask_coeffs, 1)` per level.

**For keypoint detection:**
- A `KeypointSepBNHead` adds a keypoint branch: `nn.Conv2d(feat_channels, K * 3, 1)` per level.

**Recommended approach:** Parameterize the base head with `reg_dims` and add optional branch constructors. Subclassing is cleaner than over-parameterizing the base class:

```python
class RotatedSepBNHead(SepBNHead):
    def __init__(self, ..., angle_range=(-180, 180)):
        super().__init__(...)
        # Override reg convs to output 5 instead of 4
        # Override forward() to decode rotated boxes
```

**Verdict: Minimal extension. Subclass heads with task-specific branches; parameterize reg_dims.**

### 2.3. det_loss Dispatcher -- Adding Loss Adapters

The `det_loss()` function is the central loss computation. It currently handles two tasks: classification and box regression. The structure is:

```python
def det_loss(model_out, gt_boxes_list, gt_labels_list, cls_loss, box_loss, box_weight, l1_weight, assigner):
    # Per-batch loop:
    #   1. Assign targets via assigner
    #   2. Compute cls loss
    #   3. Compute box loss (IoU-based + L1)
    # Return total_loss, loss_dict
```

**For rotated detection:**
- Add an `angle_loss` parameter and `angle_weight`.
- The box loss adapter must accept `(N, 5)` boxes and use rotated IoU.
- The assigner must use rotated IoU for cost computation.

**For instance segmentation:**
- Add a `mask_loss` parameter and `mask_weight`.
- After the existing cls+box computation, add mask loss computation for positive anchors.
- The mask loss typically operates on decoded mask coefficients vs. binary mask targets.

**For keypoint detection:**
- Add a `keypoint_loss` parameter and `keypoint_weight`.
- Compute keypoint regression loss (e.g., smooth L1 on visible keypoints).

**Recommended approach:** Introduce a `LossAdapter` protocol or simply extend `det_loss()` with optional parameters:

```python
def det_loss(
    model_out, gt_boxes_list, gt_labels_list,
    cls_loss="qfl", box_loss="giou", box_weight=3.5, l1_weight=0.25,
    assigner=None,
    # Extensions:
    angle_loss=None, angle_weight=1.0,
    mask_loss=None, mask_weight=1.0,
    keypoint_loss=None, keypoint_weight=1.0,
):
```

Each additional loss is only computed when the corresponding loss function and model output fields are present. This keeps the function backward-compatible and avoids conditional complexity.

**Verdict: Minimal extension. Add optional loss parameters; compute conditionally based on output fields.**

### 2.4. Universal Assigner Protocol -- Rotated IoU

The assigner protocol is well-designed but the SimOTA implementation uses `box_iou()` from `core.iou`, which only handles axis-aligned boxes. For rotated detection:

- A `rotated_box_iou()` function must be added to `core/iou.py`.
- The assigner must accept `(A, 5)` boxes and `(G, 5)` ground truth boxes.
- The center-based cost computation in SimOTA must handle rotated box centers (which are already part of the xywha representation).

**Recommended approach:** Add a `rotated_box_iou()` function alongside `box_iou()`. The assigner can accept an `iou_fn` parameter:

```python
def dynamic_soft_label_assign(
    ...,
    iou_fn=box_iou,  # Default to axis-aligned; pass rotated_box_iou for rotated tasks
):
```

This keeps the assigner protocol clean and avoids duplicating the entire assigner for rotated boxes.

**Verdict: Minimal extension. Add rotated IoU; parameterize the assigner with iou_fn.**

### 2.5. NMS for Rotated Boxes

`multiclass_nms()` in `core/nms.py` uses `torchvision.ops.batched_nms`, which only supports axis-aligned boxes. For rotated detection:

- A `rotated_nms()` function must be implemented (e.g., using rotated IoU for suppression).
- The `postprocess()` method in `SingleStageDetector` must dispatch to the correct NMS based on box dimensionality or an explicit flag.

**Recommended approach:** Add `rotated_nms()` to `core/nms.py` and parameterize `postprocess()`:

```python
def postprocess(self, outputs, x, nms_fn=multiclass_nms_batch):
```

Or detect automatically: if `pred_boxes.shape[-1] == 5`, use rotated NMS.

**Verdict: Minimal extension. Add rotated NMS; parameterize or auto-detect in postprocess.**

### 2.6. Dataset Layer -- Extended Annotations

`DetDataset` currently reads COCO-format annotations with `[x, y, w, h]` bounding boxes. For extended tasks:

**Rotated detection:**
- Read `bbox` as `[x, y, w, h, angle]` from rotated COCO formats (e.g., DOTA, HRSC).
- The `get_raw_sample()` method must return 5-dim boxes.

**Instance segmentation:**
- Read `segmentation` field from COCO annotations (polygon or RLE format).
- Convert polygons to binary masks at the image or feature level.
- The `get_raw_sample()` method must return mask targets alongside boxes.

**Keypoint detection:**
- Read `keypoints` field from COCO annotations (K x 3 arrays with visibility flags).
- The `get_raw_sample()` method must return keypoint targets.

**Recommended approach:** Extend `DetDataset` with a `task_type` parameter or add task-specific subclasses:

```python
class RotatedDetDataset(DetDataset):
    def get_raw_sample(self, index):
        # Returns (img, bboxes_xywha, labels, img_id, ori_shape)
        ...


class SegmDataset(DetDataset):
    def get_raw_sample(self, index):
        # Returns (img, bboxes, labels, masks, img_id, ori_shape)
        ...
```

Correspondingly, extend `det_collate()` to handle the additional fields.

**Verdict: Minimal extension. Subclass DetDataset; extend det_collate conditionally.**

### 2.7. Validation & Evaluation Pipeline

`val_one_epoch()` in `det_runner.py` is tightly coupled to axis-aligned bbox detection:

- It extracts `boxes`, `scores`, `labels` from detections.
- It formats predictions as `[x1, y1, w, h]` for COCO evaluation.
- The `COCOEvaluator` likely uses `pycocotools` which only supports axis-aligned boxes.

**For rotated detection:**
- A `RotatedCOCOEvaluator` using a rotated IoU evaluation API is needed.
- The prediction format in `val_one_epoch()` must handle 5-dim boxes.

**For instance segmentation:**
- The evaluator must handle mask predictions (either as polygons or RLE).
- The `val_one_epoch()` must extract and format mask predictions.

**Recommended approach:** Parameterize `val_one_epoch()` with an `output_formatter` callable or add task-specific validation functions. The evaluator can be injected (it already is via the `evaluator` parameter).

**Verdict: Moderate extension. The evaluator injection already exists; the output formatting in val_one_epoch needs parameterization.**

---

## 3. Areas Requiring More Significant Redesign

### 3.1. Two-Stage Detection

Two-stage detectors (Faster R-CNN, Cascade R-CNN) have a fundamentally different forward pass:

1. **Stage 1 (RPN):** Generate region proposals from feature maps.
2. **Stage 2 (ROI Head):** Extract ROI features and classify/regress each proposal.

The current `SingleStageDetector` cannot express this two-phase computation. A new `TwoStageDetector` base class is needed:

```python
class TwoStageDetector(nn.Module):
    def __init__(self, backbone, neck, rpn, roi_head, ...):
        ...
    def forward(self, x, training_override=None):
        feats = self.extract_feat(x)
        proposals = self.rpn(feats)  # Stage 1
        outputs = self.roi_head(feats, proposals)  # Stage 2
        ...
```

Key differences from single-stage:
- The RPN produces proposals, not final detections.
- The ROI head operates on cropped/resampled features (ROIAlign).
- The loss computation is two-phase: RPN loss + ROI head loss.
- The assigner for the ROI head uses proposal-GT matching, not anchor-GT matching.

**Impact on existing abstractions:**
- `SingleStageDetector` is not reusable; a parallel `TwoStageDetector` is needed.
- `det_loss()` is not directly applicable; a `two_stage_loss()` dispatcher is needed.
- The Universal Assigner Protocol can be reused for the ROI head stage with a different matching strategy.
- `DetectionOutput` can be reused for the final output.
- `generic_train()` works without changes (it only cares about `loss_fn(outputs, targets)`).

**Verdict: Requires a new detector base class and a new loss dispatcher. The DI pattern and training runner hold. No global registry needed.**

### 3.2. Query-Based Transformers (DETR Family)

DETR-style detectors replace the anchor-based head with a set of learnable object queries processed by a transformer decoder. The forward pass is:

1. Backbone + optional neck produce feature maps.
2. Feature maps are flattened and positional-encoded.
3. Object queries attend to features via transformer decoder layers.
4. Each query produces a class prediction and a box prediction.

Key architectural differences:
- No anchor centers or strides; queries are the "priors."
- No NMS; Hungarian matching produces unique assignments.
- The loss uses bipartite matching, not greedy assignment.

**Impact on existing abstractions:**
- `SingleStageDetector` can be adapted: the "head" becomes a transformer decoder, and `anchor_centers`/`anchor_strides` become irrelevant.
- `DetectionOutput` needs to drop or make optional the anchor fields.
- `det_loss()` cannot be used; a `detr_loss()` with Hungarian matching is fundamentally different.
- The assigner protocol is replaced by a Hungarian matcher.
- `generic_train()` works without changes.

**Recommended approach:** Create a `TransformerDetector` base class that inherits the composition pattern but replaces the anchor-based head with a query-based decoder. The output contract can be a subclass of `DetectionOutput` or a parallel `TransformerOutput` dataclass.

**Verdict: Requires a new detector base class and a new loss paradigm. The DI pattern holds; the anchor-centric abstractions (assigner, NMS, anchor centers) do not apply.**

### 3.3. Core IoU/NMS as a Monolithic Module

Currently, `core/iou.py` and `core/nms.py` contain only axis-aligned implementations. As the library grows to support rotated boxes, masks, and potentially other geometries, these modules will accumulate specialized functions:

- `box_iou`, `rotated_box_iou`, `mask_iou`
- `multiclass_nms`, `rotated_nms`

**Recommended approach:** Introduce a lightweight geometry abstraction without over-engineering:

```python
# core/geometry.py -- defines the box representation protocol
class BoxFormat(Enum):
    XYXY = "xyxy"       # (N, 4) axis-aligned
    XYWHA = "xywha"     # (N, 5) rotated
    MASK = "mask"       # (N, H, W) or (N, M) prototypes

# core/iou.py -- dispatches based on format
def compute_iou(boxes1, boxes2, format=BoxFormat.XYXY, ...):
    if format == BoxFormat.XYXY:
        return box_iou(boxes1, boxes2)
    elif format == BoxFormat.XYWHA:
        return rotated_box_iou(boxes1, boxes2)
    ...
```

This avoids a global registry while providing a clean dispatch mechanism. The assigner and NMS can accept a `format` or `iou_fn` parameter.

**Verdict: Moderate extension. Add a geometry dispatch layer; keep functions explicit and avoid registries.**

---

## 4. What NOT to Do

### 4.1. No Global Registries

The current explicit MAP pattern (HEAD_MAP, NECK_MAP, etc.) is superior to MMDetection's `@MODULE_REGISTRY.register()` decorator pattern. Do not introduce:

- Dynamic class registration via decorators.
- String-based class lookup with lazy imports.
- Configuration-driven instantiation that hides the actual class being used.

The explicit maps are small, type-checkable, and IDE-friendly. They can grow to include rotated and segmentation variants without losing clarity.

### 4.2. No Monolithic Base Classes

Do not try to make `SingleStageDetector` handle every paradigm (single-stage, two-stage, transformer, keypoint, segmentation) through flags and conditional logic. This creates an unmaintainable god class. Instead:

- Keep `SingleStageDetector` for anchor-free single-stage detection.
- Add `TwoStageDetector` for RPN + ROI head architectures.
- Add `TransformerDetector` for query-based architectures.
- Share common logic (checkpoint loading, feature extraction) via mixins or utility functions.

### 4.3. No MMDetection-Style Complexity

Avoid:
- Deep inheritance hierarchies with 10+ levels.
- Configuration files that are harder to read than code.
- Build functions that parse nested dicts to instantiate components.
- Runtime type dispatch based on string tags.

The KISS principle means: explicit composition, small maps, callable DI, and clear dataclass contracts.

---

## 5. Summary Matrix

| Component | Rotated Detection | Instance Segmentation | Two-Stage | Transformers | Keypoints |
|---|---|---|---|---|---|
| SingleStageDetector | Holds | Holds | New base class | New base class | Holds |
| DetectionOutput | Add pred_angles or widen pred_boxes | Add pred_masks | Holds | Optional anchor fields | Add pred_keypoints |
| Head (SepBN/YOLOX) | Subclass with reg_dims=5 | Subclass with mask branch | New RPN + ROI heads | Transformer decoder head | Subclass with KP branch |
| det_loss | Add angle_loss param | Add mask_loss param | New two_stage_loss | New detr_loss | Add keypoint_loss param |
| Assigner Protocol | Add iou_fn param | Holds | Reused for ROI stage | Replaced by Hungarian | Holds |
| NMS | Add rotated_nms | Holds (for boxes) | Holds (for proposals) | Not needed | Holds |
| Dataset | Subclass for xywha | Subclass for masks | Holds | Holds | Subclass for keypoints |
| Validation | New evaluator | Extend formatter | Extend formatter | Extend formatter | Extend formatter |
| generic_train | Holds | Holds | Holds | Holds | Holds |
| Build Maps | Extend HEAD_MAP | Extend HEAD_MAP | Extend HEAD_MAP | Extend HEAD_MAP | Extend HEAD_MAP |

---

## 6. Recommended Extension Path

### Phase 1: Rotated Detection (Lowest Risk)

1. Add `rotated_box_iou()` to `core/iou.py`.
2. Add `rotated_nms()` to `core/nms.py`.
3. Add `pred_angles: Optional[torch.Tensor] = None` to `DetectionOutput` (or widen `pred_boxes` to `(B, A, 5)`).
4. Create `RotatedSepBNHead` subclass with `reg_dims=5` and rotated box decoding.
5. Extend `det_loss()` with `angle_loss` and `angle_weight` parameters.
6. Add `iou_fn` parameter to the assigner protocol and SimOTA implementation.
7. Parameterize `postprocess()` in `SingleStageDetector` with `nms_fn`.
8. Create `RotatedDetDataset` for reading rotated COCO annotations.
9. Add rotated evaluator and extend `val_one_epoch()` formatting.
10. Register in HEAD_MAP, ASSIGNER_MAP, BOX_LOSS_MAP.

### Phase 2: Instance Segmentation (Medium Risk)

1. Add `pred_masks: Optional[torch.Tensor] = None` to `DetectionOutput`.
2. Create `MaskSepBNHead` with prototype + coefficient branches.
3. Extend `det_loss()` with `mask_loss` and `mask_weight`.
4. Create `SegmDataset` for reading segmentation annotations.
5. Extend `val_one_epoch()` to format mask predictions.
6. Add mask evaluator.
7. Register in HEAD_MAP.

### Phase 3: Two-Stage & Transformers (Higher Risk, Independent Branches)

1. Create `TwoStageDetector` base class with RPN + ROI head composition.
2. Create `detr_loss()` with Hungarian matching.
3. Create `TransformerDetector` base class.
4. Each can reuse `generic_train()`, `DetectionOutput`, and the DI pattern.

---

## 7. Conclusion

The LucidDet KISS DI architecture is fundamentally sound and scalable. The DI mechanism (explicit maps, callable injection, resolve functions) does not need to change. The Universal Assigner Protocol is well-designed and can be parameterized for different IoU functions. The `generic_train()` runner is paradigm-agnostic.

The primary constraint is the tight coupling to axis-aligned 4-dimensional bounding boxes throughout the output contract, head implementations, loss dispatcher, assigner, NMS, and evaluation pipeline. This coupling can be broken through:

1. **Optional fields** in `DetectionOutput` for new prediction types.
2. **Head subclasses** for task-specific branches (rotated, mask, keypoint).
3. **Optional parameters** in `det_loss()` for additional loss components.
4. **Parameterized assigners** with injectable IoU functions.
5. **New detector base classes** for fundamentally different paradigms (two-stage, transformers).

All of these extensions preserve the KISS DI principle: no global registries, no dynamic class loading, no deep inheritance hierarchies. The architecture can support rotated detection, instance segmentation, keypoints, two-stage detection, and transformer-based detection while remaining clean, type-safe, and maintainable.
