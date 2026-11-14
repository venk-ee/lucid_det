"""
Base dataclasses and schemas weight conversion system.
Provides immutable model presets, checkpoint envelope metadata, and conversion report schemas.
"""

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelWeightPreset:
    """
    Immutable specification of a model weight preset.

    Attributes:
        name: Unique preset identifier (e.g. 'rtmdet_s', 'yolox_s').
        family: Model family name ('rtmdet', 'yolox', 'yolov1').
        arch: Architecture variant/scale ('tiny', 's', 'm', 'l', 'x', 'darknet24').
        builder_fn: Factory builder function name in models.zoo.
        num_classes: Number of detection classes (default: 80 for COCO, 20 for VOC).
        url: Official remote download URL for pretrained weights.
        cache_hints: Local filesystem paths to check before downloading.
        color_space: Image color space expected by the model ('RGB' or 'BGR').
        mean: Normalization channel means.
        std: Normalization channel standard deviations.
        input_size: Default input image resolution (H, W).
        description: Human-readable description of preset.
    """

    name: str
    family: str
    arch: str
    builder_fn: str
    num_classes: int = 80
    url: Optional[str] = None
    cache_hints: Tuple[str, ...] = field(default_factory=tuple)
    color_space: str = "RGB"
    mean: Tuple[float, ...] = (123.675, 116.28, 103.53)
    std: Tuple[float, ...] = (58.395, 57.12, 57.375)
    input_size: Tuple[int, int] = (640, 640)
    description: str = ""


@dataclass
class CheckpointEnvelope:
    """
    Self-describing metadata envelope embedded within saved converted checkpoints.
    Enables tools/infer.py and downstream consumers to automatically adapt preprocessing.
    """

    version: str = "1.0"
    family: str = ""
    arch: str = ""
    num_classes: int = 80
    color_space: str = "RGB"
    mean: Tuple[float, ...] = (123.675, 116.28, 103.53)
    std: Tuple[float, ...] = (58.395, 57.12, 57.375)
    input_size: Tuple[int, int] = (640, 640)
    source: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize envelope to python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointEnvelope":
        """Deserialize envelope from python dictionary, ignoring unknown extra keys."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class ConversionReport:
    """
    Detailed audit report produced after converting a weight preset.
    Tracks converted tensors, strict key matches, forward verification, and timings.
    """

    preset_name: str
    output_path: str
    total_weights_converted: int = 0
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    forward_pass_verified: bool = False
    time_elapsed_sec: float = 0.0
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Returns True if converted cleanly without missing/unexpected keys or errors."""
        return (
            len(self.missing_keys) == 0
            and len(self.unexpected_keys) == 0
            and self.forward_pass_verified
            and self.error_message is None
        )
