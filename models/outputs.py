"""
Model prediction outputs schema and container classes (dataclass contract).
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple
import torch


@dataclass
class DetectionOutput:
    """
    Standard output contract for single-stage anchor-free detectors.
    Compatible with dictionary indexing and attribute access.
    """

    pred_cls: torch.Tensor
    pred_boxes: torch.Tensor
    anchor_centers: torch.Tensor
    anchor_strides: torch.Tensor
    detections: Optional[List[Dict[str, torch.Tensor]]] = None

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def __setitem__(self, k: str, v: Any) -> None:
        setattr(self, k, v)

    def __contains__(self, k: str) -> bool:
        return hasattr(self, k)

    def get(self, k: str, default: Any = None) -> Any:
        return getattr(self, k, default)

    def keys(self) -> List[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]

    def values(self) -> List[Any]:
        return [getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None]

    def items(self) -> List[Tuple[str, Any]]:
        return [(f.name, getattr(self, f.name)) for f in fields(self) if getattr(self, f.name) is not None]
