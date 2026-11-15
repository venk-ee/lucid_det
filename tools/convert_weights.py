"""
Decoupled, dataclass-based model weight converter CLI.
Converts official pretrained checkpoints (OpenMMLab, Megvii, Darknet) into clean,
pure-PyTorch LucidDet checkpoints with embedded metadata envelopes and weight validation.

Usage:
    uv run python tools/convert_weights.py --list
    uv run python tools/convert_weights.py --preset rtmdet_s
    uv run python tools/convert_weights.py --all
"""

import argparse
import io
import os
import pickle
import sys
import time
import urllib.request
from typing import Any, List, Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import models.zoo as zoo
from weights import (
    CheckpointEnvelope,
    ConversionReport,
    ModelWeightPreset,
    get_converter,
    get_preset,
    list_presets,
)


def safe_load_checkpoint(path: str, device: torch.device = torch.device("cpu")) -> Any:
    """
    Safely load a PyTorch checkpoint from disk, mocking mmengine/mmcv classes
    to allow reading OpenMMLab pickles without mmengine dependencies.
    """

    class SafeUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if any(module.startswith(pkg) for pkg in ("mmengine", "mmcv", "mmdet", "mmyolo")):
                return type(name, (), {})
            return super().find_class(module, name)

    class SafePickleModule:
        @staticmethod
        def load(f: Any, **kwargs: Any) -> Any:
            return SafeUnpickler(f).load()

        @staticmethod
        def loads(data: bytes, **kwargs: Any) -> Any:
            return SafeUnpickler(io.BytesIO(data)).load()

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception:
        # Retry with mock safe pickle module
        return torch.load(
            path, map_location=device, pickle_module=SafePickleModule, weights_only=False
        )


def resolve_weight_source(
    preset: ModelWeightPreset, custom_path: Optional[str], download_dir: str
) -> str:
    """
    Resolve weights source file path.
    Checks:
    1. Custom user path
    2. Preset cache hints
    3. Remote URL download
    """
    if custom_path:
        if not os.path.isfile(custom_path):
            raise FileNotFoundError(f"Specified custom weights file not found: {custom_path}")
        return custom_path

    # Check cache hints
    for hint in preset.cache_hints:
        if os.path.isfile(hint):
            return hint

    # Download from URL if available
    if preset.url:
        os.makedirs(download_dir, exist_ok=True)
        filename = os.path.basename(preset.url.split("?")[0])
        dest_path = os.path.join(download_dir, filename)
        if not os.path.isfile(dest_path):
            print(f"Downloading {preset.name} from {preset.url}...")
            urllib.request.urlretrieve(preset.url, dest_path)
            print(f"Downloaded to {dest_path}")
        return dest_path

    raise FileNotFoundError(
        f"No weights found for preset '{preset.name}'. "
        f"Checked cache hints {preset.cache_hints} and no URL provided."
    )


def convert_and_verify_preset(
    preset: ModelWeightPreset,
    output_dir: str,
    custom_path: Optional[str] = None,
    device_str: str = "cpu",
) -> ConversionReport:
    """Execute complete conversion, strict load verification, and atomic saving for a preset."""
    t0 = time.time()
    out_path = os.path.join(output_dir, f"{preset.name}.pth")
    report = ConversionReport(preset_name=preset.name, output_path=out_path)

    try:
        device = torch.device(device_str)
        download_dir = os.path.join(output_dir, "downloads")
        source_path = resolve_weight_source(preset, custom_path, download_dir)
        print(f"[{preset.name}] Source: {source_path}")

        # 1. Load raw weights
        raw_sd = safe_load_checkpoint(source_path, device=torch.device("cpu"))

        # 2. Convert to LucidDet state dict
        converter = get_converter(preset.family)
        clean_sd = converter(raw_sd, preset)
        report.total_weights_converted = len(clean_sd)

        # 3. Model construction and test strict loading
        builder_fn = getattr(zoo, preset.builder_fn, None)
        if builder_fn is None:
            raise AttributeError(
                f"Model builder function '{preset.builder_fn}' not found in models.zoo"
            )

        if preset.family == "yolov1":
            model = builder_fn(num_classes=preset.num_classes)
        else:
            model = builder_fn(arch=preset.arch, num_classes=preset.num_classes)

        model_sd = model.state_dict()
        missing = [k for k in model_sd if k not in clean_sd]
        unexpected = [k for k in clean_sd if k not in model_sd]
        report.missing_keys = missing
        report.unexpected_keys = unexpected

        if missing or unexpected:
            err_msg = f"Key mismatch: {len(missing)} missing, {len(unexpected)} unexpected."
            report.error_message = err_msg
            print(f"  [WARN] {err_msg}")
            if missing:
                print(f"    Missing sample: {missing[:5]}")
            if unexpected:
                print(f"    Unexpected sample: {unexpected[:5]}")
        else:
            # Strict load
            model.load_state_dict(clean_sd, strict=True)

            # 4. Model forward pass verification
            model.to(device)
            model.eval()
            h, w = preset.input_size
            dummy_in = torch.randn(1, 3, h, w, device=device)
            with torch.no_grad():
                out = model(dummy_in, training_override=False)
            assert out is not None, "Model forward pass returned None"
            report.forward_pass_verified = True

            # 5. Atomic checkpoint save with metadata envelope
            envelope = CheckpointEnvelope(
                version="1.0",
                family=preset.family,
                arch=preset.arch,
                num_classes=preset.num_classes,
                color_space=preset.color_space,
                mean=preset.mean,
                std=preset.std,
                input_size=preset.input_size,
                source=source_path,
            )

            os.makedirs(output_dir, exist_ok=True)
            tmp_path = f"{out_path}.tmp"
            torch.save(
                {
                    "model": clean_sd,
                    "meta": envelope.to_dict(),
                },
                tmp_path,
            )
            os.replace(tmp_path, out_path)
            print(f"  [OK] Saved verified checkpoint to {out_path}")

    except Exception as e:
        report.error_message = str(e)
        print(f"  [FAIL] Error during conversion: {e}")

    report.time_elapsed_sec = time.time() - t0
    return report


def print_presets_table(presets: List[ModelWeightPreset]) -> None:
    """Print registered presets in a clean table format."""
    print("=" * 90)
    print(
        f"{'PRESET':<16} {'FAMILY':<10} {'ARCH':<8} {'CLASSES':<8} {'INPUT':<10} {'CACHE / URL':<35}"
    )
    print("=" * 90)
    for p in presets:
        cache_status = (
            "Available"
            if any(os.path.isfile(h) for h in p.cache_hints)
            else ("URL" if p.url else "None")
        )
        inp_str = f"{p.input_size[0]}x{p.input_size[1]}"
        print(
            f"{p.name:<16} {p.family:<10} {p.arch:<8} {p.num_classes:<8} {inp_str:<10} {cache_status:<35}"
        )
    print("=" * 90)


def print_reports_summary(reports: List[ConversionReport]) -> None:
    """Print tabular conversion audit summary."""
    print("\n" + "=" * 90)
    print("WEIGHT CONVERSION SUMMARY REPORT")
    print("=" * 90)
    print(
        f"{'PRESET':<16} {'CONVERTED':<12} {'MISSING':<10} {'UNEXPECTED':<12} {'FORWARD':<10} {'TIME (s)':<10} {'STATUS':<8}"
    )
    print("-" * 90)
    for r in reports:
        status = "PASS" if r.is_success else "FAIL"
        fwd_str = "OK" if r.forward_pass_verified else "FAIL"
        print(
            f"{r.preset_name:<16} {r.total_weights_converted:<12} {len(r.missing_keys):<10} "
            f"{len(r.unexpected_keys):<12} {fwd_str:<10} {r.time_elapsed_sec:<10.2f} {status:<8}"
        )
    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LucidDet Weight Converter CLI")
    parser.add_argument(
        "--list", action="store_true", help="List all registered model weight presets"
    )
    parser.add_argument(
        "--preset", type=str, default=None, help="Preset name to convert (e.g. rtmdet_s)"
    )
    parser.add_argument("--all", action="store_true", help="Convert all available presets")
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="Optional custom weights file override"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for verification forward pass",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print_presets_table(list_presets())
        return

    presets_to_convert: List[ModelWeightPreset] = []
    if args.all:
        presets_to_convert = list_presets()
    elif args.preset:
        presets_to_convert = [get_preset(args.preset)]
    else:
        print("Error: Specify --preset <name>, --all, or --list.")
        sys.exit(1)

    reports: List[ConversionReport] = []
    for preset in presets_to_convert:
        report = convert_and_verify_preset(
            preset=preset,
            output_dir=args.output_dir,
            custom_path=args.weights if len(presets_to_convert) == 1 else None,
            device_str=args.device,
        )
        reports.append(report)

    print_reports_summary(reports)

    all_passed = all(r.is_success for r in reports)
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
