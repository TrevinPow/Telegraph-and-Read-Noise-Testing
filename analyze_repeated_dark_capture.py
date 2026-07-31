from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

PATTERN = re.compile(r"_exp(\d+)us_n(\d+)\.tiff$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze repeated dark-frame sets using an external pixel-wise residual reference."
    )
    parser.add_argument("--capture-dir", type=Path, required=True, help="Directory containing the TIFF frames and optional JSON metadata.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for CSV and JSON results; defaults beside the capture directory.")
    parser.add_argument("--frames-per-set", type=int, default=8, help="Number of consecutive frames in each set.")
    parser.add_argument("--z-threshold", type=float, default=4.38, help="Positive-tail Z-score threshold.")
    return parser.parse_args()


args = parse_args()
CAPTURE = args.capture_dir.resolve()
OUT = (args.output_dir or CAPTURE.parent / f"{CAPTURE.name}_analysis").resolve()
OUT.mkdir(parents=True, exist_ok=True)
if args.frames_per_set < 2:
    raise ValueError("--frames-per-set must be at least 2")

files = []
for path in CAPTURE.glob("*.tiff"):
    match = PATTERN.search(path.name)
    if match:
        files.append((int(match.group(1)), int(match.group(2)), path))
files.sort()
if len(files) % args.frames_per_set:
    raise RuntimeError(f"Frame count {len(files)} is not divisible by --frames-per-set")

chunks = [files[start:start + args.frames_per_set] for start in range(0, len(files), args.frames_per_set)]


def read_residual(chunk):
    stack = np.stack([np.asarray(Image.open(path), dtype=np.float32) for _, _, path in chunk])
    return stack - stack.mean(axis=0, keepdims=True)


image_shape = np.asarray(Image.open(files[0][2])).shape
total_sum = np.zeros(image_shape, dtype=np.float64)
total_sum2 = np.zeros(image_shape, dtype=np.float64)
for chunk in chunks:
    residual = read_residual(chunk).astype(np.float64)
    total_sum += residual.sum(axis=0)
    total_sum2 += np.square(residual).sum(axis=0)

rows = []
for set_index, chunk in enumerate(chunks):
    target = read_residual(chunk).astype(np.float64)
    target_sum = target.sum(axis=0)
    target_sum2 = np.square(target).sum(axis=0)
    n_ref = (len(chunks) - 1) * args.frames_per_set
    ref_sum = total_sum - target_sum
    ref_sum2 = total_sum2 - target_sum2
    ref_mean = ref_sum / n_ref
    ref_var = (ref_sum2 - n_ref * np.square(ref_mean)) / (n_ref - 1)
    ref_std = np.sqrt(np.maximum(ref_var, 0.0))
    z = np.divide(target - ref_mean, ref_std, out=np.zeros_like(target), where=ref_std > 0)
    outlier = z > args.z_threshold
    temperatures = []
    for _, _, path in chunk:
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            value = metadata.get("sensor_temp_c")
            if value is None:
                value = metadata.get("selected_settings", {}).get("DeviceTemperature")
            if value is not None:
                temperatures.append(float(value))
    rows.append({
        "set_index": set_index,
        "exposure_us": chunk[0][0],
        "reference_sets": len(chunks) - 1,
        "reference_frames": n_ref,
        "positive_z_gt_4_38_pixel_frames": int(outlier.sum()),
        "positive_z_gt_4_38_fraction": float(outlier.mean()),
        "positive_z_gt_4_38_percentage": float(100 * outlier.mean()),
        "max_positive_z": float(z.max()),
        "sensor_temp_mean_c": float(np.mean(temperatures)) if temperatures else None,
        "sensor_temp_min_c": float(np.min(temperatures)) if temperatures else None,
        "sensor_temp_max_c": float(np.max(temperatures)) if temperatures else None,
    })

with (OUT / "per_set_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

total_outliers = sum(row["positive_z_gt_4_38_pixel_frames"] for row in rows)
total_tested = len(rows) * args.frames_per_set * int(np.prod(image_shape))
summary = {
    "capture_directory": str(CAPTURE),
    "sets": len(rows),
    "frames": len(files),
    "frames_per_set": args.frames_per_set,
    "exposure_us": sorted({exposure for exposure, _, _ in files}),
    "image_shape": list(image_shape),
    "positive_z_gt_threshold_pixel_frames": total_outliers,
    "tested_pixel_frames": int(total_tested),
    "positive_z_gt_threshold_fraction": total_outliers / total_tested,
    "positive_z_gt_threshold_percentage": 100 * total_outliers / total_tested,
    "z_threshold": args.z_threshold,
    "sensor_temperature_c": {
        "min": min(row["sensor_temp_min_c"] for row in rows),
        "max": max(row["sensor_temp_max_c"] for row in rows),
    },
    "method": "Each frame set is converted to pixel-wise residuals by subtracting its temporal mean. Each target set is compared against the residual mean and sample standard deviation from the other sets, with positive Z above the selected threshold counted.",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
