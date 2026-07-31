from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent / "analysis_work_emva1288_10V_maxcool" / "emva1288_10V_maxcool"
OUT = Path(__file__).resolve().parent / "imx541_emva1288_10V_maxcool_diagnostic"
OUT.mkdir(exist_ok=True)

pattern = re.compile(r"_(bright|dark)_sn[^_]+_exp(\d+)us_n(\d+)\.tiff$")
groups: dict[tuple[str, str], list[tuple[int, Path]]] = {}
for path in ROOT.rglob("*.tiff"):
    m = pattern.search(path.name)
    if not m:
        continue
    category, exposure, index = m.group(1), m.group(2), int(m.group(3))
    groups.setdefault((category, exposure), []).append((index, path))

rows = []
shape = None
dtype = None
for (category, exposure), members in sorted(groups.items()):
    members.sort()
    if [i for i, _ in members] != list(range(8)):
        raise RuntimeError(f"Expected n00..n07 for {(category, exposure)}, got {[i for i, _ in members]}")
    stack = np.stack([np.asarray(Image.open(path)) for _, path in members]).astype(np.float64)
    if shape is None:
        shape = stack.shape[1:]
        dtype = np.asarray(Image.open(members[0][1])).dtype.name
    if stack.shape[1:] != shape:
        raise RuntimeError(f"Inconsistent image shape: {stack.shape[1:]} vs {shape}")

    mean_image = stack.mean(axis=0)
    std_image = stack.std(axis=0, ddof=0)
    nonzero = std_image > 0
    z = np.zeros_like(stack)
    z = np.divide(stack - mean_image, std_image, out=z, where=np.broadcast_to(nonzero, stack.shape))
    positive_outliers = z > 4.38
    rows.append({
        "category": category,
        "exposure_us": int(exposure),
        "n_frames": 8,
        "height": int(shape[0]),
        "width": int(shape[1]),
        "image_mean_adu": float(stack.mean()),
        "image_std_adu": float(stack.std()),
        "temporal_std_median_adu": float(np.median(std_image)),
        "temporal_std_mean_adu": float(std_image.mean()),
        "temporal_std_p99_adu": float(np.percentile(std_image, 99)),
        "positive_z_gt_4_38_count": int(positive_outliers.sum()),
        "positive_z_gt_4_38_fraction": float(positive_outliers.mean()),
        "zero_temporal_std_pixel_count": int((~nonzero).sum()),
    })

with (OUT / "per_set_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "zip": "emva1288_10V_maxcool.zip",
    "sets": len(rows),
    "sets_by_category": {cat: sum(r["category"] == cat for r in rows) for cat in ("dark", "bright")},
    "frames_per_set": 8,
    "image_shape": shape,
    "image_dtype": dtype,
    "z_threshold": 4.38,
    "z_definition": "(frame - mean of the same 8-frame set) / population std of the same set, positive tail only",
    "total_positive_outliers": int(sum(r["positive_z_gt_4_38_count"] for r in rows)),
    "total_tested_pixel_frames": int(sum(r["n_frames"] * r["height"] * r["width"] for r in rows)),
    "note": "This literal within-set Z calculation cannot produce z > sqrt(8-1)=2.646 for finite nonzero stacks, so a 4.38 threshold requires an external reference mean/std or a specified leave-one-out/reference-window method.",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
