from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


BASE = Path(__file__).resolve().parent
ROOT = BASE / "analysis_work_emva1288_10V_maxcool" / "emva1288_10V_maxcool"
OUT = BASE / "imx541_emva1288_10V_maxcool_external_reference"
OUT.mkdir(exist_ok=True)
PATTERN = re.compile(r"_(bright|dark)_sn[^_]+_exp(\d+)us_n(\d+)\.tiff$")

groups: dict[tuple[str, str], list[tuple[int, Path]]] = {}
for path in ROOT.rglob("*.tiff"):
    match = PATTERN.search(path.name)
    if match:
        category, exposure, index = match.group(1), match.group(2), int(match.group(3))
        groups.setdefault((category, exposure), []).append((index, path))

ordered_groups: dict[str, list[tuple[str, list[tuple[int, Path]]]]] = {"dark": [], "bright": []}
for (category, exposure), members in sorted(groups.items(), key=lambda x: (x[0][0], int(x[0][1]))):
    members.sort()
    if [index for index, _ in members] != list(range(8)):
        raise RuntimeError(f"Expected n00..n07 for {(category, exposure)}")
    ordered_groups[category].append((exposure, members))


def read_stack(members: list[tuple[int, Path]]) -> np.ndarray:
    return np.stack([np.asarray(Image.open(path), dtype=np.float32) for _, path in members])


def residuals(stack: np.ndarray) -> np.ndarray:
    return stack - stack.mean(axis=0, keepdims=True)


all_rows = []
for category in ("dark", "bright"):
    category_groups = ordered_groups[category]
    first_stack = read_stack(category_groups[0][1])
    shape = first_stack.shape[1:]
    total_frames = len(category_groups) * 8
    total_sum = np.zeros(shape, dtype=np.float64)
    total_sum2 = np.zeros(shape, dtype=np.float64)

    for _, members in category_groups:
        r = residuals(read_stack(members)).astype(np.float64)
        total_sum += r.sum(axis=0)
        total_sum2 += np.square(r).sum(axis=0)

    for exposure, members in category_groups:
        stack = read_stack(members)
        r = residuals(stack).astype(np.float64)
        n_ref = total_frames - 8
        ref_sum = total_sum - r.sum(axis=0)
        ref_sum2 = total_sum2 - np.square(r).sum(axis=0)
        ref_mean = ref_sum / n_ref
        ref_var = (ref_sum2 - n_ref * np.square(ref_mean)) / (n_ref - 1)
        ref_std = np.sqrt(np.maximum(ref_var, 0.0))
        z = np.divide(r - ref_mean, ref_std, out=np.zeros_like(r), where=ref_std > 0)
        outlier = z > 4.38
        rows_by_frame = outlier.sum(axis=(1, 2))
        fraction_by_frame = outlier.mean(axis=(1, 2))
        all_rows.append({
            "category": category,
            "exposure_us": int(exposure),
            "n_frames": 8,
            "reference_sets": len(category_groups) - 1,
            "reference_frames": n_ref,
            "image_mean_adu": float(stack.mean()),
            "external_reference_mean_residual_adu": float(ref_mean.mean()),
            "external_reference_std_median_adu": float(np.median(ref_std)),
            "external_reference_std_mean_adu": float(ref_std.mean()),
            "external_reference_std_p99_adu": float(np.percentile(ref_std, 99)),
            "positive_z_gt_4_38_pixel_frames": int(outlier.sum()),
            "positive_z_gt_4_38_fraction": float(outlier.mean()),
            "max_positive_z": float(z.max()),
            "frames_with_positive_outlier": int((rows_by_frame > 0).sum()),
            "max_frame_outlier_fraction": float(fraction_by_frame.max()),
        })

with (OUT / "per_set_external_reference_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)

aggregate = {}
for category in ("dark", "bright"):
    rows = [row for row in all_rows if row["category"] == category]
    total_outliers = sum(row["positive_z_gt_4_38_pixel_frames"] for row in rows)
    total_tested = sum(row["n_frames"] * 1128 * 1128 for row in rows)
    aggregate[category] = {
        "sets": len(rows),
        "tested_pixel_frames": total_tested,
        "positive_z_gt_4_38_pixel_frames": total_outliers,
        "positive_z_gt_4_38_fraction": total_outliers / total_tested,
        "mean_per_set_fraction": float(np.mean([row["positive_z_gt_4_38_fraction"] for row in rows])),
        "median_per_set_fraction": float(np.median([row["positive_z_gt_4_38_fraction"] for row in rows])),
    }

summary = {
    "zip": "emva1288_10V_maxcool.zip",
    "z_threshold": 4.38,
    "reference_method": "For each category, each target set is compared against the pixel-wise residual distribution from all other exposure sets in that category. Each set residual is frame minus that set's pixel-wise 8-frame mean, so exposure-dependent brightness is removed before pooling.",
    "reference_standard_deviation": "Sample standard deviation of the other-set residual frames, with leave-one-set-out exclusion.",
    "image_shape": [1128, 1128],
    "aggregate": aggregate,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
